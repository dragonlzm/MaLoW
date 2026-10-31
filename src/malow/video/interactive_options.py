"""Shared, dependency-light interactive evaluation options and run identity."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

PIXEL_OPTIONS = ("encoder_video_max_pixels", "encoder_video_longest_edge", "base_video_max_pixels")


def add_interactive_arguments(parser):
    import argparse
    parser.add_argument("--checkpoint-progress", action="store_true", default=argparse.SUPPRESS,
                        help="Durably save answers and text updates; reuse them when rerunning the same output.")
    parser.add_argument("--streaming-group-by-video", action="store_true", default=argparse.SUPPRESS,
                        help="Group StreamingBench questions across original annotation entries.")
    parser.add_argument("--text-cache-source", default=argparse.SUPPRESS,
                        help="Read-only exported caption/summary bundle; write fresh answers to output-dir.")
    parser.add_argument("--text-cache-mode", choices=("summaries", "captions"), default=argparse.SUPPRESS)
    parser.add_argument("--text-cache-miss", choices=("error", "generate"), default=argparse.SUPPRESS)
    parser.add_argument("--text-cache-base-only", action="store_true", default=argparse.SUPPRESS,
                        help="Answer with unadapted Qwen, cached text, and the current two-second observation.")
    parser.add_argument("--text-summary", action="store_true", default=argparse.SUPPRESS,
                        help="Augment fixed-gap LoRA answers with bounded base-model visual summaries.")
    parser.add_argument("--eval-mode", choices=("offline", "interactive", "fixed-gap"), default="offline",
                        help="Full-prefix evaluation (default), question-driven memory, or fixed-gap memory.")
    parser.add_argument("--rollout-step-seconds", type=float, default=None,
                        help="Fixed-gap video-clock interval (default: 2 seconds); questions also catch up.")
    parser.add_argument("--interactive-max-segment-frames", type=int, default=None,
                        help="Maximum frames per encoder forward; defaults to checkpoint micro cap or 766.")
    parser.add_argument("--interactive-verify-sessions", type=int, default=2,
                        help="Verify this many sessions per worker against identical-frame single passes; 0 disables.")
    parser.add_argument("--encoder-video-max-pixels", type=int, default=None,
                        help="Per-frame pre-cap; interactive default 262144, offline 0 (disabled).")
    parser.add_argument("--encoder-video-longest-edge", type=int, default=None,
                        help="Encoder whole-video pixel budget; interactive default 1000000000, offline 0 (processor default).")
    parser.add_argument("--base-video-max-pixels", type=int, default=None,
                        help="Base-tail per-frame pre-cap; default 0 (disabled).")


def resolve_interactive_arguments(parser, args, *, worker=False):
    active = args.eval_mode != "offline"
    source = getattr(args, "text_cache_source", None)
    if source:
        if not getattr(args, "text_summary", False) or not getattr(args, "checkpoint_progress", False):
            parser.error("--text-cache-source requires --text-summary and --checkpoint-progress")
        args.text_cache_source = str(Path(source).resolve())
        if not hasattr(args, "text_cache_mode"): args.text_cache_mode = "summaries"
        if not hasattr(args, "text_cache_miss"): args.text_cache_miss = "error"
    elif any(hasattr(args, name) for name in ("text_cache_mode", "text_cache_miss", "text_cache_base_only")):
        parser.error("Text-cache options require --text-cache-source")
    if getattr(args, "text_cache_base_only", False):
        args.interactive_verify_sessions = 0
        if worker and (args.base_tail_seconds != 2 or args.base_input_frame_number != 2):
            parser.error("Base-only replay requires the last two seconds / two frames")
    if (getattr(args, "checkpoint_progress", False) or getattr(args, "streaming_group_by_video", False)) and not active:
        parser.error("Progress recovery and video grouping require interactive or fixed-gap evaluation")
    if getattr(args, "streaming_group_by_video", False) and (getattr(args, "benchmark", None) == "ovo" or hasattr(args, "annotation_root")):
        parser.error("--streaming-group-by-video is only supported by StreamingBench")
    if getattr(args, "text_summary", False) and args.eval_mode != "fixed-gap":
        parser.error("--text-summary requires --eval-mode fixed-gap")
    if args.rollout_step_seconds is not None:
        if not math.isfinite(args.rollout_step_seconds) or args.rollout_step_seconds <= 0:
            parser.error("--rollout-step-seconds must be finite and positive")
        if args.eval_mode != "fixed-gap":
            parser.error("--rollout-step-seconds requires --eval-mode fixed-gap")
    if args.eval_mode == "fixed-gap" and args.rollout_step_seconds is None:
        args.rollout_step_seconds = 2.0
    if args.interactive_max_segment_frames is not None and args.interactive_max_segment_frames < 2:
        parser.error("--interactive-max-segment-frames must allow at least one two-frame block")
    if args.interactive_verify_sessions < 0:
        parser.error("--interactive-verify-sessions must be nonnegative")
    for name, default in zip(PIXEL_OPTIONS, (262144 if active else 0, 1000000000 if active else 0, 0)):
        value = getattr(args, name)
        if value is None:
            setattr(args, name, default)
        elif value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if worker and active:
        if not args.c2l_checkpoint or args.no_video:
            parser.error("Interactive evaluation requires a video C2L checkpoint and video input")
        if not args.base_input_video or not args.base_with_last_2seconds:
            parser.error("Interactive evaluation requires --base-input-video --base-with-last-2seconds")
        if args.batch_size != 1:
            parser.error("Interactive evaluation requires --batch-size 1 (one video per session)")
        if args.frame_per_lora not in (None, 2):
            parser.error("Interactive evaluation requires two-frame temporal blocks")
        if getattr(args, "context_time", -1) > 0:
            parser.error("Interactive sessions require full history; --context-time must be <= 0")
        if getattr(args, "use_streamingbench_frame_sampling", False):
            parser.error("Interactive evaluation uses its own fixed 1 FPS grid")
    return args


def guard_interactive_run(args, world_size):
    """Lock and compare configuration before appending any interactive predictions."""
    if args.eval_mode == "offline":
        return
    import fcntl
    if getattr(args, "text_cache_source", None):
        from malow.video.text_cache import init_text_cache
        init_text_cache(args)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in
              {"output_dir", "progress_ranks", "dist_barrier_timeout_seconds"}}
    if getattr(args, "text_cache_source", None):
        config["external_text_cache_sha256"] = args._text_cache_identity
    config.update(world_size=world_size, interactive_sample_fps=1.0, interactive_format_version=1)
    if getattr(args, "text_summary", False):
        from malow.video.text_summary import experiment_identity
        config["text_summary_settings"] = experiment_identity()
    if getattr(args, "streaming_group_by_video", False):
        config["streaming_grouping_version"] = 1
    if getattr(args, "checkpoint_progress", False):
        from malow.video.progress import FORMAT_VERSION, file_identity
        config["recovery_version"] = FORMAT_VERSION
        sources = ("progress.py", "interactive.py", "interactive_options.py", "text_summary.py",
                   "ovo.py" if hasattr(args, "annotation_root") else "streaming.py")
        if getattr(args, "text_cache_source", None):
            sources += ("text_cache.py",)
        config["recovery_implementation_sha256"] = hashlib.sha256(
            b"".join((Path(__file__).parent / name).read_bytes() for name in sources)).hexdigest()
        annotation_root = Path(getattr(args, "annotation_root", None) or args.data_root)
        config["annotation_fingerprints"] = {
            str(p.relative_to(annotation_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(annotation_root.rglob("*")) if p.suffix in (".json", ".parquet")}
        config["model_file_identities"] = [file_identity(p)
            for directory in (Path(args.model_id), Path(args.c2l_checkpoint))
            for p in sorted(directory.glob("*")) if p.suffix in (".json", ".safetensors", ".py")]
    config["video_environment"] = {name: os.environ.get(name, "") for name in (
        "MALOW_VIDEO_ROOT", "STREAMINGBENCH_VIDEO_PATH_PREFIX_MAP", "FORCE_QWENVL_VIDEO_READER") }
    metadata = Path(args.c2l_checkpoint) / "c2l_metadata.json"
    if metadata.is_file():
        config["checkpoint_metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    config["checkpoint_files"] = {
        name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for name in ("model.safetensors", "c2l_metadata.safetensors")
        if (path := Path(args.c2l_checkpoint) / name).is_file()
    }
    manifest = root / "interactive_run_config.json"
    with (root / ".interactive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved != config:
                changed = sorted(k for k in saved.keys() | config.keys() if saved.get(k) != config.get(k))
                raise ValueError(f"Incompatible interactive output directory ({', '.join(changed)}). Use a fresh --output-dir.")
        else:
            if any(root.rglob("*.jsonl")) or (root / "summary.json").exists() or (root / "run_config.json").exists():
                raise ValueError("Existing output has no interactive run identity. Use a fresh --output-dir.")
            if getattr(args, "checkpoint_progress", False):
                from malow.video.progress import atomic_json
                atomic_json(manifest, config)
            else:
                temporary = manifest.with_suffix(".tmp")
                temporary.write_text(json.dumps(config, indent=2) + "\n")
                temporary.replace(manifest)


def init_resolution_log(args, rank):
    if getattr(args, "checkpoint_progress", False):
        import fcntl
        from malow.video.progress import read_journal
        lock = (Path(args.output_dir) / f".recovery-rank-{rank}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"Another recovery worker owns rank {rank}")
        args._progress_lock = lock
        for prefix in ("resolution", "verification", "text_summary"):
            read_journal(Path(args.output_dir) / f"{prefix}_rank_{rank:05d}.jsonl")
    if args.eval_mode != "offline":
        args._resolution_log = str(Path(args.output_dir) / f"resolution_rank_{rank:05d}.jsonl")
        args._verification_log = str(Path(args.output_dir) / f"verification_rank_{rank:05d}.jsonl")
        if getattr(args, "text_summary", False):
            args._text_summary_log = str(Path(args.output_dir) / f"text_summary_rank_{rank:05d}.jsonl")


def write_interactive_report(args, rank, sessions, task=None):
    from malow.video.interactive import summarize_verifications, verification_verdict, enforce_verification
    from dataclasses import asdict
    summary = summarize_verifications(sessions)
    verdict = verification_verdict(summary) if args.interactive_verify_sessions else "DISABLED: interactive verification was not requested."
    report = {
        "eval_mode": args.eval_mode, "sessions": len(sessions),
        "blocks_consumed": sum(s.consumed_blocks for s in sessions),
        "segments": sum(s.segments for s in sessions),
        "turns": sum(s.turns for s in sessions),
        "fallback_turns": sum(s.fallback_turns for s in sessions),
        "verification": summary, "verdict": verdict,
        "resolved_sampling": asdict(sessions[0].config) if sessions else None,
    }
    root = Path(args.output_dir) / task if task else Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"interactive_rank_{rank:05d}.json").write_text(json.dumps(report, indent=2) + "\n")
    if verdict:
        print(verdict, flush=True)
    enforce_verification(summary, verdict)


def select_verification_session(args):
    """Share the worker's verification allowance across StreamingBench tasks."""
    remaining = getattr(args, "_verification_remaining", args.interactive_verify_sessions)
    args._verification_remaining = max(0, remaining - 1)
    return remaining > 0
