"""Shared, dependency-light interactive evaluation options and run identity."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

PIXEL_OPTIONS = ("encoder_video_max_pixels", "encoder_video_longest_edge", "base_video_max_pixels")


def add_interactive_arguments(parser):
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

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in
              {"output_dir", "progress_ranks", "dist_barrier_timeout_seconds"}}
    config.update(world_size=world_size, interactive_sample_fps=1.0, interactive_format_version=1)
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
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(config, indent=2) + "\n")
            temporary.replace(manifest)


def init_resolution_log(args, rank):
    if args.eval_mode != "offline":
        args._resolution_log = str(Path(args.output_dir) / f"resolution_rank_{rank:05d}.jsonl")
        args._verification_log = str(Path(args.output_dir) / f"verification_rank_{rank:05d}.jsonl")


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
