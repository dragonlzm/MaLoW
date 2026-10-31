"""Portable launcher for the paper's MaLoW video evaluation preset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from malow.video.interactive_options import add_interactive_arguments, resolve_interactive_arguments, PIXEL_OPTIONS
from malow.checkpoints import FORMAT, FORMAT_VERSION, METADATA_FILENAME

OBSERVATION_PRESETS = {"last2s": (2.0, 2), "last4s": (4.0, 4), "last8s": (8.0, 8)}


def resolve_observation_preset(checkpoint, preset="auto"):
    """Read only observation scalars; never load weights or reconstruct configs.

    Legacy release metadata omits base_tail_seconds, which means two seconds.
    A preset explicitly chosen by the caller overrides the checkpoint settings.
    """
    if preset != "auto":
        if preset not in OBSERVATION_PRESETS:
            raise ValueError(f"Unknown observation preset: {preset}")
        return preset
    path = Path(checkpoint).expanduser() / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Automatic preset selection requires {path}. Download the checkpoint first, "
            "or choose --preset last2s, --preset last4s, or --preset last8s explicitly for a command-only dry run."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT or document.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported MaLoW metadata format/version in {path}")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be an object in {path}")
    seconds = metadata.get("base_tail_seconds", 2.0)
    frames = metadata.get("base_input_frame_number", 2)
    if type(seconds) in (int, float) and type(frames) is int:
        for name, values in OBSERVATION_PRESETS.items():
            if (seconds, frames) == values:
                return name
    raise ValueError(
        f"Unsupported observation settings in {path}: base_tail_seconds={seconds!r}, "
        f"base_input_frame_number={frames!r}. Expected 2 seconds/2 frames, 4 seconds/4 frames, or 8 seconds/8 frames; "
        "use --preset last2s, --preset last4s, or --preset last8s only to deliberately override these settings."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ovo", "streaming"), required=True)
    parser.add_argument("--checkpoint", required=True, help="Directory downloaded with malow.download.")
    parser.add_argument("--base-model", required=True, help="Matching Qwen3.5 or Qwen3-VL HF ID or local directory.")
    parser.add_argument("--preset", choices=("auto", *OBSERVATION_PRESETS), default="auto",
                        help="Observation window: auto reads checkpoint metadata (legacy default 2s/2 frames); explicit presets override it.")
    parser.add_argument("--data-root", required=True, help="OVO parquet or StreamingBench JSON annotation root.")
    parser.add_argument("--video-root", help="OVO chunked_videos or StreamingBench videos directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=1, help="Data-parallel GPU workers; reference runs use 4.")
    parser.add_argument("--limit", type=int, help="Per-task smoke-test entry limit; not a full benchmark score.")
    parser.add_argument("--offline", action="store_true", help="Require models/data to already be downloaded.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without loading models or writing results.")
    add_interactive_arguments(parser)
    args = parser.parse_args(argv)
    resolve_interactive_arguments(parser, args)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.benchmark == "ovo" and not args.video_root:
        parser.error("OVO requires --video-root pointing to chunked_videos")
    return args


def build_command(args):
    preset = resolve_observation_preset(args.checkpoint, args.preset)
    tail_seconds, base_frames = OBSERVATION_PRESETS[preset]
    cmd = [sys.executable]
    if args.num_processes > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={args.num_processes}", "--module", f"malow.video.{args.benchmark}"]
    else:
        cmd += ["-m", f"malow.video.{args.benchmark}"]
    cmd += ["--model-id", args.base_model, "--c2l-checkpoint", args.checkpoint,
            "--output-dir", args.output_dir, "--batch-size", "1", "--dtype", "bfloat16",
            "--device-map", "auto", "--vision-encoder-only", "--num-frames", "766", "--max-frames", "766",
            "--video-fps", "1.0", "--video-min-pixels", "1024", "--video-max-pixels", "262144",
            "--video-total-pixels", "0", "--temperature", "0", "--system-prompt", "",
            "--enable-thinking", "false", "--block-lora", "true", "--frame-per-lora", "2",
            "--base-input-video", "--base-input-frame-number", str(base_frames), "--base-with-last-2seconds",
            "--base-tail-seconds", str(tail_seconds), "--user-defined-scaling", "1.0"]
    if args.benchmark == "ovo":
        cmd += ["--tasks", "ovobench", "--annotation-root", args.data_root,
                "--video-root", args.video_root, "--no-auto-download-videos", "--max-new-tokens", "128", "--num-beams", "1"]
    else:
        cmd += ["--model-family", "auto", "--trust-remote-code", "--data-root", args.data_root,
                "--context-time", "-1", "--max-new-tokens", "16", "--proactive-step", "1", "--proactive-extra-seconds", "4"]
        for task in ("real", "omni", "sqa", "proactive"):
            cmd += ["--task", task]
    if args.eval_mode != "offline":
        cmd += ["--eval-mode", args.eval_mode,
                "--interactive-verify-sessions", str(args.interactive_verify_sessions)]
        if args.rollout_step_seconds is not None:
            cmd += ["--rollout-step-seconds", str(args.rollout_step_seconds)]
        if args.interactive_max_segment_frames is not None:
            cmd += ["--interactive-max-segment-frames", str(args.interactive_max_segment_frames)]
    for name in PIXEL_OPTIONS:
        if args.eval_mode != "offline" or getattr(args, name):
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for flag in ("text_summary", "checkpoint_progress", "streaming_group_by_video", "text_cache_base_only"):
        if getattr(args, flag, False):
            cmd += ["--" + flag.replace("_", "-")]
    for name in ("text_cache_source", "text_cache_mode", "text_cache_miss"):
        if hasattr(args, name):
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return cmd


def main(argv=None):
    args = parse_args(argv)
    cmd = build_command(args)
    env = os.environ.copy()
    env.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.video_root and args.benchmark == "streaming":
        env["MALOW_VIDEO_ROOT"] = str(Path(args.video_root).expanduser().resolve())
    if args.dry_run:
        if env.get("MALOW_VIDEO_ROOT"):
            print("MALOW_VIDEO_ROOT=" + shlex.quote(env["MALOW_VIDEO_ROOT"]))
        if args.offline:
            print("HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
        print(shlex.join(cmd))
        return 0
    for value, label in ((args.checkpoint, "checkpoint"), (args.data_root, "data root")):
        if not Path(value).expanduser().is_dir():
            raise FileNotFoundError(f"Missing {label}: {value}")
    if args.video_root and not Path(args.video_root).expanduser().is_dir():
        raise FileNotFoundError(f"Missing video root: {args.video_root}")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
