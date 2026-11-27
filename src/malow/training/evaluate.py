"""Evaluate a training checkpoint with the unchanged public video protocols."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    for benchmark in ("ovo", "streaming"):
        parser.add_argument(f"--{benchmark}-data-root", required=True)
        parser.add_argument(f"--{benchmark}-video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Limit each evaluator for a GPU smoke run.")
    parser.add_argument("--offline", action="store_true", help="Disable network model downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Print the six commands without executing them.")
    args = parser.parse_args(argv)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def build_commands(args):
    """Keep all model, sampling, and scoring choices in malow.video.eval."""
    commands = []
    for benchmark in ("ovo", "streaming"):
        for mode in ("offline", "interactive", "fixed-gap"):
            suffix = "fixed-gap-2s" if mode == "fixed-gap" else mode
            command = [
                sys.executable, "-m", "malow.video.eval",
                "--benchmark", benchmark, "--eval-mode", mode,
                "--checkpoint", args.checkpoint, "--base-model", args.base_model,
                "--preset", "auto",
                "--data-root", getattr(args, f"{benchmark}_data_root"),
                "--video-root", getattr(args, f"{benchmark}_video_root"),
                "--output-dir", str(Path(args.output_dir) / f"{benchmark}-{suffix}"),
                "--num-processes", str(args.num_processes),
            ]
            if mode == "fixed-gap":
                command += ["--rollout-step-seconds", "2"]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            if args.offline:
                command += ["--offline"]
            commands.append(command)
    return commands


def main(argv=None):
    args = parse_args(argv)
    for command in build_commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
