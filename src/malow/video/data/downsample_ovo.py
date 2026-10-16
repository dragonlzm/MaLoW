#!/usr/bin/env python3
"""Process-pool temporal video downsampling with progress reporting.

Defaults:
- inputs: /media/user/easystore/huggingface/videomme and
  /media/user/easystore/huggingface/ovobench
- output: /media/user/easystore/huggingface/videomme-2fps-min64
- target rate: 2 FPS
- minimum frames: 64

Videos with fewer than --min-frames source frames are copied unchanged.
sample usage: python dataset_preparation/downsample_videos_temporal_parallel.py --workers 4 --ffmpeg-threads 1

"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

DEFAULT_VIDEOMME_ROOT = Path("/media/user/easystore/huggingface/videomme")
DEFAULT_OVOBENCH_ROOT = Path("/media/user/easystore/huggingface/ovobench")
DEFAULT_OUTPUT_ROOT = Path("/media/user/easystore/huggingface/videomme-2fps-min64")


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration: float
    frames: int | None
    codec: str


@dataclass(frozen=True)
class VideoJob:
    input_path: str
    input_root: str
    output_root: str
    output_prefix: str


@dataclass(frozen=True)
class WorkerConfig:
    target_fps: float
    min_frames: int
    ffmpeg_threads: int
    crf: int
    preset: str
    ffmpeg: str
    ffprobe: str
    overwrite: bool
    dry_run: bool
    output_extension: str | None = None
    audio_mode: str = "copy"


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    action: str
    reason: str
    source_fps: float | None = None
    target_fps: float | None = None
    source_frames: int | None = None
    source_duration: float | None = None
    width: int | None = None
    height: int | None = None


def default_input_roots() -> tuple[Path, ...]:
    return (DEFAULT_VIDEOMME_ROOT, DEFAULT_OVOBENCH_ROOT)


def parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" not in rate:
        return float(rate)
    numerator, denominator = rate.split("/", 1)
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return 0.0
    return float(numerator) / denominator_float


def ffprobe(video_path: Path, ffprobe_bin: str) -> VideoMetadata:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe failed with code {completed.returncode}")

    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video stream")

    stream = streams[0]
    fmt = payload.get("format") or {}
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    raw_frames = stream.get("nb_frames")
    frames = int(raw_frames) if raw_frames and str(raw_frames).isdigit() else None
    if frames is None and fps > 0.0 and duration > 0.0:
        frames = max(1, int(round(fps * duration)))

    return VideoMetadata(
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        duration=duration,
        frames=frames,
        codec=stream.get("codec_name") or "",
    )


def is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def normalize_roots(paths: Iterable[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return roots


def output_prefix_for(input_root: Path, input_roots: list[Path]) -> Path:
    ancestors = [
        candidate
        for candidate in input_roots
        if candidate != input_root and is_relative_to(input_root, candidate)
    ]
    if ancestors:
        nearest_ancestor = max(ancestors, key=lambda path: len(path.parts))
        return input_root.relative_to(nearest_ancestor)

    if len(input_roots) > 1 and input_root != input_roots[0]:
        return Path(input_root.name)

    return Path()


def nested_roots_for(input_root: Path, input_roots: list[Path]) -> list[Path]:
    return [
        candidate
        for candidate in input_roots
        if candidate != input_root and is_relative_to(candidate, input_root)
    ]


def iter_videos(
    input_root: Path,
    output_root: Path,
    extensions: set[str],
    excluded_roots: Iterable[Path] = (),
) -> Iterable[Path]:
    output_root_resolved = output_root.resolve()
    excluded_roots_resolved = [
        root.resolve()
        for root in excluded_roots
        if root.resolve() != input_root.resolve()
    ]
    for root, dirnames, filenames in os.walk(input_root):
        root_path = Path(root)
        kept_dirnames = []
        for dirname in dirnames:
            child = (root_path / dirname).resolve()
            if child == output_root_resolved or output_root_resolved in child.parents:
                continue
            if any(is_relative_to(child, excluded_root) for excluded_root in excluded_roots_resolved):
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() in extensions:
                yield path


def choose_target_fps(metadata: VideoMetadata, target_fps: float, min_frames: int) -> tuple[float | None, str]:
    if metadata.frames is not None and metadata.frames < min_frames:
        return None, f"copy original: source has fewer than {min_frames} frames"

    if metadata.fps <= 0.0 or metadata.duration <= 0.0:
        return None, "missing fps or duration metadata"

    min_frame_fps = min_frames / metadata.duration
    desired_fps = max(target_fps, min_frame_fps)
    desired_fps = min(metadata.fps, desired_fps)
    expected_frames = int(math.floor(metadata.duration * desired_fps + 1e-6))

    if desired_fps >= metadata.fps or expected_frames >= (metadata.frames or expected_frames):
        return None, "source fps is already at or below requested temporal rate"

    return desired_fps, "downsample"


def normalize_extension(extension: str | None) -> str | None:
    """``mp4`` and ``.mp4`` both mean the same thing."""
    if extension is None:
        return None
    extension = extension.strip().lower()
    if not extension:
        return None
    return extension if extension.startswith(".") else f".{extension}"


def apply_output_extension(output_path: Path, output_extension: str | None) -> Path:
    if output_extension is None:
        return output_path
    return output_path.with_suffix(output_extension)


def audio_arguments(audio_mode: str) -> list[str]:
    """Mapping is only widened away from the default to stay backward compatible.

    ``-c:a copy`` cannot put Opus or Vorbis into an mp4 container, and ``-map 0``
    drags subtitle and data streams along that mp4 often rejects, so the non-copy
    modes narrow the mapping to the first video stream.
    """
    if audio_mode == "copy":
        return ["-map", "0", "-c:a", "copy"]
    if audio_mode == "aac":
        return ["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
    if audio_mode == "none":
        return ["-map", "0:v:0", "-an"]
    raise ValueError(f"unknown audio mode: {audio_mode!r}")


def remux_video(input_path: Path, output_path: Path, config: WorkerConfig) -> None:
    """Change container without resampling, for sources below the frame floor.

    Tries a video stream copy first and falls back to re-encoding when the source
    codec cannot live in the target container.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()

    for video_args in (["-c:v", "copy"],
                       ["-c:v", "libx264", "-preset", config.preset,
                        "-crf", str(config.crf), "-pix_fmt", "yuv420p"]):
        command = [config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(input_path)]
        command += audio_arguments(config.audio_mode)
        command += video_args
        command += ["-threads", str(config.ffmpeg_threads), str(temporary_path)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            os.replace(temporary_path, output_path)
            return
        if temporary_path.exists():
            temporary_path.unlink()
    raise RuntimeError(f"could not remux {input_path} into {output_path.suffix}")


def copy_video(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    shutil.copy2(input_path, temporary_path)
    os.replace(temporary_path, output_path)


def run_ffmpeg(
    input_path: Path,
    output_path: Path,
    target_fps: float,
    config: WorkerConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()

    command = [
        config.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        *audio_arguments(config.audio_mode),
        "-vf",
        f"fps=fps={target_fps:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(config.ffmpeg_threads),
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg failed with code {completed.returncode}")

    os.replace(temporary_path, output_path)


def process_one(job: VideoJob, config: WorkerConfig) -> ProcessResult:
    input_path = Path(job.input_path)
    input_root = Path(job.input_root)
    output_root = Path(job.output_root)
    output_prefix = Path(job.output_prefix)
    relative_path = input_path.relative_to(input_root)
    output_path = apply_output_extension(
        output_root / output_prefix / relative_path, config.output_extension
    )

    if output_path.exists() and not config.overwrite:
        return ProcessResult(str(input_path), str(output_path), "skip", "output exists")

    try:
        metadata = ffprobe(input_path, config.ffprobe)
        selected_fps, reason = choose_target_fps(metadata, config.target_fps, config.min_frames)
        if config.dry_run:
            action = "copy" if selected_fps is None else "downsample"
        elif selected_fps is None:
            # A byte copy would mislabel the container when the extension is forced.
            if output_path.suffix.lower() != input_path.suffix.lower():
                remux_video(input_path, output_path, config)
                action = "remux"
            else:
                copy_video(input_path, output_path)
                action = "copy"
        else:
            run_ffmpeg(input_path, output_path, selected_fps, config)
            action = "downsample"

        return ProcessResult(
            input_path=str(input_path),
            output_path=str(output_path),
            action=action,
            reason=reason,
            source_fps=metadata.fps,
            target_fps=selected_fps,
            source_frames=metadata.frames,
            source_duration=metadata.duration,
            width=metadata.width,
            height=metadata.height,
        )
    except Exception as exc:
        return ProcessResult(str(input_path), str(output_path), "error", str(exc))


def write_manifest_line(handle, result: ProcessResult) -> None:
    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    handle.flush()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "?"
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds_int:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds_int:02d}s"
    return f"{seconds_int:d}s"


def should_print_progress(index: int, total: int, interval: int, action: str) -> bool:
    if action == "error":
        return True
    if index == 1 or index == total:
        return True
    return interval > 0 and index % interval == 0


def print_progress(index: int, total: int, counts: dict[str, int], start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    rate = index / elapsed if elapsed > 0.0 else 0.0
    remaining = total - index
    eta = remaining / rate if rate > 0.0 else None
    percent = (100.0 * index / total) if total else 100.0
    print(
        (
            f"[{index}/{total} {percent:5.1f}%] "
            f"counts={counts} elapsed={format_seconds(elapsed)} "
            f"rate={rate:.2f} videos/s eta={format_seconds(eta)}"
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Input video root. Repeat for multiple roots. Defaults to "
            f"{DEFAULT_VIDEOMME_ROOT} and {DEFAULT_OVOBENCH_ROOT}."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL processing manifest. Defaults to <output-root>/downsample_manifest.jsonl.",
    )
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--min-frames", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads passed to each ffmpeg process. Total threads roughly workers * ffmpeg_threads.",
    )
    parser.add_argument(
        "--output-extension",
        default=None,
        help=(
            "Force every output into this container, e.g. .mp4. Defaults to keeping "
            "each input's own extension."
        ),
    )
    parser.add_argument(
        "--audio",
        choices=("copy", "aac", "none"),
        default="copy",
        help=(
            "Audio handling. 'copy' (default) preserves the previous behaviour but "
            "cannot put Opus/Vorbis into mp4; 'aac' re-encodes; 'none' drops audio "
            "and maps only the first video stream."
        ),
    )
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help="Append to the manifest instead of replacing it.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N completed videos. Use 0 to print only first, last, and errors.",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(VIDEO_EXTENSIONS)),
        help="Comma-separated file extensions to process.",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[VideoJob]:
    using_default_input_roots = args.input_root is None
    input_roots = normalize_roots(args.input_root or default_input_roots())
    output_root = args.output_root.resolve()

    existing_input_roots: list[Path] = []
    for input_root in input_roots:
        if input_root.exists():
            existing_input_roots.append(input_root)
        elif using_default_input_roots:
            print(f"Warning: skipping missing default input root: {input_root}", file=sys.stderr, flush=True)
        else:
            raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not existing_input_roots:
        raise FileNotFoundError(f"No input roots exist: {', '.join(str(root) for root in input_roots)}")

    extensions = {item.strip().lower() for item in args.extensions.split(",") if item.strip()}
    extensions = {item if item.startswith(".") else f".{item}" for item in extensions}

    jobs: list[VideoJob] = []
    for input_root in existing_input_roots:
        output_prefix = output_prefix_for(input_root, existing_input_roots)
        excluded_roots = nested_roots_for(input_root, existing_input_roots)
        for video_path in iter_videos(input_root, output_root, extensions, excluded_roots):
            jobs.append(
                VideoJob(
                    input_path=str(video_path),
                    input_root=str(input_root),
                    output_root=str(output_root),
                    output_prefix=str(output_prefix),
                )
            )

    if args.max_videos > 0:
        jobs = jobs[: args.max_videos]

    print(
        f"Found {len(jobs)} videos under {', '.join(str(root) for root in existing_input_roots)}",
        flush=True,
    )
    return jobs


def config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        target_fps=args.target_fps,
        min_frames=args.min_frames,
        ffmpeg_threads=args.ffmpeg_threads,
        crf=args.crf,
        preset=args.preset,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        output_extension=normalize_extension(args.output_extension),
        audio_mode=args.audio,
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.manifest is None:
        args.manifest = output_root / "downsample_manifest.jsonl"

    jobs = build_jobs(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs under {output_root}", flush=True)
    print(f"Writing manifest to {args.manifest}", flush=True)
    print(f"Using {args.workers} process workers and {args.ffmpeg_threads} ffmpeg threads per worker", flush=True)

    counts: dict[str, int] = {}
    start_time = time.monotonic()
    config = config_from_args(args)
    manifest_mode = "a" if args.append_manifest else "w"

    with args.manifest.open(manifest_mode, encoding="utf-8") as manifest_handle:
        if args.workers <= 1:
            for index, job in enumerate(jobs, 1):
                result = process_one(job, config)
                counts[result.action] = counts.get(result.action, 0) + 1
                write_manifest_line(manifest_handle, result)
                if should_print_progress(index, len(jobs), args.progress_interval, result.action):
                    print_progress(index, len(jobs), counts, start_time)
                    if result.action == "error":
                        print(f"  error: {result.input_path}: {result.reason}", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_one, job, config) for job in jobs]
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    result = future.result()
                    counts[result.action] = counts.get(result.action, 0) + 1
                    write_manifest_line(manifest_handle, result)
                    if should_print_progress(index, len(jobs), args.progress_interval, result.action):
                        print_progress(index, len(jobs), counts, start_time)
                        if result.action == "error":
                            print(f"  error: {result.input_path}: {result.reason}", flush=True)

    print(f"Done. counts={counts}", flush=True)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
