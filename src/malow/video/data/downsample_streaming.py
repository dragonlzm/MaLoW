#!/usr/bin/env python3
"""Create a temporal-downsampled StreamingBench offline tree.

The output tree mirrors the prepared StreamingBench layout:

  <output-root>/annotations/questions_*.json
  <output-root>/videos/*.mp4
  <output-root>/downsample_manifest.jsonl

Videos with at most --max-frames source frames are copied unchanged. Longer
videos are re-encoded with libx264 at CRF 28 by default, sampling up to 2 FPS
while preserving full duration and capping the output at --max-frames frames.
If probing or conversion fails, the original video is copied instead.

With --no-max-frame-cap, videos are sampled at --target-fps with no upper frame
cap. If --min-frames is set and sampling at --target-fps would produce fewer
than that many frames, the original video is copied unchanged.
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
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path(
    "/run/user/1000/gvfs/smb-share:server=research.drive.wisc.edu,share=li968/"
    "zliu2346/huggingface/streamingbench"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/user/easystore/huggingface/streamingbench-2fps-max768-v2"
)
DEFAULT_ANNOTATION_VIDEO_ROOT = Path(
    "/run/user/1000/gvfs/smb-share:server=research.drive.wisc.edu,share=li968/"
    "zliu2346/huggingface/streamingbench-2fps-max768-v2/videos"
)
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    duration: float
    frames: int | None
    width: int
    height: int
    codec: str


@dataclass(frozen=True)
class VideoJob:
    input_path: str
    output_path: str


@dataclass(frozen=True)
class WorkerConfig:
    target_fps: float
    max_frames: int
    min_frames: int
    no_max_frame_cap: bool
    ffmpeg_threads: int
    crf: int
    preset: str
    ffmpeg: str
    ffprobe: str
    work_dir: str
    overwrite: bool
    dry_run: bool


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
    expected_output_frames: int | None = None
    width: int | None = None
    height: int | None = None


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
        fps=fps,
        duration=duration,
        frames=frames,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        codec=stream.get("codec_name") or "",
    )


def selected_fps(
    metadata: VideoMetadata,
    target_fps: float,
    max_frames: int,
    min_frames: int,
    no_max_frame_cap: bool,
) -> tuple[float | None, str, int | None]:
    if no_max_frame_cap:
        if metadata.fps <= 0.0 or metadata.duration <= 0.0:
            return None, "copy original: missing fps or duration metadata", None

        selected = min(target_fps, metadata.fps)
        expected_frames = max(1, int(math.floor(metadata.duration * selected + 1e-6)))
        if min_frames > 0 and expected_frames < min_frames:
            return None, f"copy original: {target_fps:g} fps would produce fewer than {min_frames} frames", expected_frames
        if selected >= metadata.fps:
            return None, "copy original: source fps is already at or below requested temporal rate", metadata.frames
        return selected, "downsample without max-frame cap", expected_frames

    if metadata.frames is not None and metadata.frames <= max_frames:
        return None, f"copy original: source has at most {max_frames} frames", metadata.frames
    if metadata.fps <= 0.0 or metadata.duration <= 0.0:
        return None, "copy original: missing fps or duration metadata", None

    capped_fps = min(target_fps, max_frames / metadata.duration)
    capped_fps = min(capped_fps, metadata.fps)
    expected_frames = max(1, int(math.floor(metadata.duration * capped_fps + 1e-6)))
    if expected_frames > max_frames:
        capped_fps *= max_frames / expected_frames
        expected_frames = max(1, int(math.floor(metadata.duration * capped_fps + 1e-6)))

    if capped_fps >= metadata.fps and metadata.frames is not None and metadata.frames <= max_frames:
        return None, "copy original: source is already within frame cap", metadata.frames
    return capped_fps, "downsample", expected_frames


def local_temp_path(work_dir: Path, output_path: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=output_path.suffix,
        dir=work_dir,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def publish_video(local_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}")
    shutil.copy2(local_path, temporary_path)
    os.replace(temporary_path, output_path)


def copy_video(input_path: Path, output_path: Path, config: WorkerConfig) -> None:
    temporary_local = local_temp_path(Path(config.work_dir), output_path)
    try:
        shutil.copy2(input_path, temporary_local)
        publish_video(temporary_local, output_path)
    finally:
        if temporary_local.exists():
            temporary_local.unlink()


def run_ffmpeg(input_path: Path, output_path: Path, fps: float, config: WorkerConfig) -> None:
    temporary_path = local_temp_path(Path(config.work_dir), output_path)

    command = [
        config.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-vf",
        f"fps=fps={fps:.8f}",
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
        "-c:a",
        "copy",
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
    try:
        publish_video(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_one(job: VideoJob, config: WorkerConfig) -> ProcessResult:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    if output_path.exists() and not config.overwrite:
        return ProcessResult(str(input_path), str(output_path), "skip", "output exists")

    try:
        metadata = ffprobe(input_path, config.ffprobe)
    except Exception as exc:
        if config.dry_run:
            return ProcessResult(str(input_path), str(output_path), "copy", f"probe failed: {exc}")
        copy_video(input_path, output_path, config)
        return ProcessResult(str(input_path), str(output_path), "copy", f"probe failed: {exc}")

    fps, reason, expected_frames = selected_fps(
        metadata,
        config.target_fps,
        config.max_frames,
        config.min_frames,
        config.no_max_frame_cap,
    )
    try:
        if config.dry_run:
            action = "copy" if fps is None else "downsample"
        elif fps is None:
            copy_video(input_path, output_path, config)
            action = "copy"
        else:
            run_ffmpeg(input_path, output_path, fps, config)
            action = "downsample"
    except Exception as exc:
        if config.dry_run:
            action = "copy"
            reason = f"conversion failed, would copy original: {exc}"
        else:
            copy_video(input_path, output_path, config)
            action = "copy_on_error"
            reason = f"conversion failed, copied original: {exc}"

    return ProcessResult(
        input_path=str(input_path),
        output_path=str(output_path),
        action=action,
        reason=reason,
        source_fps=metadata.fps,
        target_fps=fps,
        source_frames=metadata.frames,
        source_duration=metadata.duration,
        expected_output_frames=expected_frames,
        width=metadata.width,
        height=metadata.height,
    )


def iter_videos(videos_dir: Path, extensions: set[str]) -> Iterable[Path]:
    for root, _, filenames in os.walk(videos_dir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix.lower() in extensions:
                yield path


def iter_entries(data: Any):
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from iter_entries(value)
    elif isinstance(data, list):
        for item in data:
            yield from iter_entries(item)


def referenced_video_names(source_annotations: Path) -> set[str]:
    names: set[str] = set()
    for source_file in sorted(source_annotations.glob("questions*.json")):
        with source_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for entry in iter_entries(data):
            if not isinstance(entry, dict) or "video_path" not in entry:
                continue
            name = Path(str(entry["video_path"])).name
            if name:
                names.add(name)
    return names


def rewrite_annotations(source_annotations: Path, output_annotations: Path, annotation_video_root: Path) -> int:
    count = 0
    for source_file in sorted(source_annotations.glob("questions*.json")):
        with source_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for entry in iter_entries(data):
            if not isinstance(entry, dict) or "video_path" not in entry:
                continue
            entry["video_path"] = str((annotation_video_root / Path(str(entry["video_path"])).name).resolve())
        output_file = output_annotations / source_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        count += 1
    return count


def write_manifest_line(handle, result: ProcessResult) -> None:
    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


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


def print_progress(index: int, total: int, counts: dict[str, int], start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    rate = index / elapsed if elapsed > 0.0 else 0.0
    eta = (total - index) / rate if rate > 0.0 else None
    percent = (100.0 * index / total) if total else 100.0
    print(
        f"[{index}/{total} {percent:5.1f}%] counts={counts} "
        f"elapsed={format_seconds(elapsed)} rate={rate:.2f} videos/s eta={format_seconds(eta)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--annotation-video-root",
        type=Path,
        default=DEFAULT_ANNOTATION_VIDEO_ROOT,
        help=(
            "Video root written into annotation JSONs. Defaults to the final research-drive "
            "StreamingBench 2fps/max768 videos directory."
        ),
    )
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument(
        "--min-frames",
        type=int,
        default=0,
        help="With --no-max-frame-cap, copy originals when target FPS would produce fewer than this many frames.",
    )
    parser.add_argument(
        "--no-max-frame-cap",
        action="store_true",
        help="Sample at --target-fps without lowering FPS to satisfy --max-frames.",
    )
    parser.add_argument(
        "--referenced-only",
        action="store_true",
        help="Only process videos referenced by annotation JSON video_path fields.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/streamingbench_downsample_work"),
        help="Local directory for manifests and temporary encode files before publishing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append-manifest", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(VIDEO_EXTENSIONS)),
        help="Comma-separated file extensions to process.",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[VideoJob]:
    source_root = args.source_root.expanduser().resolve()
    videos_dir = source_root / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing source videos directory: {videos_dir}")

    output_videos = args.output_root.expanduser().resolve() / "videos"
    extensions = {item.strip().lower() for item in args.extensions.split(",") if item.strip()}
    extensions = {item if item.startswith(".") else f".{item}" for item in extensions}

    referenced = referenced_video_names(source_root / "annotations") if args.referenced_only else None
    videos = sorted(iter_videos(videos_dir, extensions))
    if referenced is not None:
        videos = [video for video in videos if video.name in referenced]
        missing = sorted(referenced - {video.name for video in videos})
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} referenced videos, first few: {missing[:10]}")

    jobs = [VideoJob(str(video), str(output_videos / video.relative_to(videos_dir))) for video in videos]
    if args.max_videos > 0:
        jobs = jobs[: args.max_videos]
    print(f"Found {len(jobs)} videos under {videos_dir}", flush=True)
    return jobs


def config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        no_max_frame_cap=args.no_max_frame_cap,
        ffmpeg_threads=args.ffmpeg_threads,
        crf=args.crf,
        preset=args.preset,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        work_dir=str(args.work_dir.expanduser().resolve()),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_annotations = output_root / "annotations"
    output_videos = output_root / "videos"
    annotation_video_root = args.annotation_video_root.expanduser().resolve()
    manifest = output_root / "downsample_manifest.jsonl"
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_work_path = work_dir / f"{output_root.name}_downsample_manifest.jsonl"

    jobs = build_jobs(args)
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_count = rewrite_annotations(args.source_root.resolve() / "annotations", output_annotations, annotation_video_root)
    print(f"Wrote {annotation_count} annotation files under {output_annotations}", flush=True)
    print(f"Annotation video paths point under {annotation_video_root}", flush=True)
    print(f"Writing videos under {output_videos}", flush=True)
    print(f"Writing working manifest to {manifest_work_path}", flush=True)
    print(f"Publishing final manifest to {manifest}", flush=True)
    print(f"Using {args.workers} process workers and {args.ffmpeg_threads} ffmpeg threads per worker", flush=True)

    counts: dict[str, int] = {}
    start_time = time.monotonic()
    config = config_from_args(args)
    manifest_mode = "a" if args.append_manifest else "w"

    with manifest_work_path.open(manifest_mode, encoding="utf-8") as manifest_handle:
        if args.workers <= 1:
            for index, job in enumerate(jobs, 1):
                result = process_one(job, config)
                counts[result.action] = counts.get(result.action, 0) + 1
                write_manifest_line(manifest_handle, result)
                if index == 1 or index == len(jobs) or (
                    args.progress_interval > 0 and index % args.progress_interval == 0
                ):
                    print_progress(index, len(jobs), counts, start_time)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_one, job, config) for job in jobs]
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    result = future.result()
                    counts[result.action] = counts.get(result.action, 0) + 1
                    write_manifest_line(manifest_handle, result)
                    if index == 1 or index == len(jobs) or (
                        args.progress_interval > 0 and index % args.progress_interval == 0
                    ):
                        print_progress(index, len(jobs), counts, start_time)

    run_manifest = {
        "source_root": str(args.source_root.expanduser().resolve()),
        "output_root": str(output_root),
        "target_fps": args.target_fps,
        "max_frames": args.max_frames,
        "min_frames": args.min_frames,
        "no_max_frame_cap": args.no_max_frame_cap,
        "referenced_only": args.referenced_only,
        "crf": args.crf,
        "preset": args.preset,
        "counts": counts,
        "annotations": str(output_annotations),
        "videos": str(output_videos),
        "annotation_video_root": str(annotation_video_root),
    }
    run_manifest_work_path = work_dir / f"{output_root.name}_manifest.json"
    run_manifest_work_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    shutil.copy2(manifest_work_path, manifest)
    shutil.copy2(run_manifest_work_path, output_root / "manifest.json")
    print(f"Done. counts={counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
