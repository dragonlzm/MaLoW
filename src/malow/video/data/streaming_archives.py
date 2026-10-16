#!/usr/bin/env python
"""Prepare StreamingBench annotations/videos from cached dataset ZIP archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

TASK_FILES = (
    "questions_real.json",
    "questions_real_stream.json",
    "questions_omni.json",
    "questions_omni_stream.json",
    "questions_sqa_stream.json",
    "questions_sqa.json",
    "questions_proactive.json",
    "questions_proactive_50.json",
)

REAL_TIME_PREFIX = "Real-Time Visual Understanding"
ARCHIVE_SUFFIX_OVERRIDES = {
    "Proactive Output": "proactive",
    "Proactive Output_1-25": "proactive",
    "Proactive Output_26-50": "proactive",
    "Sequential Question Answering": "sqa",
    "Sequential Question Answering_1-25": "sqa",
    "Sequential Question Answering_26-50": "sqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare StreamingBench from cached Hugging Face ZIP archives.")
    parser.add_argument("--annotation-root", required=True, help="StreamingBench src/data directory with questions_*.json.")
    parser.add_argument("--archive-root", required=True, help="Directory containing cached StreamingBench .zip archives.")
    parser.add_argument("--output-root", required=True, help="Prepared output root with annotations/ and videos/.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite prepared annotations.")
    parser.add_argument("--strict", action="store_true", help="Fail if any referenced video is missing.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_objects(child)


def expected_video_names(annotation_root: Path) -> set[str]:
    names: set[str] = set()
    for filename in TASK_FILES:
        path = annotation_root / filename
        if not path.exists():
            continue
        for obj in iter_objects(load_json(path)):
            video_path = obj.get("video_path") if isinstance(obj, dict) else None
            if isinstance(video_path, str):
                names.add(Path(video_path).name)
    return names


def archive_suffix(archive: Path) -> str:
    stem = archive.stem
    if stem.startswith(REAL_TIME_PREFIX):
        return "real"
    if stem in ARCHIVE_SUFFIX_OVERRIDES:
        return ARCHIVE_SUFFIX_OVERRIDES[stem]
    return stem.replace(" ", "_")


def sample_name(member_name: str) -> str | None:
    parts = Path(member_name).parts
    for part in parts:
        if part.startswith("sample_"):
            return part
    return None


def output_name(archive: Path, member_name: str, expected: set[str]) -> str | None:
    basename = Path(member_name).name
    if basename in expected:
        return basename
    sample = sample_name(member_name)
    if not sample:
        return None
    candidate = f"{sample}_{archive_suffix(archive)}.mp4"
    return candidate if candidate in expected else None


def extract_needed_videos(archive_root: Path, videos_dir: Path, expected: set[str]) -> set[str]:
    videos_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    archives = sorted(archive_root.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No .zip archives found under {archive_root}")

    for archive in archives:
        copied = 0
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or "__MACOSX" in Path(info.filename).parts:
                    continue
                if not info.filename.lower().endswith(".mp4"):
                    continue
                name = output_name(archive, info.filename, expected)
                if not name:
                    continue
                dest = videos_dir / name
                found.add(name)
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                with zf.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                copied += 1
        print(f"{archive.name}: extracted/updated {copied} referenced videos")
    return found


def rewrite_annotations(
    annotation_root: Path,
    archive_root: Path,
    output_root: Path,
    expected: set[str],
    found: set[str],
    overwrite: bool,
) -> set[str]:
    annotations_dir = output_root / "annotations"
    videos_dir = output_root / "videos"
    missing: set[str] = set()
    written = []

    for filename in TASK_FILES:
        src_file = annotation_root / filename
        if not src_file.exists():
            continue
        data = load_json(src_file)
        for obj in iter_objects(data):
            if not isinstance(obj, dict):
                continue
            video_path = obj.get("video_path")
            if not isinstance(video_path, str):
                continue
            name = Path(video_path).name
            if name in expected:
                obj["video_path"] = str((videos_dir / name).resolve())
                if name not in found:
                    missing.add(name)
        write_json(annotations_dir / filename, data, overwrite)
        written.append(filename)

    manifest = {
        "annotation_root": str(annotation_root),
        "archive_root": str(archive_root),
        "output_root": str(output_root),
        "annotations": written,
        "expected_videos": len(expected),
        "found_videos": len(found),
        "missing_videos": sorted(missing),
    }
    write_json(output_root / "manifest.json", manifest, True)
    return missing


def main() -> None:
    args = parse_args()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Missing annotation root: {annotation_root}")
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Missing archive root: {archive_root}")

    expected = expected_video_names(annotation_root)
    if not expected:
        raise RuntimeError(f"No video_path entries found in {annotation_root}")

    found = extract_needed_videos(archive_root, output_root / "videos", expected)
    missing = rewrite_annotations(annotation_root, archive_root, output_root, expected, found, args.overwrite)

    print(f"Prepared StreamingBench data at {output_root}")
    print(f"Expected videos: {len(expected)}")
    print(f"Found videos: {len(found)}")
    print(f"Missing videos: {len(missing)}")
    if missing:
        for name in sorted(missing)[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print("  ...")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
