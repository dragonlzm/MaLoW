"""Prepare and validate MaLoW training annotations and video datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
import tarfile
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Iterator


def registry() -> dict:
    return json.loads(files('malow.training').joinpath('datasets.json').read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    """Reject traversal and platform-dependent archive member names."""
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name}')
    return Path(*pure.parts)


def safe_target(root: Path, name: str) -> Path:
    relative = safe_relative(name)
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes destination: {name}')
    # Do not overwrite through links even if they resolve inside the root.
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f'Symlink in destination: {current}')
        current = current.parent
    return target


def _archive_destination(name: str, root: Path, anchor: str | None) -> Path | None:
    relative = safe_relative(name)
    if anchor:
        if anchor not in relative.parts:
            # Directory wrappers are harmless; files outside the declared root are rejected below.
            return None
        parts = relative.parts[relative.parts.index(anchor) + 1:]
        if not parts:
            return root
        relative = Path(*parts)
    return safe_target(root, relative.as_posix())


def safe_extract(archive: Path, destination: Path, *, member_anchor: str | None = None,
                 zip_password: str | bytes | None = None) -> int:
    """Extract regular files only; no links, devices, traversal or out-of-root payloads."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    seen: set[str] = set()

    def write_file(name: str, source) -> None:
        nonlocal extracted
        target = _archive_destination(name, destination, member_anchor)
        if target is None or target == destination:
            raise ValueError(f'File outside expected archive root {member_anchor!r}: {name}')
        if target.as_posix() in seen:
            raise ValueError(f'Duplicate archive member: {name}')
        seen.add(target.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.extract-', delete=False) as out:
            temporary = Path(out.name)
            try:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, target)
        extracted += 1

    if zipfile.is_zipfile(archive):
        password = zip_password.encode('utf-8') if isinstance(zip_password, str) else zip_password
        with zipfile.ZipFile(archive) as handle:
            for item in handle.infolist():
                safe_relative(item.filename)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError(f'Unsupported ZIP member: {item.filename}')
                if item.is_dir():
                    continue
                with handle.open(item, pwd=password) as source:
                    write_file(item.filename, source)
    else:
        with tarfile.open(archive, 'r|*') as handle:
            for item in handle:
                safe_relative(item.name)
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError(f'Unsupported TAR member: {item.name}')
                source = handle.extractfile(item)
                if source is None:
                    raise ValueError(f'Unreadable TAR member: {item.name}')
                with source:
                    write_file(item.name, source)
    if not extracted:
        raise ValueError(f'Archive contains no files: {archive}')
    return extracted


def resolve_training_sources(annotations_root: str | Path, videos_root: str | Path) -> tuple[list[str], list[str]]:
    spec = registry()
    annotations_root, videos_root = Path(annotations_root).resolve(), Path(videos_root).resolve()
    profiles = [str(annotations_root / stream['profile_dir']) for stream in spec['streams']]
    roots = [str(videos_root / spec['video_roots'][stream['video_source']]) for stream in spec['streams']]
    return profiles, roots


def resolve_video_path(path: str, video_root: Path) -> Path:
    """Resolve original profiling paths without depending on the original machine."""
    original = Path(path)
    if original.is_absolute():
        if video_root.name not in original.parts:
            raise ValueError(f'Video path has no expected {video_root.name!r} anchor: {path}')
        parts = original.parts[original.parts.index(video_root.name) + 1:]
        if not parts:
            raise ValueError(f'Video path identifies a directory: {path}')
        relative = Path(*parts).as_posix()
    else:
        relative = path
    return safe_target(video_root, relative)


def iter_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected object at {path}:{line_number}')
            yield row


def validate_training_data(
    annotations_root: str | Path,
    videos_root: str | Path,
    *, check_videos: bool = True,
    manifest: dict | None = None,
    check_readable: bool = True,
) -> dict:
    """Validate all streams and references, failing instead of silently dropping videos."""
    spec = manifest or registry()
    annotations_root, videos_root = Path(annotations_root), Path(videos_root)
    report = {'passed': False, 'streams': [], 'records': 0, 'turns': 0, 'files': 0,
              'videos_checked': check_videos, 'missing_video_count': 0, 'annotation_sha256': {}}
    missing: set[str] = set()
    checked: set[str] = set()
    errors = []
    for stream in spec['streams']:
        folder = annotations_root / stream['profile_dir']
        paths = sorted(folder.glob('*.jsonl'))
        if len(paths) != stream['shards']:
            errors.append(f"{stream['id']}: expected {stream['shards']} shards, found {len(paths)}")
        rows = turns = 0
        r2_indices = set()
        root = videos_root / spec['video_roots'][stream['video_source']]
        for path in paths:
            if path.is_symlink():
                raise ValueError(f'Annotation symlink is not supported: {path}')
            report['annotation_sha256'][path.relative_to(annotations_root).as_posix()] = sha256(path)
            for row in iter_rows(path):
                rows += 1
                questions = row.get('turns') or [row]
                turns += len(questions)
                for turn in questions:
                    if not isinstance(turn, dict) or not isinstance(turn.get('question'), str) or not turn['question'].strip():
                        raise ValueError(f'Missing question in {path}, record {rows}')
                    if not any(isinstance(turn.get(k), str) and turn[k].strip() for k in ('expected_answer', 'generated_answer', 'reference_answer')):
                        raise ValueError(f'Missing answer in {path}, record {rows}')
                if stream['video_source'] == 'r2':
                    if row.get('verification_status') != 'model_verified' or row.get('expected_answer') not in ('A', 'B', 'C', 'D'):
                        raise ValueError(f'Invalid model-verified R2 record in {path}')
                    index = row.get('source_index')
                    if index is None or index in r2_indices:
                        raise ValueError(f'Duplicate/missing R2 source_index in {path}')
                    r2_indices.add(index)
                videos = row.get('videos') or (row.get('original_record') or {}).get('videos')
                if not isinstance(videos, list) or not videos:
                    raise ValueError(f'Missing videos in {path}, record {rows}')
                for video in videos:
                    name = video.get('path') if isinstance(video, dict) else video
                    if not isinstance(name, str):
                        raise ValueError(f'Invalid video specification in {path}')
                    resolved = resolve_video_path(name, root)
                    if isinstance(video, dict):
                        start = video.get('clip_start_seconds', 0)
                        end = video.get('clip_end_seconds')
                        if not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
                            raise ValueError(f'Invalid clip start in {path}')
                        if end is not None and (not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start):
                            raise ValueError(f'Invalid clip end in {path}')
                    key = str(resolved)
                    if check_videos and key not in checked:
                        checked.add(key)
                        if not resolved.is_file() or resolved.stat().st_size == 0:
                            missing.add(key)
        if rows != stream['rows']:
            errors.append(f"{stream['id']}: expected {stream['rows']} records, found {rows}")
        report['streams'].append({'id': stream['id'], 'records': rows, 'turns': turns, 'files': len(paths)})
        report['records'] += rows
        report['turns'] += turns
        report['files'] += len(paths)
    report['missing_video_count'] = len(missing)
    report['unique_videos_checked'] = len(checked)
    if missing:
        errors.append(f'{len(missing)} missing/empty video files; examples: {sorted(missing)[:5]}')
    if errors:
        raise ValueError('Training data validation failed:\n' + '\n'.join(errors))
    if check_videos and check_readable:
        check_video_readability(checked)
    report['video_readability_checked'] = check_videos and check_readable
    report['passed'] = True
    report['original_r2_validation_present'] = (annotations_root / 'event_order_r2_model_verified/validation_model_verified.json').is_file()
    return report


def probe_video(path: str) -> str | None:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=codec_name,width,height', '-of', 'json', path],
            capture_output=True, text=True, timeout=60, check=False,
        )
        streams = json.loads(result.stdout).get('streams', []) if result.returncode == 0 else []
        if not streams or not streams[0].get('codec_name') or streams[0].get('width', 0) <= 0 or streams[0].get('height', 0) <= 0:
            return path
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return path
    return None


def check_video_readability(paths: set[str]) -> None:
    if not shutil.which('ffprobe'):
        raise RuntimeError('ffprobe is required for video readability preflight; install FFmpeg.')
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [result for result in pool.map(probe_video, sorted(paths)) if result]
    if failures:
        raise ValueError(f'{len(failures)} unreadable videos; examples: {failures[:5]}')


def audit_data(data_root: Path, *, check_videos: bool = True) -> dict:
    """Audit the public replay layout or an existing source-profile installation."""
    from . import replay

    annotations, videos = data_root / 'annotations', data_root / 'videos'
    manifest_path = annotations / 'manifest.json'
    if not manifest_path.exists() and not any(annotations.glob('replay-*.jsonl')):
        return validate_training_data(annotations, videos, check_videos=check_videos)

    report = replay.validate(manifest_path)
    checked = set()
    if check_videos:
        manifest = replay.load_manifest(manifest_path)
        roots = [safe_target(videos, source['video_root']) for source in manifest['sources']]
        references = set()
        for shard in manifest['shards']:
            for row in iter_rows(annotations / shard['file']):
                root = roots[row['source'][0]]
                media = row['record'].get('videos')
                if not media:
                    raise ValueError(f"Missing video reference at replay position {row['position']}")
                for video in media:
                    references.add((root, video['path']))
        paths = {safe_target(root, name) for root, name in references}
        checked = {str(path) for path in paths}
        missing = sorted(str(path) for path in paths if not path.is_file() or not path.stat().st_size)
        if missing:
            raise ValueError(f'{len(missing)} missing/empty video files; examples: {missing[:5]}')
        check_video_readability(checked)
    return {**report, 'passed': True, 'videos_checked': check_videos, 'unique_videos_checked': len(checked),
            'missing_video_count': 0, 'video_readability_checked': check_videos}


def prepare_annotations(source: Path, output: Path, r2_source: Path | None = None) -> dict:
    """Copy complete private profiles byte-for-byte, without importing video payloads."""
    validate_training_data(source, output.parent / 'videos', check_videos=False)
    for stream in registry()['streams']:
        for path in sorted((source / stream['profile_dir']).glob('*.jsonl')):
            target = safe_target(output, path.relative_to(source).as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
            if sha256(path) != sha256(target):
                raise ValueError(f'Copy checksum mismatch: {path}')
    if r2_source:
        for path in sorted(r2_source.rglob('*')):
            relative = path.relative_to(r2_source)
            if relative.parts[0] == 'videos' or path.name == '.DS_Store' or not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f'Unexpected source symlink: {path}')
            target = safe_target(output / 'event_order_r2_model_verified', relative.as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
    return validate_training_data(output, output.parent / 'videos', check_videos=False)


def verify_file(path: Path, item: dict) -> None:
    if path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
        raise ValueError(f'Size/checksum mismatch: {path}')


def download_data(data_root: Path, *, annotations_revision: str | None = None,
                  videos_revision: str | None = None, components: str = 'all', dry_run: bool = False) -> list[dict]:
    spec = registry()
    annotation_revision = annotations_revision or spec.get('annotation_revision')
    video_revision = videos_revision or spec.get('video_revision')
    plan = []
    if components in ('all', 'annotations'):
        plan.append({'kind': 'annotations', 'repo_id': spec['annotation_repo'], 'revision': annotation_revision,
                     'include': 'annotations/*', 'destination': str(data_root / 'annotations')})
    if components in ('all', 'videos'):
        for item in spec['source_archives']:
            plan.append({**item, 'kind': 'source_video', 'repo_id': spec['video_repo'],
                         'revision': video_revision,
                         'destination': str(data_root / 'videos' / item['destination'])})
        plan.append({'kind': 'r2_video', 'repo_id': spec['video_repo'], 'revision': video_revision,
                     'manifest': 'release_manifest.json', 'destination': str(data_root)})
    if dry_run:
        return plan
    if any(not item['revision'] for item in plan):
        raise ValueError('A release is not published/pinned yet. Supply --annotations-revision and/or --videos-revision with commit hashes.')
    for item in plan:
        revision = item['revision']
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError(f'Use an immutable 40-character commit revision for {item["repo_id"]}')
    from huggingface_hub import hf_hub_download
    if components in ('all', 'videos'):
        manifest_path = hf_hub_download(repo_id=spec['video_repo'], repo_type='dataset',
                                        revision=video_revision, filename='release_manifest.json')
        release = json.loads(Path(manifest_path).read_text())
        if release.get('status') != 'complete':
            raise ValueError(f"Release {spec['video_repo']} is incomplete ({release.get('status', 'unknown')}); no data archives downloaded")
        published = release.get('source_archives', [])
        expected = spec['source_archives']
        keys = ('id', 'filename', 'size_bytes', 'sha256', 'member_anchor', 'destination')
        if [{key: row.get(key) for key in keys} for row in published] != [
                {key: row.get(key) for key in keys} for row in expected]:
            raise ValueError('Video release source archive inventory differs from the pinned dataset registry')
    for item in plan:
        if item['kind'] == 'annotations':
            from huggingface_hub import snapshot_download
            from .replay import validate

            source = dict(repo_id=item['repo_id'], repo_type='dataset', revision=item['revision'])
            pinned_manifest = Path(hf_hub_download(**source, filename='annotations/manifest.json'))
            snapshot_download(**source, allow_patterns=item['include'], local_dir=data_root)
            local_manifest = Path(item['destination']) / 'manifest.json'
            if sha256(local_manifest) != sha256(pinned_manifest):
                raise ValueError('Downloaded annotations do not match the requested revision')
            validate(local_manifest)
        elif item['kind'] == 'source_video':
            path = Path(hf_hub_download(repo_id=item['repo_id'], repo_type='dataset', revision=item['revision'], filename=item['filename']))
            verify_file(path, item)
            safe_extract(path, Path(item['destination']), member_anchor=item['member_anchor'],
                         zip_password=item.get('zip_password'))
        else:
            for artifact in release['archives']:
                path = Path(hf_hub_download(repo_id=item['repo_id'], repo_type='dataset', revision=item['revision'], filename=artifact['filename']))
                verify_file(path, artifact)
                safe_extract(path, data_root)
    return plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Copy supplied private annotations, preserving their bytes.')
    prepare.add_argument('--annotations-source', required=True, type=Path)
    prepare.add_argument('--r2-source', type=Path, help='Also copy non-video R2 provenance; no videos are copied.')
    prepare.add_argument('--data-root', type=Path, required=True)
    audit = commands.add_parser('audit', help='Validate all rows and required media.')
    audit.add_argument('--data-root', type=Path, required=True)
    audit.add_argument('--annotations-only', action='store_true')
    download = commands.add_parser('download', help='Download immutable HF releases and verify every archive.')
    download.add_argument('--data-root', type=Path, required=True)
    download.add_argument('--annotations-revision')
    download.add_argument('--videos-revision')
    download.add_argument('--components', choices=['all', 'annotations', 'videos'], default='all')
    download.add_argument('--dry-run', action='store_true')
    extract = commands.add_parser('extract', help='Safely unpack a locally downloaded archive.')
    extract.add_argument('--archive', type=Path, required=True)
    extract.add_argument('--destination', type=Path, required=True)
    extract.add_argument('--member-anchor')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare_annotations(args.annotations_source, args.data_root / 'annotations', args.r2_source)
    elif args.command == 'audit':
        result = audit_data(args.data_root, check_videos=not args.annotations_only)
    elif args.command == 'extract':
        result = {'files': safe_extract(args.archive, args.destination, member_anchor=args.member_anchor)}
    else:
        result = download_data(args.data_root, annotations_revision=args.annotations_revision, videos_revision=args.videos_revision,
                               components=args.components, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
