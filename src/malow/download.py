"""Download released MaLoW artifacts and their optional Qwen base models."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path


def registry() -> dict:
    return json.loads(files('malow').joinpath('release_models.json').read_text())


def download_plan(model: str, output: str | Path, with_base: bool = False) -> list[dict]:
    manifest = registry()
    selected = list(manifest['models']) if model == 'all' else [model]
    artifacts = []
    seen = set()
    for name in selected:
        if name not in manifest['models']:
            raise ValueError(f'Unknown model: {name}')
        spec = manifest['models'][name]
        for dependency in [*spec.get('dependencies', []), name]:
            item = manifest['models'][dependency]
            if item['repo_id'] not in seen:
                artifacts.append({**item, 'local_dir': str(Path(output) / item['repo_id'].split('/')[-1]), 'kind': 'model'})
                seen.add(item['repo_id'])
        if with_base:
            base = manifest['bases'][spec['base']]
            if base['repo_id'] not in seen:
                artifacts.append({**base, 'local_dir': str(Path(output) / base['repo_id'].split('/')[-1]), 'kind': 'base'})
                seen.add(base['repo_id'])
    return artifacts


def verify_checksums(directory: Path) -> None:
    """Verify the release inventory when the downloaded repository provides it."""
    inventory = directory / 'SHA256SUMS'
    if not inventory.is_file():
        raise FileNotFoundError(f'Missing release inventory: {inventory}')
    for line in inventory.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix('*')
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()):
            raise ValueError(f'Invalid inventory path: {name}')
        digest = hashlib.sha256()
        with target.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[*registry()['models'], 'all'])
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    parser.add_argument('--with-base', action='store_true', help='Also download the required Qwen base model (several GB).')
    parser.add_argument('--dry-run', action='store_true', help='Print repository revisions and destinations without downloading.')
    args = parser.parse_args(argv)
    plan = download_plan(args.model, args.output, args.with_base)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    from huggingface_hub import snapshot_download
    for item in plan:
        if not item.get('revision'):
            raise ValueError(f"No published revision recorded for {item['repo_id']}")
        print(f"Downloading {item['repo_id']} to {item['local_dir']}")
        snapshot_download(repo_id=item['repo_id'], revision=item['revision'], local_dir=item['local_dir'])
        if item['kind'] == 'model':
            verify_checksums(Path(item['local_dir']))
    print('Downloads complete. See the evaluation guide for the checkpoint and base-model arguments.')


if __name__ == '__main__':
    main()
