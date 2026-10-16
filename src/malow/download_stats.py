"""Save daily public Hub counters without downloading checkpoint files.

This optional maintenance command has no model-library dependencies and is not
called by the checkpoint downloader or evaluation runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from malow.download import registry


def fetch_counts(repo_id: str) -> dict:
    """Read aggregate counters, retrying only transient network/server errors."""
    query = urlencode({'expand[]': ['downloads', 'downloadsAllTime']}, doseq=True)
    url = f'https://huggingface.co/api/models/{quote(repo_id, safe="/")}?{query}'
    request = Request(url, headers={'User-Agent': 'MaLoW-download-stats/1.0'})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if data.get('id') != repo_id:
        raise ValueError(f'Unexpected repository in response for {repo_id}')
    result = {'repo_id': repo_id}
    for source, target in [('downloads', 'downloads_30d'), ('downloadsAllTime', 'downloads_all_time')]:
        value = data.get(source)
        if type(value) is not int or value < 0:
            raise ValueError(f'Missing or invalid {source} for {repo_id}')
        result[target] = value
    result['observed_at'] = datetime.now(timezone.utc).isoformat()
    return result


def validate_snapshot(snapshot: dict, day: str, repo_ids: list[str]) -> None:
    """Do not silently accept a corrupt or partial existing daily snapshot."""
    if snapshot.get('schema_version') != 1 or snapshot.get('date') != day:
        raise ValueError('Invalid snapshot version/date')
    rows = snapshot.get('models', [])
    if sorted(row['repo_id'] for row in rows) != sorted(repo_ids):
        raise ValueError('Snapshot must contain exactly the released checkpoints')
    for row in rows:
        timestamp = datetime.fromisoformat(row['observed_at'])
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError('Snapshot observation timestamps must be UTC')
        for key in ('downloads_30d', 'downloads_all_time'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f'Invalid snapshot counter: {key}')


def save_daily_snapshot(output_dir: Path, repo_ids: list[str], *, now: datetime | None = None) -> Path:
    """Keep the first complete successful observation for each UTC day."""
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    day = now.astimezone(timezone.utc).date().isoformat()
    target = output_dir / f'{day}.json'
    if target.exists():
        validate_snapshot(json.loads(target.read_text()), day, repo_ids)
        return target
    snapshot = {'schema_version': 1, 'date': day,
                'models': [fetch_counts(repo_id) for repo_id in sorted(repo_ids)]}
    validate_snapshot(snapshot, day, repo_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically, without overwriting a concurrent run.
    with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            validate_snapshot(json.loads(target.read_text()), day, repo_ids)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    repo_ids = sorted({item['repo_id'] for item in registry()['models'].values()})
    path = save_daily_snapshot(args.output_dir, repo_ids)
    print(f'Daily snapshot: {path}')


if __name__ == '__main__':
    main()
