"""Opt-in durable evaluation journals and immutable recovery records.

No model tensors are serialized: callers replay the visual recurrence and restore
validated text records at the same segment boundaries.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

FORMAT_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path):
    path = Path(path)
    if not path.exists():
        ensure_directory(path.parent)
        try:
            path.mkdir()
        except FileExistsError:
            pass
        sync_directory(path.parent)


def atomic_json(path, value):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def read_journal(path):
    """Repair only an unterminated final record; preserve its bytes for inspection."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        end = data.rfind(b'\n') + 1
        backup = path.with_name(path.name + '.incomplete-' + uuid.uuid4().hex)
        with backup.open('xb') as stream:
            stream.write(data[end:]); stream.flush(); os.fsync(stream.fileno())
        with path.open('r+b') as stream:
            stream.truncate(end); stream.flush(); os.fsync(stream.fileno())
        sync_directory(path.parent)
        data = data[:end]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


class PredictionJournal:
    def __init__(self, path, key, expected):
        self.path = Path(path)
        self.key = key
        self.expected = set(expected)
        self.rows = {}
        for row in read_journal(path):
            identity = key(row)
            if identity not in self.expected:
                raise ValueError(f'Unexpected prediction ID {identity!r} in {path}')
            if identity in self.rows:
                raise ValueError(f'Duplicate prediction ID {identity!r} in {path}')
            self.rows[identity] = row

    def append(self, row):
        identity = self.key(row)
        if identity not in self.expected:
            raise ValueError(f'Unexpected prediction ID {identity!r}')
        if identity in self.rows:
            if self.rows[identity] != row:
                raise ValueError(f'Conflicting prediction for {identity!r}')
            return
        ensure_directory(self.path.parent)
        with self.path.open('a') as stream:
            stream.write(canonical(row) + '\n'); stream.flush(); os.fsync(stream.fileno())
        sync_directory(self.path.parent)
        self.rows[identity] = row

    def require_complete(self):
        if self.expected != self.rows.keys():
            raise ValueError(f'Incomplete predictions in {self.path}: {len(self.rows)}/{len(self.expected)}')


def bind_session(args, session, key):
    if getattr(args, 'checkpoint_progress', False):
        session.progress_key = key
        previous = cache_get(args, session, 'report', 'latest')
        if previous is not None:
            session.verifications = previous['verifications']
            session.skipped_verifications = previous['skipped_verifications']


def cache_path(args, session, kind, key):
    if not getattr(args, 'checkpoint_progress', False):
        return None
    identity = [FORMAT_VERSION, session.progress_key, kind, key]
    return Path(args.output_dir) / 'recovery' / digest(session.progress_key) / kind / (digest(identity) + '.json')


def cache_get(args, session, kind, key):
    path = cache_path(args, session, kind, key)
    if path is None or not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('key') != key or record.get('sha256') != digest(record.get('value')):
        raise ValueError(f'Corrupt recovery record: {path}')
    return record['value']


def cache_put(args, session, kind, key, value):
    path = cache_path(args, session, kind, key)
    if path is not None:
        atomic_json(path, {'key': key, 'value': value, 'sha256': digest(value)})


def save_session(args, session):
    value = {name: getattr(session, name) for name in (
        'consumed_blocks', 'segments', 'turns', 'fallback_turns', 'verifications', 'skipped_verifications')}
    cache_put(args, session, 'report', 'latest', value)


def saved_session(args, config, key):
    session = SimpleNamespace(progress_key=key, config=config)
    value = cache_get(args, session, 'report', 'latest')
    if value is None:
        raise ValueError(f'Completed predictions have no durable session report: {key}')
    session.__dict__.update(value)
    return session


def ovo_key(row):
    return (row['task_name'], row['id'], row.get('round_idx'))


def guard_video_inputs(args, rank, paths):
    if not getattr(args, 'checkpoint_progress', False):
        return
    identities = [file_identity(path) for path in sorted(set(map(str, paths)))]
    path = Path(args.output_dir) / f'recovery_inputs_rank_{rank}.json'
    if path.exists():
        if json.loads(path.read_text()) != identities:
            raise ValueError('Video inputs changed; use a fresh output directory')
    else:
        atomic_json(path, identities)


def validate_video_inputs(output):
    for path in Path(output).glob('recovery_inputs_rank_*.json'):
        for expected in json.loads(path.read_text()):
            if file_identity(expected[0]) != expected:
                raise ValueError(f'Video input changed: {expected[0]}')
