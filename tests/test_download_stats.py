from datetime import datetime, timezone
from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from malow import download_stats as stats


NOW = datetime(2026, 10, 16, 9, 17, tzinfo=timezone.utc)
REPOS = ['owner/model-a', 'owner/model-b']


def row(repo_id, count=0):
    return {'repo_id': repo_id, 'observed_at': NOW.isoformat(),
            'downloads_30d': count, 'downloads_all_time': count}


def test_fetch_uses_only_metadata_api(monkeypatch):
    requests = []
    def open_request(request, timeout):
        requests.append((request.full_url, timeout))
        return BytesIO(json.dumps({'id': REPOS[0], 'downloads': 2, 'downloadsAllTime': 9}).encode())
    monkeypatch.setattr(stats, 'urlopen', open_request)
    result = stats.fetch_counts(REPOS[0])
    assert result['downloads_30d'] == 2
    assert result['downloads_all_time'] == 9
    assert requests == [('https://huggingface.co/api/models/owner/model-a?expand%5B%5D=downloads&expand%5B%5D=downloadsAllTime', 30)]


@pytest.mark.parametrize('value', [None, -1, True, '4'])
def test_missing_or_invalid_counter_is_not_zero(monkeypatch, value):
    monkeypatch.setattr(stats, 'urlopen', lambda *a, **k: BytesIO(json.dumps(
        {'id': REPOS[0], 'downloads': 1, 'downloadsAllTime': value}).encode()))
    with pytest.raises(ValueError, match='downloadsAllTime'):
        stats.fetch_counts(REPOS[0])


@pytest.mark.parametrize('error', [URLError('temporary'), HTTPError('url', 429, 'limited', {}, None),
                                 HTTPError('url', 503, 'unavailable', {}, None)])
def test_transient_failure_retries(monkeypatch, error):
    calls = []
    sleeps = []
    def open_request(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise error
        return BytesIO(json.dumps({'id': REPOS[0], 'downloads': 0, 'downloadsAllTime': 0}).encode())
    monkeypatch.setattr(stats, 'urlopen', open_request)
    monkeypatch.setattr(stats.time, 'sleep', sleeps.append)
    assert stats.fetch_counts(REPOS[0])['downloads_all_time'] == 0
    assert sleeps == [1, 2]


def test_permanent_failure_does_not_retry(monkeypatch):
    def fail(*a, **k):
        raise HTTPError('url', 404, 'missing', {}, None)
    monkeypatch.setattr(stats, 'urlopen', fail)
    monkeypatch.setattr(stats.time, 'sleep', lambda _: pytest.fail('Unexpected retry'))
    with pytest.raises(HTTPError):
        stats.fetch_counts(REPOS[0])


def test_persistent_failure_stops_after_three_attempts(monkeypatch):
    calls = []
    def fail(*a, **k):
        calls.append(1)
        raise URLError('offline')
    monkeypatch.setattr(stats, 'urlopen', fail)
    monkeypatch.setattr(stats.time, 'sleep', lambda _: None)
    with pytest.raises(URLError):
        stats.fetch_counts(REPOS[0])
    assert len(calls) == 3


def test_complete_snapshot_and_idempotent_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'fetch_counts', row)
    path = stats.save_daily_snapshot(tmp_path, REPOS, now=NOW)
    original = path.read_bytes()
    assert path.name == '2026-10-16.json'
    assert len(json.loads(original)['models']) == 2
    monkeypatch.setattr(stats, 'fetch_counts', lambda _: pytest.fail('Rerun must not fetch'))
    assert stats.save_daily_snapshot(tmp_path, REPOS, now=NOW) == path
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_partial_failure_writes_nothing(tmp_path, monkeypatch):
    def fetch(repo_id):
        if repo_id == REPOS[1]:
            raise URLError('offline')
        return row(repo_id)
    monkeypatch.setattr(stats, 'fetch_counts', fetch)
    with pytest.raises(URLError):
        stats.save_daily_snapshot(tmp_path, REPOS, now=NOW)
    assert not list(tmp_path.iterdir())


def test_existing_incomplete_snapshot_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / '2026-10-16.json'
    target.write_text(json.dumps({'schema_version': 1, 'date': '2026-10-16', 'models': [row(REPOS[0])]}))
    monkeypatch.setattr(stats, 'fetch_counts', lambda _: pytest.fail('Invalid history must not be overwritten'))
    with pytest.raises(ValueError, match='exactly'):
        stats.save_daily_snapshot(tmp_path, REPOS, now=NOW)


def test_counter_corrections_preserve_raw_values(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'fetch_counts', lambda repo: row(repo, 10))
    stats.save_daily_snapshot(tmp_path, REPOS, now=NOW)
    monkeypatch.setattr(stats, 'fetch_counts', lambda repo: row(repo, 8))
    path = stats.save_daily_snapshot(tmp_path, REPOS, now=NOW.replace(day=17))
    assert json.loads(path.read_text())['models'][0]['downloads_all_time'] == 8
