import hashlib
from pathlib import Path

import pytest

from malow.download import download_plan, main, verify_checksums


def test_video_includes_checkpoint_and_matching_base():
    plan = download_plan('video-2b', 'checkpoints', with_base=True)
    assert [item['repo_id'] for item in plan] == [
        'zhuomingliu/MaLoW-Qwen3.5-2B',
        'Qwen/Qwen3.5-2B',
    ]


def test_all_downloads_deduplicate_shared_base():
    plan = download_plan('all', 'checkpoints', with_base=True)
    assert len(plan) == 11
    assert len({item['repo_id'] for item in plan}) == 11
    assert sum(item['kind'] == 'model' for item in plan) == 6


def test_inventory_detects_corruption(tmp_path):
    target = tmp_path / 'weights'
    target.write_bytes(b'correct')
    (tmp_path / 'SHA256SUMS').write_text(hashlib.sha256(b'correct').hexdigest() + '  weights\n')
    verify_checksums(tmp_path)
    target.write_bytes(b'incorrect')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify_checksums(tmp_path)


def test_inventory_rejects_path_escape(tmp_path):
    (tmp_path / 'SHA256SUMS').write_text('0' * 64 + '  ../outside\n')
    with pytest.raises(ValueError, match='Invalid inventory path'):
        verify_checksums(tmp_path)


def test_last4s_reuses_qwen4b_base():
    plan = download_plan('video-4b-last4s', 'checkpoints', with_base=True)
    assert [item['repo_id'] for item in plan] == [
        'zhuomingliu/MaLoW-Qwen3.5-4B-Last4s', 'Qwen/Qwen3.5-4B',
    ]


def test_last8s_uses_qwen3vl_instruct_base():
    plan = download_plan('video-qwen3vl-4b-last8s', 'checkpoints', with_base=True)
    assert [item['repo_id'] for item in plan] == [
        'zhuomingliu/MaLoW-Qwen3-VL-4B-Last8s', 'Qwen/Qwen3-VL-4B-Instruct',
    ]
    assert plan[1]['revision'] == 'ebb281ec70b05090aa6165b016eac8ec08e71b17'


def test_all_snapshot_downloads_preserve_revisions_and_validate_added_config(tmp_path, monkeypatch):
    import huggingface_hub
    plan = download_plan('all', tmp_path, with_base=True)
    calls = []
    def snapshot_download(*, repo_id, revision, local_dir):
        calls.append({'repo_id': repo_id, 'revision': revision, 'local_dir': local_dir})
        folder = Path(local_dir)
        folder.mkdir()
        if repo_id.startswith('zhuomingliu/'):
            payloads = {'model.safetensors': b'unchanged-weights', 'config.json': b'{"format":"malow-memory-module"}'}
            for name, data in payloads.items():
                (folder / name).write_bytes(data)
            (folder / 'SHA256SUMS').write_text(''.join(
                hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name, data in payloads.items()))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    main(['--model', 'all', '--with-base', '--output', str(tmp_path)])
    assert calls == [{key: item[key] for key in ('repo_id', 'revision', 'local_dir')} for item in plan]
