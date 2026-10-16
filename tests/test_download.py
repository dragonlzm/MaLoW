import hashlib
from pathlib import Path

import pytest

from malow.download import download_plan, verify_checksums


def test_vla_includes_stage1_and_one_base():
    plan = download_plan('vla', 'checkpoints', with_base=True)
    assert [item['repo_id'] for item in plan] == [
        'zhuomingliu/Qwen3.5-2B-MLP',
        'zhuomingliu/MaLoW-Qwen3.5-2B-MLP',
        'Qwen/Qwen3.5-2B',
    ]


def test_all_downloads_deduplicate_shared_base_and_dependency():
    plan = download_plan('all', 'checkpoints', with_base=True)
    assert len(plan) == 13
    assert len({item['repo_id'] for item in plan}) == 13


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
