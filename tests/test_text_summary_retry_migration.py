import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('retry_migration', Path(__file__).resolve().parents[1] / 'scripts/migrate_text_summary_retry_recovery.py')
migration = importlib.util.module_from_spec(spec); spec.loader.exec_module(migration)


def fixture():
    old = {name: name.encode() for name in migration.SOURCES}
    new = dict(old, **{'text_summary.py': b'new failure-only retry implementation'})
    settings = migration.text_summary.experiment_identity()
    settings.pop('compression_attempts'); settings.pop('fresh_repair_prompt_sha256')
    settings['implementation_sha256'] = migration.sha(old['text_summary.py'])
    saved = {'checkpoint_progress': True, 'text_summary': True, 'annotation_root': '/data',
             'recovery_implementation_sha256': migration.sha(b''.join(old[n] for n in migration.SOURCES)),
             'text_summary_settings': settings, 'world_size': 8}
    return saved, old, new


def test_migration_changes_only_explicit_implementation_identities():
    saved, old, new = fixture()
    result = migration.upgraded_manifest(saved, old, new)
    assert {k for k in saved if saved[k] != result[k]} == {'recovery_implementation_sha256', 'text_summary_settings'}
    assert 'compression_attempts' not in saved['text_summary_settings']
    assert result['text_summary_settings']['compression_attempts'] == 4


@pytest.mark.parametrize('damage', ['predecessor', 'budget', 'unrelated_source'])
def test_migration_rejects_unreviewed_changes(damage):
    saved, old, new = fixture()
    if damage == 'predecessor': saved['recovery_implementation_sha256'] = 'other'
    if damage == 'budget': saved['text_summary_settings']['summary_max_tokens'] = 999
    if damage == 'unrelated_source': new['ovo.py'] = b'other'
    with pytest.raises(ValueError): migration.upgraded_manifest(saved, old, new)
