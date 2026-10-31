"""Recovery tests interrupt the actual evaluation loops, not a duplicate model."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from malow.video import progress, streaming, ovo, text_summary, interactive, sampling
import test_video_interactive_streaming as stream_tests
import test_video_interactive_ovo as ovo_tests
from test_video_text_summary import Tokenizer


@pytest.fixture
def streaming_stubs():
    case = stream_tests.InteractiveStreamingBenchTests()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


@pytest.fixture
def ovo_stubs():
    case = ovo_tests.InteractiveSessionTests()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


def question(time, name='q'):
    return {'time_stamp': time, 'question': name, 'options': ['A. x', 'B. y'], 'answer': 'A'}


def test_journal_durability_tail_repair_and_duplicates(tmp_path):
    path = tmp_path / 'answers.jsonl'
    j = progress.PredictionJournal(path, lambda r: r['id'], {1, 2})
    j.append({'id': 1, 'prediction': 'A'})
    with path.open('ab') as f:
        f.write(b'{"id":2,"prediction":"\xe6')
    recovered = progress.PredictionJournal(path, lambda r: r['id'], {1, 2})
    assert list(recovered.rows) == [1]
    assert list(tmp_path.glob('*.incomplete-*'))
    recovered.append({'id': 2, 'prediction': 'B'})
    recovered.require_complete()
    with pytest.raises(ValueError, match='Conflicting'):
        recovered.append({'id': 1, 'prediction': 'C'})
    with path.open('a') as f:
        f.write('{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        progress.PredictionJournal(path, lambda r: r['id'], {1, 2})


def test_grouping_preserves_ids_context_and_single_worker(streaming_stubs):
    data = [{'video_path': '/videos/a.mp4', 'questions': [question('00:00:30', 'late'), question('00:00:10', 'early')]},
            {'video_path': '/videos/b.mp4', 'questions': [question('00:00:10')]},
            {'video_path': '/videos/a.mp4', 'questions': [question('00:00:10', 'other entry')]}]
    groups = streaming.group_video_entries(data, 'sqa', Path('/ann.json'))
    ids = [streaming.question_id('sqa', i, j, q) for i, entry in groups for j, q in enumerate(entry['questions'])]
    assert set(ids) == {'sqa:0:0:00:00:30', 'sqa:0:1:00:00:10', 'sqa:1:0:00:00:10', 'sqa:2:0:00:00:10'}
    qs = groups[0][1]['questions']
    assert 'Question: late' in qs[1]['_source_context']
    assert qs[2]['_source_context'] == ''
    model = stream_tests.FakeModel()
    args = stream_tests.make_args(streaming_group_by_video=True)
    with mock.patch.object(streaming, 'generate_with_loras_single', return_value='A'):
        streaming.evaluate_mc_task_interactive('sqa', groups, Path('/ann.json'), model, object(), args,
                                               Path('/unused'), 0, 1)
    rows = streaming_stubs.written
    assert [r['question'] for r in rows[:3]] == ['early', 'other entry', 'late']
    assert [r['entry_index'] for r in rows[:3]] == [0, 2, 0]
    assert [r['question_index'] for r in rows[:3]] == [1, 0, 0]
    assert sum(model.states[0].segments) == 28
    assert len(model.states) == 2


@pytest.mark.parametrize('interrupt_after', [1, 2])
def test_grouped_streaming_resume_skips_answers_but_replays_history(tmp_path, streaming_stubs, monkeypatch, interrupt_after):
    monkeypatch.setattr(streaming, 'file_identity', lambda p: [str(p), 1, 0])
    data = [{'video_path': '/videos/a.mp4', 'questions': [question(t)]} for t in ('00:00:10', '00:00:40')]
    groups = streaming.group_video_entries(data, 'real', Path('/ann.json'))
    args = stream_tests.make_args(output_dir=str(tmp_path), checkpoint_progress=True,
                                 streaming_group_by_video=True, eval_mode='fixed-gap', rollout_step_seconds=30)
    path = tmp_path / 'real/rank_00000.jsonl'
    append = progress.PredictionJournal.append
    calls = []
    def interrupted(self, row):
        append(self, row)
        if len(self.rows) == interrupt_after:
            raise RuntimeError('power loss')
    def run(model):
        return streaming.evaluate_mc_task_interactive('real', groups, Path('/ann.json'), model, object(), args, path, 0, 1)
    with mock.patch.object(streaming, 'generate_with_loras_single', side_effect=lambda *a, **k: calls.append(1) or 'A'):
        with mock.patch.object(progress.PredictionJournal, 'append', interrupted), pytest.raises(RuntimeError, match='power loss'):
            run(stream_tests.FakeModel())
        resumed = stream_tests.FakeModel()
        tracked = run(resumed)
        assert len(calls) == 2
        assert len(progress.read_journal(path)) == 2
        assert tracked[0].turns == 2
        if interrupt_after == 1:
            assert sum(resumed.states[0].segments) == 38
        else:
            assert not resumed.states


def test_ovo_saves_before_worker_finishes_and_resumes(tmp_path, ovo_stubs):
    args = ovo_tests.make_args(output_dir=str(tmp_path), checkpoint_progress=True)
    sessions = ovo.build_interactive_sessions(ovo_stubs.units, args)
    append = progress.PredictionJournal.append
    def interrupted(self, row):
        append(self, row)
        if len(self.rows) == 2:
            raise RuntimeError('power loss')
    with mock.patch.object(progress.PredictionJournal, 'append', interrupted), pytest.raises(RuntimeError):
        ovo.run_interactive_sessions(ovo_tests.FakeModel(), ovo_tests.FakeProcessor(), sessions, args, 0)
    assert len(progress.read_journal(tmp_path / 'rank_0_predictions.jsonl')) == 2
    model = ovo_tests.FakeModel()
    rows, tracked = ovo.run_interactive_sessions(model, ovo_tests.FakeProcessor(), sessions, args, 0)
    assert len(rows) == 3 and len({progress.ovo_key(r) for r in rows}) == 3
    assert sum(model.states[0].segments) == 28
    assert tracked[0].turns == 3 and tracked[0].fallback_turns == 1
    model = ovo_tests.FakeModel()
    ovo.run_interactive_sessions(model, ovo_tests.FakeProcessor(), sessions, args, 0)
    assert not model.states


@pytest.mark.parametrize('stage', ['caption', 'summary'])
def test_text_cache_survives_interruption_and_preserves_memory(tmp_path, monkeypatch, stage):
    args = SimpleNamespace(checkpoint_progress=True, output_dir=str(tmp_path))
    video = tmp_path / 'video.mp4'; video.touch()
    class Processor:
        tokenizer = Tokenizer()
        def __call__(self, **kw): return kw
    processor = Processor()
    monkeypatch.setattr(text_summary, 'chat_text', lambda *a: 'prompt')
    monkeypatch.setattr(sampling, 'load_video_frames_at_times', lambda p, times: list(times))
    monkeypatch.setattr(sampling, 'processor_predecoded_video_inputs', lambda *a, **k: {})
    value = {key: [] for key in text_summary.KEYS}; value['events'] = ['[0,2)s: cup moves']
    generated = []
    def generate(model, processor, inputs, limit):
        generated.append(limit)
        return 'a cup moves' if limit == text_summary.CAPTION_TOKENS else json.dumps(value)
    monkeypatch.setattr(text_summary, 'base_generate', generate)
    def session():
        return SimpleNamespace(progress_key=['real', 'v'], text_summary=text_summary.TextSummaryState(origin=0, end=0),
                               config=interactive.InteractiveConfig())
    original = progress.cache_put
    def interrupted(args, session, kind, key, value):
        original(args, session, kind, key, value)
        if kind == stage: raise RuntimeError('power loss')
    with mock.patch.object(progress, 'cache_put', interrupted), pytest.raises(RuntimeError):
        text_summary.update_memory(None, processor, session(), str(video), args, [0., 1.])
    resumed = session()
    text_summary.update_memory(None, processor, resumed, str(video), args, [0., 1.])
    assert resumed.text_summary.memory == value
    assert resumed.text_summary.revision == 1
    assert generated == [text_summary.CAPTION_TOKENS, text_summary.COMPRESSION_TOKENS]
    record = next((tmp_path / 'recovery').glob('*/summary/*.json'))
    corrupt = json.loads(record.read_text()); corrupt['value']['tokens'] += 1
    record.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match='Corrupt'):
        text_summary.update_memory(None, processor, session(), str(video), args, [0., 1.])


def test_proactive_resume_reuses_poll_outcomes(tmp_path, streaming_stubs, monkeypatch):
    monkeypatch.setattr(streaming, 'file_identity', lambda p: [str(p), 1, 0])
    args = stream_tests.make_args(output_dir=str(tmp_path), checkpoint_progress=True)
    entry = {'video_path': '/videos/a.mp4', 'questions': [{
        'time_stamp': '00:00:20', 'ground_truth_time_stamp': '00:00:26',
        'question': 'When?', 'ground_truth_output': 'ready'}]}
    path = tmp_path / 'proactive/rank_00000.jsonl'
    calls = []
    def answer(*a, **k):
        calls.append(1)
        return 'no' if len(calls) == 1 else 'yes' if len(calls) == 2 else 'ready'
    original = progress.cache_put
    interrupted = False
    def persist_then_stop(args, session, kind, key, value):
        nonlocal interrupted
        original(args, session, kind, key, value)
        if kind == 'answer' and not interrupted:
            interrupted = True
            raise RuntimeError('power loss')
    def run():
        return streaming.evaluate_proactive_task_interactive([(0, entry)], Path('/ann.json'),
            stream_tests.FakeModel(), object(), args, path, 0, 1)
    # The first poll falls back; save/reuse that response too.
    with mock.patch.object(streaming, 'generate_c2l_batch', side_effect=lambda *a, **k: [answer()]), \
         mock.patch.object(streaming, 'generate_with_loras_single', side_effect=answer), \
         mock.patch.object(streaming, 'can_decode_video_time_range', return_value=True):
        # interactive_answer imports cache_put at module scope.
        with mock.patch.object(streaming, 'cache_put', persist_then_stop), pytest.raises(RuntimeError):
            run()
        run()
    assert len(calls) == 3
    row = progress.read_journal(path)[0]
    assert row['prediction'] == 'ready'
    assert [r['content'] for r in row['dialog_history'] if r['role'] == 'assistant'] == ['no', 'yes', 'ready']


def test_recovery_identity_rejects_annotation_change(tmp_path):
    from malow.video.interactive_options import guard_interactive_run
    annotations = tmp_path / 'data'; annotations.mkdir()
    source = annotations / 'questions.json'; source.write_text('[]')
    output = tmp_path / 'out'
    args = SimpleNamespace(eval_mode='fixed-gap', output_dir=str(output), c2l_checkpoint=str(tmp_path / 'checkpoint'),
                           model_id=str(tmp_path / 'base'), data_root=str(annotations), checkpoint_progress=True)
    guard_interactive_run(args, 1)
    guard_interactive_run(args, 1)
    with pytest.raises(ValueError, match='world_size'):
        guard_interactive_run(args, 2)
    source.write_text('[{}]')
    with pytest.raises(ValueError, match='annotation_fingerprints'):
        guard_interactive_run(args, 1)


def test_worker_lock_rejects_concurrent_resume(tmp_path):
    from malow.video.interactive_options import init_resolution_log
    first = SimpleNamespace(output_dir=str(tmp_path), eval_mode='fixed-gap', checkpoint_progress=True)
    second = SimpleNamespace(output_dir=str(tmp_path), eval_mode='fixed-gap', checkpoint_progress=True)
    init_resolution_log(first, 0)
    try:
        with pytest.raises(RuntimeError, match='Another recovery worker'):
            init_resolution_log(second, 0)
    finally:
        first._progress_lock.close()


def test_video_input_changes_rejected(tmp_path):
    video = tmp_path / 'v.mp4'; video.write_bytes(b'video')
    args = SimpleNamespace(output_dir=str(tmp_path), checkpoint_progress=True)
    progress.guard_video_inputs(args, 0, [video])
    video.write_bytes(b'different video')
    with pytest.raises(ValueError, match='Video inputs changed'):
        progress.guard_video_inputs(args, 0, [video])


def test_grouping_and_recovery_flags_are_opt_in_and_forwarded():
    from malow.video.eval import parse_args, build_command
    common = ['--benchmark', 'streaming', '--checkpoint', 'ckpt', '--base-model', 'base',
              '--data-root', 'data', '--output-dir', 'out', '--preset', 'last2s']
    legacy = parse_args(common)
    assert not hasattr(legacy, 'checkpoint_progress') and not hasattr(legacy, 'streaming_group_by_video')
    with pytest.raises(SystemExit):
        parse_args(common + ['--checkpoint-progress'])
    args = parse_args(common + ['--eval-mode', 'fixed-gap', '--checkpoint-progress', '--streaming-group-by-video'])
    cmd = build_command(args)
    assert '--checkpoint-progress' in cmd and '--streaming-group-by-video' in cmd
    with pytest.raises(SystemExit):
        parse_args(['--benchmark', 'ovo', *common[2:], '--video-root', 'videos', '--eval-mode', 'fixed-gap', '--streaming-group-by-video'])
