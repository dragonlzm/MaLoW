import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from malow.video import text_cache, text_summary, progress, interactive, streaming, ovo
from test_video_text_summary import Tokenizer
import test_video_interactive_streaming as st
import test_video_interactive_ovo as ot


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'interactive_run_config.json').write_text('{}')
    video = tmp_path / 'video.mp4'; video.write_bytes(b'video contents')
    old = {k: [] for k in text_summary.KEYS}
    memory = {**old, 'events': ['[0,2)s cup moves']}
    key = {'video': progress.file_identity(video), 'frame_times': [0., 1.], 'previous_memory': progress.digest(old), 'revision': 0}
    args = Namespace(checkpoint_progress=True, output_dir=str(source))
    session = SimpleNamespace(progress_key=['real', str(video)])
    progress.cache_put(args, session, 'caption', key, {'caption': 'A cup moves.', 'seconds': 1.})
    progress.cache_put(args, session, 'summary', key, {'memory': memory,
        'tokens': len(Tokenizer().encode(text_summary.serialize(memory))), 'seconds': 2., 'normalizations': []})
    destination = tmp_path / 'bundle'
    result = text_cache.export_bundle(source, destination, 'streaming')
    return destination, video, key, memory, result


def test_bundle_roundtrip_checksums_and_relocated_video(bundle, tmp_path):
    destination, video, key, memory, result = bundle
    cache = text_cache.TextCache(destination)
    relocated = tmp_path / 'relocated'; relocated.mkdir()
    copy = relocated / video.name; copy.write_bytes(video.read_bytes())
    assert cache.get('summary', copy, key['frame_times'], key['previous_memory'], 0)['memory'] == memory
    assert result['segments'] == 1
    copy.write_bytes(b'wrong contents')
    with pytest.raises(ValueError, match='video content mismatch'):
        text_cache.TextCache(destination).get('caption', copy, key['frame_times'], '', 0)
    with (destination / 'records.jsonl').open('a') as f: f.write('broken')
    with pytest.raises(ValueError, match='checksum'):
        text_cache.TextCache(destination)


def test_summary_replay_never_calls_model_and_keeps_source_read_only(bundle, tmp_path, monkeypatch):
    destination, video, key, expected, _ = bundle
    before = {p.name: text_cache.file_sha(p) for p in destination.iterdir() if p.is_file()}
    args = Namespace(checkpoint_progress=True, output_dir=str(tmp_path / 'fresh'),
                     _external_text_cache=text_cache.TextCache(destination), text_cache_mode='summaries')
    session = SimpleNamespace(progress_key=['new-session'], config=interactive.InteractiveConfig(),
                              text_summary=text_summary.TextSummaryState(origin=0,end=0))
    monkeypatch.setattr(text_summary, 'base_generate', lambda *a: pytest.fail('Replay must not generate text'))
    text_summary.update_memory(None, SimpleNamespace(tokenizer=Tokenizer()), session, str(video), args, [0.,1.])
    assert session.text_summary.memory == expected and session.text_summary.end == 2
    assert before == {p.name: text_cache.file_sha(p) for p in destination.iterdir() if p.is_file()}
    assert not list(Path(args.output_dir).rglob('*predictions*'))


def test_strict_missing_summary_does_not_use_a_future_memory(bundle, tmp_path):
    destination, video, key, expected, _ = bundle
    args = Namespace(checkpoint_progress=True, output_dir=str(tmp_path/'fresh'),
                     _external_text_cache=text_cache.TextCache(destination), text_cache_mode='summaries')
    session = SimpleNamespace(progress_key=['s'], config=interactive.InteractiveConfig(),
                              text_summary=text_summary.TextSummaryState(origin=0,end=0))
    with pytest.raises(ValueError, match='No matching external'):
        text_summary.update_memory(None, SimpleNamespace(tokenizer=Tokenizer()), session, str(video), args, [0.,1.,2.,3.])
    assert session.text_summary.revision == 0


def test_caption_mode_recompresses_without_visual_captioning(bundle, tmp_path, monkeypatch):
    destination, video, key, expected, _ = bundle
    args = Namespace(checkpoint_progress=True, output_dir=str(tmp_path/'fresh'),
                     _external_text_cache=text_cache.TextCache(destination), text_cache_mode='captions')
    session = SimpleNamespace(progress_key=['s'], config=interactive.InteractiveConfig(),
                              text_summary=text_summary.TextSummaryState(origin=0,end=0))
    class Processor:
        tokenizer=Tokenizer()
        def __call__(self, **kw): return kw
    monkeypatch.setattr(text_summary, 'chat_text', lambda *a: 'prompt')
    calls=[]
    monkeypatch.setattr(text_summary, 'base_generate', lambda *a: calls.append(a[-1]) or json.dumps(expected))
    text_summary.update_memory(None, Processor(), session, str(video), args, [0.,1.])
    assert calls == [text_summary.COMPRESSION_TOKENS]


def test_base_history_only_advances_clock_and_text(monkeypatch):
    model = SimpleNamespace(new_interactive_state=lambda: None,
                            interactive_update=lambda *a: pytest.fail('Base-only must not encode history'))
    session = interactive.new_session(model, interactive.InteractiveConfig(rollout_step_seconds=30))
    seen=[];session.on_segment=lambda video,times:seen.append(times)
    monkeypatch.setattr(interactive, 'video_duration_seconds', lambda *a: None)
    monkeypatch.setattr(interactive, 'build_segment_encoder_inputs', lambda *a,**k:pytest.fail('Historical visual input'))
    args=Namespace(text_cache_base_only=True)
    interactive.advance_session(model,None,session,'','video',65,args,None)
    assert session.consumed_blocks==31 and list(map(len,seen))==[28,30,4]


@pytest.mark.parametrize('benchmark', ['streaming','ovo'])
def test_base_answers_never_generate_lora_even_for_early_questions(benchmark):
    case = st.InteractiveStreamingBenchTests() if benchmark=='streaming' else ot.InteractiveSessionTests()
    case.setUp()
    try:
        runner=streaming if benchmark=='streaming' else ovo
        model=st.FakeModel() if benchmark=='streaming' else ot.FakeModel()
        model.interactive_update=lambda *a:pytest.fail('Base-only encoder was called')
        model.interactive_generate_weights=lambda *a:pytest.fail('LoRA weights were generated')
        args=(st.make_args if benchmark=='streaming' else ot.make_args)(text_cache_base_only=True)
        with mock.patch.object(text_summary,'generate_base_answer',return_value='A') as base:
            if benchmark=='streaming':
                data={'video_path':'/videos/a.mp4','questions':[
                    {'time_stamp':t,'question':'q','options':['A. x'],'answer':'A'} for t in ('00:00:01','00:00:10')]}
                runner.evaluate_mc_task_interactive('real',[(0,data)],Path('/ann'),model,object(),args,Path('/unused'),0,1)
                assert base.call_count==2
            else:
                runner.run_interactive_sessions(model,ot.FakeProcessor(),runner.build_interactive_sessions(case.units,args),args,0)
                assert base.call_count==3
    finally:case.doCleanups()
