from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from malow.video import interactive as live, text_summary as memory
from malow.video.eval import parse_args, build_command
from malow.video.interactive_options import guard_interactive_run


def launcher(*extra):
    return parse_args(["--benchmark", "ovo", "--checkpoint", "checkpoint", "--base-model", "base",
                       "--data-root", "data", "--video-root", "video", "--output-dir", "out",
                       "--preset", "last2s", *extra])


def test_opt_in_and_legacy_identity(tmp_path):
    args = launcher("--eval-mode", "fixed-gap")
    assert not hasattr(args, "text_summary")
    assert "--text-summary" not in build_command(args)
    args.output_dir = str(tmp_path)
    args.c2l_checkpoint = "checkpoint"
    guard_interactive_run(args, 1)
    config = json.loads((tmp_path / "interactive_run_config.json").read_text())
    assert not any("summary" in key for key in config)
    args.text_summary = True
    with pytest.raises(ValueError, match="text_summary"):
        guard_interactive_run(args, 1)


@pytest.mark.parametrize("mode", ["offline", "interactive"])
def test_summary_rejects_other_modes(mode):
    with pytest.raises(SystemExit):
        launcher("--eval-mode", mode, "--text-summary")


@pytest.mark.parametrize("benchmark", ["ovo", "streaming"])
def test_worker_receives_opt_in(benchmark, monkeypatch):
    from malow.video import ovo, streaming
    args = launcher("--eval-mode", "fixed-gap", "--rollout-step-seconds", "10", "--text-summary")
    args.benchmark = benchmark
    command = build_command(args)
    monkeypatch.setattr("sys.argv", ["worker", *command[3:]])
    parsed = (ovo if benchmark == "ovo" else streaming).parse_args()
    assert parsed.text_summary and parsed.rollout_step_seconds == 10


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def test_summary_validation_and_budget():
    summary = {key: [] for key in memory.KEYS}
    summary["events"] = ["[0,2)s: person puts red cup on table"]
    result, count = memory.validate_summary("```json\n" + json.dumps(summary) + "\n```", Tokenizer())
    assert result == summary and count <= 1024
    with pytest.raises(ValueError):
        memory.validate_summary('{"events": []}', Tokenizer())
    summary["events"] = ["x" * 1024]
    with pytest.raises(ValueError, match="maximum"):
        memory.validate_summary(json.dumps(summary), Tokenizer())
    summary["events"] = [None]
    with pytest.raises(ValueError, match="nonempty"):
        memory.validate_summary(json.dumps(summary), Tokenizer())


@pytest.mark.parametrize("case", json.loads(
    (Path(__file__).parent / "fixtures/text_summary_format_failures.json").read_text()
), ids=lambda case: case["name"])
def test_captured_production_failure_is_normalized_losslessly(case, monkeypatch):
    # Character tokenizer here tests preservation, not Qwen's vocabulary. The
    # canonical budget is tested separately, including its escaping overhead.
    monkeypatch.setattr(memory, "SUMMARY_TOKENS", 10000)
    original, _end = json.JSONDecoder().raw_decode(case["raw"])
    result, _count, changes = memory.normalize_summary(case["raw"], Tokenizer())
    assert set(result) == set(original)
    for key in memory.KEYS:
        assert [json.loads(entry) for entry in result[key]] == original[key]
    assert any(change.startswith("object_entry:") for change in changes)
    assert ("trailing_quote" in changes) == (case["name"].startswith("initial"))


def test_mixed_entries_preserve_order_types_and_quoted_text():
    value = {key: [] for key in memory.KEYS}
    record = {"id": "C1", "count": 2, "obj": 'cups marked "水"', "time": [0, 8],
              "details": {"uncertain": False}}
    value["counts"] = ["[0,2)s: one cup", record, "[8s]: last observed count two"]
    result, _count, changes = memory.normalize_summary(json.dumps(value), Tokenizer())
    assert result["counts"][0] == value["counts"][0]
    assert json.loads(result["counts"][1]) == record
    assert result["counts"][2] == value["counts"][2]
    assert changes == ["object_entry:counts[1]"]
    # Canonical records are stable when they re-enter the compressor as old memory.
    again, _count, changes = memory.normalize_summary(memory.serialize(result), Tokenizer())
    assert again == result and changes == []


def test_budget_counts_normalization_escaping_overhead():
    value = {key: [] for key in memory.KEYS}
    value["text"] = [{"desc": '"' * 250}]
    raw = memory.serialize(value)
    assert len(raw) <= memory.SUMMARY_TOKENS
    with pytest.raises(ValueError, match="maximum"):
        memory.normalize_summary(raw, Tokenizer())


@pytest.mark.parametrize("suffix", ['""', " another response", '{}'])
def test_normalization_does_not_discard_trailing_content(suffix):
    raw = memory.serialize({key: [] for key in memory.KEYS})
    with pytest.raises(ValueError, match="Unexpected content"):
        memory.normalize_summary(raw + suffix, Tokenizer())


@pytest.mark.parametrize("entry", [None, 3, True, [], {}, ""])
def test_invalid_entries_are_not_coerced_to_facts(entry):
    value = {key: [] for key in memory.KEYS}
    value["events"] = [entry]
    with pytest.raises(ValueError, match="nonempty"):
        memory.normalize_summary(json.dumps(value), Tokenizer())


def test_normalization_rejects_truncation_duplicate_keys_and_nonfinite_numbers():
    value = {key: [] for key in memory.KEYS}
    raw = memory.serialize(value)
    with pytest.raises(ValueError):
        memory.normalize_summary(raw[:-1], Tokenizer())
    with pytest.raises(ValueError, match="Duplicate"):
        memory.normalize_summary(raw[:-1] + ',"events":[]}', Tokenizer())
    for token in ("NaN", "Infinity", "-Infinity", "1e999"):
        raw = memory.serialize(value).replace('"events":[]', '"events":[{"count":' + token + '}]')
        with pytest.raises(ValueError, match="Nonfinite"):
            memory.normalize_summary(raw, Tokenizer())


def test_object_output_updates_memory_without_model_repair(monkeypatch, tmp_path):
    from malow.video import sampling
    state = memory.TextSummaryState(origin=0, end=0)
    session = SimpleNamespace(text_summary=state, config=live.InteractiveConfig())
    args = Namespace(_text_summary_log=str(tmp_path / "trace.jsonl"))
    class Processor:
        tokenizer = Tokenizer()
        def __call__(self, **kwargs):
            return kwargs
    monkeypatch.setattr(memory, "chat_text", lambda processor, content: "chat")
    monkeypatch.setattr(sampling, "load_video_frames_at_times", lambda path, times: list(times))
    monkeypatch.setattr(sampling, "processor_predecoded_video_inputs", lambda *a, **k: {})
    value = {key: [] for key in memory.KEYS}
    value["events"] = [{"desc": "person places cup", "time": "0-2s"}]
    responses = iter(["[0,2)s person places cup", json.dumps(value) + '"'])
    monkeypatch.setattr(memory, "base_generate", lambda *args: next(responses))
    memory.update_memory(None, Processor(), session, "video", args, [0, 1])
    assert state.end == 2 and state.revision == 1
    assert json.loads(state.memory["events"][0]) == value["events"][0]
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["caption", "summary"]
    assert rows[-1]["attempts"] == 1
    assert rows[-1]["normalizations"] == ["trailing_quote", "object_entry:events[0]"]


def test_base_chat_preserves_disabled_thinking_template():
    suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "user input" + suffix
    assert memory.chat_text(Processor(), [{"type": "text", "text": "caption"}]).endswith(suffix)


def test_ovo_opt_in_distributed_timeout(monkeypatch):
    import torch
    from datetime import timedelta
    from malow.video import ovo
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs))
    ovo.init_distributed()
    ovo.init_distributed(timeout_seconds=86400)
    assert calls == [{"backend": "gloo"}, {"backend": "gloo", "timeout": timedelta(days=1)}]


@pytest.mark.parametrize("fail", [False, True])
def test_base_generation_restores_lora_after_exception(fail):
    original, adapted = object(), object()
    module = SimpleNamespace(forward=adapted, forward_orig=original, patched_forward=True)
    base = SimpleNamespace(modules=lambda: [module])
    model = SimpleNamespace(base_model=base, generated_loras={"unchanged": 1})
    try:
        with memory.unadapted_base(model) as unadapted:
            assert unadapted is base and module.forward is original
            if fail:
                raise RuntimeError("generation failed")
    except RuntimeError:
        pass
    assert module.forward is adapted and module.patched_forward
    assert model.generated_loras == {"unchanged": 1}


def test_unadapted_base_matches_original_numerically():
    import torch
    layer = torch.nn.Linear(2, 2, bias=False)
    sample = torch.tensor([[1., 2.]])
    original = layer(sample).detach()
    layer.forward_orig = layer.forward
    layer.forward = lambda value: layer.forward_orig(value) + 10
    layer.patched_forward = True
    model = SimpleNamespace(base_model=layer)
    assert torch.allclose(layer(sample), original + 10)
    with memory.unadapted_base(model) as base:
        assert torch.equal(base(sample), original)
    assert torch.allclose(layer(sample), original + 10)


def test_summary_uses_exact_committed_history_and_replays(monkeypatch):
    model = SimpleNamespace(new_interactive_state=lambda: [],
                            interactive_update=lambda inputs, state: state.append(inputs))
    monkeypatch.setattr(live, "video_duration_seconds", lambda _: 45)
    monkeypatch.setattr(live, "build_segment_encoder_inputs", lambda p, t, v, times, *a, **k: times)
    observed = []
    def update(model, processor, session, path, args, times):
        observed.append((session.text_summary.session_id, list(times)))
        session.text_summary.end = times[-1] + 1
        session.text_summary.revision += 1
    monkeypatch.setattr(memory, "update_memory", update)
    config = live.InteractiveConfig(rollout_step_seconds=10)
    def replay():
        session = live.new_session(model, config)
        for now in [3, 10, 15, 15, 20, 100]:
            live.advance_session(model, None, session, "encoder", "video", now,
                                 Namespace(text_summary=True), None)
        return session
    first = replay()
    first_intervals = [times for sid, times in observed]
    assert first_intervals[:3] == [list(range(8)), list(range(8, 12)), list(range(12, 18))]
    assert all(len(times) <= 10 for times in first_intervals)
    assert [t for times in first_intervals for t in times] == list(range(42))
    assert first.text_summary.end == 42
    second = replay()
    assert observed[len(first_intervals):] == [(second.text_summary.session_id, times) for times in first_intervals]
    assert second.text_summary.session_id != first.text_summary.session_id
    live.finish_session(first)
    assert first.on_segment is None and not hasattr(first, "text_summary")


def test_answer_memory_before_video_and_unchanged_question():
    state = memory.TextSummaryState(origin=0, end=18, revision=1)
    state.memory["states"] = ["[8s]: cup on table; later location unobserved"]
    session = SimpleNamespace(text_summary=state, origin=0, config=live.InteractiveConfig())
    messages = [{"role": "user", "content": [{"type": "video", "video": "clip"},
                 {"type": "text", "text": "Q?\nA. x\nB. y\nAnswer one letter."}]}]
    question = messages[0]["content"][-1].copy()
    memory.prepend_memory(messages, memory.answer_prefix(session, 20))
    assert messages[0]["content"][1]["type"] == "video"
    assert messages[0]["content"][-1] == question
    assert "[18s, 20s]" in messages[0]["content"][0]["text"]
    assert "does not prove" in messages[0]["content"][0]["text"]
    assert memory.answer_prefix(SimpleNamespace(), 2) is None


def test_ovo_progressive_chunks_use_current_file_for_captions(monkeypatch):
    model = SimpleNamespace(new_interactive_state=lambda: [],
                            interactive_update=lambda inputs, state: state.append(inputs))
    monkeypatch.setattr(live, "video_duration_seconds", lambda path: {"short": 10., "long": 20.}[path])
    encoded, captioned = [], []
    def encode(processor, text, path, times, *args, **kwargs):
        encoded.append((path, times))
        return times
    monkeypatch.setattr(live, "build_segment_encoder_inputs", encode)
    monkeypatch.setattr(memory, "update_memory",
                        lambda model, processor, session, path, args, times: captioned.append((path, times)))
    session = live.new_session(model, live.InteractiveConfig(rollout_step_seconds=10))
    for path, now in [("short", 10), ("long", 20)]:
        live.advance_session(model, None, session, "", path, now, Namespace(text_summary=True), None)
    assert encoded == captioned == [("short", list(range(8))), ("long", list(range(8, 18)))]


@pytest.mark.parametrize("repair_success", [False, True])
def test_caption_compression_uses_only_history_and_bounded_repairs(monkeypatch, tmp_path, repair_success):
    from malow.video import sampling
    state = memory.TextSummaryState(origin=0, end=0)
    session = SimpleNamespace(text_summary=state, config=live.InteractiveConfig())
    args = Namespace(_text_summary_log=str(tmp_path / "trace.jsonl"))
    processor = SimpleNamespace(tokenizer=Tokenizer())
    class Processor:
        tokenizer = processor.tokenizer
        def __call__(self, **kwargs):
            return kwargs
    processor = Processor()
    captured = []
    monkeypatch.setattr(memory, "chat_text", lambda processor, content: captured.append(content) or "chat")
    monkeypatch.setattr(sampling, "load_video_frames_at_times", lambda path, times: list(times))
    monkeypatch.setattr(sampling, "processor_predecoded_video_inputs", lambda *a, **k: {})
    good = {key: [] for key in memory.KEYS}
    good["events"] = ["[0,2)s person places cup"]
    responses = iter(["[0,2)s person places cup", "invalid JSON", json.dumps(good) if repair_success else "invalid", "invalid", "invalid"])
    monkeypatch.setattr(memory, "base_generate", lambda *args: next(responses))
    if repair_success:
        memory.update_memory(None, processor, session, "video", args, [0, 1])
        assert state.memory == good and state.end == 2 and state.revision == 1
    else:
        with pytest.raises(RuntimeError, match="after 4 attempts"):
            memory.update_memory(None, processor, session, "video", args, [0, 1])
        assert state.revision == 0 and state.end == 0
    assert len(captured) == (3 if repair_success else 5)
    assert "OLD_MEMORY" in captured[1][0]["text"]
    assert "NEW_CAPTION" in captured[1][0]["text"]
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["frame_times"] == [0, 1]
    assert sum(row["kind"] == "invalid_summary" for row in rows) == (1 if repair_success else 4)


def test_repetitive_failure_gets_fresh_evidence_only_repair(monkeypatch):
    from malow.video import sampling
    case = json.loads((Path(__file__).parent / 'fixtures/text_summary_repetition_failure.json').read_text())
    state = memory.TextSummaryState(origin=0, end=28, revision=1, memory=case['old_memory'])
    session = SimpleNamespace(text_summary=state, config=live.InteractiveConfig())
    class Processor:
        tokenizer = Tokenizer()
        def __call__(self, **kwargs): return kwargs
    prompts = []
    monkeypatch.setattr(memory, 'chat_text', lambda p, c: prompts.append(c[0]['text'] if c[0]['type']=='text' else 'caption') or 'chat')
    monkeypatch.setattr(sampling, 'load_video_frames_at_times', lambda p, t: list(t))
    monkeypatch.setattr(sampling, 'processor_predecoded_video_inputs', lambda *a, **k: {})
    good = {key: [] for key in memory.KEYS}; good['events'] = ['[28,58)s person remains seated']
    outputs = iter([case['caption'], *case['failed_outputs'], json.dumps(good)])
    monkeypatch.setattr(memory, 'base_generate', lambda *a: next(outputs))
    memory.update_memory(None, Processor(), session, 'video', Namespace(), case['frame_times'])
    assert state.memory == good and state.end == 58 and state.revision == 2
    assert len(prompts) == 4
    assert 'PREVIOUS_INVALID_OUTPUT' not in prompts[3]
    assert case['caption'] in prompts[3] or memory.serialize(case['caption'])[1:-1] in prompts[3]
    for bad in case['failed_outputs']:
        assert bad not in prompts[3]
        with pytest.raises(ValueError):
            memory.normalize_summary(bad, Tokenizer())
