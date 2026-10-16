"""Scheduling, CLI compatibility, resolution and output isolation."""
from argparse import Namespace
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from malow.video import interactive as live, sampling
from malow.video.eval import parse_args, build_command
from malow.video.interactive_options import guard_interactive_run


class Recorder:
    frame_per_lora = 2
    interleave_timestamp = False

    def new_interactive_state(self):
        return []

    def interactive_update(self, inputs, state):
        state.append(inputs)


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(live, "video_duration_seconds", lambda _: 3000.0)
    monkeypatch.setattr(live, "build_segment_encoder_inputs", lambda processor, text, path, times, *a, **k: times)
    def drive(config, questions, origin=0):
        model = Recorder()
        session = live.new_session(model, config, origin)
        added = []
        for t in questions:
            added.append(live.advance_session(model, None, session, "", "clip", t, Namespace(), None))
        return session, added
    return drive


@pytest.mark.parametrize("tail", [2.0, 4.0, 8.0])
@pytest.mark.parametrize("step", [0.5, 2.0, 3.3, 10.0])
def test_fixed_gap_and_questions_consume_identical_causal_grid(scheduler, tail, step):
    questions = [1.0, 3.0, 10.5, 10.5, 11.0, 20.0, 1200.7]
    config = live.InteractiveConfig(tail_seconds=tail, max_segment_blocks=7)
    question, qadded = scheduler(config, questions)
    fixed, fadded = scheduler(replace(config, rollout_step_seconds=step), questions)
    frames = lambda s: [t for segment in s.state for t in segment]
    assert qadded == fadded
    assert frames(question) == frames(fixed) == list(range(2 * question.consumed_blocks))
    assert len(frames(fixed)) > 766
    assert len(set(frames(fixed))) == len(frames(fixed))
    assert max(map(len, fixed.state)) <= 14
    assert max(frames(fixed)) < questions[-1] - tail


def test_fixed_ticks_remain_anchored_after_question_catchup(scheduler):
    session, added = scheduler(live.InteractiveConfig(rollout_step_seconds=5.0), [8.0, 9.1, 10.0])
    assert added == [3, 0, 1]
    assert session.next_tick == 3
    assert session.state == [[0.0, 1.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]


def test_origin_clamp_and_short_prefix(scheduler, monkeypatch):
    monkeypatch.setattr(live, "video_duration_seconds", lambda _: 40.0)
    session, added = scheduler(live.InteractiveConfig(rollout_step_seconds=2), [31, 33.9, 34, 10000], origin=30)
    assert added == [0, 0, 1, 3]
    assert session.state == [[30, 31], [32, 33], [34, 35], [36, 37]]
    assert session.last_question_time == 40


def test_non_monotonic_questions_fail(scheduler):
    with pytest.raises(ValueError, match="backwards"):
        scheduler(live.InteractiveConfig(), [10, 9])


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_fixed_gap(value):
    with pytest.raises(ValueError):
        live.InteractiveConfig(rollout_step_seconds=value)


def launcher(benchmark="ovo", *extra):
    return parse_args(["--benchmark", benchmark, "--checkpoint", "checkpoint", "--base-model", "base",
                       "--data-root", "data", "--video-root", "videos", "--output-dir", "out",
                       "--preset", "last2s", *extra])


@pytest.mark.parametrize("benchmark", ["ovo", "streaming"])
@pytest.mark.parametrize("mode", ["offline", "interactive", "fixed-gap"])
@pytest.mark.parametrize("preset,tail", [("last2s", 2), ("last8s", 8)])
def test_launcher_to_worker(benchmark, mode, monkeypatch, preset, tail):
    from malow.video import ovo, streaming
    args = launcher(benchmark, "--eval-mode", mode, "--preset", preset)
    command = build_command(args)
    monkeypatch.setattr("sys.argv", ["worker", *command[3:]])
    worker = (ovo if benchmark == "ovo" else streaming).parse_args()
    assert worker.eval_mode == mode
    assert worker.encoder_video_max_pixels == (0 if mode == "offline" else 262144)
    assert worker.encoder_video_longest_edge == (0 if mode == "offline" else 10**9)
    assert worker.base_video_max_pixels == 0
    assert worker.base_tail_seconds == tail
    assert worker.base_input_frame_number == tail
    assert worker.rollout_step_seconds == (2 if mode == "fixed-gap" else None)
    if mode == "offline":
        assert "--eval-mode" not in command
        assert "--encoder-video-max-pixels" not in command


@pytest.mark.parametrize("extra", [
    ["--rollout-step-seconds", "2"],
    ["--eval-mode", "fixed-gap", "--rollout-step-seconds", "nan"],
    ["--interactive-max-segment-frames", "1"],
    ["--interactive-verify-sessions", "-1"],
    ["--encoder-video-longest-edge", "-1"],
])
def test_invalid_cli(extra):
    with pytest.raises(SystemExit):
        launcher("ovo", *extra)


def test_checkpoint_segment_cap_and_explicit_precedence():
    model = SimpleNamespace(frame_per_lora=2, micro_c2l_forward_frame=65)
    args = Namespace(eval_mode="interactive", base_with_last_2seconds=True, base_tail_seconds=4)
    assert live.interactive_config_from_args(model, args).max_segment_blocks == 32
    args.interactive_max_segment_frames = 8
    assert live.interactive_config_from_args(model, args).max_segment_blocks == 4
    assert live.interactive_config_from_args(model, args).tail_seconds == 4
    model.micro_c2l_forward_frame = None
    args.interactive_max_segment_frames = None
    assert live.interactive_config_from_args(model, args).max_segment_blocks == 383


@pytest.mark.parametrize("change", ["model_id", "eval_mode", "base_tail_seconds", "encoder_video_max_pixels", "rollout_step_seconds"])
def test_resume_rejects_incompatible_settings(tmp_path, change):
    args = Namespace(output_dir=str(tmp_path), c2l_checkpoint="checkpoint", model_id="base", eval_mode="fixed-gap",
                     rollout_step_seconds=2, base_tail_seconds=2, encoder_video_max_pixels=262144)
    guard_interactive_run(args, 2)
    guard_interactive_run(args, 2)
    with pytest.raises(ValueError, match="world_size"):
        guard_interactive_run(args, 1)
    setattr(args, change, "changed")
    with pytest.raises(ValueError, match=change):
        guard_interactive_run(args, 2)


def test_resume_refuses_unidentified_outputs(tmp_path):
    (tmp_path / "rank_00000.jsonl").write_text("{}\n")
    args = Namespace(output_dir=str(tmp_path), c2l_checkpoint="checkpoint", eval_mode="interactive")
    with pytest.raises(ValueError, match="fresh"):
        guard_interactive_run(args, 1)


def test_resolution_policy_and_log(tmp_path):
    class Processor:
        video_processor = SimpleNamespace(size={"shortest_edge": 131072, "longest_edge": 25165824},
                                          patch_size=16, merge_size=2, temporal_patch_size=2)
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"video_grid_thw": [[1, 24, 42]], "video_metadata": "remove me"}
    processor = Processor()
    args = Namespace(encoder_video_max_pixels=262144, encoder_video_longest_edge=10**9,
                     base_video_max_pixels=0, video_min_pixels=1024, video_fps=1, eval_mode="interactive",
                     _resolution_log=str(tmp_path / "resolution.jsonl"))
    frames = [Image.new("RGB", (1280, 720))] * 2
    inputs = sampling.processor_predecoded_video_inputs(processor, "", [frames], args)
    assert "video_metadata" not in inputs
    assert processor.kwargs["videos"][0][0].size == (682, 384)
    assert processor.kwargs["size"]["longest_edge"] == 10**9
    assert processor.kwargs["do_sample_frames"] is False
    sampling.processor_predecoded_video_inputs(processor, "", [frames], args, branch="base")
    assert processor.kwargs["videos"][0] is frames
    assert "size" not in processor.kwargs
    records = [json.loads(line) for line in (tmp_path / "resolution.jsonl").read_text().splitlines()]
    assert records[0]["processed_dimensions_hw"] == [[384, 672]]
    assert records[0]["source_dimensions_hw"] == [[720, 1280]]
    assert records[1]["branch"] == "base"


def test_tail_only_loader_matches_existing_sampling(monkeypatch):
    monkeypatch.setattr(sampling, "_resolve_clip_bounds", lambda *a: (0, 10))
    calls = []
    def decode(path, count, clip_start_seconds, clip_end_seconds):
        calls.append((count, clip_start_seconds, clip_end_seconds))
        return list(range(count)), list(range(count)), clip_end_seconds
    monkeypatch.setattr(sampling, "load_video_clip_frames_with_timestamps", decode)
    frames, times = sampling.load_c2l_base_tail_frames("clip", 4, tail_seconds=4)
    assert calls == [(4, 6, 10)]
    assert len(frames) == len(times) == 4


def test_session_release_does_not_retain_model_state():
    session = live.new_session(Recorder(), live.InteractiveConfig())
    live.finish_session(session)
    assert session.state is None


def test_verification_failure_cannot_be_waived(monkeypatch):
    monkeypatch.setenv("C2L_INTERACTIVE_ALLOW_VERIFY_FAILURE", "1")
    with pytest.raises(RuntimeError):
        live.enforce_verification({"failures": 1}, "mismatch")
    assert "INCONCLUSIVE" in live.verification_verdict({"checks": 1, "failures": 0, "max_rel_diff": 0})


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280), (1024, 1024)])
def test_real_qwen_resize_is_segment_invariant_for_encfull(width, height):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    frame = sampling.resize_c2l_video_frames([[Image.new("RGB", (width, height))]], 262144, 1024)[0][0]
    dimensions = [smart_resize(t, frame.height, frame.width, factor=32, min_pixels=131072, max_pixels=10**9)
                  for t in [2, 8, 128, 766]]
    assert len(set(dimensions)) == 1
    if width == 1280:
        assert dimensions[0] == (384, 672)
        assert smart_resize(2, height, width, min_pixels=131072, max_pixels=25165824) == (704, 1280)
        assert smart_resize(766, height, width, min_pixels=131072, max_pixels=25165824) == (128, 224)


def test_real_video_processor_grid_matches_diagnostics(tmp_path):
    import torch
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    processor = Qwen3VLVideoProcessor(size={"shortest_edge": 131072, "longest_edge": 10**9})
    frames = sampling.resize_c2l_video_frames([[Image.new("RGB", (1280, 720))] * 2], 262144, 1024)[0]
    inputs = processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
    assert inputs["video_grid_thw"].tolist() == [[1, 24, 42]]
    assert torch.isfinite(inputs["pixel_values_videos"]).all()


def test_fixed_grid_decord_reader_reused_and_sampled_by_absolute_time(monkeypatch):
    import numpy as np
    import sys
    opened, requested = [], []
    class Reader:
        def __init__(self, path, **kwargs):
            opened.append(path)
        def __len__(self):
            return 100
        def get_avg_fps(self):
            return 2.0
        def get_batch(self, indices):
            requested.append(indices)
            return SimpleNamespace(asnumpy=lambda: np.stack([np.full((32, 32, 3), i, dtype=np.uint8) for i in indices]))
    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=Reader, cpu=lambda n: n))
    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "decord")
    sampling._interactive_decord_reader.cache_clear()
    try:
        a = sampling.load_video_frames_at_times("clip", [10, 11])
        b = sampling.load_video_frames_at_times("clip", [12, 13])
        assert opened == ["clip"]
        assert requested == [[20, 22], [24, 26]]
        assert [image.getpixel((0, 0))[0] for image in a + b] == [20, 22, 24, 26]
    finally:
        sampling._interactive_decord_reader.cache_clear()


def test_pyav_decoder_seeks_new_segment_without_redecoding_prefix(monkeypatch):
    import sys
    from fractions import Fraction
    calls = []
    stream = SimpleNamespace(type="video", average_rate=2, time_base=Fraction(1, 10), start_time=0)
    frames = [SimpleNamespace(pts=i * 5, to_image=lambda i=i: i) for i in range(18, 26)]
    container = SimpleNamespace(streams=[stream], seek=lambda pts, **kw: calls.append(pts),
                                decode=lambda s: iter(frames), close=lambda: calls.append("closed"))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *a: container))
    assert sampling._decode_frames_at_indices_pyav("clip", [20, 22, 24]) == [20, 22, 24]
    assert calls == [100, "closed"]


def test_verification_uses_identical_grid_and_detects_dropped_state(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(live, "build_segment_encoder_inputs", lambda processor, text, path, times, *a, **kw: times)
    class SumModel:
        def generate_weights(self, times):
            t = torch.tensor([sum(times)], dtype=torch.float32)
            return {"layer": {"A": t, "B": t * 2}}, None
        def interactive_generate_weights(self, state):
            return self.generate_weights(state)
    session = live.InteractiveSession(state=[0, 1, 2, 3], config=live.InteractiveConfig(),
                                      consumed_blocks=2, segments=2, last_segment_start=1)
    args = Namespace(_verification_log=str(tmp_path / "checks.jsonl"))
    result = live.verify_session_against_single_pass(SumModel(), None, session, "", "clip", args, None)
    assert result["ok"] and result["max_rel_diff"] == 0
    assert result["dropped_state_rel_diff"] > 0
    assert "OK:" in live.verification_verdict(live.summarize_verifications([session]))
    session.state = [2, 3]
    with pytest.raises(RuntimeError, match="verification failed"):
        live.verify_session_against_single_pass(SumModel(), None, session, "", "clip", args, None)
    assert len((tmp_path / "checks.jsonl").read_text().splitlines()) == 2
    session.consumed_blocks = 129
    assert live.verify_session_against_single_pass(SumModel(), None, session, "", "clip", args, None) is None
    assert session.skipped_verifications == 1
