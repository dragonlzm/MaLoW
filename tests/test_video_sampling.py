# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from malow.video import sampling
from malow.video.sampling import add_c2l_timestamp_inputs


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, labels, add_special_tokens=False):
        self.calls.append((labels, add_special_tokens))
        return {
            "input_ids": [[index + 10] for index, _label in enumerate(labels)],
            "attention_mask": [[1] for _label in labels],
        }

    def pad(self, features, padding=True, return_tensors="pt"):
        assert padding is True
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
            "attention_mask": torch.tensor([feature["attention_mask"] for feature in features]),
        }


class Transformers56LikeProcessor:
    """Minimal processor double that exposes metadata-dropping fallbacks."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        metadata = kwargs.get("video_metadata")
        if metadata is None:
            return {"video_grid_thw": torch.tensor([[4, 1, 1]], dtype=torch.long)}

        if "total_frames" in metadata[0]:
            raise TypeError("VideoMetadata.__init__() got an unexpected keyword argument 'total_frames'")
        if "min_pixels" in kwargs or "max_pixels" in kwargs:
            raise TypeError("VideosKwargs does not accept generic pixel kwargs")

        temporal_blocks = int(metadata[0]["total_num_frames"]) // 2
        return {
            "video_grid_thw": torch.tensor([[temporal_blocks, 1, 1]], dtype=torch.long),
            "video_metadata": metadata,
            "second_per_grid_ts": [2.0 / float(metadata[0]["fps"])],
        }


def test_timestamp_inputs_keep_sparse_predecoded_frame_times():
    tokenizer = FakeTokenizer()
    encoder_inputs = {"video_grid_thw": torch.ones((1, 3), dtype=torch.long)}

    add_c2l_timestamp_inputs(
        encoder_inputs,
        tokenizer,
        frame_per_lora=2,
        frame_timestamps=[[0.0, 2.5, 5.0, 7.5]],
        terminal_times=[10.0],
    )

    assert tokenizer.calls == [(["0s-5s", "5s-10s"], False)]
    assert encoder_inputs["timestamp_input_ids"].tolist() == [[10], [11]]
    assert encoder_inputs["timestamp_attention_mask"].tolist() == [[1], [1]]
    assert encoder_inputs["timestamp_block_owners"].tolist() == [0, 0]


def test_timestamp_inputs_use_processor_video_metadata_when_frames_are_not_predecoded():
    tokenizer = FakeTokenizer()
    encoder_inputs = {
        "video_grid_thw": torch.ones((1, 3), dtype=torch.long),
        "video_metadata": [{"frames_indices": [6, 8, 10], "fps": 2.0}],
    }

    add_c2l_timestamp_inputs(encoder_inputs, tokenizer, frame_per_lora=3)

    assert tokenizer.calls == [(["3s-6s"], False)]
    assert encoder_inputs["timestamp_block_owners"].tolist() == [0]
    assert "video_metadata" not in encoder_inputs


def test_timestamp_split_loader_keeps_source_times_after_uniform_sampling(monkeypatch):
    monkeypatch.setattr(
        sampling,
        "load_video_clip_frames_with_timestamps",
        lambda *_args, **_kwargs: (["f0", "f1", "f2", "f3"], [0.0, 2.5, 5.0, 7.5], 10.0),
    )

    encoder_frames, base_frames, encoder_times, base_times, terminal_time = (
        sampling.load_split_c2l_video_frames_with_timestamps("clip.mp4", 4, 0, 0.0, 10.0)
    )

    assert encoder_frames == ["f0", "f1", "f2", "f3"]
    assert base_frames == []
    assert encoder_times == [0.0, 2.5, 5.0, 7.5]
    assert base_times == []
    assert terminal_time == 10.0


@pytest.mark.parametrize(("frame_count", "temporal_blocks"), [(766, 383), (2046, 1023)])
def test_predecoded_processor_metadata_fps_changes_temporal_positions_without_changing_frames(
    frame_count, temporal_blocks
):
    default_processor = Transformers56LikeProcessor()
    one_fps_processor = Transformers56LikeProcessor()
    frames = [object() for _index in range(frame_count)]
    videos = [frames]
    args = SimpleNamespace(video_fps=2.0, video_min_pixels=1024, video_max_pixels=262144)

    default_inputs = sampling.processor_predecoded_video_inputs(default_processor, ["<video>"], videos, args)
    one_fps_inputs = sampling.processor_predecoded_video_inputs(
        one_fps_processor,
        ["<video>"],
        videos,
        args,
        metadata_fps=1.0,
    )

    assert default_inputs["video_grid_thw"].tolist() == [[temporal_blocks, 1, 1]]
    assert one_fps_inputs["video_grid_thw"].tolist() == [[temporal_blocks, 1, 1]]
    assert default_inputs["second_per_grid_ts"] == [1.0]
    assert one_fps_inputs["second_per_grid_ts"] == [2.0]
    assert "video_metadata" not in default_inputs
    assert "video_metadata" not in one_fps_inputs
    assert len(default_processor.calls) == 1
    assert len(one_fps_processor.calls) == 1
    default_call = default_processor.calls[0]
    one_fps_call = one_fps_processor.calls[0]
    assert default_call["videos"] is videos
    assert one_fps_call["videos"] is videos
    assert default_call["videos"][0] is frames
    assert one_fps_call["videos"][0] is frames
    assert default_call["do_sample_frames"] is False
    assert one_fps_call["do_sample_frames"] is False
    assert default_call["return_metadata"] is True
    assert one_fps_call["return_metadata"] is True
    assert default_call["fps"] == 2.0
    assert one_fps_call["fps"] == 1.0
    assert default_call["video_metadata"] == [
        {
            "fps": 2.0,
            "duration": frame_count / 2.0,
            "total_num_frames": frame_count,
            "frames_indices": list(range(frame_count)),
        }
    ]
    assert one_fps_call["video_metadata"] == [
        {
            "fps": 1.0,
            "duration": float(frame_count),
            "total_num_frames": frame_count,
            "frames_indices": list(range(frame_count)),
        }
    ]
    assert "min_pixels" not in default_call
    assert "max_pixels" not in default_call
    assert "min_pixels" not in one_fps_call
    assert "max_pixels" not in one_fps_call






























































def test_single_frame_timestamps_include_existing_odd_frame_padding():
    tokenizer = FakeTokenizer()
    inputs = {"video_grid_thw": torch.tensor([[2, 2, 2]])}
    add_c2l_timestamp_inputs(inputs, tokenizer, 1, [[0.0, 1.0, 2.0]], [3.0])
    assert inputs["timestamp_block_owners"].tolist() == [0, 0, 0, 0]
    assert tokenizer.calls == [(["0s-1s", "1s-2s", "2s-2s", "2s-3s"], False)]















def test_evaluator_tail_cli_default_values_and_validation(tail_evaluator, monkeypatch):
    import sys

    required = ["prog", "--model-id", "qwen35", "--c2l-checkpoint", "checkpoint", "--output-dir", "/tmp/output"]
    monkeypatch.setattr(sys, "argv", required)
    assert tail_evaluator.parse_args().base_tail_seconds == 2.0
    for duration in ("1", "4", "8", "16"):
        monkeypatch.setattr(sys, "argv", [*required, "--base-tail-seconds", duration])
        assert tail_evaluator.parse_args().base_tail_seconds == float(duration)
    for invalid in ("0", "-1", "nan", "inf"):
        monkeypatch.setattr(sys, "argv", [*required, "--base-tail-seconds", invalid])
        with pytest.raises(SystemExit) as error:
            tail_evaluator.parse_args()
        assert error.value.code == 2


@pytest.mark.parametrize("duration,base_frames", [(1.0, 1), (4.0, 4), (8.0, 8), (16.0, 2)])
@pytest.mark.parametrize("timestamps", [False, True])
def test_evaluator_passes_tail_to_sampler(tail_evaluator, monkeypatch, duration, base_frames, timestamps):
    runner = tail_evaluator
    args = SimpleNamespace(base_input_video=True, base_with_last_2seconds=True,
                           base_tail_seconds=duration, base_input_frame_number=base_frames,
                           max_frames=766, num_frames=766)
    model = SimpleNamespace(interleave_timestamp=timestamps, base_last_2seconds_max_frames=256)
    monkeypatch.setattr(runner, "apply_chat_template", lambda *_: ["video"])
    builder = "build_c2l_encoder_messages" if runner.__name__.endswith("streaming") else "build_video_only_messages"
    monkeypatch.setattr(runner, builder, lambda *_: [])
    if hasattr(runner, "model_input_device"):
        monkeypatch.setattr(runner, "model_input_device", lambda *_: "cpu")

    class Sampled(Exception):
        pass

    def load(path, cap, frames, **kwargs):
        assert cap == 766
        assert frames == base_frames
        assert kwargs["tail_seconds"] == duration
        raise Sampled

    loader = "load_base_last_2seconds_c2l_video_frames" + ("_with_timestamps" if timestamps else "")
    monkeypatch.setattr(runner, loader, load)
    with pytest.raises(Sampled):
        if runner.__name__ == "run_videomme_qwen35_c2l":
            runner.prepare_c2l_inputs(model, None, ["question"], ["clip.mp4"], args, "cpu")
        elif runner.__name__.endswith("ovo"):
            runner.prepare_c2l_inputs(model, None, [SimpleNamespace(video_path="clip.mp4")], args)
        else:
            runner.prepare_c2l_inputs(model, None, [("question", "clip.mp4", 5, 10)], args)

@pytest.fixture(params=["ovo", "streaming"])
def tail_evaluator(request):
    import importlib
    return importlib.import_module("malow.video." + request.param)
