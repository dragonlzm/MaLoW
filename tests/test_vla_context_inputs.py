"""Inference regression tests derived from Hippocampus master e998ec8."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


from malow.vla.c2l.blocks import FRAMES_PER_BLOCK  # noqa: E402
from malow.vla.c2l.encoder_inputs import build_context_encoder_inputs  # noqa: E402


ACTION_DIM = 7


class _Tokenizer:
    def __call__(self, labels, add_special_tokens=False):
        # One id per character keeps the label recoverable from the tensor.
        return {"input_ids": [[ord(c) for c in label] for label in labels]}

    def pad(self, features, padding=True, return_tensors="pt"):
        width = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros(len(features), width, dtype=torch.long)
        mask = torch.zeros(len(features), width, dtype=torch.long)
        for i, feature in enumerate(features):
            length = len(feature["input_ids"])
            ids[i, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            mask[i, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.video_processor = type("_VP", (), {"temporal_patch_size": FRAMES_PER_BLOCK})()
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ["<video>"]

    def __call__(self, text=None, videos=None, **kwargs):
        self.last_kwargs = kwargs
        grid = torch.tensor(
            [[len(video) // FRAMES_PER_BLOCK, 2, 2] for video in videos], dtype=torch.long
        )
        return {"video_grid_thw": grid, "pixel_values_videos": torch.zeros(1), **kwargs}


def _frames(count):
    from PIL import Image

    return [Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(count)]


def _actions(num_blocks, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(num_blocks, FRAMES_PER_BLOCK, ACTION_DIM)).astype(np.float32)
    is_empty = np.zeros((num_blocks, FRAMES_PER_BLOCK), dtype=bool)
    is_empty[0, 0] = True
    return values, is_empty


# --- the per-sample clock -----------------------------------------------------------

def test_each_sample_gets_its_own_frame_rate():
    """Bridge at 5 Hz and RT-1 at 3 Hz in one batch, which is the normal case."""
    processor = _Processor()
    inputs = build_context_encoder_inputs(
        processor,
        [_frames(4), _frames(4)],
        step_seconds=[0.2, 1.0 / 3.0],
        block_labels=[["0s-0.4s", "0.4s-0.8s"], ["0s-0.666667s", "0.666667s-1.33333s"]],
    )
    metadata = processor.last_kwargs["video_metadata"]
    assert [round(entry["fps"], 6) for entry in metadata] == [5.0, 3.0]
    assert inputs["timestamp_block_owners"].tolist() == [0, 0, 1, 1]


def test_a_scalar_clock_is_broadcast():
    processor = _Processor()
    build_context_encoder_inputs(processor, [_frames(2), _frames(2)], step_seconds=0.2)
    assert [entry["fps"] for entry in processor.last_kwargs["video_metadata"]] == [5.0, 5.0]


def test_the_processor_is_not_asked_to_resample():
    """`fps=` is a scalar the frame sampler would use; a two-clock batch has no scalar.

    `do_sample_frames=False` makes it dead anyway, so it is not passed and the real rate
    travels per sample in `video_metadata`.
    """
    processor = _Processor()
    build_context_encoder_inputs(processor, [_frames(2)], step_seconds=0.2)
    assert processor.last_kwargs["do_sample_frames"] is False
    assert "fps" not in processor.last_kwargs


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_nonpositive_clock_is_rejected(bad):
    with pytest.raises(ValueError, match="step_seconds"):
        build_context_encoder_inputs(_Processor(), [_frames(2)], step_seconds=bad)


def test_a_clock_per_sample_is_required():
    with pytest.raises(ValueError, match="step_seconds"):
        build_context_encoder_inputs(
            _Processor(), [_frames(2), _frames(2)], step_seconds=[0.2]
        )


# --- the action stream --------------------------------------------------------------

def test_actions_are_flattened_with_owners_like_the_timestamps():
    processor = _Processor()
    first, first_mask = _actions(2, seed=1)
    second, second_mask = _actions(1, seed=2)

    inputs = build_context_encoder_inputs(
        processor,
        [_frames(4), _frames(2)],
        step_seconds=0.2,
        block_labels=[["0s-0.4s", "0.4s-0.8s"], ["0s-0.4s"]],
        block_actions=[first, second],
        block_action_is_empty=[first_mask, second_mask],
    )

    assert inputs["action_history_values"].shape == (3, FRAMES_PER_BLOCK, ACTION_DIM)
    assert inputs["action_history_block_owners"].tolist() == [0, 0, 1]
    # Same owner vector as the timestamps, which is what the hypernet cross-checks.
    assert (
        inputs["action_history_block_owners"].tolist()
        == inputs["timestamp_block_owners"].tolist()
    )
    torch.testing.assert_close(
        inputs["action_history_values"][:2], torch.from_numpy(first)
    )
    torch.testing.assert_close(
        inputs["action_history_values"][2:], torch.from_numpy(second)
    )
    assert inputs["action_history_is_empty"].dtype == torch.bool


def test_no_action_keys_appear_when_the_stream_is_off():
    """The video-C2L contract: absent keys, not zero-filled ones."""
    inputs = build_context_encoder_inputs(
        _Processor(), [_frames(2)], step_seconds=0.2, block_labels=[["0s-0.4s"]]
    )
    assert not [key for key in inputs if key.startswith("action_history_")]


def test_text_prompts_are_tokenized_once_per_context_sample():
    processor = _Processor()
    inputs = build_context_encoder_inputs(
        processor,
        [_frames(4), _frames(2)],
        step_seconds=0.2,
        text_prompts=["pick up the carrot", "close the drawer"],
    )

    assert inputs["text_prompt_input_ids"].shape[0] == 2
    assert inputs["text_prompt_attention_mask"].sum(dim=1).tolist() == [18, 16]
    assert not [key for key in inputs if key.startswith("text_prompt_") and key.endswith("owners")]


def test_text_prompt_count_must_match_the_context_batch():
    with pytest.raises(ValueError, match="batch size"):
        build_context_encoder_inputs(
            _Processor(),
            [_frames(2), _frames(2)],
            step_seconds=0.2,
            text_prompts=["only one prompt"],
        )


def test_action_block_count_must_match_the_video():
    values, is_empty = _actions(1)
    with pytest.raises(ValueError, match="block counts"):
        build_context_encoder_inputs(
            _Processor(),
            [_frames(4)],
            step_seconds=0.2,
            block_actions=[values],
            block_action_is_empty=[is_empty],
        )


def test_values_and_mask_must_arrive_together():
    values, _ = _actions(1)
    with pytest.raises(ValueError, match="block_action_is_empty"):
        build_context_encoder_inputs(
            _Processor(), [_frames(2)], step_seconds=0.2, block_actions=[values]
        )
