"""Inference compatibility implementation; see SOURCE_NOTICE.md."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .blocks import FRAMES_PER_BLOCK


# The context encoder is `vision_embedder` in the v1.4.x C2L recipe: it reads
# `pixel_values_videos` / `video_grid_thw` and ignores the text entirely. The prompt is
# still built so the inputs stay valid for the `per_layer_activations` encoder, which
# does run the language model over them.
CONTEXT_ENCODER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": [{"type": "video", "video": None}]}
]


def build_context_encoder_text(processor: Any) -> str:
    """Video-only encoder prompt, matching C2L's default `c2l_encoder_context=image_only`."""
    text = processor.apply_chat_template(
        [CONTEXT_ENCODER_MESSAGES], tokenize=False, add_generation_prompt=True
    )
    return text[0] if isinstance(text, (list, tuple)) else text


def _video_metadata(videos: list[list[Any]], step_seconds: list[float]) -> list[dict[str, Any]]:
    """Per-video sampling metadata, each on that sample's own clock.

    A `bridge_rt_1` batch mixes 5 Hz and 3 Hz episodes, so this is a list of rates
    rather than one -- which is also why the processor's scalar ``fps=`` kwarg is not
    passed at all (see :func:`build_context_encoder_inputs`).
    """
    metadata = []
    for video, seconds in zip(videos, step_seconds, strict=True):
        fps = 1.0 / float(seconds)
        metadata.append(
            {
                "fps": fps,
                "duration": len(video) / max(fps, 1e-6),
                "total_num_frames": len(video),
                "frames_indices": list(range(len(video))),
            }
        )
    return metadata


def _annotate_video_temporal_patch_size(processor: Any, encoder_inputs: Any) -> None:
    """Tell the hypernet how many frames the vision tower folded into one temporal unit.

    Without this, ``_build_video_block_features`` falls back to the model config's value,
    which is right today but is exactly the number the two-frame block policy depends on.
    """
    video_grid_thw = encoder_inputs.get("video_grid_thw")
    if video_grid_thw is None or int(video_grid_thw.numel()) == 0:
        return

    video_processor = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 0) or 0)
    if temporal_patch_size < 1:
        return
    if temporal_patch_size != FRAMES_PER_BLOCK:
        raise ValueError(
            "C2L-VLA pairs frames two at a time because the Qwen vision tower folds two "
            f"frames into one temporal unit, but this processor reports "
            f"temporal_patch_size={temporal_patch_size}. Update blocks.FRAMES_PER_BLOCK "
            "and the block policy together, or the block stream will not line up."
        )

    encoder_inputs["video_temporal_patch_size"] = video_grid_thw.new_full(
        (video_grid_thw.shape[0],), temporal_patch_size
    )


def _add_timestamp_inputs(
    tokenizer: Any,
    encoder_inputs: Any,
    block_labels: list[list[str]],
    device: Any = None,
) -> None:
    """Attach the frozen timestamp text the interleaved-timestamp checkpoints expect.

    One label per temporal block, flattened across the batch, with ``block_owners``
    recording which sample each block came from -- the same layout
    ``_interleave_block_timestamps`` validates against the visual block owners.
    """
    flat_labels: list[str] = []
    owners: list[int] = []
    for sample_idx, labels in enumerate(block_labels):
        flat_labels.extend(labels)
        owners.extend([sample_idx] * len(labels))

    if not flat_labels:
        raise ValueError("C2L-VLA timestamp interleaving requires at least one temporal block.")

    tokenized = tokenizer(flat_labels, add_special_tokens=False)
    features = [
        {"input_ids": input_ids, "attention_mask": attention_mask}
        for input_ids, attention_mask in zip(
            tokenized["input_ids"],
            tokenized.get(
                "attention_mask", [[1] * len(input_ids) for input_ids in tokenized["input_ids"]]
            ),
            strict=True,
        )
    ]
    padded = tokenizer.pad(features, padding=True, return_tensors="pt")
    owners_tensor = torch.tensor(owners, dtype=torch.long)
    if device is not None:
        encoder_inputs["timestamp_input_ids"] = padded["input_ids"].to(device)
        encoder_inputs["timestamp_attention_mask"] = padded["attention_mask"].to(device)
        encoder_inputs["timestamp_block_owners"] = owners_tensor.to(device)
    else:
        encoder_inputs["timestamp_input_ids"] = padded["input_ids"]
        encoder_inputs["timestamp_attention_mask"] = padded["attention_mask"]
        encoder_inputs["timestamp_block_owners"] = owners_tensor


def _add_text_prompt_inputs(
    tokenizer: Any,
    encoder_inputs: Any,
    text_prompts: list[str],
    device: Any = None,
) -> None:
    """Attach one tokenized VLA instruction prompt per context sample."""
    if not text_prompts or any(not str(prompt).strip() for prompt in text_prompts):
        raise ValueError("C2L-VLA text-prompt interleaving requires one non-empty prompt per sample.")

    tokenized = tokenizer(text_prompts, add_special_tokens=False)
    features = [
        {"input_ids": input_ids, "attention_mask": attention_mask}
        for input_ids, attention_mask in zip(
            tokenized["input_ids"],
            tokenized.get(
                "attention_mask", [[1] * len(input_ids) for input_ids in tokenized["input_ids"]]
            ),
            strict=True,
        )
    ]
    padded = tokenizer.pad(features, padding=True, return_tensors="pt")
    if device is not None:
        encoder_inputs["text_prompt_input_ids"] = padded["input_ids"].to(device)
        encoder_inputs["text_prompt_attention_mask"] = padded["attention_mask"].to(device)
    else:
        encoder_inputs["text_prompt_input_ids"] = padded["input_ids"]
        encoder_inputs["text_prompt_attention_mask"] = padded["attention_mask"]


def _add_action_history_inputs(
    encoder_inputs: Any,
    block_actions: list[np.ndarray],
    block_action_is_empty: list[np.ndarray],
    device: Any = None,
) -> None:
    """Attach the previous-step actions, one per context frame.

    Layout mirrors the timestamp stream: ``[total_blocks, FRAMES_PER_BLOCK, action_dim]``
    flattened across the batch with an owner vector, so
    ``_interleave_block_action_history`` can validate it against the visual block owners
    the same way. ``is_empty`` marks the pre-episode slot ``a_{-1}``, whose value is
    meaningless -- the hypernet substitutes a learned embedding there. Zeros would not
    do: 0 is a perfectly ordinary normalized action, and the gripper dimension is
    normalized to a literal ``{0, 1}``.
    """
    flat_values: list[np.ndarray] = []
    flat_is_empty: list[np.ndarray] = []
    owners: list[int] = []
    for sample_idx, (values, is_empty) in enumerate(
        zip(block_actions, block_action_is_empty, strict=True)
    ):
        values = np.asarray(values, dtype=np.float32)
        is_empty = np.asarray(is_empty, dtype=bool)
        if values.ndim != 3 or values.shape[1] != FRAMES_PER_BLOCK:
            raise ValueError(
                "C2L-VLA action history must be shaped [blocks, "
                f"{FRAMES_PER_BLOCK}, action_dim] per sample; sample {sample_idx} is "
                f"{values.shape}."
            )
        if is_empty.shape != values.shape[:2]:
            raise ValueError(
                "C2L-VLA action history mask must be shaped [blocks, "
                f"{FRAMES_PER_BLOCK}]; sample {sample_idx} is {is_empty.shape} against "
                f"values {values.shape}."
            )
        flat_values.append(values)
        flat_is_empty.append(is_empty)
        owners.extend([sample_idx] * values.shape[0])

    if not owners:
        raise ValueError("C2L-VLA action interleaving requires at least one temporal block.")

    values_tensor = torch.from_numpy(np.concatenate(flat_values, axis=0))
    is_empty_tensor = torch.from_numpy(np.concatenate(flat_is_empty, axis=0))
    owners_tensor = torch.tensor(owners, dtype=torch.long)
    if device is not None:
        values_tensor = values_tensor.to(device)
        is_empty_tensor = is_empty_tensor.to(device)
        owners_tensor = owners_tensor.to(device)
    encoder_inputs["action_history_values"] = values_tensor
    encoder_inputs["action_history_is_empty"] = is_empty_tensor
    encoder_inputs["action_history_block_owners"] = owners_tensor


def build_context_encoder_inputs(
    processor: Any,
    videos: list[list[Any]],
    step_seconds: list[float] | float,
    block_labels: list[list[str]] | None = None,
    block_actions: list[np.ndarray] | None = None,
    block_action_is_empty: list[np.ndarray] | None = None,
    text_prompts: list[str] | None = None,
    encoder_text: str | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Processor inputs for one batch of context streams.

    Args:
        videos: one already-decoded frame list per sample. Its length must be a multiple
            of :data:`blocks.FRAMES_PER_BLOCK`; :func:`blocks.context_frame_steps`
            guarantees that by repeating the last frame.
        step_seconds: the control period of each sample's episode, in seconds -- 0.2 for
            bridge and SimplerEnv's WidowX setup, 1/3 for RT-1. A scalar is broadcast to
            every sample. Required rather than defaulted: a `bridge_rt_1` batch carries
            both rates, so there is no safe single value.
        block_labels: one timestamp label per temporal block per sample, from
            :mod:`.blocks`. Pass ``None`` for a checkpoint trained without
            ``interleave_timestamp``.
        block_actions / block_action_is_empty: per sample, the previous-step action for
            every context frame, shaped ``[blocks, FRAMES_PER_BLOCK, action_dim]`` and
            its "this slot is the pre-episode empty action" mask. Pass ``None`` for a
            checkpoint trained without ``interleave_action_history``.
        text_prompts: one fully formatted VLA instruction prompt per sample. Pass
            ``None`` for a checkpoint trained without ``interleave_text_prompt``.
        encoder_text: cached output of :func:`build_context_encoder_text`.
    """
    if not videos:
        raise ValueError("C2L-VLA context encoding requires at least one sample.")
    for sample_idx, video in enumerate(videos):
        if not video:
            raise ValueError(
                f"C2L-VLA sample {sample_idx} has an empty context. Samples at env step 0 have "
                "no history and must be filtered out before this point."
            )
        if len(video) % FRAMES_PER_BLOCK:
            raise ValueError(
                f"C2L-VLA sample {sample_idx} has {len(video)} context frames, which is not a "
                f"multiple of {FRAMES_PER_BLOCK}; blocks.context_frame_steps pads for this."
            )

    if isinstance(step_seconds, (int, float)):
        step_seconds = [float(step_seconds)] * len(videos)
    else:
        step_seconds = [float(seconds) for seconds in step_seconds]
    if len(step_seconds) != len(videos):
        raise ValueError(
            f"C2L-VLA context encoding got {len(step_seconds)} step_seconds for {len(videos)} samples."
        )
    if any(seconds <= 0.0 for seconds in step_seconds):
        raise ValueError(f"C2L-VLA step_seconds must all be positive, got {step_seconds}.")

    if encoder_text is None:
        encoder_text = build_context_encoder_text(processor)

    # No `fps=` kwarg: it is only read by the processor's frame sampler, which
    # `do_sample_frames=False` disables, and it is a scalar -- it could not describe a
    # batch that mixes 5 Hz bridge and 3 Hz RT-1 episodes anyway. The per-sample rate
    # rides in `video_metadata`, which is where the processor actually reads it.
    inputs = processor(
        text=[encoder_text] * len(videos),
        images=None,
        videos=videos,
        padding=True,
        return_tensors="pt",
        video_metadata=_video_metadata(videos, step_seconds),
        return_metadata=True,
        do_sample_frames=False,
    )
    # Processor-side sampling metadata, not a model input: Transformers rejects it as an
    # unused kwarg downstream.
    inputs.pop("video_metadata", None)

    _annotate_video_temporal_patch_size(processor, inputs)

    expected_blocks = [len(video) // FRAMES_PER_BLOCK for video in videos]
    video_grid_thw = inputs.get("video_grid_thw")
    if video_grid_thw is None or int(video_grid_thw.shape[0]) != len(videos):
        raise ValueError(
            "C2L-VLA context encoding expects exactly one video row per sample; got "
            f"{None if video_grid_thw is None else int(video_grid_thw.shape[0])} rows "
            f"for {len(videos)} samples."
        )
    actual_blocks = [int(row[0].item()) for row in video_grid_thw]
    if actual_blocks != expected_blocks:
        raise ValueError(
            "C2L-VLA temporal unit count disagrees with the block policy: "
            f"processor produced {actual_blocks}, block policy expects {expected_blocks}. "
            "The processor most likely resampled the frames."
        )

    if block_labels is not None:
        if [len(labels) for labels in block_labels] != expected_blocks:
            raise ValueError(
                "C2L-VLA timestamp labels do not match the block counts: "
                f"labels={[len(labels) for labels in block_labels]}, blocks={expected_blocks}."
            )
        _add_timestamp_inputs(processor.tokenizer, inputs, block_labels, device=device)

    if text_prompts is not None:
        if len(text_prompts) != len(videos):
            raise ValueError(
                "C2L-VLA text prompts must match the context batch size: "
                f"prompts={len(text_prompts)}, samples={len(videos)}."
            )
        _add_text_prompt_inputs(processor.tokenizer, inputs, text_prompts, device=device)

    if (block_actions is None) != (block_action_is_empty is None):
        raise ValueError(
            "C2L-VLA action interleaving needs both `block_actions` and `block_action_is_empty`."
        )
    if block_actions is not None:
        counts = [int(np.asarray(values).shape[0]) for values in block_actions]
        if counts != expected_blocks:
            raise ValueError(
                "C2L-VLA action history does not match the block counts: "
                f"actions={counts}, blocks={expected_blocks}."
            )
        _add_action_history_inputs(inputs, block_actions, block_action_is_empty, device=device)

    if device is not None:
        inputs = inputs.to(device)
    return inputs
