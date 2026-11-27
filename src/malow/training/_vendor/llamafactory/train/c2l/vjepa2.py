from __future__ import annotations

from typing import Any

import torch
from transformers.image_processing_base import BatchFeature


def prepare_vjepa2_encoder_inputs(
    videos: list[list[Any]],
    video_processor,
    *,
    model_config,
    window_size: int = 16,
    stride: int = 2,
    spatial_pool_size: int = 2,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Preprocess each unique frame once and describe the deferred sliding windows."""
    if window_size < 2 or window_size % 2:
        raise ValueError("V-JEPA2 window_size must be an even integer >= 2.")
    if stride < 1 or window_size % stride:
        raise ValueError("V-JEPA2 stride must be >= 1 and divide window_size.")
    if spatial_pool_size < 1:
        raise ValueError("V-JEPA2 spatial_pool_size must be >= 1.")
    tubelet_size = int(getattr(model_config, "tubelet_size", 0) or 0)
    patch_size = int(getattr(model_config, "patch_size", 0) or 0)
    image_size = int(getattr(model_config, "image_size", 0) or 0)
    if tubelet_size != stride:
        raise ValueError(
            "V-JEPA2 sliding-window stride must match the checkpoint tubelet size: "
            f"stride={stride}, tubelet_size={tubelet_size}."
        )
    if patch_size < 1 or image_size < 1 or image_size % patch_size:
        raise ValueError("V-JEPA2 image_size must be divisible by patch_size.")
    spatial_tokens = image_size // patch_size
    if spatial_tokens % spatial_pool_size:
        raise ValueError("V-JEPA2 spatial patch grid must be divisible by spatial_pool_size.")

    processed_videos = []
    frame_counts = []
    block_counts = []
    for sample_idx, original_frames in enumerate(videos):
        frames = list(original_frames)
        if not frames:
            raise ValueError(f"V-JEPA2 video sample {sample_idx} contains no frames.")
        processed = video_processor(videos=[frames], return_tensors="pt")
        pixel_values = processed["pixel_values_videos"]
        if pixel_values.dim() != 5 or pixel_values.shape[0] != 1:
            raise ValueError(
                "V-JEPA2 video processor must return [1, frames, channels, height, width] "
                f"for one source video, got {tuple(pixel_values.shape)}."
            )
        pixel_values = pixel_values[0]
        if pixel_values.shape[0] != len(frames):
            raise ValueError(
                "V-JEPA2 video processor changed the source frame count: "
                f"input={len(frames)}, output={pixel_values.shape[0]}."
            )
        if pixel_values.shape[0] % stride:
            padding = stride - pixel_values.shape[0] % stride
            pixel_values = torch.cat([pixel_values, pixel_values[-1:].expand(padding, -1, -1, -1)], dim=0)
        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)
        processed_videos.append(pixel_values)
        frame_counts.append(pixel_values.shape[0])
        block_counts.append(pixel_values.shape[0] // stride)

    pixel_values_videos = processed_videos[0] if len(processed_videos) == 1 else torch.cat(processed_videos)

    pooled_spatial_tokens = spatial_tokens // spatial_pool_size
    return {
        "pixel_values_videos": pixel_values_videos,
        "vjepa2_frame_counts": torch.tensor(frame_counts, dtype=torch.long),
        "vjepa2_block_counts": torch.tensor(block_counts, dtype=torch.long),
        "video_grid_thw": torch.tensor(
            [[count, pooled_spatial_tokens, pooled_spatial_tokens] for count in block_counts],
            dtype=torch.long,
        ),
        "video_temporal_patch_size": torch.full(
            (len(block_counts),),
            tubelet_size,
            dtype=torch.long,
        ),
    }


def model_uses_vjepa2(model) -> bool:
    ctx_encoder_args = getattr(model, "ctx_encoder_args", None)
    encoder_type = getattr(ctx_encoder_args, "ctx_encoder_type", None)
    encoder_type = getattr(encoder_type, "value", encoder_type)
    return encoder_type == "vjepa2_vision_embedder"


def prepare_vjepa2_encoder_inputs_for_model(model, videos: list[list[Any]]) -> BatchFeature:
    """Apply the V-JEPA2 preprocessing contract saved with a C2L checkpoint."""
    if not model_uses_vjepa2(model):
        raise ValueError("The C2L model does not use a V-JEPA2 context encoder.")
    processor = getattr(model, "vjepa2_video_processor", None)
    encoder_model = getattr(model, "encoder_model", None)
    if processor is None or encoder_model is None:
        raise ValueError("The V-JEPA2 C2L model is missing its video processor or encoder model.")
    parameter = next(encoder_model.parameters(), None)
    input_dtype = parameter.dtype if parameter is not None and parameter.is_floating_point() else None
    config = model.ctx_encoder_args
    return BatchFeature(
        data=prepare_vjepa2_encoder_inputs(
            videos,
            processor,
            model_config=encoder_model.config,
            window_size=int(config.vjepa2_window_size),
            stride=int(config.vjepa2_window_stride),
            spatial_pool_size=int(config.vjepa2_spatial_pool_size),
            dtype=input_dtype,
        )
    )
