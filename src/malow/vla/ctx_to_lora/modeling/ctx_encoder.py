import logging
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel

from malow.vla.ctx_to_lora.configs import CtxEncoderArguments
from malow.vla.ctx_to_lora.utils import get_base_model

logger = logging.getLogger()


VIDEOCHAT3_MODEL_TYPE = "videochat3"


def resolve_video_token_layout(base_model) -> tuple[int, int]:
    """Return ``(spatial_merge_size, temporal_patch_size)`` for a vision tower.

    Both numbers describe how many input patches/frames collapse into one language
    token, which is what the block-LoRA frame splitter needs in order to map a token
    span back to a frame span.

    Qwen towers advertise ``spatial_merge_size`` / ``temporal_patch_size`` directly on
    the module, and that path is preserved bit-for-bit. VideoChat3 advertises neither:
    it keeps ``merge_kernel_size`` (a 2-tuple) and ``temporal_merge_size`` on
    ``config.vision_config``, and its ``patch_merger`` takes a temporal MEAN over each
    ``temporal_merge_size``-frame clip, so its effective temporal patch size is 4 while
    ``temporal_patch_size`` in the config is 1 (the patch-embed stride, not the merge).
    Reading ``temporal_patch_size`` off a VideoChat3 config would therefore report 1
    and silently quadruple every computed frame span.
    """
    config = getattr(base_model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == VIDEOCHAT3_MODEL_TYPE:
        vision_config = getattr(config, "vision_config", None)
        merge_kernel_size = getattr(vision_config, "merge_kernel_size", (2, 2)) or (2, 2)
        if isinstance(merge_kernel_size, (list, tuple)):
            if len(merge_kernel_size) != 2 or merge_kernel_size[0] != merge_kernel_size[1]:
                raise ValueError(
                    "C2L block LoRA assumes a square VideoChat3 spatial merge kernel, got "
                    f"{tuple(merge_kernel_size)}."
                )
            spatial_merge_size = int(merge_kernel_size[0])
        else:
            spatial_merge_size = int(merge_kernel_size)
        temporal_merge_size = int(getattr(vision_config, "temporal_merge_size", 4) or 4)
        return max(1, spatial_merge_size), max(1, temporal_merge_size)

    visual = base_model
    if not hasattr(visual, "spatial_merge_size"):
        for attribute in ("visual", "vision_tower"):
            candidate = getattr(base_model, attribute, None)
            if candidate is None:
                inner = getattr(base_model, "model", None)
                candidate = getattr(inner, attribute, None) if inner is not None else None
            if candidate is not None and hasattr(candidate, "spatial_merge_size"):
                visual = candidate
                break

    spatial_merge_size = int(getattr(visual, "spatial_merge_size", getattr(visual, "merge_size", 1)) or 1)
    temporal_patch_size = int(getattr(visual, "temporal_patch_size", 1) or 1)
    return max(1, spatial_merge_size), max(1, temporal_patch_size)


DEFAULT_VIDEOCHAT3_VISION_CHUNK_FRAMES = 128
VIDEOCHAT3_VISION_CHUNK_FRAMES_ENV = "C2L_VIDEOCHAT3_VISION_CHUNK_FRAMES"


def videochat3_vision_chunk_frames() -> int:
    """Frames the VideoChat3 vision tower encodes per forward pass. 0 disables chunking."""
    import os

    raw = os.environ.get(VIDEOCHAT3_VISION_CHUNK_FRAMES_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_VIDEOCHAT3_VISION_CHUNK_FRAMES
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"{VIDEOCHAT3_VISION_CHUNK_FRAMES_ENV} must be an integer, got {raw!r}."
        ) from exc


def videochat3_vision_frame_chunks(
    grid_thw, max_frames: int, temporal_merge_size: int
) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    """Split ``grid_thw`` into clip-aligned encoder chunks.

    Returns ``(sub_rows, patch_start, patch_end)`` per chunk, where ``sub_rows`` are
    ``(t, h, w)`` triples and the patch bounds index the flat pixel-value rows.

    Exactness: ``VideoChat3VisionModel.forward`` first chops every grid row into
    ``temporal_merge_size``-frame clips, and each clip is its own attention segment
    (``cu_seqlens``), its own rope span, and its own ``patch_merger`` group. Nothing in
    the tower crosses a clip boundary, so encoding clip-aligned frame ranges separately
    and concatenating the results reproduces the single-pass output exactly. Only a
    row's own trailing remainder is allowed to be shorter than a clip, which is where
    the single-pass split puts its short clip too.

    Verified against the real tower (`flash_attention_2` and `sdpa` agree with the
    single pass to float32 round-off). This holds only while the tower masks by
    `cu_seqlens`: the checkpoint's `eager` branch adds its boolean mask to the logits
    instead of masking with it, so that path attends across clips and chunking it would
    change the features. `vision_config.attn_impl` is `flash_attention_2`.
    """
    if max_frames < 1:
        raise ValueError("VideoChat3 vision chunking requires max_frames >= 1.")
    temporal_merge_size = max(1, int(temporal_merge_size))

    chunks: list[tuple[list[tuple[int, int, int]], int, int]] = []
    rows: list[tuple[int, int, int]] = []
    frames = 0
    patch_start = 0
    patch_cursor = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        offset = 0
        while offset < t:
            if rows and max_frames - frames < temporal_merge_size:
                chunks.append((rows, patch_start, patch_cursor))
                rows, frames, patch_start = [], 0, patch_cursor
            room = max(max_frames - frames, temporal_merge_size)
            take = min(t - offset, room)
            if take < t - offset and take % temporal_merge_size:
                # Never hand the tower a short clip mid-row; that clip would then be
                # merged over fewer frames than the single-pass forward merges it over.
                take -= take % temporal_merge_size
            rows.append((take, h, w))
            frames += take
            patch_cursor += take * h * w
            offset += take

    if rows:
        chunks.append((rows, patch_start, patch_cursor))
    return chunks


def video_token_counts_from_grid(base_model, grid_thw) -> list[int]:
    """Number of language tokens each ``grid_thw`` row expands into.

    Qwen: ``t * h * w / spatial_merge**2`` -- every frame survives as its own token
    plane. VideoChat3: ``ceil(t / temporal_merge) * (h * w) / spatial_merge**2`` --
    ``VideoChat3VisionModel.split_grid_thws_clip_by_clip`` chops ``t`` into
    ``temporal_merge_size``-frame clips and ``patch_merger`` averages each clip down to
    a single token plane, so the naive product overcounts by ~4x.
    """
    spatial_merge_size, temporal_patch_size = resolve_video_token_layout(base_model)
    config = getattr(base_model, "config", None)
    if str(getattr(config, "model_type", "") or "").lower() != VIDEOCHAT3_MODEL_TYPE:
        return (grid_thw.prod(-1) // spatial_merge_size**2).tolist()

    counts = []
    for t, h, w in grid_thw.tolist():
        clips = -(-int(t) // temporal_patch_size)  # ceil
        counts.append(clips * (int(h) * int(w)) // spatial_merge_size**2)
    return counts


def build_vjepa2_window_indices(
    frame_counts: list[int] | torch.Tensor,
    *,
    window_size: int = 16,
    stride: int = 2,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build global indices for past-only windows over concatenated video frames."""
    if window_size < 2 or window_size % 2:
        raise ValueError("V-JEPA2 window_size must be an even integer >= 2.")
    if stride < 1 or window_size % stride:
        raise ValueError("V-JEPA2 stride must be >= 1 and divide window_size.")

    if isinstance(frame_counts, torch.Tensor):
        counts = [int(value) for value in frame_counts.reshape(-1).tolist()]
        device = frame_counts.device if device is None else device
    else:
        counts = [int(value) for value in frame_counts]
    device = device or torch.device("cpu")
    if not counts:
        raise ValueError("V-JEPA2 window indexing requires at least one video.")

    index_chunks = []
    owner_chunks = []
    block_counts: list[int] = []
    frame_offset = 0
    relative_positions = torch.arange(-window_size, 0, dtype=torch.long, device=device)
    for sample_idx, frame_count in enumerate(counts):
        if frame_count < 1 or frame_count % stride:
            raise ValueError(
                "V-JEPA2 processed frame counts must be positive multiples of the window stride: "
                f"sample={sample_idx}, frames={frame_count}, stride={stride}."
            )
        block_ends = torch.arange(stride, frame_count + 1, stride, dtype=torch.long, device=device)
        local_indices = (block_ends[:, None] + relative_positions[None, :]).clamp_(min=0)
        index_chunks.append(local_indices + frame_offset)
        owner_chunks.append(torch.full((block_ends.numel(),), sample_idx, dtype=torch.long, device=device))
        block_counts.append(block_ends.numel())
        frame_offset += frame_count

    return (
        torch.cat(index_chunks, dim=0),
        torch.cat(owner_chunks, dim=0),
        torch.tensor(block_counts, dtype=torch.long, device=device),
    )


@contextmanager
def early_exit(base_model: PreTrainedModel, exit_layer: int):
    try:
        layers = base_model.layers
        base_model.layers = layers[:exit_layer]
        yield base_model
    finally:
        base_model.layers = layers


@contextmanager
def maybe_add_batch_dim(kwargs):
    try:
        batched_input = False
        batched_attn_mask = False
        if (
            "input_ids" in kwargs
            and kwargs["input_ids"] is not None
            and len(kwargs["input_ids"].shape) == 1
        ):
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
            batched_input = True
        if (
            "attention_mask" in kwargs
            and kwargs["attention_mask"] is not None
            and isinstance(kwargs["attention_mask"], torch.Tensor)
            and len(kwargs["attention_mask"].shape) == 1
        ):
            kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)
            batched_attn_mask = True
        yield batched_input, batched_attn_mask
    finally:
        if batched_input:
            kwargs["input_ids"] = kwargs["input_ids"].squeeze(0)
        if batched_attn_mask:
            kwargs["attention_mask"] = kwargs["attention_mask"].squeeze(0)


class EarlyExit(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        base_model = get_base_model(base_model)
        if "gte" in base_model.config.name_or_path:
            base_model.encoder.layer = base_model.encoder.layer[: config.layer_idx]
        else:
            base_model.layers = base_model.layers[: config.layer_idx]

        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        model_outputs = self.base_model(**kwargs)
        return model_outputs.last_hidden_state


class EmbeddingOnly(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return the embeddings only
        return outputs.hidden_states[0]  # The first hidden state is the embeddings


class PerLayerActivations(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.keep_lm_head = getattr(config, "keep_lm_head", False)
        if not self.keep_lm_head:
            base_model = get_base_model(base_model)  # remove lm head
        else:
            base_model.lm_head = nn.Identity()

        # -1 to remove last attn block
        if config.ctx_encoder_last_layer is not None:
            last_layer = config.ctx_encoder_last_layer - 1
        else:
            last_layer = -1

        if self.keep_lm_head:
            base_model.model.layers = base_model.model.layers[:last_layer]
        else:
            base_model.layers = base_model.layers[:last_layer]
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.base_model.set_decoder(decoder)

    def get_decoder(self):
        return self.base_model.get_decoder()

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return all layers' activations except the last one
        # from embeddings to the input of the last attn block
        # Shape: (batch_size, num_layers, seq_len, hidden_size)

        if self.keep_lm_head:
            return outputs
        else:
            return torch.stack(outputs.hidden_states, dim=1)


class VisionEmbedder(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.base_model = base_model

    @property
    def config(self):
        config = self.base_model.config
        return getattr(config, "text_config", config)

    def _get_image_token_id(self) -> int:
        image_token_id = getattr(self.base_model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("multimodal_embedder requires `base_model.config.image_token_id`.")
        return image_token_id

    def _get_visual_model(self):
        if hasattr(self.base_model, "spatial_merge_size") and hasattr(self.base_model, "merger"):
            return self.base_model

        visual = getattr(self.base_model, "visual", None)
        if visual is not None:
            return visual

        model = getattr(self.base_model, "model", None)
        visual = getattr(model, "visual", None)
        if visual is not None:
            return visual

        # VideoChat3 exposes its tower as `.vision_tower` on the wrapper and on the
        # inner model; it has no `.visual` alias.
        visual = getattr(self.base_model, "vision_tower", None)
        if visual is not None:
            return visual

        visual = getattr(model, "vision_tower", None)
        if visual is not None:
            return visual

        return None

    def _get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        media_type: str = "image",
    ) -> list[torch.Tensor]:
        if pixel_values is None or grid_thw is None:
            raise ValueError(f"vision_embedder requires `{media_type}` pixel values and grid metadata.")

        feature_getter = getattr(self.base_model, f"get_{media_type}_features", None)
        if callable(feature_getter):
            image_features = self._call_feature_getter(feature_getter, pixel_values, grid_thw)
            image_features = self._extract_pooler_features(image_features)
            if (
                isinstance(image_features, tuple)
                and len(image_features) == 2
                and isinstance(image_features[0], (list, tuple))
            ):
                image_features = image_features[0]
        else:
            visual = self._get_visual_model()
            if visual is None:
                raise ValueError("vision_embedder requires a model with `get_image_features` or `.visual`.")
            pixel_values = pixel_values.type(visual.dtype)
            image_features = visual(pixel_values, grid_thw=grid_thw)
            image_features = self._extract_pooler_features(image_features)
            if isinstance(image_features, tuple):
                image_features = image_features[0]
            split_sizes = video_token_counts_from_grid(self.base_model, grid_thw)
            image_features = torch.split(image_features, split_sizes)

        # block is skipped
        if isinstance(image_features, torch.Tensor):
            split_sizes = self._split_sizes(grid_thw)
            image_features = torch.split(image_features, split_sizes)

        features = [feature for feature in image_features if feature.numel() > 0]
        if not features:
            raise ValueError(f"vision_embedder received no {media_type} features.")

        return [feature.to(dtype=features[0].dtype) for feature in features]

    def _call_feature_getter(
        self,
        feature_getter,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        """Context modulation inference helper."""
        config = getattr(self.base_model, "config", None)
        if str(getattr(config, "model_type", "") or "").lower() != VIDEOCHAT3_MODEL_TYPE:
            return feature_getter(pixel_values, grid_thw)

        max_frames = videochat3_vision_chunk_frames()
        if max_frames < 1 or grid_thw is None or int(grid_thw.numel()) == 0:
            return feature_getter(pixel_values, grid_thw)
        if int(grid_thw[:, 0].sum().item()) <= max_frames:
            return feature_getter(pixel_values, grid_thw)

        expected_patch_rows = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        if pixel_values is None or pixel_values.shape[0] != expected_patch_rows:
            # The pixel rows do not line up with the grid, so they cannot be sliced by it.
            return feature_getter(pixel_values, grid_thw)

        _, temporal_merge_size = resolve_video_token_layout(self.base_model)
        chunks = videochat3_vision_frame_chunks(grid_thw, max_frames, temporal_merge_size)
        features = []
        for rows, patch_start, patch_end in chunks:
            sub_grid = torch.tensor(rows, dtype=grid_thw.dtype, device=grid_thw.device)
            chunk_features = feature_getter(pixel_values[patch_start:patch_end], sub_grid)
            if not isinstance(chunk_features, torch.Tensor):
                raise ValueError(
                    "VideoChat3 chunked vision encoding expects tensor features, got "
                    f"{type(chunk_features).__name__}."
                )
            features.append(chunk_features)
        return torch.cat(features, dim=0)

    def _get_position_id_media_features(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        media_type: str,
        num_soft_tokens: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Features for encoders that describe patches with position ids, not a grid.

        Gemma 4 differs from Qwen in three ways that matter here: its
        ``get_video_features`` lives on the inner model rather than on the
        ``ForConditionalGeneration`` wrapper, it takes ``*_position_ids`` in place
        of a ``grid_thw`` tensor, and it returns a single flat
        ``[total_soft_tokens, hidden]`` pooler output with media and frames
        collapsed onto one axis. The per-medium split is therefore recovered from
        the input layout instead of from grid metadata.

        Only reachable when the caller has no ``grid_thw``; the Qwen path is
        untouched.
        """
        if pixel_values is None:
            raise ValueError(f"vision_embedder requires `{media_type}` pixel values.")

        feature_getter = getattr(self.base_model, f"get_{media_type}_features", None)
        if not callable(feature_getter):
            inner_model = getattr(self.base_model, "model", None)
            feature_getter = getattr(inner_model, f"get_{media_type}_features", None)
        if not callable(feature_getter):
            raise ValueError(
                f"vision_embedder could not find `get_{media_type}_features` on "
                f"{type(self.base_model).__name__} or on its inner model."
            )

        features = self._extract_pooler_features(feature_getter(pixel_values, position_ids))
        if isinstance(features, tuple):
            features = features[0]
        if not isinstance(features, torch.Tensor):
            raise ValueError(
                f"vision_embedder expected tensor {media_type} features, got {type(features).__name__}."
            )
        if features.dim() != 2:
            raise ValueError(
                f"vision_embedder expected flat [tokens, hidden] {media_type} features, "
                f"got {tuple(features.shape)}."
            )

        # Videos arrive as [media, frames, patches, pixels]; images omit the frame axis.
        num_media = int(pixel_values.shape[0])
        frames_per_medium = int(pixel_values.shape[1]) if pixel_values.dim() == 4 else 1
        total_tokens = int(features.shape[0])

        # `num_soft_tokens_per_*` is the per-frame soft-token count and is exact even
        # when media in one batch were resized differently. Fall back to a uniform
        # split when it is absent or inconsistent with the returned features.
        split_sizes: list[int] | None = None
        if num_soft_tokens is not None and int(num_soft_tokens.numel()) == num_media:
            candidate = [
                frames_per_medium * int(tokens) for tokens in num_soft_tokens.reshape(-1).tolist()
            ]
            if all(size > 0 for size in candidate) and sum(candidate) == total_tokens:
                split_sizes = candidate

        if split_sizes is None:
            units = num_media * frames_per_medium
            if units <= 0 or total_tokens % units:
                raise ValueError(
                    f"vision_embedder cannot split {total_tokens} {media_type} soft tokens evenly "
                    f"across {num_media} media x {frames_per_medium} frames."
                )
            split_sizes = [frames_per_medium * (total_tokens // units)] * num_media

        media_features = [
            feature for feature in torch.split(features, split_sizes) if feature.numel() > 0
        ]
        if not media_features:
            raise ValueError(f"vision_embedder received no {media_type} features.")
        return [feature.to(dtype=media_features[0].dtype) for feature in media_features]

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        num_soft_tokens_per_image: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if image_grid_thw is None and image_position_ids is not None:
            return self._get_position_id_media_features(
                pixel_values, image_position_ids, "image", num_soft_tokens_per_image
            )
        return self._get_visual_features(pixel_values, image_grid_thw, media_type="image")

    def _get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        video_position_ids: torch.Tensor | None = None,
        num_soft_tokens_per_video: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if video_grid_thw is None and video_position_ids is not None:
            return self._get_position_id_media_features(
                pixel_values_videos, video_position_ids, "video", num_soft_tokens_per_video
            )
        return self._get_visual_features(pixel_values_videos, video_grid_thw, media_type="video")

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in {"labels"} and v is not None}
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        video_grid_thw = kwargs.get("video_grid_thw")
        # Grid-less encoders (Gemma 4) describe patches with position ids instead.
        image_position_ids = kwargs.get("image_position_ids")
        video_position_ids = kwargs.get("video_position_ids")
        num_soft_tokens_per_image = kwargs.get("num_soft_tokens_per_image")
        num_soft_tokens_per_video = kwargs.get("num_soft_tokens_per_video")
        features = []
        if pixel_values is not None or image_grid_thw is not None:
            features.extend(
                self._get_image_features(
                    pixel_values, image_grid_thw, image_position_ids, num_soft_tokens_per_image
                )
            )
        if pixel_values_videos is not None or video_grid_thw is not None:
            features.extend(
                self._get_video_features(
                    pixel_values_videos, video_grid_thw, video_position_ids, num_soft_tokens_per_video
                )
            )
        if not features:
            raise ValueError("vision_embedder requires image or video encoder inputs.")
        features = [feature.to(dtype=features[0].dtype) for feature in features]
        padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
        attn_mask = torch.zeros(
            (len(features), padded_features.shape[1]),
            dtype=torch.long,
            device=padded_features.device,
        )
        for idx, feature in enumerate(features):
            attn_mask[idx, : feature.shape[0]] = 1

        return padded_features, attn_mask

    def _split_sizes(self, image_grid_thw: torch.Tensor) -> list[int]:
        return video_token_counts_from_grid(self.base_model, image_grid_thw)

    def video_token_layout(self) -> tuple[int, int]:
        """``(spatial_merge_size, temporal_patch_size)`` for the wrapped vision tower."""
        return resolve_video_token_layout(self.base_model)

    def _extract_pooler_features(self, image_features):
        if hasattr(image_features, "pooler_output"):
            return image_features.pooler_output
        if isinstance(image_features, Mapping) and "pooler_output" in image_features:
            return image_features["pooler_output"]
        return image_features


class MultimodalEmbedder(VisionEmbedder):
    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in {"labels"} and v is not None}
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        if input_ids is None:
            raise ValueError("multimodal_embedder requires `input_ids`.")
        if input_ids.dim() != 2:
            raise ValueError(f"multimodal_embedder expects 2D input_ids, got shape {tuple(input_ids.shape)}.")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "multimodal_embedder expects a 2D attention_mask for EncoderAggregator, "
                f"got shape {tuple(attention_mask.shape)}."
            )

        image_features = self._get_image_features(pixel_values, image_grid_thw)
        image_token_id = self._get_image_token_id()
        input_embeddings = self.base_model.get_input_embeddings()
        inputs_embeds = input_embeddings(input_ids).clone()
        image_embeds = torch.cat(image_features, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if hasattr(self.base_model.model, "get_placeholder_mask"):
            image_mask, _ = self.base_model.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
        else:
            raise NotImplementedError
            image_token_mask = input_ids == image_token_id
            image_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[image_mask].numel() != image_embeds.numel():
                n_image_tokens = image_token_mask.sum().item()
                raise ValueError(
                    "multimodal_embedder image token count mismatch: "
                    f"input_ids contain {n_image_tokens} image tokens, "
                    f"but image features contain {image_embeds.shape[0]} tokens."
                )

        if not image_mask.any():
            raise ValueError(
                "multimodal_embedder received image features but input_ids contain no image placeholder tokens."
            )

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds, attention_mask


class VJEPA2VisionEmbedder(nn.Module):
    """Extract the pooled final tubelet from each past-only V-JEPA2 window."""

    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.base_model = base_model
        self.window_size = int(getattr(config, "vjepa2_window_size", 16))
        self.window_stride = int(getattr(config, "vjepa2_window_stride", 2))
        self.micro_batch_size = int(getattr(config, "vjepa2_micro_batch_size", 32))
        self.spatial_pool_size = int(getattr(config, "vjepa2_spatial_pool_size", 2))
        if self.window_size < 2 or self.window_size % 2:
            raise ValueError("V-JEPA2 window size must be an even integer >= 2.")
        if self.window_stride < 1 or self.window_size % self.window_stride:
            raise ValueError("V-JEPA2 window stride must be >= 1 and divide the window size.")
        if self.micro_batch_size < 1:
            raise ValueError("V-JEPA2 micro batch size must be >= 1.")
        if self.spatial_pool_size < 1:
            raise ValueError("V-JEPA2 spatial pool size must be >= 1.")

    @property
    def config(self):
        return self.base_model.config

    def _select_pooled_tail(self, hidden_states: torch.Tensor) -> torch.Tensor:
        model_config = self.base_model.config
        image_size = int(model_config.image_size)
        patch_size = int(model_config.patch_size)
        spatial_tokens = image_size // patch_size
        tokens_per_tubelet = spatial_tokens**2
        if hidden_states.dim() != 3 or hidden_states.shape[1] % tokens_per_tubelet:
            raise ValueError(
                "Unexpected V-JEPA2 hidden-state layout: "
                f"shape={tuple(hidden_states.shape)}, tokens_per_tubelet={tokens_per_tubelet}."
            )
        if spatial_tokens % self.spatial_pool_size:
            raise ValueError("V-JEPA2 spatial patch grid is not divisible by the configured pool size.")

        tail = hidden_states[:, -tokens_per_tubelet:]
        tail = tail.reshape(hidden_states.shape[0], spatial_tokens, spatial_tokens, hidden_states.shape[-1])
        pooled_size = spatial_tokens // self.spatial_pool_size
        tail = tail.reshape(
            hidden_states.shape[0],
            pooled_size,
            self.spatial_pool_size,
            pooled_size,
            self.spatial_pool_size,
            hidden_states.shape[-1],
        ).mean(dim=(2, 4))
        return tail.reshape(hidden_states.shape[0], pooled_size**2, hidden_states.shape[-1])

    @torch.no_grad()
    def forward(self, **kwargs):
        pixel_values = kwargs.get("pixel_values_videos")
        frame_counts = kwargs.get("vjepa2_frame_counts")
        block_counts = kwargs.get("vjepa2_block_counts")
        if pixel_values is None or frame_counts is None or block_counts is None:
            raise ValueError(
                "vjepa2_vision_embedder requires compact pixel values, frame counts, and block counts."
            )
        if pixel_values.dim() != 4:
            raise ValueError(
                "V-JEPA2 compact pixel values must have shape [total_frames, channels, height, width], "
                f"got {tuple(pixel_values.shape)}."
            )
        frame_counts = frame_counts.reshape(-1).to(dtype=torch.long)
        block_counts = block_counts.reshape(-1).to(dtype=torch.long)
        if frame_counts.numel() != block_counts.numel():
            raise ValueError("V-JEPA2 frame counts and block counts must have one entry per video.")
        if int(frame_counts.sum().item()) != pixel_values.shape[0]:
            raise ValueError("V-JEPA2 frame counts do not match the compact pixel-value tensor.")

        parameter = next(self.base_model.parameters(), None)
        model_device = parameter.device if parameter is not None else pixel_values.device
        model_dtype = parameter.dtype if parameter is not None and parameter.is_floating_point() else pixel_values.dtype
        pixel_values = pixel_values.to(device=model_device, dtype=model_dtype)
        window_indices, owners, derived_block_counts = build_vjepa2_window_indices(
            frame_counts,
            window_size=self.window_size,
            stride=self.window_stride,
            device=model_device,
        )
        if not torch.equal(block_counts.to(device=model_device), derived_block_counts):
            raise ValueError("V-JEPA2 block counts do not match the compact frame counts.")

        pooled_chunks = []
        for start in range(0, window_indices.shape[0], self.micro_batch_size):
            end = min(start + self.micro_batch_size, window_indices.shape[0])
            windows = pixel_values[window_indices[start:end]]
            outputs = self.base_model(
                pixel_values_videos=windows,
                skip_predictor=True,
            )
            pooled_chunks.append(self._select_pooled_tail(outputs.last_hidden_state))
            del outputs, windows
        pooled_windows = torch.cat(pooled_chunks, dim=0)

        sample_features = []
        for sample_idx, expected_count in enumerate(derived_block_counts.tolist()):
            blocks = pooled_windows[owners == sample_idx]
            if blocks.shape[0] != expected_count:
                raise ValueError(
                    f"V-JEPA2 sample {sample_idx} expected {expected_count} windows, got {blocks.shape[0]}."
                )
            sample_features.append(blocks.reshape(-1, blocks.shape[-1]))

        padded_features = pad_sequence(sample_features, batch_first=True, padding_value=0.0)
        attention_mask = torch.zeros(
            padded_features.shape[:2],
            dtype=torch.long,
            device=padded_features.device,
        )
        for sample_idx, feature in enumerate(sample_features):
            attention_mask[sample_idx, : feature.shape[0]] = 1
        return padded_features, attention_mask


class CTX_ENCODER_TYPE(str, Enum):
    EARLY_EXIT = "early_exit"
    EMBED_ONLY = "embed_only"
    PER_LAYER_ACTIVATIONS = "per_layer_activations"
    VISION_EMBEDDER = "vision_embedder"
    VJEPA2_VISION_EMBEDDER = "vjepa2_vision_embedder"
    MULTIMODAL_EMBEDDER = "multimodal_embedder"


CTX_ENCODER_CLS = {
    CTX_ENCODER_TYPE.EARLY_EXIT: EarlyExit,
    CTX_ENCODER_TYPE.EMBED_ONLY: EmbeddingOnly,
    CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS: PerLayerActivations,
    CTX_ENCODER_TYPE.VISION_EMBEDDER: VisionEmbedder,
    CTX_ENCODER_TYPE.VJEPA2_VISION_EMBEDDER: VJEPA2VisionEmbedder,
    CTX_ENCODER_TYPE.MULTIMODAL_EMBEDDER: MultimodalEmbedder,
}
