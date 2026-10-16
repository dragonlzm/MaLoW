# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel

from malow.video.ctx_to_lora.configs import CtxEncoderArguments
from malow.video.ctx_to_lora.utils import get_base_model

logger = logging.getLogger()


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
            image_features = feature_getter(pixel_values, grid_thw)
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
            split_sizes = (grid_thw.prod(-1) // visual.spatial_merge_size**2).tolist()
            image_features = torch.split(image_features, split_sizes)

        # block is skipped
        if isinstance(image_features, torch.Tensor):
            split_sizes = self._split_sizes(grid_thw)
            image_features = torch.split(image_features, split_sizes)

        features = [feature for feature in image_features if feature.numel() > 0]
        if not features:
            raise ValueError(f"vision_embedder received no {media_type} features.")

        return [feature.to(dtype=features[0].dtype) for feature in features]

    def _get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        return self._get_visual_features(pixel_values, image_grid_thw, media_type="image")

    def _get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        return self._get_visual_features(pixel_values_videos, video_grid_thw, media_type="video")

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in {"labels"} and v is not None}
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        video_grid_thw = kwargs.get("video_grid_thw")
        features = []
        if pixel_values is not None or image_grid_thw is not None:
            features.extend(self._get_image_features(pixel_values, image_grid_thw))
        if pixel_values_videos is not None or video_grid_thw is not None:
            features.extend(self._get_video_features(pixel_values_videos, video_grid_thw))
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
        visual = self._get_visual_model()
        merge_size = getattr(visual, "spatial_merge_size", 1)
        return (image_grid_thw.prod(-1) // merge_size**2).tolist()

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
