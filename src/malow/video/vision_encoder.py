# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import transformers
from transformers import PretrainedConfig, PreTrainedModel


@dataclass(frozen=True)
class VisionEncoderLoadSpec:
    model_class_name: str
    config_attribute: str
    checkpoint_key_mapping: dict[str, str]
    output_hidden_size_attribute: str


@dataclass(frozen=True)
class LoadedVisionEncoder:
    model: PreTrainedModel
    feature_config: PretrainedConfig


VISION_ENCODER_LOAD_SPECS = {
    "qwen3_5": VisionEncoderLoadSpec(
        model_class_name="Qwen3_5VisionModel",
        config_attribute="vision_config",
        checkpoint_key_mapping={r"^model\.visual\.(.*)$": r"\1"},
        output_hidden_size_attribute="out_hidden_size",
    ),
    "qwen3_vl": VisionEncoderLoadSpec(
        model_class_name="Qwen3VLVisionModel",
        config_attribute="vision_config",
        checkpoint_key_mapping={r"^model\.visual\.(.*)$": r"\1"},
        output_hidden_size_attribute="out_hidden_size",
    ),
}


def load_c2l_vision_encoder(
    model_name_or_path: str | Path,
    model_config: PretrainedConfig,
    **from_pretrained_kwargs: Any,
) -> LoadedVisionEncoder:
    """Load only the projected vision tower from a full multimodal checkpoint."""
    model_type = getattr(model_config, "model_type", None)
    spec = VISION_ENCODER_LOAD_SPECS.get(model_type)
    if spec is None:
        raise ValueError(f"Unsupported C2L vision-only encoder model type: {model_type}")

    model_class = getattr(transformers, spec.model_class_name, None)
    if model_class is None:
        raise ImportError(
            f"C2L vision-only loading for {model_type} requires Transformers to export "
            f"{spec.model_class_name}."
        )

    vision_config = deepcopy(getattr(model_config, spec.config_attribute))
    load_result = model_class.from_pretrained(
        model_name_or_path,
        config=vision_config,
        key_mapping=spec.checkpoint_key_mapping,
        output_loading_info=True,
        **from_pretrained_kwargs,
    )
    vision_model, loading_info = load_result
    missing_keys = sorted(loading_info.get("missing_keys", ()))
    mismatched_keys = sorted(loading_info.get("mismatched_keys", ()), key=str)
    error_messages = loading_info.get("error_msgs", ()) or loading_info.get("conversion_errors", ())
    if missing_keys or mismatched_keys or error_messages:
        raise RuntimeError(
            "Failed to load the complete C2L vision encoder: "
            f"missing={missing_keys}, mismatched={mismatched_keys}, errors={error_messages}"
        )

    vision_model.requires_grad_(False)
    vision_model.eval()
    feature_config = deepcopy(vision_config)
    feature_config.hidden_size = getattr(vision_config, spec.output_hidden_size_attribute)
    return LoadedVisionEncoder(model=vision_model, feature_config=feature_config)

