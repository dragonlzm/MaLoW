"""Inference compatibility implementation; see SOURCE_NOTICE.md."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from malow.vla.ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    resolve_temporal_state_options,
)

from malow.checkpoints import load_checkpoint_metadata
from malow.vla.constants import VLA_ADAPTER_DIR, VLA_CONFIG_NAME, VLA_TRAINABLE_WEIGHTS_NAME
from .merge import assert_adapter_merged
from .modeling import C2LVLAModelWrapper


logger = logging.getLogger(__name__)


def load_c2l_vla_state(checkpoint_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    metadata = load_checkpoint_metadata(checkpoint_dir, runtime="vla")

    for candidate in (checkpoint_dir / "model.safetensors",):
        if candidate.is_file():
            if candidate.suffix == ".safetensors":
                tensors = load_file(str(candidate), device="cpu")
            else:
                tensors = torch.load(candidate, map_location="cpu", weights_only=True)
            break
    else:
        raise FileNotFoundError(f"No hypernet weights found in {checkpoint_dir}.")

    state = dict(tensors)
    state.update(metadata)
    missing = {"base_model_name_or_path", "hypernet_config", "ctx_encoder_args"}.difference(state)
    if missing:
        raise ValueError(f"C2L-VLA checkpoint metadata is missing {sorted(missing)}.")
    return state


def load_merged_vla_backbone(
    model_name_or_path: str,
    stage1_checkpoint: Path,
    dtype: torch.dtype,
) -> Any:
    """Raw Qwen with the stage-1 VLA adapter merged, i.e. exactly what C2L trained against."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    adapter_dir = stage1_checkpoint / VLA_ADAPTER_DIR
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Stage-1 checkpoint {stage1_checkpoint} has no {VLA_ADAPTER_DIR}/; without merging "
            "it the hypernet would modulate an un-finetuned backbone."
        )

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=True
    )
    logger.info("Merging the stage-1 VLA LoRA adapter: %s", adapter_dir)
    base_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    base_model = base_model.merge_and_unload()
    assert_adapter_merged(base_model, adapter_dir)
    return base_model


def _load_processor(checkpoint_dir: Path, model_name_or_path: str) -> Any:
    """Inference compatibility implementation; see SOURCE_NOTICE.md."""
    from transformers import AutoProcessor

    for source in (checkpoint_dir, Path(model_name_or_path)):
        if not source.exists():
            continue
        try:
            processor = AutoProcessor.from_pretrained(str(source), trust_remote_code=True)
            break
        except Exception:
            logger.info("Processor is not loadable from %s; trying next source.", source)
    else:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.init_kwargs["padding_side"] = "left"
    return processor


def _copy_vision_tower(base_model: Any) -> Any:
    visual = getattr(base_model, "visual", None)
    if visual is None:
        visual = getattr(getattr(base_model, "model", None), "visual", None)
    if visual is None:
        raise ValueError("Could not find the vision tower on the merged VLA model.")
    return deepcopy(visual)


def _patch_hidden_size_for_hypernet(model: Any) -> None:
    config = model.config
    text_config = getattr(config, "text_config", config)
    if not hasattr(config, "hidden_size") and hasattr(text_config, "hidden_size"):
        setattr(config, "hidden_size", text_config.hidden_size)


def load_c2l_vla_model(
    checkpoint_path: str,
    stage1_checkpoint_path: str,
    model_name_or_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[C2LVLAModelWrapper, Any]:
    """Load a trained C2L-VLA checkpoint. Returns `(model, processor)`."""
    from omegaconf import OmegaConf

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    stage1_dir = Path(stage1_checkpoint_path).expanduser().resolve()
    state = load_c2l_vla_state(checkpoint_dir)

    if model_name_or_path is None:
        model_name_or_path = str(state["base_model_name_or_path"])

    processor = _load_processor(checkpoint_dir, model_name_or_path)

    base_model = load_merged_vla_backbone(model_name_or_path, stage1_dir, dtype)
    _patch_hidden_size_for_hypernet(base_model)
    base_model.requires_grad_(False)
    base_model.eval()
    base_model.config.use_cache = False
    # The released metadata names the source repository, while an offline base
    # model may have been downloaded to any local directory.
    base_model.name_or_path = str(state["base_model_name_or_path"])

    # The merged model's own vision tower, not the raw checkpoint's: stage 1 trains with
    # `freeze_vision_tower=false`, so `lora_target: all` covers the tower and the merged
    encoder_model = _copy_vision_tower(base_model)
    encoder_model.requires_grad_(False)
    encoder_model.eval()

    hypernet_config = state["hypernet_config"]
    ctx_encoder_args = state["ctx_encoder_args"]
    temporal_options = resolve_temporal_state_options(state, hypernet_config)

    modulated_model = ModulatedPretrainedModel(
        base_model,
        hypernet_config,
        ctx_encoder_args,
        encoder_model=encoder_model,
        image_context_encoder=True,
        use_sequence_packing=False,
        block_lora=bool(state.get("block_lora", True)),
        frame_per_lora=int(state.get("frame_per_lora", 2)),
        normalize_add_lora=bool(state.get("normalize_add_lora", False)),
        interleave_timestamp=bool(state.get("interleave_timestamp", True)),
        interleave_text_prompt=bool(state.get("interleave_text_prompt", False)),
        # Absent from every checkpoint written before the action stream existed, which is
        # exactly the "off" the default expresses.
        interleave_action_history=bool(state.get("interleave_action_history", False)),
        micro_c2l_forward_frame=state.get("micro_c2l_forward_frame", None),
        **temporal_options,
    )
    modulated_model.load_state_dict(state)

    vla_config_path = checkpoint_dir / VLA_CONFIG_NAME
    if not vla_config_path.is_file():
        raise FileNotFoundError(f"No {VLA_CONFIG_NAME} in {checkpoint_dir}.")
    vla_config = OmegaConf.create(json.loads(vla_config_path.read_text(encoding="utf-8")))

    model = C2LVLAModelWrapper(
        modulated_model=modulated_model, processor=processor, vla_config=vla_config
    )

    trainable_path = checkpoint_dir / VLA_TRAINABLE_WEIGHTS_NAME
    if not trainable_path.is_file():
        raise FileNotFoundError(
            f"No {VLA_TRAINABLE_WEIGHTS_NAME} in {checkpoint_dir}; the action decoder would "
            "run from random initialization."
        )
    vla_state = torch.load(trainable_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(vla_state, strict=False)
    serious_unexpected = [name for name in unexpected if not name.startswith("qwen_model.")]
    if serious_unexpected:
        raise RuntimeError(f"Unexpected C2L-VLA checkpoint keys: {serious_unexpected[:10]}")

    missing_vla = [name for name in missing if not name.startswith("qwen_model.")]
    if missing_vla:
        raise RuntimeError(f"Missing action checkpoint weights: {missing_vla[:10]}")

    # Cast only the non-hypernet halves; see the module docstring.
    for name, module in model.named_children():
        if name != "qwen_model":
            module.to(dtype=dtype)
    model.to(device)
    modulated_model.hypernet.to(dtype=torch.float32)
    modulated_model.device = next(modulated_model.base_model.parameters()).device
    model.eval()

    supported, reason = modulated_model.supports_interactive_c2l()
    if not supported:
        raise ValueError(f"This C2L-VLA checkpoint cannot be evaluated interactively: {reason}")
    if int(modulated_model.frame_per_lora) != 2:
        raise ValueError(
            "C2L-VLA evaluation pairs history frames two at a time; this checkpoint has "
            f"frame_per_lora={modulated_model.frame_per_lora}."
        )
    return model, processor
