"""Load the released MaLoW video memory module without pickle or trainers."""
from pathlib import Path
from typing import Any

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


def remove_empty_think_block(texts: str | list[str]) -> str | list[str]:
    if isinstance(texts, str):
        return texts.replace(EMPTY_THINK_BLOCK, "")
    return [text.replace(EMPTY_THINK_BLOCK, "") for text in texts]


def load_c2l_state(checkpoint: str) -> dict[str, Any]:
    from malow.checkpoints import load_checkpoint_metadata
    from safetensors.torch import load_file
    path = Path(checkpoint)
    if not path.is_dir():
        raise ValueError("--checkpoint must be a released checkpoint directory.")
    state = load_file(str(path / "model.safetensors"), device="cpu")
    metadata = load_checkpoint_metadata(path, runtime="video")
    required = {"base_model_name_or_path", "hypernet_config", "ctx_encoder_args"}
    if missing := required.difference(metadata):
        raise ValueError(f"Missing inference metadata: {sorted(missing)}")
    if overlap := state.keys() & metadata.keys():
        raise ValueError(f"Tensor and metadata keys overlap: {sorted(overlap)}")
    state.update(metadata)
    return state


def resolve_block_lora_options(state, args):
    requested = getattr(args, "block_lora", "auto")
    enabled = bool(state.get("block_lora", False)) if requested == "auto" else requested == "true"
    frames = getattr(args, "frame_per_lora", None)
    frames = int(state.get("frame_per_lora", 4) if frames is None else frames)
    if frames < 1:
        raise ValueError("--frame-per-lora must be positive")
    state.update(block_lora=enabled, frame_per_lora=frames)
    return enabled, frames


def resolve_micro_c2l_forward_frame_option(state, args):
    cap = getattr(args, "micro_c2l_forward_frame", None)
    if cap is None:
        cap = state.get("micro_c2l_forward_frame")
    if cap is not None and int(cap) < 1:
        raise ValueError("--micro-c2l-forward-frame must be positive")
    state["micro_c2l_forward_frame"] = None if cap is None else int(cap)
    return state["micro_c2l_forward_frame"]


def load_c2l_model(args, local_rank: int, world_size: int):
    import torch
    from transformers import AutoModelForImageTextToText
    from malow.video.ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    from malow.video.vision_encoder import load_c2l_vision_encoder

    state = load_c2l_state(args.c2l_checkpoint)
    hypernet_config, encoder_args = state["hypernet_config"], state["ctx_encoder_args"]
    encoder_type = getattr(encoder_args.ctx_encoder_type, "value", encoder_args.ctx_encoder_type)
    if encoder_type != "vision_embedder":
        raise ValueError("This release supports the Qwen3.5 and Qwen3-VL vision_embedder checkpoints only.")
    block_lora, frame_per_lora = resolve_block_lora_options(state, args)
    micro_cap = resolve_micro_c2l_forward_frame_option(state, args)
    device_map = args.device_map
    if world_size > 1 and device_map == "auto":
        device_map = {"": f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"}
    base = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=args.dtype, device_map=device_map)
    text_config = getattr(base.config, "text_config", base.config)
    if not hasattr(base.config, "hidden_size"):
        base.config.hidden_size = text_config.hidden_size
    base.requires_grad_(False)
    base.eval()
    # Metadata identifies the original model; weights may be downloaded to any path.
    base.name_or_path = state["base_model_name_or_path"]
    encoder = load_c2l_vision_encoder(args.model_id, base.config, dtype=args.dtype, device_map=device_map).model
    model = ModulatedPretrainedModel(
        base, hypernet_config, encoder_args, encoder_model=encoder,
        image_context_encoder=True, use_sequence_packing=False,
        user_defined_scaling=args.user_defined_scaling,
        block_lora=block_lora, frame_per_lora=frame_per_lora,
        micro_c2l_forward_frame=micro_cap,
        temporal_sum_at_query=state.get("temporal_sum_at_query", False),
        normalize_add_query=state.get("normalize_add_query", False),
        temporal_aggregator=state.get("temporal_aggregator", getattr(hypernet_config, "temporal_aggregator", None)),
        temporal_query_weighting=state.get("temporal_query_weighting", getattr(hypernet_config, "temporal_query_weighting", None)),
        temporal_query_gate_threshold=state.get("temporal_query_gate_threshold", getattr(hypernet_config, "temporal_query_gate_threshold", 0.05)),
    )
    model.load_state_dict(state)
    model.requires_grad_(False)
    model.eval()
    return model
