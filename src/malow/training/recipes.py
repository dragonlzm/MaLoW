"""Resolved model-verified recipes from the pinned Hippocampus launchers."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

SOURCE_COMMIT = "e257adeb497dc02d07d112cbb6372c29017e5a40"
MIXTURE = "0.574360,0.181881,0.028005,0.028005,0.034587,0.024706,0.024716,0.024716,0.017455,0.0012893333,0.0006446667,0.014542,0.014542,0.015276,0.015275"
PRESETS = {
    "video-9b": ("9B", 2.0, 2),
    "video-4b": ("4B", 2.0, 2),
    "video-4b-last4s": ("4B", 4.0, 4),
    "video-qwen3vl-4b-last8s": ("qwen3-vl-4b", 8.0, 8),
    "video-2b": ("2B", 2.0, 2),
    "video-0.8b": ("0.8B", 2.0, 2),
}

# YAML defaults followed by the source launcher's explicit overrides. video_fps
# remains 2.0 for Qwen3.5: the base-tail sampler independently uses 1 FPS.
COMMON = {
    "image_max_pixels": 262144, "video_max_pixels": 262144,
    "video_min_pixels": 1024, "video_fps": 2.0, "video_maxlen": 32,
    "trust_remote_code": True, "stage": "c2l", "do_train": True,
    "finetuning_type": "lora", "lora_rank": 8,
    "lora_target": "down_proj,up_proj,gate_proj,self_attn.v_proj,linear_attn.in_proj_qkv",
    "c2l_gen_lora_l1_reg_coef": 0.01, "c2l_per_rank_gen": True,
    "c2l_ctx_encoder_type": "vision_embedder", "c2l_aggregator_type": "encoder_aggregator",
    "c2l_encoder_aggregator_num_layers": 4, "c2l_quantize_ctx_encoder": False,
    "c2l_strict_target_alignment": True, "c2l_loss_type": "sft_expected_or_generated",
    "template": "qwen3_5_nothink", "streaming": True, "cutoff_len": 2048,
    "dataloader_num_workers": 4, "dataloader_persistent_workers": True,
    "logging_steps": 10, "save_steps": 2800, "plot_loss": True,
    "overwrite_output_dir": False, "save_only_model": False, "report_to": "none",
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    "learning_rate": 8e-5, "max_steps": 2800, "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1, "bf16": True, "fp16": False, "ddp_timeout": 180000000,
    "resume_from_checkpoint": None, "mix_strategy": "interleave_over",
    "interleave_probs": MIXTURE, "c2l_use_valid_slots": True,
    "c2l_hypernet_latent_size": 512, "c2l_encoder_aggregator_d_model": 512,
    "c2l_encoder_aggregator_query_per_layer": True,
    "c2l_encoder_aggregator_query_per_module": False,
    "c2l_encoder_aggregator_query_per_rank": False,
    "c2l_hypernet_head_per_layer": False, "c2l_hypernet_head_per_module": True,
    "c2l_hypernet_head_per_rank": True, "c2l_streaming_repeat": True,
    "block_lora": True, "frame_per_lora": 2, "normalize_add_lora": False,
    "temporal_sum_at_query": True, "normalize_add_query": True,
    "temporal_aggregator": "gated_residual", "temporal_deltanet_num_layers": 4,
    "temporal_query_weighting": "raw_pooled_cosine_gate",
    "temporal_query_gate_threshold": 0.05, "interleave_timestamp": True,
    "base_input_video": True, "base_with_last_2seconds": True,
    "base_last_2seconds_max_frames": 256, "video_merge_extend": False,
    "video_merge_extend_perc": 0.0, "weight_by_sample": True,
    "c2l_vision_encoder_only": True, "c2l_vjepa2_model_name_or_path": "",
    "c2l_vjepa2_window_size": 16, "c2l_vjepa2_window_stride": 2,
    "c2l_vjepa2_spatial_pool_size": 2, "c2l_vjepa2_micro_batch_size": 32,
}


def resolve_recipe(model: str) -> dict:
    """Return an independent config with the same pinned base as evaluation."""
    if model not in PRESETS:
        raise ValueError(f"Unknown training model: {model}")
    base, seconds, frames = PRESETS[model]
    registry = json.loads((Path(__file__).parents[1] / "release_models.json").read_text())
    base_info = registry["bases"][base]
    config = deepcopy(COMMON)
    config.update(model_name_or_path=base_info["repo_id"], model_revision=base_info["revision"],
                  base_tail_seconds=seconds, base_input_frame_number=frames, run_name=f"malow-{model}")
    if base == "qwen3-vl-4b":
        config.update(template="qwen3_vl_nothink", video_fps=1.0,
                      lora_target="down_proj,up_proj,gate_proj,self_attn.v_proj")
    return config
