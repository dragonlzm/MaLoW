from __future__ import annotations

from math import log

import torch
from torch import Tensor, nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


def _largest_head_dim(latent_size: int) -> int:
    """Choose a Qwen3.5 linear-attention head width that exactly divides the latent size."""
    for head_dim in (128, 64, 32, 16, 8, 4, 2, 1):
        if head_dim <= latent_size and latent_size % head_dim == 0:
            return head_dim
    raise ValueError(f"Temporal Gated DeltaNet latent size must be positive, got {latent_size}.")


class _GatedDeltaNetBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = Qwen3_5GatedDeltaNet(config, layer_idx)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return hidden_states + self.mixer(
            self.input_norm(hidden_states),
            attention_mask=attention_mask,
        )


class TemporalHistoryGate(nn.Module):
    """The history residual's learned scale, without any temporal mixer."""

    def __init__(self, num_base_layers: int, history_gate_init: float = 0.1):
        super().__init__()
        if num_base_layers < 1:
            raise ValueError("Temporal history gate num_base_layers must be >= 1.")
        if not 0.0 < history_gate_init < 1.0:
            raise ValueError("Temporal history gate initialization must be strictly between 0 and 1.")
        initial_logit = log(history_gate_init / (1.0 - history_gate_init))
        self.history_gate_logits = nn.Parameter(
            torch.full((num_base_layers,), initial_logit, dtype=torch.float32)
        )


class TemporalGatedDeltaNet(nn.Module):
    """Configurable causal Gated DeltaNet blocks plus static per-base-layer history gates."""

    def __init__(
        self,
        latent_size: int,
        num_base_layers: int,
        num_hidden_layers: int = 2,
        history_gate_init: float = 0.1,
        use_history_gate: bool = True,
    ):
        super().__init__()
        if latent_size < 1:
            raise ValueError("Temporal Gated DeltaNet latent_size must be >= 1.")
        if num_base_layers < 1:
            raise ValueError("Temporal Gated DeltaNet num_base_layers must be >= 1.")
        if num_hidden_layers < 1:
            raise ValueError("Temporal Gated DeltaNet num_hidden_layers must be >= 1.")
        if not 0.0 < history_gate_init < 1.0:
            raise ValueError("Temporal history gate initialization must be strictly between 0 and 1.")

        head_dim = _largest_head_dim(latent_size)
        num_value_heads = latent_size // head_dim
        num_key_heads = max(1, num_value_heads // 2)
        while num_value_heads % num_key_heads != 0:
            num_key_heads -= 1

        config = Qwen3_5TextConfig(
            hidden_size=latent_size,
            num_hidden_layers=num_hidden_layers,
            layer_types=["linear_attention"] * num_hidden_layers,
            linear_num_value_heads=num_value_heads,
            linear_num_key_heads=num_key_heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            use_cache=False,
        )
        self.blocks = nn.ModuleList(
            [_GatedDeltaNetBlock(config, layer_idx) for layer_idx in range(num_hidden_layers)]
        )
        if use_history_gate:
            initial_logit = log(history_gate_init / (1.0 - history_gate_init))
            self.history_gate_logits = nn.Parameter(
                torch.full((num_base_layers,), initial_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("history_gate_logits", None)

    def forward(self, sequence: Tensor, attention_mask: Tensor) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence, attention_mask)
        return sequence
