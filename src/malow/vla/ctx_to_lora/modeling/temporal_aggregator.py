from __future__ import annotations

import logging
from functools import lru_cache
from math import log

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

logger = logging.getLogger(__name__)


def _largest_head_dim(latent_size: int) -> int:
    """Choose a Qwen3.5 linear-attention head width that exactly divides the latent size."""
    for head_dim in (128, 64, 32, 16, 8, 4, 2, 1):
        if head_dim <= latent_size and latent_size % head_dim == 0:
            return head_dim
    raise ValueError(f"Temporal Gated DeltaNet latent size must be positive, got {latent_size}.")


@lru_cache(maxsize=8)
def multitoken_continuation_supported(device_type: str, dtype: torch.dtype) -> bool:
    """Does this build continue a cached prefix through a *multi-token* forward?

    Transformers <= 5.6.0 gates the linear-attention cache on ``seq_len == 1``::

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(...) and seq_len == 1
        )

    so feeding a warm cache a segment of two or more tokens silently discards both the
    recurrent and the conv state -- the result is bit-identical to running that segment with
    no history at all. Later builds added a continuation branch that prepends the cached conv
    context and passes ``initial_state`` into the chunked kernel.

    This probes the behaviour rather than the version string, because the fused
    FLA / causal-conv1d kernels used on CUDA are a different code path from the torch
    fallbacks. It is measured once per (device, dtype) and is deliberately conservative: any
    failure to demonstrate continuation returns False, which selects the slower but always
    correct step-by-step path in :meth:`TemporalGatedDeltaNet.forward`.
    """
    try:
        device = torch.device(device_type)
        probe = TemporalGatedDeltaNet(32, 1, num_hidden_layers=1, use_history_gate=False)
        probe = probe.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            for parameter in probe.parameters():
                parameter.normal_(0, 0.3)
            for block in probe.blocks:
                # g = -exp(A_log) * softplus(a + dt_bias); a tiny A drives g to 0 so the
                # recurrent state is never forgotten and a dropped carry cannot hide.
                block.mixer.A_log.fill_(log(1e-4))
                block.mixer.dt_bias.fill_(-4.0)
                block.mixer.conv1d.weight.normal_(0, 0.5)

            sequence = torch.randn(2, 6, 32, device=device, dtype=dtype)
            mask = torch.ones(2, 6, dtype=torch.long, device=device)
            whole = probe._forward_blocks(sequence, mask, None)

            cache = probe.new_cache()
            probe._forward_blocks(sequence[:, :3], mask[:, :3], cache)
            warm = probe._forward_blocks(sequence[:, 3:], mask[:, 3:], cache)
            cold = probe._forward_blocks(sequence[:, 3:], mask[:, 3:], None)

        target = whole[:, 3:].float()
        scale = target.abs().amax().clamp(min=1e-6)
        warm_error = float((warm.float() - target).abs().amax() / scale)
        cold_error = float((cold.float() - target).abs().amax() / scale)
        # Require both that the continuation is right and that the probe could have seen it
        # be wrong; otherwise "supported" would be an untested claim.
        return warm_error < 1e-2 and cold_error > 1e-1
    except Exception as error:  # noqa: BLE001 - never let a probe break the model
        logger.warning(
            "Temporal Gated DeltaNet continuation probe failed (%s); falling back to "
            "step-by-step streaming.",
            error,
        )
        return False


class _GatedDeltaNetBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = Qwen3_5GatedDeltaNet(config, layer_idx)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        cache_params: Cache | None = None,
    ) -> Tensor:
        return hidden_states + self.mixer(
            self.input_norm(hidden_states),
            cache_params=cache_params,
            attention_mask=attention_mask,
        )


class TemporalGatedDeltaNet(nn.Module):
    """Inference compatibility implementation; see SOURCE_NOTICE.md."""

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
        # Kept so a streaming caller can build a matching linear-attention cache. It is a
        # plain config object, not a submodule, so checkpoints are unaffected.
        self.deltanet_config = config
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

    def new_cache(self) -> Cache:
        """Fresh linear-attention state for one interactive stream."""
        return DynamicCache(config=self.deltanet_config)

    def _forward_blocks(
        self,
        sequence: Tensor,
        attention_mask: Tensor,
        cache_params: Cache | None,
    ) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence, attention_mask, cache_params=cache_params)
        return sequence

    def _forward_one_block_at_a_time(
        self,
        sequence: Tensor,
        attention_mask: Tensor,
        cache_params: Cache,
    ) -> Tensor:
        """Advance a warm cache one token at a time.

        Every layer here is causal and stateful, so feeding the whole stack token ``t`` and
        then token ``t+1`` is equivalent to feeding it both at once -- and the single-token
        path is the one every Transformers build consumes cached state on. This is the
        fallback for builds whose multi-token forward ignores the cache.
        """
        return torch.cat(
            [
                self._forward_blocks(
                    sequence[:, index : index + 1],
                    attention_mask[:, index : index + 1],
                    cache_params,
                )
                for index in range(sequence.shape[1])
            ],
            dim=1,
        )

    def forward(
        self,
        sequence: Tensor,
        attention_mask: Tensor,
        cache_params: Cache | None = None,
    ) -> Tensor:
        if cache_params is None:
            return self._forward_blocks(sequence, attention_mask, None)

        # A cold cache is safe to fill with a single multi-token call on every build: there is
        # no prior state to consume, and the final state is stored correctly either way. Only
        # *continuing* a warm cache needs the capability check.
        warm = bool(cache_params.has_previous_state(0)) if len(cache_params.layers) else False
        if not warm or sequence.shape[1] == 1:
            return self._forward_blocks(sequence, attention_mask, cache_params)

        first_parameter = next(self.parameters(), None)
        dtype = first_parameter.dtype if first_parameter is not None else sequence.dtype
        if multitoken_continuation_supported(sequence.device.type, dtype):
            return self._forward_blocks(sequence, attention_mask, cache_params)

        logger.info(
            "This Transformers build ignores cached linear-attention state on multi-token "
            "forwards; streaming temporal blocks one at a time instead.",
        )
        return self._forward_one_block_at_a_time(sequence, attention_mask, cache_params)
