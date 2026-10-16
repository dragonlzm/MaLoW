# Interactive inference adapted from Hippocampus master f4b15d9. See NOTICE.
"""Bounded per-video memory for released Qwen Video C2L checkpoints."""
from contextlib import nullcontext
from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping
import torch
from torch import Tensor
import torch.nn.functional as F

@dataclass
class C2LInteractiveState:
    deltanet_cache: Any = None
    query_sum: Tensor | None = None
    weight_sum: float = 0.0
    block_count: int = 0
    last_temporal_output: Tensor | None = None
    previous_pooled_block: Tensor | None = None
    segment_count: int = 0

    def has_context(self):
        return self.block_count > 0


class InteractiveC2LMixin:
    def supports_interactive_c2l(self):
        if not self.image_context_encoder or not self.block_lora or not self.temporal_sum_at_query:
            return False, "requires a video block-LoRA checkpoint with temporal_sum_at_query=True"
        if self.temporal_aggregator not in ("gated_residual", "only"):
            return False, "requires temporal_aggregator gated_residual or only"
        if self.temporal_query_weighting not in (None, "raw_pooled_cosine_gate"):
            return False, "supported history weighting: none or raw_pooled_cosine_gate"
        if self.frame_per_lora != 2:
            return False, "interactive evaluation requires frame_per_lora=2"
        return True, ""

    def _interactive_block_embeddings(self, inputs: Mapping[str, Tensor]):
        kwargs = {k: v for k, v in inputs.items() if k != "labels"}
        mask = kwargs.get("attention_mask")
        features = self.ctx_encoder(**kwargs)
        if isinstance(features, tuple):
            features, returned_mask = features
            if returned_mask is not None:
                mask = returned_mask
        if features.shape[0] != 1:
            raise ValueError("Interactive C2L requires one video per session update.")
        blocks, masks, owners, pooled = self._build_video_block_features(
            features, mask, kwargs.get("video_grid_thw"),
            kwargs.get("video_temporal_patch_size"), kwargs.get("video_metadata"),
            kwargs.get("timestamp_input_ids"), kwargs.get("timestamp_attention_mask"),
            kwargs.get("timestamp_block_owners"),
        )
        if not len(blocks) or bool((owners != 0).any()):
            raise ValueError("Interactive segment must contain blocks from exactly one video.")
        step = self._micro_c2l_forward_blocks() or len(blocks)
        embeddings = [self.hypernet.aggregate_lora_embeddings(blocks[i:i+step], masks[i:i+step], None)
                      for i in range(0, len(blocks), step)]
        return torch.cat(embeddings, dim=0), pooled

    @torch.inference_mode()
    def interactive_update(self, ctx_encoder_inputs, state):
        if state.deltanet_cache is None:
            raise ValueError("Create interactive state with new_interactive_state().")
        embeddings, pooled = self._interactive_block_embeddings(ctx_encoder_inputs)
        count = len(embeddings)
        weights = embeddings.new_ones(count, dtype=torch.float32)
        if self.temporal_query_weighting == "raw_pooled_cosine_gate":
            from .temporal_weighting import raw_pooled_cosine_distances
            owners = torch.zeros(count, dtype=torch.long, device=pooled.device)
            scores = raw_pooled_cosine_distances(pooled, owners)
            if state.previous_pooled_block is not None:
                scores[0] = (1 - F.cosine_similarity(
                    pooled[:1].float(), state.previous_pooled_block.to(pooled.device).float().unsqueeze(0),
                    dim=-1)[0]).clamp(0, 2)
            weights = (scores >= self.hypernet.temporal_query_weighting.threshold).float()
            if state.block_count == 0:
                weights[0] = 1
            state.previous_pooled_block = pooled[-1].detach().clone()
        segment_sum = (embeddings.float() * weights.reshape(-1, 1, 1, 1, 1)).sum(0, keepdim=True)
        state.query_sum = segment_sum if state.query_sum is None else state.query_sum + segment_sum
        state.weight_sum += float(weights.sum())
        state.block_count += count
        layers, modules, ranks = self._unique_query_slices()
        state.last_temporal_output = self._run_temporal_deltanet(
            embeddings[:, layers, modules, ranks, :],
            torch.zeros(count, dtype=torch.long, device=embeddings.device), 1,
            torch.tensor([count], device=embeddings.device), deltanet_cache=state.deltanet_cache,
        )
        state.segment_count += 1
        return state

    def _unique_query_slices(self) -> tuple[slice, slice, slice]:
        """Query axes the encoder actually varies; the rest are broadcast copies."""
        aggregator_config = self.hypernet.config.aggregator_config
        return (
            slice(None) if getattr(aggregator_config, "query_per_layer", False) else slice(0, 1),
            slice(None) if getattr(aggregator_config, "query_per_module", True) else slice(0, 1),
            slice(None) if getattr(aggregator_config, "query_per_rank", True) else slice(0, 1),
        )

    def _run_temporal_deltanet(
        self,
        unique_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
        block_counts: Tensor,
        deltanet_cache: Any = None,
    ) -> Tensor:
        """Run the causal Gated DeltaNet over per-sample block streams and keep the last output.

        Returns ``[num_samples, unique_layers, unique_modules, unique_rank, latent]``. When
        ``deltanet_cache`` is given the linear-attention state is read from and written back
        to it, so a streaming caller can continue an earlier prefix; the returned last output
        is then the last output of the *whole* prefix seen so far.
        """
        latent_size = unique_embeddings.shape[-1]
        unique_layers, unique_modules, unique_rank = unique_embeddings.shape[1:4]

        max_blocks = int(block_counts.max().item())
        padded = unique_embeddings.new_zeros(
            (num_samples, max_blocks, unique_layers, unique_modules, unique_rank, latent_size)
        )
        temporal_mask = torch.zeros(
            (num_samples, max_blocks),
            device=unique_embeddings.device,
            dtype=torch.long,
        )
        for sample_idx in range(num_samples):
            sample_embeddings = unique_embeddings[block_owners == sample_idx]
            sample_blocks = sample_embeddings.shape[0]
            padded[sample_idx, :sample_blocks] = sample_embeddings
            temporal_mask[sample_idx, :sample_blocks] = 1

        sequence = padded.permute(0, 2, 3, 4, 1, 5).reshape(
            num_samples * unique_layers * unique_modules * unique_rank,
            max_blocks,
            latent_size,
        )
        stream_mask = temporal_mask[:, None, None, None, :].expand(
            num_samples, unique_layers, unique_modules, unique_rank, max_blocks
        ).reshape(-1, max_blocks)

        temporal_module = self.hypernet.temporal_aggregator
        first_parameter = next(temporal_module.parameters(), None)
        if (
            not sequence.is_cuda
            and first_parameter is not None
            and sequence.is_floating_point()
            and sequence.dtype != first_parameter.dtype
        ):
            sequence = sequence.to(first_parameter.dtype)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if sequence.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            # Offline and training keep the original two-argument call, so any temporal
            # module that predates the streaming cache still works.
            temporal_outputs = (
                temporal_module(sequence, stream_mask)
                if deltanet_cache is None
                else temporal_module(sequence, stream_mask, cache_params=deltanet_cache)
            )
        temporal_outputs = temporal_outputs.reshape(
            num_samples,
            unique_layers,
            unique_modules,
            unique_rank,
            max_blocks,
            latent_size,
        )
        return torch.stack(
            [
                temporal_outputs[sample_idx, :, :, :, int(block_counts[sample_idx].item()) - 1]
                for sample_idx in range(num_samples)
            ],
            dim=0,
        )

    def _combine_temporal_and_history(
        self,
        last_outputs: Tensor,
        history_embeddings: Tensor,
        target_shape: tuple[int, int, int, int, int],
    ) -> Tensor:
        """`normalize(deltanet last token) + gate * normalize(history mean)` per base layer.

        ``history_embeddings`` is the per-sample mean (or weighted mean) of the block queries;
        only its direction matters because it is re-normalized here.
        """
        num_samples, num_layers, num_modules, rank, latent_size = target_shape
        last_outputs = last_outputs.expand(num_samples, num_layers, num_modules, rank, latent_size)

        if self.temporal_aggregator == "only":
            return last_outputs
        if self.temporal_aggregator != "gated_residual":
            raise ValueError(f"Unsupported temporal aggregator mode: {self.temporal_aggregator!r}.")

        normalized_temporal = F.normalize(last_outputs.float(), dim=-1, eps=1e-6) * sqrt(
            self.hypernet.config.latent_size
        )
        normalized_history = F.normalize(history_embeddings.float(), dim=-1, eps=1e-6) * sqrt(
            self.hypernet.config.latent_size
        )
        history_gates = self.hypernet.temporal_aggregator.history_gate_logits.sigmoid().to(
            device=normalized_temporal.device,
            dtype=normalized_temporal.dtype,
        )
        if history_gates.numel() != num_layers:
            raise ValueError(
                "Temporal history gate count must match the number of generated base-model layers."
            )
        return normalized_temporal + history_gates.view(1, num_layers, 1, 1, 1) * normalized_history

    def new_interactive_state(self) -> C2LInteractiveState:
        """Open an interactive C2L session for a single video."""
        supported, reason = self.supports_interactive_c2l()
        if not supported:
            raise ValueError(f"This C2L checkpoint cannot run interactively: {reason}")
        return C2LInteractiveState(deltanet_cache=self.hypernet.temporal_aggregator.new_cache())

    @torch.inference_mode()
    def interactive_generate_weights(self, state: C2LInteractiveState):
        """LoRA factors for everything the session has consumed so far."""
        if not state.has_context() or state.query_sum is None or state.last_temporal_output is None:
            raise ValueError(
                "Interactive C2L has no context yet; feed at least one segment with "
                "interactive_update() before generating weights."
            )

        _, num_layers, num_modules, rank, latent_size = state.query_sum.shape
        if state.weight_sum <= 0:
            raise ValueError("Interactive C2L temporal history has zero retained weight.")
        history_embeddings = state.query_sum / float(state.weight_sum)
        temporal_embeddings = self._combine_temporal_and_history(
            state.last_temporal_output,
            history_embeddings,
            (1, num_layers, num_modules, rank, latent_size),
        )
        lora_dict, generated_layernorms = self.hypernet.generate_weights_from_lora_embeddings(
            temporal_embeddings
        )
        if generated_layernorms is not None:
            raise NotImplementedError("Interactive C2L does not support generated layernorm modules.")

        if self.user_defined_scaling != 1:
            for module in lora_dict:
                lora_dict[module]["A"] = lora_dict[module]["A"] * self.user_defined_scaling
                lora_dict[module]["B"] = lora_dict[module]["B"] * self.user_defined_scaling
        return lora_dict, None
