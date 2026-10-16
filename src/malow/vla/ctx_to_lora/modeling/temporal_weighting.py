"""Weighting and exact-zero gating of temporal block queries.

Each temporal block carries a scalar score ``s_b`` (V-JEPA2 predictor surprise or
the pooled-cosine control). The score is standardised against the running
statistics of the blocks seen so far in the same video, so the weight of block
``b`` never depends on a future block:

    z_b = (s_b - running_mean_b) / running_std_b      (0 for the warm-up prefix)
The linear modes use ``w_b = clamp(1 + alpha * clamp(z_b, +/-c), min=0)``.
The deterministic gate instead retains the warm-up prefix and later blocks with
``z_b >= threshold``. Both paths normalize their retained weights per video.

Only scalar running moments are needed to extend the gate to a streaming
setting; pooled-cosine scoring additionally keeps one preceding feature vector.

``raw_pooled_cosine_gate`` is intentionally separate: it reproduces the V1.5.1
experiment's unstandardized adjacent-block distance threshold. Streaming needs
only the preceding pooled vector and a running retained-query sum/count.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


TEMPORAL_QUERY_WEIGHTING_MODES = (
    "vjepa_surprise",
    "pooled_cosine",
    "pooled_cosine_gate",
    "raw_pooled_cosine_gate",
)


def _causal_running_z_scores(
    scores: Tensor,
    block_owners: Tensor,
    num_samples: int,
    *,
    warmup_blocks: int,
    eps: float,
) -> Tensor:
    scores = scores.reshape(-1).float()
    owners = block_owners.reshape(-1).to(dtype=torch.long, device=scores.device)
    if scores.shape != owners.shape:
        raise ValueError(
            "Temporal query weighting needs one score per block: "
            f"scores={tuple(scores.shape)}, owners={tuple(owners.shape)}."
        )
    if owners.numel() and bool((owners[1:] < owners[:-1]).any()):
        raise ValueError("Temporal query weighting requires blocks grouped by sample in temporal order.")

    counts = torch.bincount(owners, minlength=num_samples)
    if counts.shape[0] != num_samples or bool((counts == 0).any()):
        raise ValueError("Temporal query weighting requires at least one block per sample.")

    max_blocks = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(owners.numel(), device=scores.device) - starts[owners]

    padded = scores.new_zeros((num_samples, max_blocks))
    mask = scores.new_zeros((num_samples, max_blocks))
    padded[owners, positions] = scores
    mask[owners, positions] = 1.0

    seen = torch.cumsum(mask, dim=1)
    running_sum = torch.cumsum(padded * mask, dim=1)
    running_sq = torch.cumsum(padded * padded * mask, dim=1)
    running_mean = running_sum / seen.clamp(min=1.0)
    running_var = (running_sq / seen.clamp(min=1.0)) - running_mean * running_mean
    running_std = running_var.clamp(min=0.0).sqrt()

    z = (padded - running_mean) / running_std.clamp(min=eps)
    z = torch.where((seen > warmup_blocks) & (running_std > eps), z, torch.zeros_like(z))
    return z[owners, positions]


class TemporalQueryWeighting(nn.Module):
    """Turn per-block scores into normalised per-block query weights."""

    def __init__(
        self,
        alpha_init: float = 0.0,
        clamp: float = 3.0,
        warmup_blocks: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if clamp <= 0:
            raise ValueError("Temporal query weight clamp must be positive.")
        if warmup_blocks < 1:
            raise ValueError("Temporal query weight warmup_blocks must be >= 1.")
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.clamp = float(clamp)
        self.warmup_blocks = int(warmup_blocks)
        self.eps = float(eps)

    def running_z_scores(self, scores: Tensor, block_owners: Tensor, num_samples: int) -> Tensor:
        """Causal z-scores: block b is standardised against blocks 0..b of its own video."""
        return _causal_running_z_scores(
            scores,
            block_owners,
            num_samples,
            warmup_blocks=self.warmup_blocks,
            eps=self.eps,
        )

    def forward(self, scores: Tensor, block_owners: Tensor, num_samples: int) -> Tensor:
        """Return per-block weights that sum to 1 within each sample."""
        z = self.running_z_scores(scores, block_owners, num_samples)
        owners = block_owners.reshape(-1).to(dtype=torch.long, device=z.device)
        alpha = self.alpha.to(device=z.device, dtype=z.dtype)
        weights = (1.0 + alpha * z.clamp(-self.clamp, self.clamp)).clamp(min=0.0)

        totals = weights.new_zeros(num_samples).index_add_(0, owners, weights)
        return weights / totals.clamp(min=self.eps)[owners]


class TemporalQueryGate(nn.Module):
    """Deterministic causal gate with exact-zero, uniformly weighted dropped blocks."""

    def __init__(
        self,
        threshold: float = 0.5,
        warmup_blocks: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("Temporal query gate threshold must be finite.")
        if warmup_blocks < 1:
            raise ValueError("Temporal query gate warmup_blocks must be >= 1.")
        self.threshold = float(threshold)
        self.warmup_blocks = int(warmup_blocks)
        self.eps = float(eps)

    def forward(self, scores: Tensor, block_owners: Tensor, num_samples: int) -> Tensor:
        """Return exact-zero gates normalized to a uniform mean over retained blocks."""
        z = _causal_running_z_scores(
            scores,
            block_owners,
            num_samples,
            warmup_blocks=self.warmup_blocks,
            eps=self.eps,
        )
        owners = block_owners.reshape(-1).to(dtype=torch.long, device=z.device)

        counts = torch.bincount(owners, minlength=num_samples)
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(owners.numel(), device=z.device) - starts[owners]
        retained = (positions < self.warmup_blocks) | (z >= self.threshold)
        gates = retained.to(dtype=z.dtype)
        totals = gates.new_zeros(num_samples).index_add_(0, owners, gates)
        return gates / totals.clamp(min=self.eps)[owners]

    def streaming_weights(
        self,
        scores: Tensor,
        *,
        prior_count: int,
        prior_sum: float,
        prior_squared_sum: float,
    ) -> tuple[Tensor, int, float, float]:
        """Gate one continuation segment and return unnormalized 0/1 weights plus new statistics."""
        scores = scores.reshape(-1).float()
        if scores.numel() == 0:
            raise ValueError("Temporal query gate streaming update requires at least one score.")

        seen = torch.arange(
            prior_count + 1,
            prior_count + scores.numel() + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        running_sum = scores.cumsum(dim=0) + float(prior_sum)
        running_squared_sum = scores.square().cumsum(dim=0) + float(prior_squared_sum)
        running_mean = running_sum / seen
        running_var = (running_squared_sum / seen) - running_mean.square()
        running_std = running_var.clamp(min=0.0).sqrt()
        z = (scores - running_mean) / running_std.clamp(min=self.eps)
        z = torch.where(
            (seen > self.warmup_blocks) & (running_std > self.eps),
            z,
            torch.zeros_like(z),
        )
        retained = (seen <= self.warmup_blocks) | (z >= self.threshold)
        weights = retained.to(dtype=scores.dtype)
        return (
            weights,
            prior_count + int(scores.numel()),
            float(running_sum[-1].item()),
            float(running_squared_sum[-1].item()),
        )


class RawPooledCosineGate(nn.Module):
    """Uniformly average blocks retained by a raw adjacent-cosine threshold."""

    def __init__(self, threshold: float = 0.05, eps: float = 1e-6):
        super().__init__()
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("Raw pooled-cosine gate threshold must be finite.")
        if not 0.0 <= float(threshold) <= 2.0:
            raise ValueError("Raw pooled-cosine gate threshold must be between 0 and 2.")
        self.threshold = float(threshold)
        self.eps = float(eps)

    def forward(self, scores: Tensor, block_owners: Tensor, num_samples: int) -> Tensor:
        scores = scores.reshape(-1).float()
        owners = block_owners.reshape(-1).to(dtype=torch.long, device=scores.device)
        if scores.shape != owners.shape:
            raise ValueError(
                "Raw pooled-cosine gating needs one score per block: "
                f"scores={tuple(scores.shape)}, owners={tuple(owners.shape)}."
            )
        if owners.numel() and bool((owners[1:] < owners[:-1]).any()):
            raise ValueError("Raw pooled-cosine gating requires blocks grouped by sample in temporal order.")

        counts = torch.bincount(owners, minlength=num_samples)
        if counts.shape[0] != num_samples or bool((counts == 0).any()):
            raise ValueError("Raw pooled-cosine gating requires at least one block per sample.")
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(owners.numel(), device=scores.device) - starts[owners]
        retained = (positions == 0) | (scores >= self.threshold)
        gates = retained.to(dtype=scores.dtype)
        totals = gates.new_zeros(num_samples).index_add_(0, owners, gates)
        if bool((totals == 0).any()):
            raise RuntimeError("Raw pooled-cosine gating dropped every block for a sample.")
        return gates / totals.clamp(min=self.eps)[owners]

    def streaming_weights(self, scores: Tensor, *, prior_count: int) -> Tensor:
        """Unnormalized 0/1 gates for a continuation of one video's block stream."""
        scores = scores.reshape(-1).float()
        if scores.numel() == 0:
            raise ValueError("Raw pooled-cosine streaming update requires at least one score.")
        retained = scores >= self.threshold
        # Segment boundaries do not start a new video. Only the session's first
        # block is unconditionally retained, even at a threshold above one.
        if prior_count == 0:
            retained[0] = True
        return retained.to(dtype=scores.dtype)


def pooled_cosine_scores(
    pooled_blocks: Tensor,
    block_owners: Tensor,
) -> Tensor:
    """`1 - cos(pooled_b, pooled_{b-1})` within each sample; 0 for each sample's first block."""
    pooled = pooled_blocks.float()
    owners = block_owners.reshape(-1).to(dtype=torch.long, device=pooled.device)
    if pooled.dim() != 2 or pooled.shape[0] != owners.shape[0]:
        raise ValueError(
            "Pooled cosine scoring expects one pooled vector per block: "
            f"pooled={tuple(pooled.shape)}, owners={tuple(owners.shape)}."
        )

    scores = pooled.new_zeros(pooled.shape[0])
    if pooled.shape[0] < 2:
        return scores
    similarity = torch.nn.functional.cosine_similarity(pooled[1:], pooled[:-1], dim=-1)
    same_sample = owners[1:] == owners[:-1]
    scores[1:] = torch.where(same_sample, 1.0 - similarity, torch.zeros_like(similarity))
    return scores


def raw_pooled_cosine_distances(
    pooled_blocks: Tensor,
    block_owners: Tensor,
) -> Tensor:
    """Raw adjacent-block ``1 - cosine_similarity`` with each video's first score set to one."""
    pooled = pooled_blocks.float()
    owners = block_owners.reshape(-1).to(dtype=torch.long, device=pooled.device)
    if pooled.dim() != 2 or pooled.shape[0] != owners.shape[0]:
        raise ValueError(
            "Raw pooled-cosine scoring expects one pooled vector per block: "
            f"pooled={tuple(pooled.shape)}, owners={tuple(owners.shape)}."
        )
    if owners.numel() and bool((owners[1:] < owners[:-1]).any()):
        raise ValueError("Raw pooled-cosine scoring requires blocks grouped by sample in temporal order.")

    scores = pooled.new_ones(pooled.shape[0])
    if pooled.shape[0] < 2:
        return scores
    similarity = torch.nn.functional.cosine_similarity(pooled[1:], pooled[:-1], dim=-1)
    same_sample = owners[1:] == owners[:-1]
    distances = (1.0 - similarity).clamp(min=0.0, max=2.0)
    scores[1:] = torch.where(same_sample, distances, torch.ones_like(distances))
    return scores
