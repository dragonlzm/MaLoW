# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
"""Raw pooled-cosine weighting for temporal C2L query histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


TEMPORAL_QUERY_WEIGHTING_MODES = ("raw_pooled_cosine", "raw_pooled_cosine_gate")


def raw_pooled_cosine_distances(
    pooled_blocks: Tensor,
    block_owners: Tensor,
) -> Tensor:
    """Return adjacent-block cosine distance, with score 1 for each video's first block."""
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

    similarities = torch.nn.functional.cosine_similarity(pooled[1:], pooled[:-1], dim=-1)
    same_sample = owners[1:] == owners[:-1]
    distances = (1.0 - similarities).clamp(min=0.0, max=2.0)
    scores[1:] = torch.where(same_sample, distances, torch.ones_like(distances))
    return scores


def _normalize_per_sample(weights: Tensor, block_owners: Tensor, num_samples: int, eps: float) -> Tensor:
    owners = block_owners.reshape(-1).to(dtype=torch.long, device=weights.device)
    if weights.shape != owners.shape:
        raise ValueError(
            "Temporal query weighting needs one weight per block: "
            f"weights={tuple(weights.shape)}, owners={tuple(owners.shape)}."
        )
    counts = torch.bincount(owners, minlength=num_samples)
    if counts.shape[0] != num_samples or bool((counts == 0).any()):
        raise ValueError("Temporal query weighting requires at least one block per sample.")
    totals = weights.new_zeros(num_samples).index_add_(0, owners, weights)
    return weights / totals.clamp(min=eps)[owners]


class RawPooledCosineWeighting(nn.Module):
    """Normalize raw adjacent-block cosine distances into per-video query weights."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, scores: Tensor, block_owners: Tensor, num_samples: int) -> Tensor:
        scores = scores.reshape(-1).float().clamp(min=0.0, max=2.0)
        return _normalize_per_sample(scores, block_owners, num_samples, self.eps)


class RawPooledCosineGate(nn.Module):
    """Uniformly average the first block and later blocks meeting a raw-distance threshold."""

    def __init__(self, threshold: float = 0.05, eps: float = 1e-6):
        super().__init__()
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("Raw pooled-cosine gate threshold must be finite.")
        if not 0.0 <= threshold <= 2.0:
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

        counts = torch.bincount(owners, minlength=num_samples)
        if counts.shape[0] != num_samples or bool((counts == 0).any()):
            raise ValueError("Raw pooled-cosine gating requires at least one block per sample.")
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(owners.numel(), device=scores.device) - starts[owners]
        retained = (positions == 0) | (scores >= self.threshold)
        return _normalize_per_sample(retained.to(dtype=scores.dtype), owners, num_samples, self.eps)
