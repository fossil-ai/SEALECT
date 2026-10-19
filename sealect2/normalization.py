"""
normalization.py

Score normalization strategies applied to each component before fusion.

- identity        : raw scores (A0raw diagnostic; the literal "no normalization")
- perquery_z      : per-query z-score over the full candidate pool.
                    THIS is what shipped SEALECT uses (legacy zscore_torch), so
                    A0 (baseline) uses it and reproduces the paper exactly.
- global_z        : z-score using (mu, sigma) estimated from TRAINING data only
                    (A1). Stats are passed in; see estimate_global_stats().

All operate row-wise on a [B, Ncand] tensor (B queries in a batch).
"""

from __future__ import annotations

import torch

EPS = 1e-6  # matches legacy zscore_torch


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def perquery_z(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + EPS)


def global_z(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return (x - mu) / (sigma + EPS)


NORMALIZERS = {
    "identity": "identity",
    "perquery_z": "perquery_z",
    "global_z": "global_z",
}


def estimate_global_stats(component_rows: torch.Tensor) -> tuple[float, float]:
    """Estimate (mu, sigma) over a pool of component values (train only).

    component_rows: [Nq_train, Ncand] tensor of a single component's raw scores
    computed on TRAINING queries. Returns scalar mean/std over all finite values.
    """
    flat = component_rows.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    return float(flat.mean().item()), float(flat.std(unbiased=False).item())
