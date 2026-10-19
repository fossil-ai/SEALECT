"""
features.py

Candidate-independent query reliability features r_q (for A3/A4 gating) and
candidate geometry pi_j (for A4). All derived from quantities SEALECT already
computes; no new perception system.

r_q = [ H(context yaw prior), log1p(N_gamma), |U_obs|, pose_confidence,
        horizon_confidence, water_frac, sky_frac, land_frac, horizon_y_norm ]

pi_j = [ pitch_bin/17, yaw_bin/17, roll_bin/17, y_bin/7, log_area, aspect_ratio ]
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch

EPS = 1e-8
RELIABILITY_KEYS = [
    "H_ctx_yaw", "log_N_gamma", "n_obs", "pose_confidence",
    "horizon_confidence", "water_frac", "sky_frac", "land_frac", "horizon_y_norm",
]
RELIABILITY_DIM = len(RELIABILITY_KEYS)
GEOM_DIM = 6


def build_ctx_counts(train_pairs: List[Dict]) -> Counter:
    """N_gamma: number of TRAIN observations per context_key (context support)."""
    return Counter(str(p.get("context_key")) for p in train_pairs)


def build_obs_counts(queries: List[Dict]) -> Dict[int, int]:
    """|U_obs|: vessels sharing an image (minus self is handled at use time)."""
    per_image = defaultdict(int)
    for q in queries:
        per_image[int(q.get("image_id", -1))] += 1
    return per_image


def reliability_vector(q: Dict, prior: Dict, ctx_counts: Counter,
                       obs_per_image: Dict[int, int]) -> np.ndarray:
    ck = str(q.get("context_key"))
    st = str(q.get("scene_type"))
    h_ctx = prior.get("context_yaw_entropy", {}).get(
        ck, prior.get("scene_yaw_entropy", {}).get(st, 0.0))
    n_gamma = ctx_counts.get(ck, 0)
    n_obs = max(0, obs_per_image.get(int(q.get("image_id", -1)), 1) - 1)
    x = [
        float(h_ctx),
        math.log1p(float(n_gamma)),
        float(n_obs),
        float(q.get("pose_confidence", 0.0)),
        float(q.get("horizon_confidence", 0.0)),
        float(q.get("water_frac", 0.0)),
        float(q.get("sky_frac", 0.0)),
        float(q.get("land_frac", 0.0)),
        float(q.get("horizon_y_norm", 0.0)),
    ]
    return np.asarray(x, dtype=np.float32)


def reliability_matrix(queries, prior, ctx_counts, obs_per_image) -> torch.Tensor:
    return torch.from_numpy(np.stack([
        reliability_vector(q, prior, ctx_counts, obs_per_image) for q in queries
    ]))


def candidate_geometry(candidates: List[Dict]) -> torch.Tensor:
    rows = []
    for c in candidates:
        rows.append([
            int(c.get("pitch_bin", 0)) / 17.0,
            int(c.get("yaw_bin", 0)) / 17.0,
            int(c.get("roll_bin", 0)) / 17.0,
            int(c.get("y_bin", 0)) / 7.0,
            math.log(max(float(c.get("area_frac", EPS)), EPS)),
            float(c.get("aspect_ratio", 0.0)),
        ])
    return torch.tensor(rows, dtype=torch.float32)


def standardize_columns(x: torch.Tensor, stats=None):
    """Z-standardize feature columns; return (x_std, (mu,sigma)). Reuse stats at eval."""
    if stats is None:
        mu = x.mean(dim=0, keepdim=True)
        sigma = x.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        stats = (mu, sigma)
    return (x - stats[0]) / stats[1], stats
