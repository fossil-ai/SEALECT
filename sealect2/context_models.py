"""
context_models.py

Shared helpers for the B-series context-model ablations. All keep the empirical
scene->pose structure and the v1 fusion (per-query z + fixed lambda); they only
change how S_context(j) is produced. Reuses the frozen legacy scoring function
`context_score_numpy` wherever the change can be expressed as a different prior.

B1 PMI      : S_c - log p(pi)              (marginal from train)
B2 smoothed : add-alpha categorical prior  (rebuilt from train counts)
B3 kernel   : soft RBF over scene features (drop hard context bin)   [context_kernel.py]
B4 attention: learned scene relevance       (weights empirical evidence) [B4 folder]
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import torch

import legacy_bridge as lb

N_YAW = 18
N_POSE = 18
N_YBIN = 8


def load_ctx_assets():
    """Return (legacy, train_pairs, candidates, base_prior)."""
    legacy = lb.load_legacy()
    candidates = legacy.load_json(legacy.TRAIN_CANDIDATES)
    train_pairs = legacy.load_json(legacy.SPLIT_DIR / "train_scene_object_pairs.json")
    base_prior = legacy.load_json(legacy.PRIOR_JSON)
    return legacy, train_pairs, candidates, base_prior


def context_matrix_from_prior(legacy, prior, queries, candidates) -> torch.Tensor:
    """[Nq,Ncand] S_context using the frozen scorer with a (possibly modified) prior."""
    cache = legacy.build_context_cache(queries, candidates, prior)
    rows = [torch.from_numpy(cache[(str(q.get("context_key")), str(q.get("scene_type")))])
            for q in queries]
    return torch.stack(rows, dim=0).float()


# ---------------------------------------------------------------- B1 marginal
def _norm_counter(counter, n_bins):
    tot = sum(counter.values())
    return {str(b): (counter.get(b, 0) / tot if tot else 0.0) for b in range(n_bins)}


def build_marginal_prior(train_pairs):
    """A prior dict with a single '__G__' context = global marginal p(pi)."""
    pitch = Counter(int(p["pitch_bin"]) for p in train_pairs)
    yaw = Counter(int(p["yaw_bin"]) for p in train_pairs)
    roll = Counter(int(p["roll_bin"]) for p in train_pairs)
    # scale by ybin
    by_yb = defaultdict(list)
    for p in train_pairs:
        by_yb[int(p["y_bin"])].append(math.log(max(float(p["area_frac"]), 1e-8)))
    scale = {}
    for yb, vals in by_yb.items():
        a = np.asarray(vals)
        scale[str(yb)] = {"log_area_mean": float(a.mean()),
                          "log_area_std": float(max(a.std(), 0.15))}
    # ybin | pose
    ybin_pose = defaultdict(Counter)
    for p in train_pairs:
        pk = f"{int(p['pitch_bin'])}_{int(p['yaw_bin'])}_{int(p['roll_bin'])}"
        ybin_pose[pk][int(p["y_bin"])] += 1
    ybin_given = {pk: {str(k): v / sum(c.values()) for k, v in c.items()}
                  for pk, c in ybin_pose.items()}
    G = "__G__"
    return {
        "context_pitch_hist": {G: _norm_counter(pitch, N_POSE)},
        "context_yaw_hist": {G: _norm_counter(yaw, N_YAW)},
        "context_roll_hist": {G: _norm_counter(roll, N_POSE)},
        "context_yaw_weight": {G: 1.0},
        "scale_by_context_ybin": {G: scale},
        "ybin_given_context_pose": {G: ybin_given},
    }


def marginal_candidate_vec(legacy, train_pairs, candidates) -> torch.Tensor:
    """[Ncand] marginal context score log p(pi_j) (query-independent)."""
    mprior = build_marginal_prior(train_pairs)
    q = {"context_key": "__G__", "scene_type": "__G__"}
    vec = legacy.context_score_numpy(q, candidates, mprior)
    return torch.from_numpy(np.asarray(vec)).float()


# ---------------------------------------------------------------- B2 smoothed
def build_smoothed_prior(base_prior, train_pairs, alpha=1.0):
    """Add-alpha smoothing of the categorical context distributions (pitch/yaw/
    roll and ybin|pose). Scale (Gaussian) and yaw_weight copied from base."""
    prior = dict(base_prior)  # shallow; we replace the categorical sections
    by_ctx_pitch = defaultdict(Counter)
    by_ctx_yaw = defaultdict(Counter)
    by_ctx_roll = defaultdict(Counter)
    by_ctx_ybpose = defaultdict(lambda: defaultdict(Counter))
    for p in train_pairs:
        ck = str(p.get("context_key"))
        by_ctx_pitch[ck][int(p["pitch_bin"])] += 1
        by_ctx_yaw[ck][int(p["yaw_bin"])] += 1
        by_ctx_roll[ck][int(p["roll_bin"])] += 1
        pk = f"{int(p['pitch_bin'])}_{int(p['yaw_bin'])}_{int(p['roll_bin'])}"
        by_ctx_ybpose[ck][pk][int(p["y_bin"])] += 1

    def smooth_cat(counter, n_bins):
        n = sum(counter.values())
        return {str(b): (counter.get(b, 0) + alpha) / (n + alpha * n_bins) for b in range(n_bins)}

    prior["context_pitch_hist"] = {ck: smooth_cat(c, N_POSE) for ck, c in by_ctx_pitch.items()}
    prior["context_yaw_hist"] = {ck: smooth_cat(c, N_YAW) for ck, c in by_ctx_yaw.items()}
    prior["context_roll_hist"] = {ck: smooth_cat(c, N_POSE) for ck, c in by_ctx_roll.items()}
    ybin_given = {}
    for ck, posemap in by_ctx_ybpose.items():
        ybin_given[ck] = {pk: smooth_cat(c, N_YBIN) for pk, c in posemap.items()}
    prior["ybin_given_context_pose"] = ybin_given
    return prior
