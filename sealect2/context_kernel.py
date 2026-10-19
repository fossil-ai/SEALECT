"""
context_kernel.py

B3 continuous kernel-conditioned empirical prior. Drops the hard context bin
gamma = kappa(c) and instead weights every training scene by an RBF kernel over
(standardized) scene features:

    w_i(c_q) = softmax_i( -||c_q - c_i||^2 / (2 h^2) )
    p(pi | c_q) = sum_i w_i(c_q) K_pi(pi, pi_i)     (empirical, nonparametric)

We keep the same factorized scoring form and weights as the v1 empirical prior
(2*pitch + yaw + roll + 0.7*scale + 0.5*ybin), but each factor is a
kernel-weighted training histogram instead of a hard-bin histogram. Still fully
empirical (no learned params) -- this isolates "hard quantization" as the
weakness, not empirical estimation itself. Bandwidth h is swept.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def scene_bank(legacy, records, stats=None):
    X = np.stack([legacy.scene_features(r) for r in records]).astype(np.float32)
    if stats is None:
        mu = X.mean(0)
        sigma = X.std(0) + 1e-6
        stats = (mu, sigma)
    Xs = (X - stats[0]) / stats[1]
    return torch.from_numpy(Xs), stats


def _ints(records, key):
    return torch.tensor([int(r.get(key, 0)) for r in records], dtype=torch.long)


def kernel_context_matrix(legacy, train_pairs, queries, candidates,
                          h=1.0, batch=256, device=None, eps=1e-8):
    device = device or legacy.DEVICE
    Cx, stats = scene_bank(legacy, train_pairs)             # anchors [Na,d]
    Qx, _ = scene_bank(legacy, queries, stats)              # queries [Nq,d]
    Cx = Cx.to(device)

    oh_pitch = F.one_hot(_ints(train_pairs, "pitch_bin").clamp(0, 17), 18).float().to(device)
    oh_yaw = F.one_hot(_ints(train_pairs, "yaw_bin").clamp(0, 17), 18).float().to(device)
    oh_roll = F.one_hot(_ints(train_pairs, "roll_bin").clamp(0, 17), 18).float().to(device)
    oh_yb = F.one_hot(_ints(train_pairs, "y_bin").clamp(0, 7), 8).float().to(device)
    a_logarea = torch.tensor([math.log(max(float(p.get("area_frac", 1e-8)), 1e-8))
                              for p in train_pairs], device=device)

    c_pitch = _ints(candidates, "pitch_bin").clamp(0, 17).to(device)
    c_yaw = _ints(candidates, "yaw_bin").clamp(0, 17).to(device)
    c_roll = _ints(candidates, "roll_bin").clamp(0, 17).to(device)
    c_yb = _ints(candidates, "y_bin").clamp(0, 7).to(device)
    c_logarea = torch.tensor([math.log(max(float(c.get("area_frac", 1e-8)), 1e-8))
                              for c in candidates], device=device)

    Nq, Ncand = len(queries), len(candidates)
    out = torch.empty(Nq, Ncand, dtype=torch.float32)
    inv2h2 = 1.0 / (2.0 * h * h)

    for s in range(0, Nq, batch):
        e = min(s + batch, Nq)
        Qb = Qx[s:e].to(device)                              # [B,d]
        d2 = torch.cdist(Qb, Cx) ** 2                        # [B,Na]
        W = torch.softmax(-d2 * inv2h2, dim=1)               # [B,Na]

        lp_pitch = torch.log(W @ oh_pitch + eps)             # [B,18]
        lp_yaw = torch.log(W @ oh_yaw + eps)
        lp_roll = torch.log(W @ oh_roll + eps)
        lp_yb = torch.log(W @ oh_yb + eps)                   # [B,8]
        mu = W @ a_logarea                                    # [B]
        var = torch.clamp(W @ (a_logarea ** 2) - mu ** 2, min=0.02)
        sig = torch.sqrt(var)

        scale_term = (-0.5 * torch.log(2 * math.pi * var)[:, None]
                      - 0.5 * ((c_logarea[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        Sc = (2.0 * lp_pitch[:, c_pitch] + lp_yaw[:, c_yaw] + lp_roll[:, c_roll]
              + 0.7 * scale_term + 0.5 * lp_yb[:, c_yb])
        out[s:e] = Sc.cpu()
    return out
