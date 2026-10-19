"""A3 — query-adaptive fusion (tiny gating network).

lambda(q) = softmax(g(r_q)); S(j) = sum_k lambda_k(q) * S~_k(j). The gate g is a
tiny MLP over candidate-INDEPENDENT reliability features r_q (context-prior
entropy, context support N_gamma, |U_obs|, pose/horizon confidence, water/sky/
land fractions, horizon). Per-query weights, shared across candidates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Variant
import fitting
import features as feat


class GateNet(nn.Module):
    def __init__(self, d_r, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_r, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def weights(self, rq):                      # rq [B,d_r] -> [B,3]
        return F.softmax(self.net(rq), dim=-1)

    def score(self, n, c, o, rq=None, geom=None):
        w = self.weights(rq)                    # [B,3]
        return w[:, 0:1] * n + w[:, 1:2] * c + w[:, 2:3] * o


class A3AdaptiveGating(Variant):
    name = "A3_adaptive_gating"
    kind = "fusion"
    stage = 1
    normalizer = "perquery_z"
    learned = True
    description = "Tiny per-query gating lambda(q)=softmax(g(r_q)) over reliability features; scores frozen."

    def __init__(self):
        self.head = GateNet(feat.RELIABILITY_DIM)
        self._rq_all = None
        self._prior = None
        self._ctx_counts = None
        self._rq_stats = None
        self._device = "cpu"

    def prepare(self, trainctx=None):
        if trainctx is None:
            raise NotImplementedError("A3 needs the train context.")
        self._prior = trainctx.prior
        self._ctx_counts = trainctx.ctx_counts
        self._rq_stats = trainctx.reliability_stats
        fwd = lambda m, n, c, o, rq, geom: m.score(n, c, o, rq=rq)
        self.head, info = fitting.fit(self.head, fwd, trainctx, epochs=80, lr=1e-2)
        self.params_added = info["params_added"]; self.train_time_s = info["train_time_s"]
        return self

    def on_eval_start(self, raw, device="cpu"):
        self._device = device
        obs_per_image = feat.build_obs_counts(raw.queries)   # |U_obs| within eval images
        rq = feat.reliability_matrix(raw.queries, self._prior, self._ctx_counts, obs_per_image)
        mu, sigma = self._rq_stats
        rq = (rq - mu) / sigma                                # same standardization as train
        self._rq_all = rq.to(device)
        self.head = self.head.to(device)

    @torch.no_grad()
    def fuse(self, n, c, o, meta=None):
        rq = self._rq_all[meta["start"]:meta["end"]]
        return self.head.score(n, c, o, rq=rq)


def build(**_):
    return A3AdaptiveGating()
