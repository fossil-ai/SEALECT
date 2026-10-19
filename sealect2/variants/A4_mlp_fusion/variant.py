"""A4 — nonlinear MLP fusion.

z_j = [ S~_n, S~_c, S~_o, r_q, pi_j ]; S(j) = f_theta(z_j), a small MLP. Tests
whether nonlinear interactions between the evidence sources (and query/candidate
geometry) beat linear stacking. Neural retrieval encoder is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from base import Variant
import fitting
import features as feat


class MLPHead(nn.Module):
    def __init__(self, d_r, geom_dim, hidden=32):
        super().__init__()
        in_dim = 3 + d_r + geom_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("geom_mu", torch.zeros(geom_dim))
        self.register_buffer("geom_sigma", torch.ones(geom_dim))

    def score(self, n, c, o, rq=None, geom=None):
        B, M = n.shape
        nco = torch.stack([n, c, o], dim=-1)              # [B,M,3]
        rqx = rq[:, None, :].expand(B, M, rq.shape[-1])   # [B,M,d_r]
        g = (geom - self.geom_mu) / self.geom_sigma
        if g.dim() == 2:                                   # [M,6] shared -> [B,M,6]
            g = g[None, :, :].expand(B, M, g.shape[-1])
        z = torch.cat([nco, rqx, g], dim=-1)
        return self.net(z).squeeze(-1)                     # [B,M]


class A4MLPFusion(Variant):
    name = "A4_mlp_fusion"
    kind = "fusion"
    stage = 1
    normalizer = "perquery_z"
    learned = True
    keep = False   # DROPPED: worse than v1 on all metrics at K=20/100 (overfits sampled ranking objective)
    eval_batch = 32
    description = "Small MLP over [S~n,S~c,S~o,r_q,pi_j]; nonlinear evidence interactions, encoder frozen."

    def __init__(self):
        self.head = MLPHead(feat.RELIABILITY_DIM, feat.GEOM_DIM)
        self._rq_all = None
        self._geom = None
        self._prior = self._ctx_counts = self._rq_stats = None

    def prepare(self, trainctx=None):
        if trainctx is None:
            raise NotImplementedError("A4 needs the train context.")
        self._prior = trainctx.prior
        self._ctx_counts = trainctx.ctx_counts
        self._rq_stats = trainctx.reliability_stats
        # geom standardization stats from train samples
        g = trainctx.geom.reshape(-1, feat.GEOM_DIM)
        self.head.geom_mu.copy_(g.mean(0))
        self.head.geom_sigma.copy_(g.std(0, unbiased=False) + 1e-6)
        fwd = lambda m, n, c, o, rq, geom: m.score(n, c, o, rq=rq, geom=geom)
        self.head, info = fitting.fit(self.head, fwd, trainctx, epochs=80, lr=5e-3)
        self.params_added = info["params_added"]; self.train_time_s = info["train_time_s"]
        return self

    def on_eval_start(self, raw, device="cpu"):
        obs_per_image = feat.build_obs_counts(raw.queries)
        rq = feat.reliability_matrix(raw.queries, self._prior, self._ctx_counts, obs_per_image)
        mu, sigma = self._rq_stats
        self._rq_all = ((rq - mu) / sigma).to(device)
        self._geom = feat.candidate_geometry(raw.candidates).to(device)
        self.head = self.head.to(device)

    @torch.no_grad()
    def fuse(self, n, c, o, meta=None):
        rq = self._rq_all[meta["start"]:meta["end"]]
        return self.head.score(n, c, o, rq=rq, geom=self._geom)


def build(**_):
    return A4MLPFusion()
