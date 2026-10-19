"""A5 — tiny attention fusion over evidence tokens.

Represent the three scores as tokens t_n, t_c, t_o (scalar -> d via a shared
projection + a learned per-source type embedding), prepend a CLS token, process
with a 1-layer self-attention block, and read S(j) = w . CLS. Tests whether
learned interactions between evidence sources beat MLP/linear fusion. Kept tiny
(low d, 1 layer) on purpose.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from base import Variant
import fitting


class TinyAttn(nn.Module):
    def __init__(self, d=16, nhead=2):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.cls = nn.Parameter(torch.zeros(1, d))
        self.enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                              dim_feedforward=2 * d, batch_first=True)
        self.w = nn.Linear(d, 1)

    CHUNK = 16384  # cap sequences per kernel launch (CUDA grid limit)

    def _score_flat(self, s):                                     # s [N,3]
        tok = self.embed(s.unsqueeze(-1)) + self.type_emb[None]   # [N,3,d]
        cls = self.cls.expand(s.shape[0], 1, -1)                  # [N,1,d]
        seq = torch.cat([cls, tok], dim=1)                        # [N,4,d]
        out = self.enc(seq)[:, 0]                                 # [N,d]
        return self.w(out).squeeze(-1)                            # [N]

    def score(self, n, c, o, rq=None, geom=None):
        B, M = n.shape
        s = torch.stack([n, c, o], dim=-1).reshape(B * M, 3)      # [N,3]
        outs = [self._score_flat(s[i:i + self.CHUNK]) for i in range(0, s.shape[0], self.CHUNK)]
        return torch.cat(outs, dim=0).reshape(B, M)


class A5TinyAttention(Variant):
    name = "A5_tiny_attention"
    kind = "fusion"
    stage = 1
    normalizer = "perquery_z"
    learned = True
    keep = False   # DROPPED: Pareto-dominated by A1 (>= recall, > NDCG, > plausibility, 0 params)
    eval_batch = 64
    description = "1-layer self-attention over 3 evidence tokens -> CLS score; tests learned source interactions."

    def __init__(self):
        self.head = TinyAttn()

    def prepare(self, trainctx=None):
        if trainctx is None:
            raise NotImplementedError("A5 needs the train context.")
        fwd = lambda m, n, c, o, rq, geom: m.score(n, c, o)
        self.head, info = fitting.fit(self.head, fwd, trainctx, epochs=80, lr=5e-3, batch_q=128)
        self.params_added = info["params_added"]; self.train_time_s = info["train_time_s"]
        return self

    def on_eval_start(self, raw, device="cpu"):
        self.head = self.head.to(device)

    @torch.no_grad()
    def fuse(self, n, c, o, meta=None):
        return self.head.score(n, c, o)


def build(**_):
    return A5TinyAttention()
