"""A2 — learned global linear fusion.

lambda = softmax(theta) (3 learnable scalars); S = sum_k lambda_k * S~_k.
Scores frozen; only theta is optimized on TRAIN with the listwise ranking loss.
Uses per-query z-norm (same as A0) so the comparison is purely fixed-vs-learned
weights at the baseline's operating normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Variant
import fitting


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(3))

    def weights(self):
        return F.softmax(self.theta, dim=0)

    def score(self, n, c, o, rq=None, geom=None):
        w = self.weights()
        return w[0] * n + w[1] * c + w[2] * o


class A2LearnedLinear(Variant):
    name = "A2_learned_linear"
    kind = "fusion"
    stage = 1
    normalizer = "perquery_z"
    learned = True
    description = "Learned global linear weights lambda=softmax(theta) (3 params), ranking loss, scores frozen."

    def __init__(self):
        self.head = LinearHead()

    def prepare(self, trainctx=None):
        if trainctx is None:
            raise NotImplementedError("A2 needs the train context.")
        fwd = lambda m, n, c, o, rq, geom: m.score(n, c, o)
        self.head, info = fitting.fit(self.head, fwd, trainctx, epochs=80, lr=5e-2)
        self.params_added = info["params_added"]; self.train_time_s = info["train_time_s"]
        w = self.head.weights().detach().cpu().tolist()
        print(f"    [A2] learned lambda (n,c,o) = {[round(x,3) for x in w]}")
        return self

    @torch.no_grad()
    def fuse(self, n, c, o, meta=None):
        return self.head.score(n, c, o)


def build(**_):
    return A2LearnedLinear()
