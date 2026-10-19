"""B3 - continuous kernel-conditioned empirical prior.

Replaces the hard context lookup gamma=kappa(c) with an RBF kernel over
standardized scene features; p(pi|c_q) becomes a kernel-weighted empirical
histogram over the training library. Still nonparametric (no learned params).
Tests whether hard quantization -- not empirical estimation -- is the weak part.
Bandwidth h sweepable (--h via run-id variants). Fusion held at v1.
"""

from __future__ import annotations

from base import Variant
import context_models as cm
import context_kernel as ck


class B3KernelPrior(Variant):
    name = "B3_kernel_prior"
    kind = "context"
    stage = 2
    normalizer = "perquery_z"
    h = 1.0
    description = "Kernel-conditioned empirical prior (RBF over scene features, no hard bin). Fusion = v1."

    def prepare(self, trainctx=None):
        self._legacy, self._train_pairs, self._cands, _ = cm.load_ctx_assets()
        return self

    def context_component(self, raw):
        return ck.kernel_context_matrix(self._legacy, self._train_pairs,
                                        raw.queries, raw.candidates, h=self.h)


def build(h=1.0, lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = B3KernelPrior()
    v.h = float(h)
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
