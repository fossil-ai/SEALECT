"""B2 - additively-smoothed empirical prior.

Keep empirical estimation but replace the raw normalized histograms with
add-alpha smoothed categorical distributions:
    p(u|gamma) = (N(u,gamma) + alpha) / (N(gamma) + alpha*|U|),
applied to pitch/yaw/roll and ybin|pose. Removes the -log(eps) cliffs for unseen
bins. alpha sweepable via --lambda_c? no -- via ALPHA class attr / run-id suffix.
Fusion held at v1.
"""

from __future__ import annotations

from base import Variant
import context_models as cm


class B2SmoothedPrior(Variant):
    name = "B2_smoothed_prior"
    kind = "context"
    stage = 2
    normalizer = "perquery_z"
    alpha = 1.0
    description = "Add-alpha smoothed empirical prior (categorical pitch/yaw/roll, ybin|pose). Fusion = v1."

    def prepare(self, trainctx=None):
        self._legacy, train_pairs, self._cands, base_prior = cm.load_ctx_assets()
        self._prior = cm.build_smoothed_prior(base_prior, train_pairs, alpha=self.alpha)
        return self

    def context_component(self, raw):
        return cm.context_matrix_from_prior(self._legacy, self._prior, raw.queries, raw.candidates)


def build(alpha=1.0, lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = B2SmoothedPrior()
    v.alpha = float(alpha)
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
