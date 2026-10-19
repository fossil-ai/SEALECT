"""B1 - PMI / likelihood-ratio context score.

Replace S_c = log p(pi_j | gamma_q) with the pointwise mutual information
S_c^PMI = log[ p(pi_j|gamma_q) / p(pi_j) ], estimating the marginal p(pi) from
training data. Rewards geometry that is *specifically* informative for the scene
rather than globally common. Fusion held at v1 (per-query z + fixed lambda).
"""

from __future__ import annotations

from base import Variant
import context_models as cm


class B1PMIContext(Variant):
    name = "B1_pmi_context"
    kind = "context"
    stage = 2
    normalizer = "perquery_z"
    description = "PMI context: log p(pi|gamma) - log p(pi), marginal from train. Fusion = v1."

    def prepare(self, trainctx=None):
        legacy, train_pairs, candidates, _ = cm.load_ctx_assets()
        self._marginal = cm.marginal_candidate_vec(legacy, train_pairs, candidates)  # [Ncand]
        return self

    def context_component(self, raw):
        # PMI = base empirical log p(pi|gamma) - marginal log p(pi)
        return raw.context - self._marginal[None, :]


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = B1PMIContext()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
