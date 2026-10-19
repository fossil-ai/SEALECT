"""A1 — score-normalized fixed fusion (train-only global z).

Same three scores and fixed lambda as A0, but each component is normalized with
(mu, sigma) estimated from TRAINING data only (not per query). Isolates whether
the baseline's behavior is merely a score-scale artifact.

Requires training-component statistics: call prepare(trainctx) where trainctx
exposes .neural/.context/.obs matrices computed on TRAIN queries. Wired in
Stage 1 (train-split component builder). Until then this variant will refuse to
run rather than silently use test statistics.
"""

from base import Variant


class A1GlobalZ(Variant):
    name = "A1_global_z"
    kind = "fusion"
    stage = 1
    normalizer = "global_z"
    params_added = 0          # estimated stats, not learned params
    learned = False
    description = "Fixed lambda with train-only global z-norm (isolates score-scale effects)."

    needs_train = True

    def prepare(self, trainctx=None):
        if trainctx is None:
            raise NotImplementedError(
                "A1_global_z requires the train context (global stats). "
                "run_ablation builds it automatically."
            )
        self.global_stats = dict(trainctx.global_stats)  # {"neural":(mu,sig),...}
        return self


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = A1GlobalZ()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
