"""A0 — current SEALECT baseline (reproduction target).

Per-query z-norm of each component + fixed lambda linear fusion. This is exactly
what the paper ships (run `sweep_ln1_lc1_lo0p10`) and the gate every other
variant is judged against. Do not add smoothing/normalization beyond the
per-query z-score the original already applies.
"""

from base import Variant


class A0Baseline(Variant):
    name = "A0_baseline"
    kind = "fusion"
    stage = 1
    normalizer = "perquery_z"
    params_added = 0
    learned = False
    description = "Shipped SEALECT: per-query z-norm + fixed lambda (reproduces paper Table 1)."


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = A0Baseline()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
