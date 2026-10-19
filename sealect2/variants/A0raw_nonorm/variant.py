"""A0raw — literal spec-A0: raw scores, no normalization.

S = ln*S_neural + lc*S_context + lo*S_obs with NO per-component normalization.
Diagnostic only: isolates how much the shipped baseline relies on per-query
z-scoring (compare against A0_baseline and A1_global_z).
"""

from base import Variant


class A0Raw(Variant):
    name = "A0raw_nonorm"
    kind = "fusion"
    stage = 1
    normalizer = "identity"
    params_added = 0
    learned = False
    description = "Raw scores, no normalization (literal spec A0); diagnostic for score-scale sensitivity."


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = A0Raw()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
