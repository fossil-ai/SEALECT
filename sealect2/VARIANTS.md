# Variant tracker

One folder per variant under `variants/`. Delete a folder (or set `keep = False`) to drop a loser; the study picks up the change automatically. Regenerate with `python registry.py`.

| name | kind | stage | normalizer | params_added | learned | keep | has_run | Recall@20 | NDCG@20 | PS@20 | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_baseline | fusion | 1 | perquery_z | 0 | False | True | True | 0.5984 | 0.8429 | 2.3536 | Shipped SEALECT: per-query z-norm + fixed lambda (reproduces paper Table 1). |
| A0raw_nonorm | fusion | 1 | identity | 0 | False | True | True | 0.1828 | 0.8924 | 1.9903 | Raw scores, no normalization (literal spec A0); diagnostic for score-scale sensitivity. |
| A1_global_z | fusion | 1 | global_z | 0 | False | True | True | 0.6210 | 0.8384 | 2.3361 | Fixed lambda with train-only global z-norm (isolates score-scale effects). |
| A2_learned_linear | fusion | 1 | perquery_z | 3 | True | True | True | 0.5001 | 0.8583 | 2.3657 | Learned global linear weights lambda=softmax(theta) (3 params), ranking loss, scores frozen. |
| A3_adaptive_gating | fusion | 1 | perquery_z | 211 | True | True | True | 0.4832 | 0.8589 | 2.3315 | Tiny per-query gating lambda(q)=softmax(g(r_q)) over reliability features; scores frozen. |
| A4_mlp_fusion | fusion | 1 | perquery_z | 1697 | True | False | True | 0.4002 | 0.8327 | 2.1105 | Small MLP over [S~n,S~c,S~o,r_q,pi_j]; nonlinear evidence interactions, encoder frozen. |
| A5_tiny_attention | fusion | 1 | perquery_z | 2337 | True | False | True | 0.6163 | 0.8172 | 2.2128 | 1-layer self-attention over 3 evidence tokens -> CLS score; tests learned source interactions. |
| B1_pmi_context | context | 2 | perquery_z | 0 | False | True | True | 0.6014 | 0.8224 | 2.2562 | PMI context: log p(pi|gamma) - log p(pi), marginal from train. Fusion = v1. |
| B2_smoothed_prior | context | 2 | perquery_z | 0 | False | True | True | 0.5918 | 0.8437 | 2.3526 | Add-alpha smoothed empirical prior (categorical pitch/yaw/roll, ybin|pose). Fusion = v1. |
| B3_kernel_prior | context | 2 | perquery_z | 0 | False | True | True | 0.6266 | 0.8480 | 2.4182 | Kernel-conditioned empirical prior (RBF over scene features, no hard bin). Fusion = v1. |
| B4_attention_prior | context | 2 | perquery_z | 1489 | True | False | True | 0.5986 | 0.8453 | 2.3903 | Attention-weighted empirical prior: learned scene relevance over anchor histograms. Fusion = v1. |
| C1_geo_hardneg | neural_training | 3 | perquery_z | 0 | True | True | True | 0.6287 | 0.8494 | 2.4351 | Neural retriever trained with geometry hard-negative margin loss; evaluated with B3 context + v1 fusion. |
