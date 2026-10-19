#!/usr/bin/env python3
"""Fusion-weight sensitivity for SEALECT (kernel, h=1.0). Perturbs lambda=(n,g,o)
around the shipped operating point (1,1,0.10). Same evaluators as the paper.
(1,1,0.10) must reproduce B3_kernel_prior (0.6266/0.8480/2.4182 @K20)."""
import sys, json
from pathlib import Path
HERE = Path("/path/to/sealect_root/sealect2")
sys.path.insert(0, str(HERE))
import importlib.util
import torch, legacy_bridge as lb, driver, eval_common
spec = importlib.util.spec_from_file_location("b3", str(HERE / "variants/B3_kernel_prior/variant.py"))
b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[lambda] device={device}")
raw = lb.build_raw_components(use_cache=True, device=device)

# (lambda_n, lambda_g, lambda_o); lambda_g == harness lambda_c (context/geom term)
settings = [(1,1,0.10),(1,0.5,0.10),(1,2.0,0.10),(1,1,0.0),(1,1,0.25),(0.75,1,0.10),(1.25,1,0.10)]
rows = {}
for (ln, lg, lo) in settings:
    v = b3.build(h=1.0, lambda_n=ln, lambda_c=lg, lambda_o=lo)
    v.prepare()
    rid = f"B3_lam_{ln}_{lg}_{lo}".replace('.', 'p')
    driver.run(v, raw, [20, 100], rid, device)
    m = eval_common.score_exp1(rid, (20, 100))
    rows[f"{ln},{lg},{lo}"] = m
    print(f"[lambda] ({ln},{lg},{lo}): R@20={m['Recall@20']:.4f} NDCG@20={m['NDCG@20']:.4f} "
          f"PS@20={m['PS@20']:.4f} | R@100={m['Recall@100']:.4f} NDCG@100={m['NDCG@100']:.4f} PS@100={m['PS@100']:.4f}")

(HERE / "runs" / "lambda_sweep.json").write_text(json.dumps(rows, indent=2))
print("[lambda] wrote runs/lambda_sweep.json")
