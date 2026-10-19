#!/usr/bin/env python3
"""Bandwidth robustness sweep for the B3 kernel-conditioned prior.

Reuses the exact frozen components + original paper evaluators. Deterministic:
h=1.0 must reproduce runs_summary.csv B3_kernel_prior (0.6266/0.8480/2.4182 @K20).
"""
import sys, json
from pathlib import Path

HERE = Path("/path/to/sealect_root/sealect2")
sys.path.insert(0, str(HERE))

import importlib.util
import torch
import legacy_bridge as lb
import driver
import eval_common

spec = importlib.util.spec_from_file_location("b3", str(HERE / "variants/B3_kernel_prior/variant.py"))
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[sweep] device={device}")
raw = lb.build_raw_components(use_cache=True, device=device)

rows = {}
for h in [0.5, 1.0, 2.0]:
    v = b3.build(h=h)
    v.prepare()
    run_id = f"B3_h{h:g}"
    driver.run(v, raw, [20, 100], run_id, device)
    met = eval_common.score_exp1(run_id, (20, 100))
    rows[h] = met
    print(f"[sweep] h={h}: " + " ".join(f"{k}={met[k]:.4f}" for k in sorted(met)))

out = HERE / "runs" / "bandwidth_sweep.json"
out.write_text(json.dumps(rows, indent=2))
print(f"[sweep] wrote {out}")
