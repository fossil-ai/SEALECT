#!/usr/bin/env python3
"""
verify_baseline.py

Gate for the whole study: A0_baseline must reproduce shipped SEALECT before any
new variant is trusted. Runs the A0 variant via the registry/driver and scores
it with the ORIGINAL paper metric code.

Target (paper Table 1, SEALECT lambda_o=0.10):
    K=20  : Recall 0.5984  NDCG 0.8429  PS 2.3536
    K=100 : Recall 0.7665  NDCG 0.8268  PS 2.1419
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import legacy_bridge as lb
import registry
import driver

TARGET = {
    20: {"Recall@20": 0.5984, "NDCG@20": 0.8429, "PS@20": 2.3536},
    100: {"Recall@100": 0.7665, "NDCG@100": 0.8268, "PS@100": 2.1419},
}
TOL = 1e-3


def load_summarizer():
    f = lb.EXP1_DIR / "summarize_exp1_retrieval_results.py"
    spec = importlib.util.spec_from_file_location("legacy_summarize", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = lb.build_raw_components(use_cache=True, device=device)
    variant = registry.load("A0_baseline")
    base = driver.run(variant, raw, [20, 100], "A0_baseline", device)

    summ = load_summarizer()
    ok = True
    print("\n==== A0 reproduction check ====")
    for k in (20, 100):
        results = json.load(open(base / f"topk_{k}" / "retrieval_results.json"))
        m = summ.summarize_one(results, k)
        # persist metrics next to the run so the tracker can show them
        (base / f"topk_{k}" / "metrics.json").write_text(json.dumps(m, indent=2))
        print(f"K={k}")
        for metric, tgt in TARGET[k].items():
            got = m[metric]
            d = abs(got - tgt)
            if d > TOL:
                ok = False
            print(f"  {metric:12s} got={got:.4f}  target={tgt:.4f}  |Δ|={d:.4f}  {'OK' if d<=TOL else 'MISMATCH'}")
    registry.write_manifest()
    print("\nRESULT:", "A0 reproduces shipped SEALECT [OK]" if ok else "A0 DOES NOT MATCH -> investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
