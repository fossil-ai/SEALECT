#!/usr/bin/env python3
"""
run_ablation.py

Thin CLI: pick a variant by folder name, run it over the frozen components, emit
`retrieval_results.json`, and refresh the tracker.

    python run_ablation.py --variant A0_baseline
    python run_ablation.py --variant A0raw_nonorm --lambda_o 1.0
    python run_ablation.py --list

Output -> sealect2/runs/<run_id>/topk_{K}/retrieval_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import legacy_bridge as lb
import registry
import driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", help="variant folder name, e.g. A0_baseline")
    ap.add_argument("--run-id", default=None, help="output dir name (default: variant name)")
    ap.add_argument("--lambda_n", type=float, default=1.0)
    ap.add_argument("--lambda_c", type=float, default=1.0)
    ap.add_argument("--lambda_o", type=float, default=0.10)
    ap.add_argument("--topks", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--emit-into-exp1", action="store_true")
    ap.add_argument("--list", action="store_true", help="list discovered variants and exit")
    args = ap.parse_args()

    if args.list or not args.variant:
        print("Discovered variants:")
        for n in registry.list_variant_names():
            v = registry.load(n)
            print(f"  {n:18s} [{v.kind}/stage{v.stage}] {v.description}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = registry.load(args.variant, lambda_n=args.lambda_n,
                            lambda_c=args.lambda_c, lambda_o=args.lambda_o)
    run_id = args.run_id or args.variant

    raw = lb.build_raw_components(use_cache=not args.no_cache, device=device)

    # Learned fusions (A2-A5) and global-z normalization (A1) fit on TRAIN data.
    needs_train = getattr(variant, "learned", False) or getattr(variant, "needs_train", False) \
        or variant.normalizer == "global_z"
    if needs_train:
        import train_fit
        tc = train_fit.build_train_context(use_cache=not args.no_cache)
        variant.prepare(tc)
    else:
        variant.prepare()

    driver.run(variant, raw, args.topks, run_id, device, emit_into_exp1=args.emit_into_exp1)

    if not args.emit_into_exp1:
        import eval_common
        met = eval_common.score_exp1(run_id, tuple(args.topks))
        print(f"[run_ablation] {run_id} exp1: " +
              " ".join(f"{kk}={vv:.4f}" for kk, vv in met.items()
                       if any(kk.startswith(p) for p in ("Recall", "NDCG", "PS"))))
    registry.write_manifest()


if __name__ == "__main__":
    main()
