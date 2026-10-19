#!/usr/bin/env python3
"""
print_exp2_distributional_report_grounded_sam2.py

Reads Experiment 2 distributional metrics and prints compact tables for K=20/K=100.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/path/to/sealect_root/experiments/experiment2/distributional_metrics_grounded_sam2")
TOP_KS = [20, 100]

MODES_TO_PRINT = ["yaw", "joint_pose", "pose_family", "ybin", "scale_bin"]

METHOD_ORDER = [
    "random",
    "clip",
    "dino",
    "prior_only",
    "clip_prior",
    "dino_prior",
    "neural_only_dinofeat",
    "sealect_dinofeat",
    "sweep_neural_only",
    "sweep_context_only",
    "sweep_obs_only",
    "sweep_ln1_lc1_lo0p10",
    "sweep_ln1_lc1_lo0p50",
    "sweep_ln1_lc1_lo1p00",
]


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def fnum(x, digits=4):
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:.{digits}f}"
    except Exception:
        return "nan"


def method_sort_key(m):
    if m in METHOD_ORDER:
        return METHOD_ORDER.index(m)
    if m.startswith("sweep_"):
        return len(METHOD_ORDER)
    return len(METHOD_ORDER) + 1


def main():
    print("\n")
    print("EXPERIMENT 2 DISTRIBUTIONAL REPORT")
    print("")

    for top_k in TOP_KS:
        path = ROOT / f"k{top_k}" / "distributional_metrics_summary.csv"
        if not path.exists():
            print(f"\nMissing: {path}")
            continue

        rows = read_csv(path)
        by_mode = defaultdict(list)
        for r in rows:
            by_mode[r["mode"]].append(r)

        print(f"\n\n######## K={top_k} ########")

        for mode in MODES_TO_PRINT:
            subset = by_mode.get(mode, [])
            if not subset:
                continue

            subset = sorted(subset, key=lambda r: method_sort_key(r["method"]))

            print(f"\n--- MODE: {mode} ---")
            print(
                "method".ljust(28),
                "JS".rjust(10),
                "H_gap".rjust(10),
                "H_gt".rjust(10),
                "H_ret".rjust(10),
                "ctx".rjust(6),
            )
            print("-" * 82)

            for r in subset:
                print(
                    r["method"].ljust(28),
                    fnum(r["mean_js_divergence"]).rjust(10),
                    fnum(r["mean_entropy_gap"]).rjust(10),
                    fnum(r["mean_gt_entropy"]).rjust(10),
                    fnum(r["mean_retrieved_entropy"]).rjust(10),
                    str(r["num_contexts_eval"]).rjust(6),
                )

    print("\nLegend:")
    print("  JS     = mean Jensen-Shannon divergence across context_key bins")
    print("  H_gap  = mean |retrieved entropy - ground-truth entropy|")
    print("  H_gt    = mean ground-truth entropy")
    print("  H_ret   = mean retrieved entropy")
    print("  ctx     = number of context_key bins evaluated")


if __name__ == "__main__":
    main()
