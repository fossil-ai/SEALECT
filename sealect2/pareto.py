#!/usr/bin/env python3
"""
pareto.py

Read runs_summary.csv and report a compact comparison + the Pareto-optimal
variants across retrieval quality, plausibility, and parameter cost.

Objectives (K=20, all "higher is better" after sign flips):
    quality       = NDCG            (max)
    recall        = Recall          (max)
    plausibility  = joint_ll        (max)  [= -violation direction]
    cost          = -params_added   (max)  [cheaper is better]

A variant is Pareto-optimal if no other variant is >= on every objective and
strictly > on at least one. v1 (A0_baseline) is marked for reference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import legacy_bridge as lb

CSV = lb.SEALECT2_DIR / "runs_summary.csv"
OBJ = [("NDCG", 1), ("Recall", 1), ("joint_ll", 1), ("params_added", -1)]  # sign: +max, -min


def _load(k=20):
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if int(r["K"]) != k:
                continue
            for c in ("NDCG", "Recall", "joint_ll", "params_added", "PS", "joint_violation"):
                r[c] = float(r[c]) if r[c] not in ("", "nan") else float("nan")
            rows.append(r)
    return rows


def _dominates(a, b):
    better_any = False
    for name, sign in OBJ:
        av, bv = sign * a[name], sign * b[name]
        if av < bv:
            return False
        if av > bv:
            better_any = True
    return better_any


def pareto(rows):
    front = []
    for a in rows:
        if not any(_dominates(b, a) for b in rows if b is not a):
            front.append(a["run_id"])
    return set(front)


def main(k=20):
    rows = _load(k)
    front = pareto(rows)
    v1 = next((r for r in rows if r["run_id"] == "A0_baseline"), None)

    print(f"\n=== Pareto comparison @ K={k} (quality=NDCG, recall, plausibility=joint_ll, cost=params) ===\n")
    hdr = f"{'variant':22s} {'Recall':>7s} {'NDCG':>7s} {'PS':>6s} {'jLL':>7s} {'jViol':>6s} {'params':>7s}  {'Pareto':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (-x["NDCG"], -x["Recall"])):
        star = "  ***" if r["run_id"] in front else ""
        v1tag = " (v1)" if r["run_id"] == "A0_baseline" else ""
        print(f"{r['run_id']+v1tag:22s} {r['Recall']:7.4f} {r['NDCG']:7.4f} {r['PS']:6.3f} "
              f"{r['joint_ll']:7.3f} {r['joint_violation']:6.3f} {int(r['params_added']):7d}{star}")
    print(f"\nPareto-optimal: {sorted(front)}")
    if v1:
        print(f"v1 SEALECT (A0_baseline) Pareto-optimal: {'YES' if 'A0_baseline' in front else 'no'}")


if __name__ == "__main__":
    main()
