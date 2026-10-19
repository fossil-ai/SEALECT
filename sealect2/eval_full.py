#!/usr/bin/env python3
"""
eval_full.py

Full metric suite for every sealect2 run, reusing the ORIGINAL paper evaluators:
  - exp1 summarize_one : Recall/NDCG/PS  (pooled + per-dataset)
  - exp2 evaluate_one_method : JS / entropy-gap / retrieved-entropy (distributional)
  - exp3 evaluate_method     : joint-pose log-likelihood + violation rate (plausibility)

Emits one row per (run, K) into runs_summary.csv with fusion/context/neural type,
params_added, train/infer time, GPU mem, pooled metrics, and per-dataset recall
(cross-dataset check). Nothing in the original experiment dirs is touched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import legacy_bridge as lb

RUNS = lb.RUNS_DIR
OUT_CSV = lb.SEALECT2_DIR / "runs_summary.csv"


def _imp(rel, name):
    p = lb.SEALECT_ROOT / "experiments" / rel
    spec = importlib.util.spec_from_file_location(name, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SUMM = _imp("experiment1/summarize_exp1_retrieval_results.py", "summ_full")
EXP2 = _imp("experiment2/evaluate_exp2_distributional_metrics_grounded_sam2.py", "exp2_full")
EXP3 = _imp("experiment3/evaluate_exp3_pose_plausibility_grounded_sam2.py", "exp3_full")


def _run_ids():
    return sorted(p.name for p in RUNS.iterdir()
                  if p.is_dir() and (p / "topk_20" / "retrieval_results.json").exists())


def _cfg(run_id, k):
    p = RUNS / run_id / f"topk_{k}" / "config.json"
    return json.load(open(p)) if p.exists() else {}


def _mode_row(rows, mode):
    for r in rows:
        if r.get("mode") == mode:
            return r
    return {}


def evaluate_all(topks=(20, 100)):
    test_pairs = json.load(open(EXP3.TEST_PAIRS_JSON))
    datasets = sorted({str(p.get("dataset")) for p in test_pairs})
    # exp3 GT once
    gt3 = {m: EXP3.build_gt_counts(test_pairs, m) for m in EXP3.MODES}
    gt_counts = {m: gt3[m][0] for m in EXP3.MODES}
    gt_support = {m: gt3[m][1] for m in EXP3.MODES}

    rows = []
    for run in _run_ids():
        for k in topks:
            path = RUNS / run / f"topk_{k}" / "retrieval_results.json"
            results = json.load(open(path))
            cfg = _cfg(run, k)
            kind = cfg.get("kind", "fusion")

            exp1 = SUMM.summarize_one(results, k)
            # per-dataset recall (cross-dataset behavior)
            per_ds = {}
            for ds in datasets:
                sub = [r for r in results if str(r.get("query", {}).get("dataset")) == ds]
                per_ds[ds] = SUMM.summarize_one(sub, k)[f"Recall@{k}"] if sub else float("nan")

            e2, _ = EXP2.evaluate_one_method(run, path, test_pairs, k)
            e3, _ = EXP3.evaluate_method(run, path, k, gt_counts, gt_support)
            jp2, yaw2, yb2, sc2 = (_mode_row(e2, m) for m in ("joint_pose", "yaw", "ybin", "scale_bin"))
            jp3, yaw3, sc3 = (_mode_row(e3, m) for m in ("joint_pose", "yaw", "scale_bin"))

            row = {
                "run_id": run, "K": k,
                "fusion_type": run if kind == "fusion" else "A0",
                "context_type": run if kind == "context" else "empirical_v1",
                "neural_training_type": run if kind == "neural_training" else "v1",
                "params_added": cfg.get("params_added", 0),
                "keep": cfg.get("keep", True),
                "train_time_s": cfg.get("train_time_s", 0.0),
                "infer_ms_per_query": cfg.get("infer_ms_per_query", ""),
                "peak_gpu_mb": cfg.get("peak_gpu_mb", ""),
                f"Recall": round(exp1[f"Recall@{k}"], 4),
                f"NDCG": round(exp1[f"NDCG@{k}"], 4),
                f"PS": round(exp1[f"PS@{k}"], 4),
                "joint_js": round(jp2.get("mean_js_divergence", float("nan")), 4),
                "joint_gap": round(jp2.get("mean_entropy_gap", float("nan")), 4),
                "joint_entropy": round(jp2.get("mean_retrieved_entropy", float("nan")), 4),
                "yaw_js": round(yaw2.get("mean_js_divergence", float("nan")), 4),
                "ybin_js": round(yb2.get("mean_js_divergence", float("nan")), 4),
                "scale_js": round(sc2.get("mean_js_divergence", float("nan")), 4),
                "joint_ll": round(jp3.get("mean_log_likelihood", float("nan")), 4),
                "joint_violation": round(jp3.get("violation_rate", float("nan")), 4),
                "yaw_violation": round(yaw3.get("violation_rate", float("nan")), 4),
                "scale_violation": round(sc3.get("violation_rate", float("nan")), 4),
            }
            for ds in datasets:
                row[f"recall_{ds}"] = round(per_ds[ds], 4) if per_ds[ds] == per_ds[ds] else ""
            rows.append(row)
            print(f"  scored {run} K={k}: R={row['Recall']} NDCG={row['NDCG']} "
                  f"jointLL={row['joint_ll']} jointViol={row['joint_violation']}")

    # write CSV
    fields = list(rows[0].keys()) if rows else []
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[eval_full] wrote {OUT_CSV}  ({len(rows)} rows, {len(datasets)} datasets)")
    return rows, datasets


if __name__ == "__main__":
    evaluate_all()
