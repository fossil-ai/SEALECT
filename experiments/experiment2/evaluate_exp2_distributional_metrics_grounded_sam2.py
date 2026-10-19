#!/usr/bin/env python3
"""
evaluate_exp2_distributional_metrics_grounded_sam2.py

Experiment 2: distributional fidelity / diversity of retrieved maritime vessel candidates.

Adapts the old Exp. 2 script to the new Experiment 1 layout:

  exp1_retrieval_quality/<method_name>/topk_20/retrieval_results.json
  exp1_retrieval_quality/<method_name>/topk_100/retrieval_results.json

It evaluates how closely retrieved candidate distributions match the held-out
test distribution within each scene context_key.

Metrics:
  JS       mean Jensen-Shannon divergence between GT and retrieved distributions
  H_gap   mean absolute entropy gap |H_retrieved - H_gt|
  H_gt     mean GT entropy
  H_ret    mean retrieved entropy
  ctx      number of evaluated context bins

Modes:
  yaw, joint_pose, pose_family, ybin, scale_bin
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 
# CONFIG
# 

ROOT = Path("/path/to/sealect_root/experiments")
DATA_ROOT = ROOT / "dataset_decide"
EXP1_ROOT = ROOT / "experiment1" / "exp1_retrieval_quality"

TEST_PAIRS_JSON = (
    DATA_ROOT
    / "splits_dino_decorrelated_kbalanced_grounded_sam2"
    / "test_scene_object_pairs.json"
)

TOP_KS = [20, 100]
OUT_ROOT = ROOT / "experiment2" / "distributional_metrics_grounded_sam2"

METHOD_NAMES = [
    "random_k20",
    "random_k100",
    "clip_k20",
    "clip_k100",
    "dino_k20",
    "dino_k100",
    "prior_only_k20",
    "prior_only_k100",
    "clip_prior_k20",
    "clip_prior_k100",
    "dino_prior_k20",
    "dino_prior_k100",
    "neural_only_dinofeat_k20",
    "neural_only_dinofeat_k100",
    "sealect_dinofeat_k20",
    "sealect_dinofeat_k100",
    "discofos_ft_k20",
    "discofos_ft_k100",
    "neural_gala_k20",
    "neural_gala_k100",
    "gala_only_k20",
    "gala_only_k100",
    "sweep_neural_only",
    "sweep_context_only",
    "sweep_obs_only",
    "sweep_ln1_lc1_lo0p10",
    "sweep_ln1_lc1_lo0p50",
    "sweep_ln1_lc1_lo1p00",
]

AUTO_INCLUDE_ALL_SWEEPS = True

MIN_GT_PER_CONTEXT = 20
MIN_RETRIEVED_PER_CONTEXT = 20

MODES = ["yaw", "joint_pose", "pose_family", "ybin", "scale_bin"]

EPS = 1e-12

# 


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def context_key(row: Dict[str, Any]) -> str:
    return str(row.get("context_key", "missing_context_key"))


def token_for_mode(row: Dict[str, Any], mode: str) -> str:
    if mode == "yaw":
        return str(int(row["yaw_bin"]))

    if mode == "pitch":
        return str(int(row["pitch_bin"]))

    if mode == "roll":
        return str(int(row["roll_bin"]))

    if mode == "joint_pose":
        return f"{int(row['pitch_bin'])}_{int(row['yaw_bin'])}_{int(row['roll_bin'])}"

    if mode == "pose_family":
        if row.get("pose_family") is not None:
            return str(row["pose_family"])
        pitch_family = row.get("pitch_family", "missing_pitch")
        yaw_family = row.get("yaw_family", "missing_yaw")
        roll_family = row.get("roll_family", "missing_roll")
        return f"{pitch_family}__{yaw_family}__{roll_family}"

    if mode == "ybin":
        return str(int(row["y_bin"]))

    if mode == "scale_bin":
        area = float(row.get("area_frac", 0.0))
        edges = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
        for i, edge in enumerate(edges):
            if area <= edge:
                return str(i)
        return str(len(edges))

    raise ValueError(f"Unknown mode: {mode}")


def normalize_counter(counter: Counter, support: List[str]) -> np.ndarray:
    arr = np.array([float(counter.get(s, 0.0)) for s in support], dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.ones(len(support), dtype=np.float64) / max(1, len(support))
    return arr / total


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * (np.log(p) / np.log(base))).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float((p[mask] * ((np.log(p[mask] + EPS) - np.log(q[mask] + EPS)) / np.log(base))).sum())


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base)


def build_ground_truth_distribution(test_pairs: List[Dict[str, Any]], mode: str) -> Dict[str, Counter]:
    out = defaultdict(Counter)
    for row in test_pairs:
        out[context_key(row)][token_for_mode(row, mode)] += 1
    return out


def build_retrieved_distribution(results: List[Dict[str, Any]], mode: str, top_k: int) -> Dict[str, Counter]:
    out = defaultdict(Counter)
    for item in results:
        query = item.get("query", {})
        k = context_key(query)
        for cand in item.get("topk", [])[:top_k]:
            try:
                tok = token_for_mode(cand, mode)
            except Exception:
                continue
            out[k][tok] += 1
    return out


def count_queries_by_context(results: List[Dict[str, Any]]) -> Counter:
    counts = Counter()
    for item in results:
        counts[context_key(item.get("query", {}))] += 1
    return counts


def clean_method_name(method_dir: str, top_k: int) -> str:
    suffix = f"_k{top_k}"
    if method_dir.endswith(suffix):
        return method_dir[: -len(suffix)]
    return method_dir


def result_path_for(method_dir: str, top_k: int) -> Path:
    return EXP1_ROOT / method_dir / f"topk_{top_k}" / "retrieval_results.json"


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


def method_sort_key(method_dir: str):
    clean = clean_method_name(clean_method_name(method_dir, 20), 100)
    if clean in METHOD_ORDER:
        return (METHOD_ORDER.index(clean), clean)
    if clean.startswith("sweep_"):
        return (len(METHOD_ORDER), clean)
    return (len(METHOD_ORDER) + 1, clean)


def discover_methods(top_k: int) -> List[str]:
    names = set()

    for m in METHOD_NAMES:
        if result_path_for(m, top_k).exists():
            names.add(m)

    if AUTO_INCLUDE_ALL_SWEEPS:
        for p in sorted(EXP1_ROOT.glob(f"sweep_*/topk_{top_k}/retrieval_results.json")):
            names.add(p.parent.parent.name)

    return sorted(names, key=method_sort_key)


def evaluate_one_method(
    method_dir: str,
    result_path: Path,
    test_pairs: List[Dict[str, Any]],
    top_k: int,
):
    results = load_json(result_path)
    query_counts = count_queries_by_context(results)
    method = clean_method_name(method_dir, top_k)

    summary_rows = []
    by_context_rows = []

    for mode in MODES:
        gt_by_context = build_ground_truth_distribution(test_pairs, mode)
        retrieved_by_context = build_retrieved_distribution(results, mode, top_k)
        context_keys = sorted(set(gt_by_context.keys()) & set(retrieved_by_context.keys()))

        js_values = []
        entropy_gaps = []
        gt_entropies = []
        retrieved_entropies = []
        gt_counts = []
        retrieved_counts = []

        for k in context_keys:
            gt_counter = gt_by_context[k]
            retrieved_counter = retrieved_by_context[k]

            num_gt = int(sum(gt_counter.values()))
            num_retrieved = int(sum(retrieved_counter.values()))

            if num_gt < MIN_GT_PER_CONTEXT:
                continue
            if num_retrieved < MIN_RETRIEVED_PER_CONTEXT:
                continue

            support = sorted(set(gt_counter.keys()) | set(retrieved_counter.keys()))
            p_gt = normalize_counter(gt_counter, support)
            p_ret = normalize_counter(retrieved_counter, support)

            js = js_divergence(p_gt, p_ret, base=2.0)
            h_gt = entropy(p_gt, base=2.0)
            h_ret = entropy(p_ret, base=2.0)
            h_gap = abs(h_gt - h_ret)

            js_values.append(js)
            entropy_gaps.append(h_gap)
            gt_entropies.append(h_gt)
            retrieved_entropies.append(h_ret)
            gt_counts.append(num_gt)
            retrieved_counts.append(num_retrieved)

            by_context_rows.append({
                "method": method,
                "method_dir": method_dir,
                "mode": mode,
                "top_k": top_k,
                "context_key": k,
                "num_gt": num_gt,
                "num_retrieved": num_retrieved,
                "num_queries": int(query_counts.get(k, 0)),
                "support_size": len(support),
                "js_divergence": js,
                "gt_entropy": h_gt,
                "retrieved_entropy": h_ret,
                "entropy_gap": h_gap,
                "retrieved_minus_gt_entropy": h_ret - h_gt,
            })

        summary_rows.append({
            "method": method,
            "method_dir": method_dir,
            "mode": mode,
            "top_k": top_k,
            "num_contexts_eval": len(js_values),
            "mean_js_divergence": float(np.mean(js_values)) if js_values else float("nan"),
            "std_js_divergence": float(np.std(js_values)) if js_values else float("nan"),
            "mean_gt_entropy": float(np.mean(gt_entropies)) if gt_entropies else float("nan"),
            "mean_retrieved_entropy": float(np.mean(retrieved_entropies)) if retrieved_entropies else float("nan"),
            "mean_entropy_gap": float(np.mean(entropy_gaps)) if entropy_gaps else float("nan"),
            "std_entropy_gap": float(np.std(entropy_gaps)) if entropy_gaps else float("nan"),
            "mean_num_gt_per_context": float(np.mean(gt_counts)) if gt_counts else float("nan"),
            "mean_num_retrieved_per_context": float(np.mean(retrieved_counts)) if retrieved_counts else float("nan"),
        })

    return summary_rows, by_context_rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not TEST_PAIRS_JSON.exists():
        raise FileNotFoundError(f"Missing test pairs: {TEST_PAIRS_JSON}")
    if not EXP1_ROOT.exists():
        raise FileNotFoundError(f"Missing Exp. 1 root: {EXP1_ROOT}")

    test_pairs = load_json(TEST_PAIRS_JSON)

    all_summary_rows = []
    all_by_context_rows = []
    missing = []

    print("Evaluating Exp. 2 distributional metrics")
    print(f"Exp1 root: {EXP1_ROOT}")
    print(f"Test pairs: {TEST_PAIRS_JSON}")

    for top_k in TOP_KS:
        out_dir = OUT_ROOT / f"k{top_k}"
        out_dir.mkdir(parents=True, exist_ok=True)

        methods = discover_methods(top_k)
        print(f"\nTOP_K={top_k}; methods found: {len(methods)}")

        summary_rows = []
        by_context_rows = []

        for method_dir in methods:
            path = result_path_for(method_dir, top_k)
            if not path.exists():
                missing.append({"method_dir": method_dir, "top_k": top_k, "path": str(path)})
                continue

            print(f"  Evaluating {method_dir}: {path}")
            s_rows, c_rows = evaluate_one_method(method_dir, path, test_pairs, top_k)
            summary_rows.extend(s_rows)
            by_context_rows.extend(c_rows)

        save_json(summary_rows, out_dir / "distributional_metrics_summary.json")
        save_json(by_context_rows, out_dir / "distributional_metrics_by_context.json")
        save_csv(summary_rows, out_dir / "distributional_metrics_summary.csv")
        save_csv(by_context_rows, out_dir / "distributional_metrics_by_context.csv")

        all_summary_rows.extend(summary_rows)
        all_by_context_rows.extend(by_context_rows)

        print(f"  Saved: {out_dir / 'distributional_metrics_summary.csv'}")

    save_json(all_summary_rows, OUT_ROOT / "distributional_metrics_summary_all.json")
    save_json(all_by_context_rows, OUT_ROOT / "distributional_metrics_by_context_all.json")
    save_csv(all_summary_rows, OUT_ROOT / "distributional_metrics_summary_all.csv")
    save_csv(all_by_context_rows, OUT_ROOT / "distributional_metrics_by_context_all.csv")

    if missing:
        save_json(missing, OUT_ROOT / "missing_methods.json")
        print("\nMissing files:")
        for item in missing:
            print(f"  {item['method_dir']} K={item['top_k']}: {item['path']}")

    print("\nDone.")
    print(f"All summary: {OUT_ROOT / 'distributional_metrics_summary_all.csv'}")


if __name__ == "__main__":
    main()
