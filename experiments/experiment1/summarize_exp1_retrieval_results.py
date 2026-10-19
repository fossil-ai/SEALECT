#!/usr/bin/env python3
"""
summarize_exp1_retrieval_results.py

Aggregate SEALECT Experiment 1 retrieval result JSONs into paper-reportable metrics.

Edit paths at the top, then run directly from PyCharm.

Expected result layout from run_retrieval_eval_all_methods.py:

EXP_ROOT/
  random_k20/topk_20/retrieval_results.json
  random_k100/topk_100/retrieval_results.json
  prior_only_k20/topk_20/retrieval_results.json
  ...

Outputs:
  EXP_ROOT/summary_exp1_retrieval_quality.csv
  EXP_ROOT/summary_exp1_retrieval_quality.json
  EXP_ROOT/summary_exp1_retrieval_quality.md

Metrics:
  - Recall@K: at least one candidate reaches RECALL_THRESHOLD relevance
  - NDCG@K: graded ranking quality using pose/placement/scale relevance
  - PS@K: mean plausibility score over top-K candidates
  - BestRel@K: mean best relevance in the top-K
  - PoseDiv@K: normalized entropy over pose keys in top-K
  - YawDiv@K: normalized entropy over yaw bins in top-K
  - SeqUnique@K: fraction of unique sequence groups in top-K
  - CandUnique@K: fraction of unique candidate IDs in top-K
"""

import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# 
# EDIT THESE ONLY
# 
EXP_ROOT = Path("/path/to/sealect_root/experiments/experiment1/exp1_retrieval_quality")

# Evaluate these K values. The script looks for */topk_{K}/retrieval_results.json.
EVAL_KS = [20, 100]

# Relevance settings. Keep these fixed for the paper unless intentionally ablated.
PITCH_TOL = 0
YAW_TOL = 1
YBIN_TOL = 0
AREA_LOG_TOL = 0.30
RECALL_THRESHOLD = 4

# Optional: only include methods with these substrings/order. Leave empty to auto-discover all.
METHOD_ORDER_HINTS = [
    # "random",
    # "prior_only",
    # "dino",
    # "clip",
    # "clip_prior",
    # "dino_prior",
    # "neural_only"
    # "sealect",
    "sweep"
]
# 

EPS = 1e-8


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_area(x: Any) -> float:
    return math.log(max(float(x), EPS))


def circular_bin_distance(a: int, b: int, n: int = 18) -> int:
    d = abs(int(a) - int(b))
    return min(d, n - d)


def relevance(query: Dict, cand: Dict) -> int:
    """Graded relevance in {0,1,2,3,4}.

    Components:
      1. pitch bin match
      2. yaw bin near-match with circular tolerance
      3. vertical placement bin match
      4. log-area/scale match
    """
    score = 0

    if abs(int(query["pitch_bin"]) - int(cand["pitch_bin"])) <= PITCH_TOL:
        score += 1

    if circular_bin_distance(int(query["yaw_bin"]), int(cand["yaw_bin"])) <= YAW_TOL:
        score += 1

    if abs(int(query["y_bin"]) - int(cand["y_bin"])) <= YBIN_TOL:
        score += 1

    if abs(log_area(query["area_frac"]) - log_area(cand["area_frac"])) <= AREA_LOG_TOL:
        score += 1

    return score


def dcg(scores: List[int]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg(scores: List[int]) -> float:
    ideal = sorted(scores, reverse=True)
    denom = dcg(ideal)
    return dcg(scores) / denom if denom > 0 else 0.0


def entropy_from_counts(counts: Counter, normalize_by: Optional[int] = None) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = v / total
        h -= p * math.log(max(p, EPS))
    if normalize_by is not None and normalize_by > 1:
        return h / math.log(normalize_by)
    if len(counts) > 1:
        return h / math.log(len(counts))
    return 0.0


def pose_key(r: Dict) -> str:
    return f"{int(r['pitch_bin'])}_{int(r['yaw_bin'])}_{int(r['roll_bin'])}"


def summarize_one(results: List[Dict], eval_k: int) -> Dict[str, float]:
    recalls = []
    ndcgs = []
    ps_scores = []
    best_rels = []
    pose_divs = []
    yaw_divs = []
    seq_unique_fracs = []
    cand_unique_fracs = []
    context_match_fracs = []
    viewpoint_match_fracs = []

    used_queries = 0
    short_queries = 0

    for row in results:
        query = row.get("query", {})
        cands = row.get("topk", [])[:eval_k]
        if not cands:
            continue

        if len(cands) < eval_k:
            short_queries += 1

        rels = [relevance(query, c) for c in cands]
        recalls.append(1.0 if max(rels) >= RECALL_THRESHOLD else 0.0)
        ndcgs.append(ndcg(rels))
        ps_scores.append(sum(rels) / max(1, len(rels)))
        best_rels.append(max(rels))

        pose_counts = Counter(pose_key(c) for c in cands)
        yaw_counts = Counter(int(c["yaw_bin"]) for c in cands)
        pose_divs.append(entropy_from_counts(pose_counts))
        yaw_divs.append(entropy_from_counts(yaw_counts, normalize_by=18))

        seqs = [str(c.get("sequence_group", "")) for c in cands]
        cids = [str(c.get("candidate_id", "")) for c in cands]
        seq_unique_fracs.append(len(set(seqs)) / max(1, len(seqs)))
        cand_unique_fracs.append(len(set(cids)) / max(1, len(cids)))

        q_context = query.get("context_key")
        q_vp = query.get("viewpoint_type")
        if q_context is not None:
            context_match_fracs.append(
                sum(1 for c in cands if c.get("context_key") == q_context) / max(1, len(cands))
            )
        if q_vp is not None:
            viewpoint_match_fracs.append(
                sum(1 for c in cands if c.get("viewpoint_type") == q_vp) / max(1, len(cands))
            )

        used_queries += 1

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    out = {
        "queries": used_queries,
        "short_queries": short_queries,
        f"Recall@{eval_k}": mean(recalls),
        f"NDCG@{eval_k}": mean(ndcgs),
        f"PS@{eval_k}": mean(ps_scores),
        f"BestRel@{eval_k}": mean(best_rels),
        f"PoseDiv@{eval_k}": mean(pose_divs),
        f"YawDiv@{eval_k}": mean(yaw_divs),
        f"SeqUnique@{eval_k}": mean(seq_unique_fracs),
        f"CandUnique@{eval_k}": mean(cand_unique_fracs),
    }

    if context_match_fracs:
        out[f"ContextMatch@{eval_k}"] = mean(context_match_fracs)
    if viewpoint_match_fracs:
        out[f"ViewpointMatch@{eval_k}"] = mean(viewpoint_match_fracs)

    return out


def discover_result_files(exp_root: Path, eval_ks: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for k in eval_ks:
        for path in sorted(exp_root.glob(f"*/topk_{k}/retrieval_results.json")):
            method = path.parent.parent.name
            rows.append({"method": method, "K": k, "path": path})
    return rows


def method_sort_key(method: str) -> tuple:
    lower = method.lower()
    for i, hint in enumerate(METHOD_ORDER_HINTS):
        if lower.startswith(hint) or hint in lower:
            return (i, lower)
    return (len(METHOD_ORDER_HINTS), lower)


def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        path.write_text("No results found.\n")
        return

    # Compact paper-facing columns.
    lines = []
    for k in EVAL_KS:
        subset = [r for r in rows if r["K"] == k]
        if not subset:
            continue
        lines.append(f"## Retrieval Quality @ K={k}\n")
        cols = [
            "method",
            f"Recall@{k}",
            f"NDCG@{k}",
            f"PS@{k}",
            f"BestRel@{k}",
            f"PoseDiv@{k}",
            f"YawDiv@{k}",
            f"SeqUnique@{k}",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in subset:
            lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
        lines.append("")

    path.write_text("\n".join(lines))


def print_interpretation(rows: List[Dict[str, Any]]):
    print("\n=== Experiment 1 Summary ===\n")
    for k in EVAL_KS:
        subset = [r for r in rows if r["K"] == k]
        if not subset:
            continue
        print(f"K={k}")
        ranked = sorted(subset, key=lambda r: r.get(f"NDCG@{k}", 0.0), reverse=True)
        for r in ranked:
            print(
                f"  {r['method']:28s} "
                f"Recall={r.get(f'Recall@{k}', 0.0):.3f}  "
                f"NDCG={r.get(f'NDCG@{k}', 0.0):.3f}  "
                f"PS={r.get(f'PS@{k}', 0.0):.3f}  "
                f"PoseDiv={r.get(f'PoseDiv@{k}', 0.0):.3f}  "
                f"SeqUnique={r.get(f'SeqUnique@{k}', 0.0):.3f}"
            )
        best = ranked[0]
        print(f"  -> Best by NDCG@{k}: {best['method']}\n")


def main():
    if not EXP_ROOT.exists():
        raise FileNotFoundError(f"EXP_ROOT does not exist: {EXP_ROOT}")

    files = discover_result_files(EXP_ROOT, EVAL_KS)
    if not files:
        raise RuntimeError(f"No retrieval_results.json files found under: {EXP_ROOT}")

    rows = []
    for item in files:
        method = item["method"]
        k = int(item["K"])
        path = item["path"]
        print(f"Reading {method} @ K={k}: {path}")
        results = load_json(path)
        metrics = summarize_one(results, k)
        rows.append({"method": method, "K": k, "path": str(path), **metrics})

    rows = sorted(rows, key=lambda r: (r["K"], method_sort_key(r["method"])))

    csv_path = EXP_ROOT / "summary_exp1_retrieval_quality.csv"
    json_path = EXP_ROOT / "summary_exp1_retrieval_quality.json"
    md_path = EXP_ROOT / "summary_exp1_retrieval_quality.md"

    write_csv(rows, csv_path)
    save_json(rows, json_path)
    write_markdown(rows, md_path)
    print_interpretation(rows)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
