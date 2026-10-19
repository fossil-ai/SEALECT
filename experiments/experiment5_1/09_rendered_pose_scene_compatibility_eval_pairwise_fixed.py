#!/usr/bin/env python3
"""
09_rendered_pose_scene_compatibility_eval.py

Rendered pose/scene compatibility evaluation for Experiment 5.

Purpose:
  Evaluate BootPlace + DiffHarmony rendered composites using the same
  pose/scene plausibility logic as the paper's retrieval experiments, rather
  than a real-vs-fake artifact classifier.

Inputs expected:
  - Experiment 1 retrieval_results.json files with query/candidate metadata
  - test_scene_object_pairs.json to estimate context-conditioned GT priors
  - bootplace_outputs_clean/k*/method/placement_metadata.json
  - bootplace_outputs_clean/k*/method/postprocess_overlap_viz_report.json
  - diffharmony_outputs/k*/method/*.png

Outputs:
  rendered_pose_scene_eval/
    ranked_rows.csv/json
    method_summary.csv/json
    pairwise_win_rates.csv/json
    summary.json

Scoring:
  score = weighted normalized log-likelihood over:
    yaw, joint_pose, ybin, scale_bin
  minus overlap/rendering penalties.

Notes:
  This script is intentionally metadata-driven. It does not train a visual
  classifier and does not use human evaluation. It tests the SEALECT claim:
  retrieved candidates should be more pose/scene compatible after controlled
  BootPlace placement + DiffHarmony harmonization.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 
# CONFIG
# 

ROOT = Path("/path/to/sealect_root/experiments")
EXP5_ROOT = ROOT / "experiment5_1"

# Ground-truth test-pair metadata used by old Exp2/Exp3 priors.
TEST_PAIRS_JSON = (
    ROOT
    / "dataset_decide"
    / "splits_dino_decorrelated_kbalanced_grounded_sam2"
    / "test_scene_object_pairs.json"
)

# Exp1 retrieval outputs. Edit if your directory names differ.
EXP1_ROOT = ROOT / "experiment1" / "exp1_retrieval_quality"

BOOTPLACE_OUT_ROOT = EXP5_ROOT / "bootplace_outputs_clean"
DIFFHARMONY_OUT_ROOT = EXP5_ROOT / "diffharmony_outputs"
OUT_ROOT = EXP5_ROOT / "rendered_pose_scene_eval"

TOPKS = [100]
METHODS = ["random", "clip", "dino", "prior_only", "dino_prior", "sealect", "gala", "discofos"]

# Map Experiment 5 method names to Exp1 retrieval directories.
# The script will try several fallbacks too.
EXP1_METHOD_CANDIDATES = {
    "random": ["random_k{K}", "random"],
    "clip": ["clip_k{K}", "clip"],
    "dino": ["dino_k{K}", "dino"],
    "prior_only": ["prior_only_k{K}", "prior_only"],
    "dino_prior": ["dino_prior_k{K}", "prior_w_dino", "dino_prior"],
    "sealect": ["sealect_dinofeat_k{K}", "sealect", "sealect_1_1"],
    "gala": ["gala_render_k{K}"],
    "discofos": ["discofos_render_k{K}"],
}

# If retrieval_results include topk lists, choose which selected candidate
# corresponds to the rendered request. Usually 0 because 02 used candidate-rank 1.
RENDERED_CANDIDATE_INDEX = 0

ALPHA = 0.5
EPS = 1e-12
LOW_PROB_THRESHOLD = 0.01

# Weights are intentionally interpretable and match paper axes.
WEIGHTS = {
    "joint_pose": 0.35,
    "ybin": 0.25,
    "scale_bin": 0.25,
    "yaw": 0.15,
}

# Penalties. Keep mild so compatibility dominates.
OVERLAP_REJECT_PENALTY = 0.50
MISSING_RENDER_PENALTY = 0.75
LOW_PROB_VIOLATION_PENALTY = 0.05

# 
# IO
# 


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stem_key(x: Any) -> str:
    return Path(str(x)).stem

# 
# Old Exp2/Exp3-compatible metadata functions
# 


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
        return f"{row.get('pitch_family', 'missing_pitch')}__{row.get('yaw_family', 'missing_yaw')}__{row.get('roll_family', 'missing_roll')}"
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


def build_gt_counts(test_pairs: List[Dict[str, Any]], mode: str):
    counts_by_context = defaultdict(Counter)
    support = set()
    for row in test_pairs:
        try:
            k = context_key(row)
            tok = token_for_mode(row, mode)
        except Exception:
            continue
        counts_by_context[k][tok] += 1
        support.add(tok)
    return counts_by_context, sorted(support)


def smoothed_probability(counter: Counter, token: str, support: List[str], alpha: float = ALPHA) -> float:
    n = sum(counter.values())
    v = len(support)
    if v <= 0:
        return 0.0
    return (counter.get(token, 0) + alpha) / (n + alpha * v)


def normalize_logp(logp: float, support_size: int) -> float:
    """Map log probability to roughly [0,1], where uniform support ~0."""
    if support_size <= 1:
        return 1.0
    # logp ranges roughly [-log(V), 0]. Normalize against uniform likelihood.
    denom = math.log(support_size + EPS)
    return max(0.0, min(1.0, 1.0 + (logp / max(EPS, denom))))

# 
# Retrieval result locating and indexing
# 


def candidate_result_paths(method: str, k: int) -> List[Path]:
    out = []
    for tmpl in EXP1_METHOD_CANDIDATES.get(method, [method]):
        name = tmpl.replace("{K}", str(k))
        out.extend([
            EXP1_ROOT / name / f"topk_{k}" / "retrieval_results.json",
            EXP1_ROOT / name / "retrieval_results.json",
            EXP1_ROOT / f"{name}_k{k}" / f"topk_{k}" / "retrieval_results.json",
        ])
    # De-duplicate preserving order.
    seen = set()
    uniq = []
    for p in out:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq


def find_retrieval_results(method: str, k: int) -> Optional[Path]:
    for p in candidate_result_paths(method, k):
        if p.exists():
            return p
    return None


def request_base_id(request_id: str) -> str:
    return request_id.split("_q")[0]


def query_group_key_from_request(request_id: str, cand: Dict[str, Any], ordinal: int) -> str:
    """Return a method-independent key for pairwise matching across retrieval methods.

    Preferred key is the original query image + annotation identity from retrieval metadata.
    This aligns random/clip/dino/prior/sealect outputs for the same underlying query.
    If those fields are unavailable, fall back to the numeric request ordinal parsed from
    method-specific request ids such as `sealect_00188_q...` or `prior_only_00018_q...`.
    """
    q_img = cand.get("query_image_id") or cand.get("query_id") or cand.get("query_image")
    q_ann = cand.get("query_annotation_id") or cand.get("query_ann_id") or cand.get("annotation_id")
    if q_img is not None and q_ann is not None:
        return f"query::{q_img}::{q_ann}"
    if q_img is not None:
        return f"query_image::{q_img}"

    base = request_base_id(request_id)
    parts = base.split("_")
    # method names can contain underscores, but the final token before _q is the zero-padded ordinal.
    if parts and parts[-1].isdigit():
        return f"ordinal::{int(parts[-1])}"
    return f"ordinal::{ordinal}"


def build_retrieval_index(results: List[Dict[str, Any]], method: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a loose index from request ids to the rendered candidate metadata.

    This supports multiple possible schemas. If exact request names are not in
    retrieval_results, it falls back to sequential order. The sequential fallback
    is okay only if 02 wrote requests in the same order as Exp1 results.
    """
    idx = {}
    for i, item in enumerate(results):
        query = item.get("query", {}) if isinstance(item, dict) else {}
        topk = item.get("topk", []) if isinstance(item, dict) else []
        cand = topk[RENDERED_CANDIDATE_INDEX] if len(topk) > RENDERED_CANDIDATE_INDEX else {}
        if not isinstance(cand, dict):
            cand = {}
        merged = dict(query)
        merged.update({f"query_{k}": v for k, v in query.items()})
        merged.update(cand)

        possible_keys = [
            item.get("request_id"),
            item.get("file_name"),
            query.get("request_id"),
            query.get("file_name"),
            query.get("image_id"),
            query.get("id"),
            f"{method}_{i:05d}",
            f"{method}_{i:04d}",
            f"{method}_{i:03d}",
        ]
        for key in possible_keys:
            if key is None:
                continue
            idx[stem_key(key)] = merged
    return idx

# 
# BootPlace/DiffHarmony metadata
# 


def load_placement_rows(k: int, method: str) -> List[Dict[str, Any]]:
    p = BOOTPLACE_OUT_ROOT / f"k{k}" / method / "placement_metadata.json"
    if not p.exists():
        return []
    rows = load_json(p)
    return rows if isinstance(rows, list) else []


def load_rejected_set(k: int, method: str) -> set:
    p = BOOTPLACE_OUT_ROOT / f"k{k}" / method / "postprocess_overlap_viz_report.json"
    rejected = set()
    if not p.exists():
        return rejected
    try:
        rep = load_json(p)
        for item in rep.get("items", []):
            if item.get("rejected_overlap"):
                rid = item.get("request_id") or item.get("file_name")
                if rid:
                    rejected.add(stem_key(rid))
    except Exception:
        pass
    return rejected


def find_harmonized(k: int, method: str, request_id: str) -> Optional[Path]:
    dh = DIFFHARMONY_OUT_ROOT / f"k{k}" / method
    base = request_base_id(request_id)
    candidates = [
        dh / f"{base}_1_1.png",
        dh / f"{request_id}_1_1.png",
        dh / f"{base}.png",
        dh / f"{request_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    if dh.exists():
        wanted = {base, f"{base}_1_1", request_id, f"{request_id}_1_1"}
        for p in dh.glob("*.png"):
            if p.stem in wanted:
                return p
    return None

# 
# Scoring
# 


def score_candidate(
    cand: Dict[str, Any],
    gt_counts_by_mode: Dict[str, Dict[str, Counter]],
    support_by_mode: Dict[str, List[str]],
) -> Dict[str, Any]:
    kctx = context_key(cand)
    terms = {}
    weighted = 0.0
    violations = 0
    valid_terms = 0

    for mode, weight in WEIGHTS.items():
        try:
            tok = token_for_mode(cand, mode)
            counts = gt_counts_by_mode[mode].get(kctx)
            support = support_by_mode[mode]
            if counts is None:
                raise KeyError(f"missing context {kctx}")
            p = smoothed_probability(counts, tok, support, ALPHA)
            logp = math.log(p + EPS)
            norm = normalize_logp(logp, len(support))
            violation = int(p < LOW_PROB_THRESHOLD)
        except Exception:
            tok = None
            p = float("nan")
            logp = float("nan")
            norm = 0.0
            violation = 1
        terms[f"{mode}_token"] = tok
        terms[f"{mode}_prob"] = p
        terms[f"{mode}_logp"] = logp
        terms[f"{mode}_norm"] = norm
        terms[f"{mode}_violation"] = violation
        weighted += weight * norm
        violations += violation
        valid_terms += int(tok is not None)

    penalty = LOW_PROB_VIOLATION_PENALTY * violations
    score = weighted - penalty
    terms.update({
        "context_key": kctx,
        "weighted_prior_score": weighted,
        "num_low_prob_violations": violations,
        "valid_metadata_terms": valid_terms,
        "metadata_penalty": penalty,
        "metadata_score": score,
    })
    return terms


def collect_rows():
    if not TEST_PAIRS_JSON.exists():
        raise FileNotFoundError(f"Missing TEST_PAIRS_JSON: {TEST_PAIRS_JSON}")

    test_pairs = load_json(TEST_PAIRS_JSON)
    gt_counts_by_mode = {}
    support_by_mode = {}
    for mode in WEIGHTS:
        counts, support = build_gt_counts(test_pairs, mode)
        gt_counts_by_mode[mode] = counts
        support_by_mode[mode] = support

    all_rows = []
    missing_retrieval = []

    for k in TOPKS:
        retrieval_indices = {}
        for method in METHODS:
            rp = find_retrieval_results(method, k)
            if rp is None:
                missing_retrieval.append({
                    "k": k,
                    "method": method,
                    "tried": [str(p) for p in candidate_result_paths(method, k)],
                })
                retrieval_indices[method] = {}
                continue
            results = load_json(rp)
            retrieval_indices[method] = build_retrieval_index(results, method)
            print(f"Loaded retrieval metadata: k{k}/{method}: {rp} ({len(retrieval_indices[method])} indexed)")

        for method in METHODS:
            placement_rows = load_placement_rows(k, method)
            rejected = load_rejected_set(k, method)
            idx = retrieval_indices.get(method, {})
            for ordinal, row in enumerate(placement_rows):
                name = row.get("file_name") or row.get("request_id") or row.get("image_id")
                if not name:
                    continue
                request_id = stem_key(name)
                base = request_base_id(request_id)

                cand = (
                    idx.get(request_id)
                    or idx.get(base)
                    or idx.get(f"{method}_{ordinal:05d}")
                    or idx.get(f"{method}_{ordinal:04d}")
                    or idx.get(f"{method}_{ordinal:03d}")
                )
                if cand is None:
                    cand = {}

                score_terms = score_candidate(cand, gt_counts_by_mode, support_by_mode)

                harmonized_path = find_harmonized(k, method, request_id)
                has_render = harmonized_path is not None
                rejected_overlap = request_id in rejected

                final_score = score_terms["metadata_score"]
                if rejected_overlap:
                    final_score -= OVERLAP_REJECT_PENALTY
                if not has_render:
                    final_score -= MISSING_RENDER_PENALTY

                query_group_key = query_group_key_from_request(request_id, cand, ordinal)

                all_rows.append({
                    "k": k,
                    "method": method,
                    "request_id": request_id,
                    "base_id": base,
                    "query_group_key": query_group_key,
                    "ordinal": ordinal,
                    "has_harmonized_render": bool(has_render),
                    "harmonized_path": str(harmonized_path) if harmonized_path else None,
                    "rejected_overlap": bool(rejected_overlap),
                    "final_score": float(final_score),
                    "overlap_penalty_applied": OVERLAP_REJECT_PENALTY if rejected_overlap else 0.0,
                    "missing_render_penalty_applied": MISSING_RENDER_PENALTY if not has_render else 0.0,
                    "paste_bbox_xyxy": row.get("paste_bbox_xyxy"),
                    **score_terms,
                })

    if missing_retrieval:
        save_json(missing_retrieval, OUT_ROOT / "missing_retrieval_metadata.json")
    return all_rows

# 
# Ranking summaries
# 


def rank_within_query(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["k"], r.get("query_group_key") or r["base_id"])].append(r)

    ranked = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda x: x["final_score"], reverse=True)
        for rank, r in enumerate(items, start=1):
            rr = dict(r)
            rr["rank_within_query"] = rank
            rr["num_methods_for_query"] = len(items)
            rr["is_top1"] = int(rank == 1)
            ranked.append(rr)
    return ranked


def method_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["k"])].append(r)
    out = []
    for (method, k), vals in sorted(groups.items()):
        scores = np.asarray([v["final_score"] for v in vals], dtype=float)
        ranks = np.asarray([v.get("rank_within_query", np.nan) for v in vals], dtype=float)
        top1 = np.asarray([v.get("is_top1", 0) for v in vals], dtype=float)
        valid_terms = np.asarray([v.get("valid_metadata_terms", 0) for v in vals], dtype=float)
        out.append({
            "method": method,
            "k": k,
            "n": len(vals),
            "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
            "score_median": float(np.median(scores)) if len(scores) else float("nan"),
            "score_std": float(np.std(scores)) if len(scores) else float("nan"),
            "mean_rank": float(np.nanmean(ranks)) if len(ranks) else float("nan"),
            "top1_rate": float(np.mean(top1)) if len(top1) else float("nan"),
            "valid_render_rate": float(np.mean([v["has_harmonized_render"] for v in vals])) if vals else float("nan"),
            "overlap_reject_rate": float(np.mean([v["rejected_overlap"] for v in vals])) if vals else float("nan"),
            "valid_metadata_terms_mean": float(np.mean(valid_terms)) if len(valid_terms) else float("nan"),
            "mean_low_prob_violations": float(np.mean([v["num_low_prob_violations"] for v in vals])) if vals else float("nan"),
        })
    return out


def pairwise_sealect(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_query_method = {}
    for r in rows:
        qkey = r.get("query_group_key") or r["base_id"]
        by_query_method[(r["k"], qkey, r["method"])] = r

    out = []
    for k in TOPKS:
        query_ids = sorted({(r.get("query_group_key") or r["base_id"]) for r in rows if r["k"] == k})
        for baseline in METHODS:
            if baseline == "sealect":
                continue
            wins = losses = ties = 0
            margins = []
            for qid in query_ids:
                s = by_query_method.get((k, qid, "sealect"))
                b = by_query_method.get((k, qid, baseline))
                if s is None or b is None:
                    continue
                margin = float(s["final_score"] - b["final_score"])
                margins.append(margin)
                if abs(margin) < 1e-9:
                    ties += 1
                elif margin > 0:
                    wins += 1
                else:
                    losses += 1
            n = wins + losses + ties
            out.append({
                "k": k,
                "comparison": f"sealect_vs_{baseline}",
                "baseline": baseline,
                "n_pairs": n,
                "sealect_wins": wins,
                "ties": ties,
                "sealect_losses": losses,
                "sealect_win_rate": wins / max(1, n),
                "tie_rate": ties / max(1, n),
                "mean_margin_sealect_minus_baseline": float(np.mean(margins)) if margins else float("nan"),
                "median_margin": float(np.median(margins)) if margins else float("nan"),
            })
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Collecting rendered pose/scene compatibility rows...")
    rows = collect_rows()
    ranked = rank_within_query(rows)
    summary = method_summary(ranked)
    pairwise = pairwise_sealect(ranked)

    save_json(ranked, OUT_ROOT / "ranked_rows.json")
    save_csv(ranked, OUT_ROOT / "ranked_rows.csv")
    save_json(summary, OUT_ROOT / "method_summary.json")
    save_csv(summary, OUT_ROOT / "method_summary.csv")
    save_json(pairwise, OUT_ROOT / "pairwise_win_rates.json")
    save_csv(pairwise, OUT_ROOT / "pairwise_win_rates.csv")

    save_json({
        "out_root": str(OUT_ROOT),
        "topks": TOPKS,
        "methods": METHODS,
        "weights": WEIGHTS,
        "penalties": {
            "overlap_reject": OVERLAP_REJECT_PENALTY,
            "missing_render": MISSING_RENDER_PENALTY,
            "low_prob_violation": LOW_PROB_VIOLATION_PENALTY,
        },
        "num_rows": len(ranked),
        "method_summary": summary,
        "pairwise_win_rates": pairwise,
    }, OUT_ROOT / "summary.json")

    print("\nMethod summary:")
    print(json.dumps(summary, indent=2))
    print("\nPairwise SEALECT win rates:")
    print(json.dumps(pairwise, indent=2))
    print(f"\nSaved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
