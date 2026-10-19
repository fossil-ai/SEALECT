"""
eval_common.py

Reuse the ORIGINAL paper metric code to score a run and to compare variants
against v1 SEALECT (== A0_baseline). exp2/exp3 (distributional + plausibility),
per-dataset breakdown, and the Pareto table are added in Task #5; this module
covers exp1 (Recall/NDCG/PS) + the vs-v1 delta table used while iterating.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import legacy_bridge as lb

RUNS = lb.RUNS_DIR
V1 = "A0_baseline"   # v1 SEALECT reference


def _summarizer():
    f = lb.EXP1_DIR / "summarize_exp1_retrieval_results.py"
    spec = importlib.util.spec_from_file_location("legacy_summ_eval", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def score_exp1(run_id: str, topks=(20, 100)) -> dict:
    summ = _summarizer()
    out = {}
    for k in topks:
        p = RUNS / run_id / f"topk_{k}" / "retrieval_results.json"
        if not p.exists():
            continue
        met = summ.summarize_one(json.load(open(p)), k)
        (p.parent / "metrics.json").write_text(json.dumps(met, indent=2))
        out.update(met)
    return out


def _load_metrics(run_id: str, k: int) -> dict:
    p = RUNS / run_id / f"topk_{k}" / "metrics.json"
    return json.load(open(p)) if p.exists() else {}


def compare_to_v1(topks=(20, 100)) -> str:
    """Return a markdown table of every scored run as a delta vs v1 SEALECT (A0)."""
    lines = []
    for k in topks:
        v1 = _load_metrics(V1, k)
        if not v1:
            continue
        lines.append(f"\n### K={k}  (Δ vs v1 SEALECT = {V1})\n")
        lines.append(f"| variant | Recall | ΔR | NDCG | ΔNDCG | PS | ΔPS |")
        lines.append(f"| --- | --- | --- | --- | --- | --- | --- |")
        runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir())
        for r in runs:
            m = _load_metrics(r, k)
            if not m:
                continue
            R, N, P = m[f"Recall@{k}"], m[f"NDCG@{k}"], m[f"PS@{k}"]
            dR = R - v1[f"Recall@{k}"]; dN = N - v1[f"NDCG@{k}"]; dP = P - v1[f"PS@{k}"]
            tag = " (v1)" if r == V1 else ""
            lines.append(f"| {r}{tag} | {R:.4f} | {dR:+.4f} | {N:.4f} | {dN:+.4f} | {P:.4f} | {dP:+.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    for p in sorted(RUNS.iterdir()):
        if p.is_dir():
            score_exp1(p.name)
    print(compare_to_v1())
