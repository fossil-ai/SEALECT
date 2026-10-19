"""
driver.py

Run one Variant over the frozen components and emit `retrieval_results.json` in
the exact schema the original evaluators expect. Variant-agnostic: it asks the
variant for its evidence sources, normalization, and fusion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import legacy_bridge as lb
from base import Variant


def run(variant: Variant, raw: lb.RawComponents, topks, run_id, device,
        batch=256, emit_into_exp1=False):
    legacy = lb.load_legacy()
    max_k = max(topks)
    nq = raw.n_queries

    neural = variant.neural_component(raw)
    context = variant.context_component(raw)
    obs = variant.obs_component(raw)

    variant.on_eval_start(raw, device)
    batch = min(batch, getattr(variant, "eval_batch", batch))

    if emit_into_exp1:
        base = lb.EXP1_DIR / "exp1_retrieval_quality" / f"sweep_s2_{run_id}"
    else:
        base = lb.RUNS_DIR / run_id

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_fuse = time.time()

    results = {k: [] for k in topks}
    for start in tqdm(range(0, nq, batch), desc=f"{run_id} fuse"):
        end = min(start + batch, nq)
        n = variant.normalize(neural[start:end].to(device), "neural")
        c = variant.normalize(context[start:end].to(device), "context")
        o = variant.normalize(obs[start:end].to(device), "obs")
        valid = raw.valid[start:end].to(device)

        meta = {"queries": raw.queries[start:end], "candidates": raw.candidates,
                "start": start, "end": end}
        fused = variant.fuse(n, c, o, meta).masked_fill(~valid, -1e9)
        vals, idxs = torch.topk(fused, k=min(max_k, fused.shape[1]), dim=1)

        idxs_cpu, vals_cpu = idxs.cpu().numpy(), vals.cpu().numpy()
        n_cpu, c_cpu, o_cpu = n.cpu(), c.cpu(), o.cpu()

        for bi, qi in enumerate(range(start, end)):
            q = raw.queries[qi]
            for k in topks:
                top = []
                for rank in range(k):
                    idx = int(idxs_cpu[bi, rank])
                    top.append(legacy.format_candidate(rank + 1, raw.candidates[idx], {
                        "neural_score_used": float(n_cpu[bi, idx]),
                        "context_score_used": float(c_cpu[bi, idx]),
                        "observed_score_used": float(o_cpu[bi, idx]),
                        "base_score": float(vals_cpu[bi, rank]),
                        "final_score": float(vals_cpu[bi, rank]),
                        "lambda_n": float(variant.lambda_n),
                        "lambda_c": float(variant.lambda_c),
                        "lambda_o": float(variant.lambda_o),
                        "lambda_d": 0.0,
                    }))
                results[k].append({"query": legacy.format_query(q), "topk": top})

    fuse_time = time.time() - t_fuse
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
    infer_ms_per_query = 1000.0 * fuse_time / max(1, nq)

    for k in topks:
        out = base / f"topk_{k}" / "retrieval_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results[k], f)
        cfg = {**variant.manifest_row(), "run_id": run_id, "top_k": k,
               "lambda_n": variant.lambda_n, "lambda_c": variant.lambda_c,
               "lambda_o": variant.lambda_o,
               "train_time_s": float(getattr(variant, "train_time_s", 0.0)),
               "infer_ms_per_query": round(infer_ms_per_query, 3),
               "peak_gpu_mb": round(peak_gpu_mb, 1)}
        with open(base / f"topk_{k}" / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[driver] wrote {out}")
    return base
