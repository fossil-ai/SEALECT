#!/usr/bin/env python3
"""
Supplementary analyses (CPU only, no retraining), supplementary Section 15 of the paper:

  1. Kernel concentration: effective number of library entries per query,
     ESS = 1 / sum_i w_i^2, plus the largest single weight and top-10 mass,
     for the B3 kernel prior at h in {0.5, 1, 2}.
  2. Observation-term fallback: fraction of test queries with no other
     annotated vessel in the scene, where C_obs reduces to the uniform baseline
     (S_obs is then constant and contributes nothing after per-query z-norm).
  3. Online latency: wall-clock per-query scoring of the full shipped
     pipeline (scene tower + neural scores, kernel prior, S_obs, z-norm fusion,
     top-K) over the 68k-candidate pool, with candidate embeddings precomputed.

Run with CUDA_VISIBLE_DEVICES="" so it never touches the GPU.
"""
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import context_kernel as ck
import context_models as cm

OUT = Path(__file__).resolve().parent / "camera_ready"
OUT.mkdir(exist_ok=True)
N_TIMING = 200
SEED = 42


def kernel_concentration(legacy, train_pairs, queries, hs=(0.5, 1.0, 2.0), batch=256):
    Cx, stats = ck.scene_bank(legacy, train_pairs)
    Qx, _ = ck.scene_bank(legacy, queries, stats)
    out = {"n_anchors": int(Cx.shape[0]), "dim": int(Cx.shape[1])}
    for h in hs:
        ess, wmax, top10 = [], [], []
        for s in range(0, Qx.shape[0], batch):
            d2 = torch.cdist(Qx[s:s + batch], Cx) ** 2
            W = torch.softmax(-d2 / (2.0 * h * h), dim=1)
            ess.append(1.0 / (W ** 2).sum(1))
            wmax.append(W.max(1).values)
            top10.append(W.topk(10, dim=1).values.sum(1))
        ess, wmax, top10 = torch.cat(ess).numpy(), torch.cat(wmax).numpy(), torch.cat(top10).numpy()
        out[f"h={h}"] = {
            "ess_median": float(np.median(ess)),
            "ess_p10": float(np.percentile(ess, 10)),
            "ess_p90": float(np.percentile(ess, 90)),
            "ess_mean": float(ess.mean()),
            "frac_ess_below_5": float((ess < 5).mean()),
            "frac_ess_below_20": float((ess < 20).mean()),
            "max_weight_median": float(np.median(wmax)),
            "top10_mass_median": float(np.median(top10)),
        }
        print(f"h={h}: {out[f'h={h}']}", flush=True)
    return out


def obs_fallback(queries):
    by_img = defaultdict(list)
    for q in queries:
        by_img[int(q["image_id"])].append(int(q["annotation_id"]))
    n_other = np.asarray([len(by_img[int(q["image_id"])]) - 1 for q in queries])
    return {
        "n_queries": int(len(queries)),
        "frac_no_other_vessel": float((n_other == 0).mean()),
        "mean_other_vessels": float(n_other.mean()),
    }


def time_online(legacy, train_pairs, queries, candidates, h=1.0, lam=(1.0, 1.0, 0.10), k=100):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(queries), size=min(N_TIMING, len(queries)), replace=False)
    sample = [queries[i] for i in idx]

    scene_enc, cand_enc = legacy.load_model()
    dino_by_key = legacy.load_dino_cache_by_key()

    # ---- offline: candidate index (tower encoding + per-candidate geometry) ----
    t0 = time.perf_counter()
    with torch.no_grad():
        cand_embs = legacy.encode_candidates(cand_enc, candidates, dino_by_key)
    t_cand_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    Cx, stats = ck.scene_bank(legacy, train_pairs)
    oh = {key: torch.nn.functional.one_hot(ck._ints(train_pairs, key).clamp(0, n - 1), n).float()
          for key, n in (("pitch_bin", 18), ("yaw_bin", 18), ("roll_bin", 18), ("y_bin", 8))}
    a_log = torch.tensor([math.log(max(float(p.get("area_frac", 1e-8)), 1e-8)) for p in train_pairs])
    c_idx = {key: ck._ints(candidates, key).clamp(0, n - 1)
             for key, n in (("pitch_bin", 18), ("yaw_bin", 18), ("roll_bin", 18), ("y_bin", 8))}
    c_log = torch.tensor([math.log(max(float(c.get("area_frac", 1e-8)), 1e-8)) for c in candidates])
    cand_t = legacy.build_candidate_tensors(candidates)
    t_library = time.perf_counter() - t0

    obs_by_image = defaultdict(list)
    for q in queries:
        obs_by_image[int(q["image_id"])].append(q)

    def z(x):
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)

    per_q = []
    with torch.no_grad():
        for q in sample:
            t0 = time.perf_counter()
            xq = torch.from_numpy(legacy.scene_features(q)).float()[None]
            qe = torch.nn.functional.normalize(scene_enc(xq), dim=1)[0]
            s_n = cand_embs @ qe
            Qx, _ = ck.scene_bank(legacy, [q], stats)
            W = torch.softmax(-(torch.cdist(Qx, Cx) ** 2) / (2 * h * h), dim=1)[0]
            lp = {key: torch.log(W @ oh[key] + 1e-8) for key in oh}
            mu = W @ a_log
            var = torch.clamp(W @ (a_log ** 2) - mu ** 2, min=0.02)
            scale = -0.5 * torch.log(2 * math.pi * var) - 0.5 * ((c_log - mu) ** 2) / var
            s_g = (2.0 * lp["pitch_bin"][c_idx["pitch_bin"]] + lp["yaw_bin"][c_idx["yaw_bin"]]
                   + lp["roll_bin"][c_idx["roll_bin"]] + 0.7 * scale + 0.5 * lp["y_bin"][c_idx["y_bin"]])
            s_o = legacy.observed_score_cuda(q, obs_by_image, cand_t)
            s = lam[0] * z(s_n) + lam[1] * z(s_g) + lam[2] * z(s_o)
            torch.topk(s, k)
            per_q.append(time.perf_counter() - t0)
    per_q = np.asarray(per_q[5:]) * 1000.0  # drop warm-up
    return {
        "device": "cpu",
        "cpu": platform.processor() or "see lscpu",
        "torch_threads": torch.get_num_threads(),
        "n_candidates": len(candidates),
        "n_library_anchors": len(train_pairs),
        "offline_candidate_tower_encode_s": round(t_cand_enc, 2),
        "offline_library_tensors_s": round(t_library, 2),
        "online_ms_per_query_median": float(np.median(per_q)),
        "online_ms_per_query_p90": float(np.percentile(per_q, 90)),
        "n_timed": int(len(per_q)),
    }


def main():
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"
    legacy, train_pairs, candidates, _ = cm.load_ctx_assets()
    queries = legacy.load_json(legacy.TEST_QUERIES)
    print(f"train_pairs={len(train_pairs)} candidates={len(candidates)} queries={len(queries)}", flush=True)

    res = {"obs_fallback": obs_fallback(queries)}
    print(res["obs_fallback"], flush=True)
    res["kernel_concentration"] = kernel_concentration(legacy, train_pairs, queries)
    res["latency"] = time_online(legacy, train_pairs, queries, candidates)
    print(res["latency"], flush=True)

    with open(OUT / "camera_ready_analysis.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT / 'camera_ready_analysis.json'}")


if __name__ == "__main__":
    main()
