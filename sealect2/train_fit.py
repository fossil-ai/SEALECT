"""
train_fit.py

Builds the TRAIN-split fitting data used by learned variants (A2-A5) and A1's
global-normalization stats. Uses the frozen legacy functions so the components
are identical in construction to the eval components; relevance labels use the
paper's own tolerances (imported from summarize_exp1_retrieval_results).

Fitting on TRAIN queries -> evaluate on TEST queries, same TRAIN candidate pool,
same validity rule, same graded relevance. Only the fusion/gating params are
learned; the three scores stay frozen.
"""

from __future__ import annotations

import importlib.util
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import legacy_bridge as lb
import features as feat

CACHE = lb.CACHE_DIR / "train_context.pt"


def _load_relevance_cfg():
    f = lb.EXP1_DIR / "summarize_exp1_retrieval_results.py"
    spec = importlib.util.spec_from_file_location("legacy_summarize2", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TrainContext:
    def __init__(self, blob):
        self.n_z = blob["n_z"]            # [Q,S] perquery-z normalized
        self.c_z = blob["c_z"]
        self.o_z = blob["o_z"]
        self.rel = blob["rel"]            # [Q,S] graded relevance 0..4
        self.geom = blob["geom"]          # [Q,S,6]
        self.rq = blob["rq"]              # [Q,d_r] standardized reliability
        self.global_stats = blob["global_stats"]     # raw (mu,sig) per component (A1)
        self.reliability_stats = blob["reliability_stats"]  # (mu,sigma) for r_q at eval
        self.prior = blob["prior"]
        self.ctx_counts = blob["ctx_counts"]
        self.meta = blob["meta"]


def build_train_context(cap=3000, n_pos=64, n_neg=192, seed=42, use_cache=True) -> TrainContext:
    lb.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and CACHE.exists():
        blob = torch.load(CACHE, map_location="cpu")
        if blob.get("cfg") == (cap, n_pos, n_neg, seed):
            print(f"[train_fit] loaded cached train context: {CACHE}")
            return TrainContext(blob)
        print("[train_fit] cache cfg changed; rebuilding.")

    legacy = lb.load_legacy()
    device = legacy.DEVICE
    rng = random.Random(seed)

    candidates = legacy.load_json(legacy.TRAIN_CANDIDATES)
    train_pairs_path = legacy.SPLIT_DIR / "train_scene_object_pairs.json"
    train_q = legacy.load_json(train_pairs_path)
    prior = legacy.load_json(legacy.PRIOR_JSON)
    dino_by_key = legacy.load_dino_cache_by_key()

    scene_enc, cand_enc = legacy.load_model()
    with torch.no_grad():
        cand_embs = legacy.encode_candidates(cand_enc, candidates, dino_by_key)

    # sample the training queries we fit on
    idx_all = list(range(len(train_q)))
    rng.shuffle(idx_all)
    sel = sorted(idx_all[:min(cap, len(train_q))])
    queries = [train_q[i] for i in sel]
    with torch.no_grad():
        q_embs = legacy.encode_queries(scene_enc, queries)

    valid_masks = legacy.build_valid_masks(queries, candidates)          # [Q,Ncand] bool (device)
    context_cache = legacy.build_context_cache(queries, candidates, prior)
    obs_by_image = defaultdict(list)
    for q in train_q:  # observed set uses full train image membership
        obs_by_image[int(q["image_id"])].append(q)
    cand_t = legacy.build_candidate_tensors(candidates)

    relmod = _load_relevance_cfg()
    geom_all = feat.candidate_geometry(candidates)                      # [Ncand,6] cpu

    # candidate arrays for vectorized relevance
    c_pitch = np.array([int(c.get("pitch_bin", 0)) for c in candidates])
    c_yaw = np.array([int(c.get("yaw_bin", 0)) for c in candidates])
    c_ybin = np.array([int(c.get("y_bin", 0)) for c in candidates])
    c_logarea = np.array([math.log(max(float(c.get("area_frac", 1e-8)), 1e-8)) for c in candidates])

    def relevance_vec(q):
        s = np.zeros(len(candidates), dtype=np.int16)
        s += (np.abs(int(q["pitch_bin"]) - c_pitch) <= relmod.PITCH_TOL)
        d = np.abs(int(q["yaw_bin"]) - c_yaw); d = np.minimum(d, 18 - d)
        s += (d <= relmod.YAW_TOL)
        s += (np.abs(int(q["y_bin"]) - c_ybin) <= relmod.YBIN_TOL)
        qla = math.log(max(float(q["area_frac"]), 1e-8))
        s += (np.abs(qla - c_logarea) <= relmod.AREA_LOG_TOL)
        return s

    S = n_pos + n_neg
    Q = len(queries)
    n_z = torch.empty((Q, S)); c_z = torch.empty((Q, S)); o_z = torch.empty((Q, S))
    rel = torch.empty((Q, S)); geom = torch.empty((Q, S, feat.GEOM_DIM))

    # running global stats over full rows (A1) -- unbiased over sampled queries x all cands
    gsum = {"neural": 0.0, "context": 0.0, "obs": 0.0}
    gsq = {"neural": 0.0, "context": 0.0, "obs": 0.0}
    gcount = 0

    ctx_counts = feat.build_ctx_counts(train_q)

    with torch.no_grad():
        for qi, q in enumerate(tqdm(queries, desc="[train_fit] build fit data")):
            neural = (cand_embs @ q_embs[qi])                       # [Ncand] device
            context = torch.from_numpy(context_cache[(str(q.get("context_key")), str(q.get("scene_type")))]).to(device)
            obs = legacy.observed_score_cuda(q, obs_by_image, cand_t)
            valid = valid_masks[qi]

            # accumulate raw global stats
            for name, vec in (("neural", neural), ("context", context), ("obs", obs)):
                gsum[name] += float(vec.sum().item())
                gsq[name] += float((vec * vec).sum().item())
            gcount += neural.numel()

            # per-query z (matches shipped normalization) computed over full pool
            def pqz(v):
                return (v - v.mean()) / (v.std(unbiased=False) + 1e-6)
            nz, cz, oz = pqz(neural).cpu().numpy(), pqz(context).cpu().numpy(), pqz(obs).cpu().numpy()

            rv = relevance_vec(q)
            valid_np = valid.cpu().numpy()
            valid_idx = np.where(valid_np)[0]
            rv_valid = rv[valid_idx]
            pos_pool = valid_idx[rv_valid >= 3]
            neg_pool = valid_idx[rv_valid < 3]
            if len(pos_pool) > n_pos:
                pos = list(np.array(pos_pool)[rng.sample(range(len(pos_pool)), n_pos)])
            else:
                pos = list(pos_pool)
            need = S - len(pos)
            if len(neg_pool) >= need:
                neg = list(np.array(neg_pool)[rng.sample(range(len(neg_pool)), need)])
            else:
                neg = list(neg_pool) + list(np.array(valid_idx)[rng.sample(range(len(valid_idx)), need - len(neg_pool))])
            samp = np.array(pos + neg)[:S]
            if len(samp) < S:  # extreme fallback
                samp = np.concatenate([samp, samp[:S - len(samp)]])

            n_z[qi] = torch.from_numpy(nz[samp]); c_z[qi] = torch.from_numpy(cz[samp]); o_z[qi] = torch.from_numpy(oz[samp])
            rel[qi] = torch.from_numpy(rv[samp].astype(np.float32))
            geom[qi] = geom_all[torch.from_numpy(samp)]

    # global stats
    global_stats = {}
    for name in ("neural", "context", "obs"):
        mu = gsum[name] / gcount
        var = max(gsq[name] / gcount - mu * mu, 0.0)
        global_stats[name] = (mu, math.sqrt(var))

    obs_per_image_train = feat.build_obs_counts(train_q)
    rq_raw = feat.reliability_matrix(queries, prior, ctx_counts, obs_per_image_train)
    rq_std, rq_stats = feat.standardize_columns(rq_raw)

    blob = {
        "cfg": (cap, n_pos, n_neg, seed),
        "n_z": n_z, "c_z": c_z, "o_z": o_z, "rel": rel, "geom": geom,
        "rq": rq_std, "global_stats": global_stats,
        "reliability_stats": (rq_stats[0], rq_stats[1]),
        "prior": prior, "ctx_counts": dict(ctx_counts),
        "meta": {"cap": cap, "n_pos": n_pos, "n_neg": n_neg, "seed": seed, "Q": Q, "S": S},
    }
    if use_cache:
        torch.save(blob, CACHE)
        print(f"[train_fit] cached train context -> {CACHE}")
    return TrainContext(blob)
