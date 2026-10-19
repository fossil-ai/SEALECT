#!/usr/bin/env python3
"""
run_sealect_exp1_lambda_sweep_cuda_dino.py

Fast CUDA lambda sweep for Experiment 1, using the retrained SEALECT+DINO model.

This script:
  1. loads the trained SEALECT+DINO q/k model
  2. encodes train candidates once
  3. encodes test queries once
  4. precomputes context prior scores once
  5. precomputes observed-pose scores once
  6. runs lambda sweeps over:
       S = lambda_n S_neural + lambda_c S_context + lambda_o S_obs

No MMR.
No retraining.
No CLI arguments.

Outputs are compatible with summarize_exp1_retrieval_results.py:
  exp1_retrieval_quality/<run_name>/topk_20/retrieval_results.json
  exp1_retrieval_quality/<run_name>/topk_100/retrieval_results.json
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 
# PATHS
# 

DATA_ROOT = Path("/path/to/sealect_root/experiments/dataset_decide")
SPLIT_DIR = DATA_ROOT / "splits_dino_decorrelated_kbalanced_grounded_sam2"
MERGED_DIR = DATA_ROOT / "merged_scene_pose_library_out_grounded_sam2"

TRAIN_CANDIDATES = SPLIT_DIR / "train_candidate_library.json"
TEST_QUERIES = SPLIT_DIR / "test_scene_object_pairs.json"
PRIOR_JSON = DATA_ROOT / "scene_pose_prior_train_grounded_sam2.json"
DINO_CACHE = SPLIT_DIR / "cache" / "candidate_dino_embeddings.pt"

EXP_ROOT = Path("exp1_retrieval_quality")
MODEL_PATH = EXP_ROOT / "sealect_neural_dino" / "sealect_neural_retriever_dino.pt"

# 
# SWEEP CONFIG
# 

TOPKS = [20, 100]

# Keep this compact for the paper. You can expand later.
# Main useful ablations:
#   neural_only:       (1, 0, 0)
#   + context:         (1, 0.25/0.5/1/2, 0)
#   + observed support:(1, 1, 0.05/0.1/0.25/0.5/1)
#   component knockouts around best setting.
SWEEP_RUNS = [
    # Component-only / ablation rows
    ("sweep_neural_only",        1.0, 0.0, 0.0),
    ("sweep_context_only",       0.0, 1.0, 0.0),
    ("sweep_obs_only",           0.0, 0.0, 1.0),

    # Context sweep with neural fixed
    ("sweep_ln1_lc0p10_lo0",     1.0, 0.10, 0.0),
    ("sweep_ln1_lc0p25_lo0",     1.0, 0.25, 0.0),
    ("sweep_ln1_lc0p50_lo0",     1.0, 0.50, 0.0),
    ("sweep_ln1_lc1p00_lo0",     1.0, 1.00, 0.0),
    ("sweep_ln1_lc2p00_lo0",     1.0, 2.00, 0.0),

    # Observed-pose sweep with neural/context fixed
    ("sweep_ln1_lc1_lo0p05",     1.0, 1.00, 0.05),
    ("sweep_ln1_lc1_lo0p10",     1.0, 1.00, 0.10),
    ("sweep_ln1_lc1_lo0p25",     1.0, 1.00, 0.25),
    ("sweep_ln1_lc1_lo0p50",     1.0, 1.00, 0.50),
    ("sweep_ln1_lc1_lo1p00",     1.0, 1.00, 1.00),

    # A few balanced alternatives
    ("sweep_ln1_lc0p50_lo0p10",  1.0, 0.50, 0.10),
    ("sweep_ln1_lc0p50_lo0p25",  1.0, 0.50, 0.25),
    ("sweep_ln1_lc2p00_lo0p10",  1.0, 2.00, 0.10),
    ("sweep_ln0p75_lc1_lo0p10",  0.75, 1.00, 0.10),
    ("sweep_ln1p25_lc1_lo0p10",  1.25, 1.00, 0.10),
]

# Runtime.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
ENCODE_BATCH = 4096
QUERY_BATCH = 256
EPS = 1e-8

NUM_YAW_BINS = 18
OBS_ALPHA = 0.7
OBS_SIGMA_PITCH = 1.5
OBS_SIGMA_YAW = 2.0
OBS_SIGMA_ROLL = 1.5

FALLBACK_EMBED_DIM = 256

VIEWPOINTS = ["nadir_aerial", "oblique_aerial", "horizon_view", "low_oblique_or_shore", "unknown_viewpoint"]
LAYOUTS = ["open_water", "open_water_uncertain", "coastal_transition", "harbor_port", "water_land_mixed", "non_maritime_or_uncertain"]
POSE_FAMILIES = [
    "nadir__front_back__level", "nadir__side__level", "nadir__oblique__level",
    "level__front_back__level", "level__side__level", "level__oblique__level",
    "level__front_back__rolled", "level__side__rolled", "level__oblique__rolled",
    "steep_up__front_back__level", "steep_up__side__level", "steep_up__oblique__level",
]
YAW_FAMILIES = ["front_back", "side", "oblique"]


# 
# IO / features
# 

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def key_for_record(r: Dict) -> Tuple[int, int]:
    return int(r["image_id"]), int(r["annotation_id"])


def one_hot(value: str, choices: List[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def scene_features(r: Dict) -> np.ndarray:
    vp = r.get("vertical_profile", {})
    water_v = vp.get("water", [0.0] * 16)
    sky_v = vp.get("sky", [0.0] * 16)
    land_v = vp.get("land", [0.0] * 16)
    other_v = vp.get("other", [0.0] * 16)

    x = [
        float(r.get("water_frac", 0.0)),
        float(r.get("sky_frac", 0.0)),
        float(r.get("land_frac", 0.0)),
        float(r.get("other_frac", 0.0)),
        float(r.get("vessel_ignore_frac", 0.0)),
        float(r.get("horizon_y_norm", 0.0)),
        float(r.get("horizon_confidence", 0.0)),
        float(r.get("water_row_mean", 0.0)),
        float(r.get("water_row_std", 0.0)),
        float(r.get("sky_row_mean", 0.0)),
        float(r.get("sky_row_std", 0.0)),
        float(r.get("land_row_mean", 0.0)),
        float(r.get("land_row_std", 0.0)),

        # Target-local query geometry. Must match training.
        float(r.get("center_x_norm", 0.0)),
        float(r.get("center_y_norm", 0.0)),
        int(r.get("y_bin", 0)) / 7.0,
        math.log(max(float(r.get("area_frac", EPS)), EPS)),
        float(r.get("aspect_ratio", 0.0)),
    ]

    x.extend([float(v) for v in water_v])
    x.extend([float(v) for v in sky_v])
    x.extend([float(v) for v in land_v])
    x.extend([float(v) for v in other_v])
    x.extend(one_hot(str(r.get("viewpoint_type", "unknown_viewpoint")), VIEWPOINTS))
    x.extend(one_hot(str(r.get("scene_layout", "non_maritime_or_uncertain")), LAYOUTS))

    return np.asarray(x, dtype=np.float32)


def base_candidate_features(r: Dict) -> List[float]:
    emb = r.get("embedding", []) or [0.0] * FALLBACK_EMBED_DIM
    x = [
        int(r.get("pitch_bin", 0)) / 17.0,
        int(r.get("yaw_bin", 0)) / 17.0,
        int(r.get("roll_bin", 0)) / 17.0,
        int(r.get("y_bin", 0)) / 7.0,
        math.log(max(float(r.get("area_frac", EPS)), EPS)),
        float(r.get("aspect_ratio", 0.0)),
        float(r.get("center_x_norm", 0.0)),
        float(r.get("center_y_norm", 0.0)),
        float(r.get("water_frac", 0.0)),
        float(r.get("sky_frac", 0.0)),
        float(r.get("land_frac", 0.0)),
        float(r.get("horizon_y_norm", 0.0)),
        float(r.get("horizon_confidence", 0.0)),
    ]
    x.extend(one_hot(str(r.get("viewpoint_type", "unknown_viewpoint")), VIEWPOINTS))
    x.extend(one_hot(str(r.get("scene_layout", "non_maritime_or_uncertain")), LAYOUTS))
    x.extend(one_hot(str(r.get("pose_family", "")), POSE_FAMILIES))
    x.extend(one_hot(str(r.get("yaw_family", "")), YAW_FAMILIES))
    x.extend(float(v) for v in emb)
    return x


def load_dino_cache_by_key():
    obj = torch.load(DINO_CACHE, map_location="cpu")
    emb = obj["embeddings"].float().numpy().astype(np.float32)
    valid_indices = list(obj["valid_indices"])

    scene_pairs = load_json(MERGED_DIR / "scene_object_pairs.json")
    all_candidates = load_json(MERGED_DIR / "candidate_library.json")
    cand_by_key = {key_for_record(c): c for c in all_candidates}
    paired = [r for r in scene_pairs if key_for_record(r) in cand_by_key]
    candidates_paired = [cand_by_key[key_for_record(r)] for r in paired]

    dino_by_key = {
        key_for_record(candidates_paired[raw_idx]): emb[j]
        for j, raw_idx in enumerate(valid_indices)
    }
    print(f"Loaded DINO cache vectors={len(dino_by_key)} dim={emb.shape[1]}")
    return dino_by_key


def candidate_features_with_dino(c: Dict, dino_by_key: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
    dino = dino_by_key[key_for_record(c)]
    dino = dino / (np.linalg.norm(dino) + EPS)
    x = base_candidate_features(c)
    x.extend(float(v) for v in dino)
    return np.asarray(x, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    scene_enc = MLP(int(ckpt["scene_dim"]), int(ckpt["emb_dim"])).to(DEVICE)
    cand_enc = MLP(int(ckpt["cand_dim"]), int(ckpt["emb_dim"])).to(DEVICE)
    scene_enc.load_state_dict(ckpt["scene_enc"])
    cand_enc.load_state_dict(ckpt["cand_enc"])
    return scene_enc.eval(), cand_enc.eval()


@torch.no_grad()
def encode_candidates(cand_enc, candidates, dino_by_key):
    xs = torch.stack([
        torch.from_numpy(candidate_features_with_dino(c, dino_by_key))
        for c in candidates
    ]).float()
    out = []
    for i in tqdm(range(0, len(xs), ENCODE_BATCH), desc="encode train candidates"):
        out.append(cand_enc(xs[i:i + ENCODE_BATCH].to(DEVICE)))
    return F.normalize(torch.cat(out, dim=0), dim=1)


@torch.no_grad()
def encode_queries(scene_enc, queries):
    xs = torch.stack([torch.from_numpy(scene_features(q)) for q in queries]).float()
    out = []
    for i in tqdm(range(0, len(xs), QUERY_BATCH), desc="encode test queries"):
        out.append(scene_enc(xs[i:i + QUERY_BATCH].to(DEVICE)))
    return F.normalize(torch.cat(out, dim=0), dim=1)


# 
# Precompute score components
# 

def build_valid_masks(queries, candidates):
    cand_img = np.asarray([int(c.get("image_id", -2)) for c in candidates], dtype=np.int64)
    cand_ann = np.asarray([int(c.get("annotation_id", -2)) for c in candidates], dtype=np.int64)
    cand_seq = np.asarray([str(c.get("sequence_group", "")) for c in candidates], dtype=object)

    masks = []
    for q in tqdm(queries, desc="build validity masks"):
        valid = ~(
            ((cand_img == int(q.get("image_id", -1))) & (cand_ann == int(q.get("annotation_id", -1))))
            | (cand_seq == str(q.get("sequence_group", "")))
        )
        masks.append(torch.from_numpy(valid.astype(np.bool_)))
    return torch.stack(masks, dim=0).to(DEVICE)


def log_prob_hist_array(hist, n_bins=18):
    arr = np.full((n_bins,), math.log(EPS), dtype=np.float32)
    for k, v in hist.items():
        idx = int(k)
        if 0 <= idx < n_bins:
            arr[idx] = math.log(max(float(v), EPS))
    return arr


def pose_key(c):
    return f"{int(c.get('pitch_bin', 0))}_{int(c.get('yaw_bin', 0))}_{int(c.get('roll_bin', 0))}"


def context_score_numpy(query, candidates, prior):
    context_key = query.get("context_key")
    scene_type = query.get("scene_type")

    def get_prob_dict(section):
        if context_key is not None and prior.get(section, {}).get(context_key) is not None:
            return prior[section][context_key]
        scene_section = section.replace("context_", "scene_")
        if scene_type is not None and prior.get(scene_section, {}).get(scene_type) is not None:
            return prior[scene_section][scene_type]
        return {}

    pitch_hist = log_prob_hist_array(get_prob_dict("context_pitch_hist"))
    yaw_hist = log_prob_hist_array(get_prob_dict("context_yaw_hist"))
    roll_hist = log_prob_hist_array(get_prob_dict("context_roll_hist"))
    yaw_weight = float(
        prior.get("context_yaw_weight", {}).get(
            context_key,
            prior.get("scene_yaw_weight", {}).get(scene_type, 1.0),
        )
    )

    pitch = np.asarray([int(c.get("pitch_bin", 0)) for c in candidates], dtype=np.int64)
    yaw = np.asarray([int(c.get("yaw_bin", 0)) for c in candidates], dtype=np.int64)
    roll = np.asarray([int(c.get("roll_bin", 0)) for c in candidates], dtype=np.int64)
    ybin = np.asarray([int(c.get("y_bin", 0)) for c in candidates], dtype=np.int64)
    log_area = np.asarray([
        math.log(max(float(c.get("area_frac", EPS)), EPS)) for c in candidates
    ], dtype=np.float32)

    s = (
        2.0 * pitch_hist[np.clip(pitch, 0, 17)]
        + yaw_weight * yaw_hist[np.clip(yaw, 0, 17)]
        + roll_hist[np.clip(roll, 0, 17)]
    )

    mu = np.full_like(log_area, math.log(1e-4), dtype=np.float32)
    sig = np.full_like(log_area, 1.5, dtype=np.float32)
    for y in range(8):
        yb = str(y)
        stats = None
        if context_key is not None:
            stats = prior.get("scale_by_context_ybin", {}).get(context_key, {}).get(yb)
        if stats is None and scene_type is not None:
            stats = prior.get("scale_by_scene_ybin", {}).get(scene_type, {}).get(yb)
        if stats is not None:
            m = ybin == y
            mu[m] = float(stats.get("log_area_mean", math.log(1e-4)))
            sig[m] = max(float(stats.get("log_area_std", 1.5)), 0.15)

    s += 0.7 * (
        -0.5 * np.log(2.0 * np.pi * sig * sig)
        - 0.5 * ((log_area - mu) / sig) ** 2
    )

    y_logp = np.full_like(log_area, math.log(EPS), dtype=np.float32)
    for i, c in enumerate(candidates):
        pk = pose_key(c)
        hist = None
        if context_key is not None:
            hist = prior.get("ybin_given_context_pose", {}).get(context_key, {}).get(pk)
        if hist is None and scene_type is not None:
            hist = prior.get("ybin_given_scene_pose", {}).get(scene_type, {}).get(pk)
        if hist:
            y_logp[i] = math.log(max(float(hist.get(str(int(ybin[i])), 0.0)), EPS))

    return (s + 0.5 * y_logp).astype(np.float32)


def build_context_cache(queries, candidates, prior):
    cache, uniq = {}, {}
    for q in queries:
        key = (str(q.get("context_key")), str(q.get("scene_type")))
        if key not in uniq:
            uniq[key] = q
    for key, q in tqdm(uniq.items(), desc="precompute S_context"):
        cache[key] = context_score_numpy(q, candidates, prior)
    return cache


def build_candidate_tensors(candidates):
    return {
        "pitch": torch.tensor([int(c.get("pitch_bin", 0)) for c in candidates], device=DEVICE, dtype=torch.float32),
        "yaw": torch.tensor([int(c.get("yaw_bin", 0)) for c in candidates], device=DEVICE, dtype=torch.float32),
        "roll": torch.tensor([int(c.get("roll_bin", 0)) for c in candidates], device=DEVICE, dtype=torch.float32),
    }


def circular_distance_tensor(a, b, n):
    d = torch.abs(a - b)
    return torch.minimum(d, torch.tensor(float(n), device=a.device) - d)


def observed_score_cuda(query, obs_by_image, cand_t):
    obs = [
        o for o in obs_by_image.get(int(query.get("image_id", -1)), [])
        if int(o.get("annotation_id", -2)) != int(query.get("annotation_id", -1))
    ]
    n = cand_t["pitch"].shape[0]
    if not obs:
        return torch.zeros(n, device=DEVICE, dtype=torch.float32)

    op = torch.tensor([int(o.get("pitch_bin", 0)) for o in obs], device=DEVICE, dtype=torch.float32)
    oy = torch.tensor([int(o.get("yaw_bin", 0)) for o in obs], device=DEVICE, dtype=torch.float32)
    orr = torch.tensor([int(o.get("roll_bin", 0)) for o in obs], device=DEVICE, dtype=torch.float32)

    dp = torch.abs(cand_t["pitch"][:, None] - op[None, :])
    dy = circular_distance_tensor(cand_t["yaw"][:, None], oy[None, :], NUM_YAW_BINS)
    dr = torch.abs(cand_t["roll"][:, None] - orr[None, :])

    kde = torch.exp(
        -0.5 * (
            (dp / OBS_SIGMA_PITCH) ** 2
            + (dy / OBS_SIGMA_YAW) ** 2
            + (dr / OBS_SIGMA_ROLL) ** 2
        )
    ).mean(dim=1)

    uniform = 1.0 / max(1.0, float(NUM_YAW_BINS * 18 * 18))
    p = (1.0 - OBS_ALPHA) * uniform + OBS_ALPHA * kde
    return torch.log(torch.clamp(p, min=EPS)).float()


def zscore_torch(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)


def precompute_components(queries, candidates, q_embs, cand_embs, context_cache, obs_by_image, cand_t):
    neural_rows = []
    context_rows = []
    obs_rows = []

    for qi, q in enumerate(tqdm(queries, desc="precompute normalized components")):
        neural_raw = cand_embs @ q_embs[qi]
        context_raw = torch.from_numpy(context_cache[(str(q.get("context_key")), str(q.get("scene_type")))]).to(DEVICE)
        obs_raw = observed_score_cuda(q, obs_by_image, cand_t)

        neural_rows.append(zscore_torch(neural_raw).detach().cpu())
        context_rows.append(zscore_torch(context_raw).detach().cpu())
        obs_rows.append(zscore_torch(obs_raw).detach().cpu())

    return {
        "neural": torch.stack(neural_rows, dim=0).to(DEVICE),
        "context": torch.stack(context_rows, dim=0).to(DEVICE),
        "obs": torch.stack(obs_rows, dim=0).to(DEVICE),
    }


# 
# Formatting
# 

def compact_context(r):
    return {
        "scene_type": r.get("scene_type"),
        "scene_layout": r.get("scene_layout"),
        "viewpoint_type": r.get("viewpoint_type"),
        "context_key": r.get("context_key"),
        "water_frac": r.get("water_frac"),
        "sky_frac": r.get("sky_frac"),
        "land_frac": r.get("land_frac"),
        "horizon_y_norm": r.get("horizon_y_norm"),
        "horizon_confidence": r.get("horizon_confidence"),
    }


def format_query(q):
    return {
        "dataset": q.get("dataset"),
        "source": q.get("source"),
        "sequence_group": q.get("sequence_group"),
        "image_id": q.get("image_id"),
        "annotation_id": q.get("annotation_id"),
        "image_name": q.get("image_name"),
        "center_x_norm": q.get("center_x_norm"),
        "center_y_norm": q.get("center_y_norm"),
        "y_bin": q.get("y_bin"),
        "area_frac": q.get("area_frac"),
        "pitch_bin": q.get("pitch_bin"),
        "yaw_bin": q.get("yaw_bin"),
        "roll_bin": q.get("roll_bin"),
        "pitch_deg": q.get("pitch_deg"),
        "yaw_deg": q.get("yaw_deg"),
        "roll_deg": q.get("roll_deg"),
        "pose_family": q.get("pose_family"),
        "yaw_family": q.get("yaw_family"),
        **compact_context(q),
    }


def format_candidate(rank, c, score_dict):
    return {
        "rank": rank,
        "candidate_id": c.get("candidate_id"),
        "dataset": c.get("dataset"),
        "source": c.get("source"),
        "sequence_group": c.get("sequence_group"),
        "image_id": c.get("image_id"),
        "annotation_id": c.get("annotation_id"),
        "image_name": c.get("image_name"),
        "crop_path": c.get("crop_path"),
        "image_path": c.get("image_path"),
        "pitch_bin": c.get("pitch_bin"),
        "yaw_bin": c.get("yaw_bin"),
        "roll_bin": c.get("roll_bin"),
        "pitch_deg": c.get("pitch_deg"),
        "yaw_deg": c.get("yaw_deg"),
        "roll_deg": c.get("roll_deg"),
        "pose_family": c.get("pose_family"),
        "yaw_family": c.get("yaw_family"),
        "area_frac": c.get("area_frac"),
        "y_bin": c.get("y_bin"),
        **compact_context(c),
        **score_dict,
    }


def run_one(run_name, top_k, queries, candidates, valid_masks, comp, ln, lc, lo):
    results = []
    n_queries = len(queries)

    for qi, q in enumerate(tqdm(queries, desc=f"{run_name} topK={top_k}")):
        final = (
            ln * comp["neural"][qi]
            + lc * comp["context"][qi]
            + lo * comp["obs"][qi]
        ).masked_fill(~valid_masks[qi], -1e9)

        vals, idxs = torch.topk(final, k=min(top_k, final.numel()))

        idxs_cpu = idxs.detach().cpu().numpy().tolist()
        vals_cpu = vals.detach().cpu().numpy().tolist()

        top = []
        for rank, (idx, score) in enumerate(zip(idxs_cpu, vals_cpu), 1):
            top.append(format_candidate(rank, candidates[idx], {
                "neural_score_used": float(comp["neural"][qi, idx].item()),
                "context_score_used": float(comp["context"][qi, idx].item()),
                "observed_score_used": float(comp["obs"][qi, idx].item()),
                "base_score": float(score),
                "final_score": float(score),
                "lambda_n": float(ln),
                "lambda_c": float(lc),
                "lambda_o": float(lo),
                "lambda_d": 0.0,
            }))

        results.append({"query": format_query(q), "topk": top})

    out = EXP_ROOT / run_name / f"topk_{top_k}" / "retrieval_results.json"
    save_json(results, out)
    save_json({
        "run_name": run_name,
        "top_k": top_k,
        "model_path": str(MODEL_PATH),
        "lambda_n": ln,
        "lambda_c": lc,
        "lambda_o": lo,
        "lambda_d": 0.0,
        "uses_cached_dino": True,
        "device": DEVICE,
        "note": "lambda sweep over normalized S_neural, S_context, S_obs; no MMR",
    }, EXP_ROOT / run_name / f"topk_{top_k}" / "config.json")
    print(f"Saved {out}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Device:", DEVICE)
    print("Loading model:", MODEL_PATH)

    scene_enc, cand_enc = load_model()

    candidates = load_json(TRAIN_CANDIDATES)
    queries = load_json(TEST_QUERIES)
    prior = load_json(PRIOR_JSON)
    dino_by_key = load_dino_cache_by_key()

    print("Train candidates:", len(candidates))
    print("Test queries:", len(queries))
    print("Sweep runs:", len(SWEEP_RUNS))

    with torch.no_grad():
        cand_embs = encode_candidates(cand_enc, candidates, dino_by_key)
        q_embs = encode_queries(scene_enc, queries)

    valid_masks = build_valid_masks(queries, candidates)
    context_cache = build_context_cache(queries, candidates, prior)

    obs_by_image = defaultdict(list)
    for q in queries:
        obs_by_image[int(q["image_id"])].append(q)

    cand_t = build_candidate_tensors(candidates)

    with torch.no_grad():
        comp = precompute_components(
            queries=queries,
            candidates=candidates,
            q_embs=q_embs,
            cand_embs=cand_embs,
            context_cache=context_cache,
            obs_by_image=obs_by_image,
            cand_t=cand_t,
        )

    for name, ln, lc, lo in SWEEP_RUNS:
        for k in TOPKS:
            run_one(name, k, queries, candidates, valid_masks, comp, ln, lc, lo)

    print("Done.")


if __name__ == "__main__":
    main()
