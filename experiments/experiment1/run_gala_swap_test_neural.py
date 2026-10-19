#!/usr/bin/env python3
"""
Fair swap test, trained-neural appearance version.

Hold the appearance retriever constant (the trained SEALECT SceneRetriever neural
score q.k) and swap only the geometry term. Compare neural + explicit generative
prior (SEALECT) against neural + learned GALA-style discriminative compatibility.
This isolates the reviewer's exact question: explicit generative prior vs learned
discriminative compatibility as the geometry signal, with appearance fixed to the
same trained neural retriever.

Companion to run_gala_swap_test.py, which runs the same swap with DINO cosine
similarity as the fixed appearance term. Here appearance is the trained neural
retriever instead, so the margins are sharper but the story is the same trade-off.
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

ROOT = Path("/path/to/sealect_root")
DATA_ROOT = ROOT / "experiments/dataset_decide"
SPLIT_DIR = DATA_ROOT / "splits_dino_decorrelated_kbalanced_grounded_sam2"
MERGED_DIR = DATA_ROOT / "merged_scene_pose_library_out_grounded_sam2"
TRAIN_PAIRS = SPLIT_DIR / "train_scene_object_pairs.json"
TRAIN_CANDIDATES = SPLIT_DIR / "train_candidate_library.json"
TEST_QUERIES = SPLIT_DIR / "test_scene_object_pairs.json"
PRIOR_JSON = DATA_ROOT / "scene_pose_prior_train_grounded_sam2.json"
DINO_CACHE = SPLIT_DIR / "cache" / "candidate_dino_embeddings.pt"
EXP_ROOT = ROOT / "experiments/experiment1/exp1_retrieval_quality"
MODEL_PATH = EXP_ROOT / "sealect_neural_dino" / "sealect_neural_retriever_dino.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EPS = 1e-8
ENCODE_BATCH = 4096
QUERY_BATCH = 256
FALLBACK_EMBED_DIM = 256
NUM_YAW_BINS = 18
VIEWPOINTS = ["nadir_aerial", "oblique_aerial", "horizon_view", "low_oblique_or_shore", "unknown_viewpoint"]
LAYOUTS = ["open_water", "open_water_uncertain", "coastal_transition", "harbor_port", "water_land_mixed", "non_maritime_or_uncertain"]
POSE_FAMILIES = ["nadir__front_back__level", "nadir__side__level", "nadir__oblique__level", "level__front_back__level", "level__side__level", "level__oblique__level", "level__front_back__rolled", "level__side__rolled", "level__oblique__rolled", "steep_up__front_back__level", "steep_up__side__level", "steep_up__oblique__level"]
YAW_FAMILIES = ["front_back", "side", "oblique"]


# 
# metric (verbatim from run_gala_swap_test.py)
# 
def log_area(x): return math.log(max(float(x), EPS))
def circ(a, b, n=18): d = abs(int(a) - int(b)); return min(d, n - d)
def relevance(q, c):
    s = 0
    if abs(int(q["pitch_bin"]) - int(c["pitch_bin"])) <= 0: s += 1
    if circ(int(q["yaw_bin"]), int(c["yaw_bin"])) <= 1: s += 1
    if abs(int(q["y_bin"]) - int(c["y_bin"])) <= 0: s += 1
    if abs(log_area(q["area_frac"]) - log_area(c["area_frac"])) <= 0.30: s += 1
    return s
def dcg(xs): return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(xs))
def ndcg(xs):
    d = dcg(sorted(xs, reverse=True)); return dcg(xs) / d if d > 0 else 0.0
def summarize(results, k):
    rec, nd, ps = [], [], []
    for row in results:
        q, cands = row["query"], row["topk"][:k]
        if not cands: continue
        rels = [relevance(q, c) for c in cands]
        rec.append(1.0 if max(rels) >= 4 else 0.0); nd.append(ndcg(rels)); ps.append(sum(rels) / max(1, len(rels)))
    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"Recall": m(rec), "NDCG": m(nd), "PS": m(ps)}
def znorm(x):
    x = np.asarray(x, np.float32); return (x - x.mean()) / (x.std() + 1e-6)


# 
# trained SceneRetriever: features + arch (verbatim from
# run_sealect_exp1_fast_cuda.py / train_sealect_neural_retriever_grounded_sam2.py)
# 
def load_json(path: Path):
    with open(path, "r") as f: return json.load(f)
def key_for_record(r: Dict) -> Tuple[int, int]: return int(r["image_id"]), int(r["annotation_id"])
def one_hot(value: str, choices: List[str]) -> List[float]: return [1.0 if value == c else 0.0 for c in choices]


def scene_features(r: Dict) -> np.ndarray:
    vp = r.get("vertical_profile", {})
    water_v = vp.get("water", [0.0] * 16); sky_v = vp.get("sky", [0.0] * 16)
    land_v = vp.get("land", [0.0] * 16); other_v = vp.get("other", [0.0] * 16)
    x = [
        float(r.get("water_frac", 0.0)), float(r.get("sky_frac", 0.0)), float(r.get("land_frac", 0.0)),
        float(r.get("other_frac", 0.0)), float(r.get("vessel_ignore_frac", 0.0)),
        float(r.get("horizon_y_norm", 0.0)), float(r.get("horizon_confidence", 0.0)),
        float(r.get("water_row_mean", 0.0)), float(r.get("water_row_std", 0.0)),
        float(r.get("sky_row_mean", 0.0)), float(r.get("sky_row_std", 0.0)),
        float(r.get("land_row_mean", 0.0)), float(r.get("land_row_std", 0.0)),
        float(r.get("center_x_norm", 0.0)), float(r.get("center_y_norm", 0.0)),
        int(r.get("y_bin", 0)) / 7.0, math.log(max(float(r.get("area_frac", EPS)), EPS)),
        float(r.get("aspect_ratio", 0.0)),
    ]
    x.extend([float(v) for v in water_v]); x.extend([float(v) for v in sky_v])
    x.extend([float(v) for v in land_v]); x.extend([float(v) for v in other_v])
    x.extend(one_hot(str(r.get("viewpoint_type", "unknown_viewpoint")), VIEWPOINTS))
    x.extend(one_hot(str(r.get("scene_layout", "non_maritime_or_uncertain")), LAYOUTS))
    return np.asarray(x, dtype=np.float32)


def base_candidate_features(r):
    emb = r.get("embedding", []) or [0.0] * FALLBACK_EMBED_DIM
    x = [int(r.get("pitch_bin", 0)) / 17.0, int(r.get("yaw_bin", 0)) / 17.0, int(r.get("roll_bin", 0)) / 17.0,
         int(r.get("y_bin", 0)) / 7.0, math.log(max(float(r.get("area_frac", EPS)), EPS)),
         float(r.get("aspect_ratio", 0.0)), float(r.get("center_x_norm", 0.0)), float(r.get("center_y_norm", 0.0)),
         float(r.get("water_frac", 0.0)), float(r.get("sky_frac", 0.0)), float(r.get("land_frac", 0.0)),
         float(r.get("horizon_y_norm", 0.0)), float(r.get("horizon_confidence", 0.0))]
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
    dino_by_key = {key_for_record(candidates_paired[raw_idx]): emb[j] for j, raw_idx in enumerate(valid_indices)}
    print(f"Loaded DINO cache vectors={len(dino_by_key)} dim={emb.shape[1]}", flush=True)
    return dino_by_key


def candidate_features_with_dino(c, dino_by_key):
    dino = dino_by_key[key_for_record(c)]
    dino = dino / (np.linalg.norm(dino) + EPS)
    x = base_candidate_features(c)
    x.extend(float(v) for v in dino)
    return np.asarray(x, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.LayerNorm(512), nn.Dropout(0.05),
                                 nn.Linear(512, 256), nn.ReLU(inplace=True), nn.LayerNorm(256), nn.Linear(256, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    scene_enc = MLP(int(ckpt["scene_dim"]), int(ckpt["emb_dim"])).to(DEVICE)
    cand_enc = MLP(int(ckpt["cand_dim"]), int(ckpt["emb_dim"])).to(DEVICE)
    scene_enc.load_state_dict(ckpt["scene_enc"]); cand_enc.load_state_dict(ckpt["cand_enc"])
    return scene_enc.eval(), cand_enc.eval()


@torch.no_grad()
def encode_candidates(cand_enc, candidates, dino_by_key):
    xs = torch.stack([torch.from_numpy(candidate_features_with_dino(c, dino_by_key)) for c in candidates]).float()
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
# explicit generative prior S_context (verbatim from run_sealect_exp1_fast_cuda.py)
# 
def log_prob_hist_array(hist, n_bins=18):
    arr = np.full((n_bins,), math.log(EPS), dtype=np.float32)
    for k, v in hist.items():
        idx = int(k)
        if 0 <= idx < n_bins: arr[idx] = math.log(max(float(v), EPS))
    return arr
def pose_key(c): return f"{int(c.get('pitch_bin', 0))}_{int(c.get('yaw_bin', 0))}_{int(c.get('roll_bin', 0))}"


def context_score_numpy(query, candidates, prior):
    context_key = query.get("context_key"); scene_type = query.get("scene_type")
    def get_prob_dict(section):
        if context_key is not None and prior.get(section, {}).get(context_key) is not None: return prior[section][context_key]
        scene_section = section.replace("context_", "scene_")
        if scene_type is not None and prior.get(scene_section, {}).get(scene_type) is not None: return prior[scene_section][scene_type]
        return {}
    pitch_hist = log_prob_hist_array(get_prob_dict("context_pitch_hist"))
    yaw_hist = log_prob_hist_array(get_prob_dict("context_yaw_hist"))
    roll_hist = log_prob_hist_array(get_prob_dict("context_roll_hist"))
    yaw_weight = float(prior.get("context_yaw_weight", {}).get(context_key, prior.get("scene_yaw_weight", {}).get(scene_type, 1.0)))
    pitch = np.asarray([int(c.get("pitch_bin", 0)) for c in candidates], dtype=np.int64)
    yaw = np.asarray([int(c.get("yaw_bin", 0)) for c in candidates], dtype=np.int64)
    roll = np.asarray([int(c.get("roll_bin", 0)) for c in candidates], dtype=np.int64)
    ybin = np.asarray([int(c.get("y_bin", 0)) for c in candidates], dtype=np.int64)
    log_area_c = np.asarray([math.log(max(float(c.get("area_frac", EPS)), EPS)) for c in candidates], dtype=np.float32)
    s = 2.0 * pitch_hist[np.clip(pitch, 0, 17)] + yaw_weight * yaw_hist[np.clip(yaw, 0, 17)] + roll_hist[np.clip(roll, 0, 17)]
    mu = np.full_like(log_area_c, math.log(1e-4), dtype=np.float32); sig = np.full_like(log_area_c, 1.5, dtype=np.float32)
    for y in range(8):
        stats = None; yb = str(y)
        if context_key is not None: stats = prior.get("scale_by_context_ybin", {}).get(context_key, {}).get(yb)
        if stats is None and scene_type is not None: stats = prior.get("scale_by_scene_ybin", {}).get(scene_type, {}).get(yb)
        if stats is not None:
            m = ybin == y; mu[m] = float(stats.get("log_area_mean", math.log(1e-4))); sig[m] = max(float(stats.get("log_area_std", 1.5)), 0.15)
    s += 0.7 * (-0.5 * np.log(2.0 * np.pi * sig * sig) - 0.5 * ((log_area_c - mu) / sig) ** 2)
    y_logp = np.full_like(log_area_c, math.log(EPS), dtype=np.float32)
    for i, c in enumerate(candidates):
        pk = pose_key(c); hist = None
        if context_key is not None: hist = prior.get("ybin_given_context_pose", {}).get(context_key, {}).get(pk)
        if hist is None and scene_type is not None: hist = prior.get("ybin_given_scene_pose", {}).get(scene_type, {}).get(pk)
        if hist: y_logp[i] = math.log(max(float(hist.get(str(int(ybin[i])), 0.0)), EPS))
    return (s + 0.5 * y_logp).astype(np.float32)


def build_context_cache(queries, candidates, prior):
    cache, uniq = {}, {}
    for q in queries:
        key = (str(q.get("context_key")), str(q.get("scene_type")))
        if key not in uniq: uniq[key] = q
    for key, q in tqdm(uniq.items(), desc="precompute S_context (prior)"):
        cache[key] = context_score_numpy(q, candidates, prior)
    return cache


# 
# learned GALA-style discriminative compatibility (verbatim from run_gala_swap_test.py)
# 
GALA_SCENE_KEYS = ["water_frac", "sky_frac", "land_frac", "other_frac", "horizon_y_norm",
                   "water_row_mean", "water_row_std", "sky_row_mean", "sky_row_std", "land_row_mean", "land_row_std"]
def gala_scene_vec(r): return [float(r.get(k) or 0.0) for k in GALA_SCENE_KEYS]
def gala_geom_vec(r, Np, Ny, Nr):
    def sc(b, N): a = 2 * math.pi * int(b) / max(1, N); return [math.sin(a), math.cos(a)]
    return (sc(r["pitch_bin"], Np) + sc(r["yaw_bin"], Ny) + sc(r["roll_bin"], Nr) +
            [int(r["y_bin"]) / 8.0, log_area(r["area_frac"]),
             math.log(max(float(r.get("aspect_ratio", 1.0)), 1e-6)),
             float(r.get("center_x_norm") or 0.5), float(r.get("center_y_norm") or 0.5)])


def train_gala(train, Np, Ny, Nr):
    S = np.asarray([gala_scene_vec(r) for r in train], np.float32)
    G = np.asarray([gala_geom_vec(r, Np, Ny, Nr) for r in train], np.float32)
    smu, ssd = S.mean(0), S.std(0) + 1e-6; gmu, gsd = G.mean(0), G.std(0) + 1e-6
    St = torch.tensor((S - smu) / ssd, device=DEVICE); Gt = torch.tensor((G - gmu) / gsd, device=DEVICE)
    model = nn.Sequential(nn.Linear(S.shape[1] + G.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    N = St.shape[0]; bs = 1024
    print("training GALA scorer...", flush=True)
    for ep in range(8):
        model.train(); perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]; s = St[idx]; gpos = Gt[idx]; gneg = Gt[idx[torch.randperm(len(idx), device=DEVICE)]]
            x = torch.cat([torch.cat([s, gpos], 1), torch.cat([s, gneg], 1)], 0)
            y = torch.cat([torch.ones(len(idx), device=DEVICE), torch.zeros(len(idx), device=DEVICE)])
            opt.zero_grad(); bce(model(x).squeeze(1), y).backward(); opt.step()
    model.eval()
    return model, (smu, ssd, gmu, gsd)


# 
def load_existing(name, k):
    p = EXP_ROOT / name / f"topk_{k}" / "retrieval_results.json"
    return load_json(p) if p.exists() else None


def main():
    print("Device:", DEVICE, flush=True)
    train = load_json(TRAIN_PAIRS)
    candidates = load_json(TRAIN_CANDIDATES)
    queries = load_json(TEST_QUERIES)
    prior = load_json(PRIOR_JSON)
    print(f"train {len(train)}  queries {len(queries)}  candidates {len(candidates)}", flush=True)
    Np = max(int(r["pitch_bin"]) for r in train) + 1; Ny = NUM_YAW_BINS; Nr = max(int(r["roll_bin"]) for r in train) + 1

    # 1. trained neural appearance term
    scene_enc, cand_enc = load_model()
    dino_by_key = load_dino_cache_by_key()
    with torch.no_grad():
        cand_embs = encode_candidates(cand_enc, candidates, dino_by_key)   # (Nc,128) normalized
        q_embs = encode_queries(scene_enc, queries)                        # (Nq,128) normalized

    # 2. explicit generative prior (geometry term A)
    context_cache = build_context_cache(queries, candidates, prior)

    # 3. learned GALA compatibility (geometry term B)
    gala_model, (gsmu, gssd, ggmu, ggsd) = train_gala(train, Np, Ny, Nr)
    Cg = torch.tensor((np.asarray([gala_geom_vec(c, Np, Ny, Nr) for c in candidates], np.float32) - ggmu) / ggsd, device=DEVICE)

    # candidate validity arrays + slim dicts for the metric
    cgroup = np.asarray([str(c.get("sequence_group", "")) for c in candidates])
    cimg = np.asarray([int(c.get("image_id", -2)) for c in candidates])
    cann = np.asarray([int(c.get("annotation_id", -2)) for c in candidates])
    keep = ["pitch_bin", "yaw_bin", "y_bin", "area_frac"]
    cslim = [{k: c.get(k) for k in keep} for c in candidates]

    methods = ["neural_only", "prior_only", "gala_only", "neural_prior", "neural_gala"]
    res = {m: [] for m in methods}
    Nc = cand_embs.shape[0]
    with torch.no_grad():
        for qi, q in enumerate(tqdm(queries, desc="score queries")):
            neural = (cand_embs @ q_embs[qi]).cpu().numpy().astype(np.float32)   # (Nc,)
            prior_raw = context_cache[(str(q.get("context_key")), str(q.get("scene_type")))]
            sv = torch.tensor((np.asarray(gala_scene_vec(q), np.float32) - gsmu) / gssd, device=DEVICE).unsqueeze(0).expand(Nc, -1)
            gala = gala_model(torch.cat([sv, Cg], 1)).squeeze(1).cpu().numpy().astype(np.float32)

            invalid = (cgroup == str(q.get("sequence_group", ""))) | \
                      ((cimg == int(q.get("image_id", -1))) & (cann == int(q.get("annotation_id", -1))))
            qslim = {k: q.get(k) for k in keep}
            combos = {
                "neural_only": znorm(neural),
                "prior_only": znorm(prior_raw),
                "gala_only": znorm(gala),
                "neural_prior": znorm(neural) + znorm(prior_raw),
                "neural_gala": znorm(neural) + znorm(gala),
            }
            for name, base in combos.items():
                sc = base.copy().astype(np.float32); sc[invalid] = -1e9
                top = np.argpartition(-sc, 100)[:100]; top = top[np.argsort(-sc[top])]
                res[name].append({"query": qslim, "topk": [cslim[j] for j in top]})

    # sanity cross-check: my neural_only should reproduce the validated neural_only baseline
    print("\n SANITY: reproduce validated baselines from disk")
    for k in (20, 100):
        for disp, name in (("neural_only (disk)", f"neural_only_dinofeat_k{k}"),
                           ("SEALECT full (disk)", f"sealect_dinofeat_k{k}")):
            r = load_existing(name, k)
            if r:
                m = summarize(r, k)
                print(f"  K={k:<3d} {disp:22s} R={m['Recall']:.3f} NDCG={m['NDCG']:.3f} PS={m['PS']:.3f}")

    print("\n FAIR SWAP TEST: geometry term swapped, neural appearance held constant")
    label = {"neural_only": "neural only (appearance)", "prior_only": "prior only (geom A)",
             "gala_only": "GALA only (geom B)", "neural_prior": "neural + prior  (SEALECT)",
             "neural_gala": "neural + GALA  (learned)"}
    for k in (20, 100):
        print(f"\n--- K={k} ---")
        print(f"{'method':30s} {'Recall':>8s} {'NDCG':>8s} {'PS':>8s}")
        for m in methods:
            s = summarize(res[m], k)
            star = "   <== swap" if m in ("neural_prior", "neural_gala") else ""
            print(f"{label[m]:30s} {s['Recall']:8.3f} {s['NDCG']:8.3f} {s['PS']:8.3f}{star}")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
