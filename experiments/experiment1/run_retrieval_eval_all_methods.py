#!/usr/bin/env python3
"""
run_retrieval_eval_all_methods.py

Unified retrieval runner for SeaPlace/Sealect experiments.

Supports:
  - random
  - prior_only
  - dino
  - clip
  - neural_only       (requires external neural score JSON)
  - sealect          (neural/visual + context prior + observed-pose support + optional MMR)
  - sealect_dino     (DINO as S_neural proxy + priors)
  - sealect_clip     (CLIP as S_neural proxy + priors)

Inputs are the DINO-decorrelated Grounded-SAM2 split outputs:
  splits_dino_decorrelated_kbalanced_grounded_sam2/
      train_candidate_library.json
      test_candidate_library.json
      test_scene_object_pairs.json
  scene_pose_prior_train_grounded_sam2.json

Retrieval pool: train candidates only.
Queries: test scene-object pairs; their crop paths are resolved from test_candidate_library.

Scoring:
  S_j = lambda_n S_neural(j)
      + lambda_c S_context(j)
      + lambda_o S_obs(j)
      + lambda_d S_div(j)

where S_div is applied greedily as MMR during top-K selection.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T

EPS = 1e-8


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def key_for_record(r: Dict) -> Tuple[int, int]:
    return (int(r["image_id"]), int(r["annotation_id"]))


def pose_key(c: Dict) -> str:
    return f"{int(c['pitch_bin'])}_{int(c['yaw_bin'])}_{int(c['roll_bin'])}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


# -----------------------------
# Prior scoring
# -----------------------------

def log_prob_lookup(prob_dict: Dict, key) -> float:
    p = float(prob_dict.get(str(key), 0.0))
    return math.log(max(p, EPS))


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), EPS)
    return -0.5 * math.log(2.0 * math.pi * sigma * sigma) - 0.5 * ((x - mu) / sigma) ** 2


def get_prob_dict(prior: Dict, section: str, context_key: Optional[str], scene_type: Optional[str]) -> Dict:
    if context_key is not None:
        d = prior.get(section, {}).get(context_key)
        if d is not None:
            return d

    if scene_type is not None:
        scene_section = section.replace("context_", "scene_")
        d = prior.get(scene_section, {}).get(scene_type)
        if d is not None:
            return d

    return {}


def get_yaw_weight(prior: Dict, context_key: Optional[str], scene_type: Optional[str]) -> float:
    if context_key is not None:
        w = prior.get("context_yaw_weight", {}).get(context_key)
        if w is not None:
            return float(w)
    if scene_type is not None:
        w = prior.get("scene_yaw_weight", {}).get(scene_type)
        if w is not None:
            return float(w)
    return 1.0


def get_scale_stats(prior: Dict, y_bin: int, context_key: Optional[str], scene_type: Optional[str]) -> Dict:
    yb = str(int(y_bin))
    if context_key is not None:
        d = prior.get("scale_by_context_ybin", {}).get(context_key, {}).get(yb)
        if d is not None:
            return d
    if scene_type is not None:
        d = prior.get("scale_by_scene_ybin", {}).get(scene_type, {}).get(yb)
        if d is not None:
            return d
    return {"log_area_mean": math.log(1e-4), "log_area_std": 1.5, "count": 0}


def get_ybin_given_pose(prior: Dict, pkey: str, context_key: Optional[str], scene_type: Optional[str]) -> Dict:
    if context_key is not None:
        d = prior.get("ybin_given_context_pose", {}).get(context_key, {}).get(pkey)
        if d is not None:
            return d
    if scene_type is not None:
        d = prior.get("ybin_given_scene_pose", {}).get(scene_type, {}).get(pkey)
        if d is not None:
            return d
    return {}


def context_prior_score(query: Dict, cand: Dict, prior: Dict) -> float:
    """S_context: global scene-conditioned pose/scale/placement likelihood."""
    context_key = query.get("context_key")
    scene_type = query.get("scene_type")

    pitch_hist = get_prob_dict(prior, "context_pitch_hist", context_key, scene_type)
    yaw_hist = get_prob_dict(prior, "context_yaw_hist", context_key, scene_type)
    roll_hist = get_prob_dict(prior, "context_roll_hist", context_key, scene_type)

    pitch_logp = log_prob_lookup(pitch_hist, int(cand["pitch_bin"]))
    yaw_logp = log_prob_lookup(yaw_hist, int(cand["yaw_bin"]))
    roll_logp = log_prob_lookup(roll_hist, int(cand["roll_bin"]))

    yaw_weight = get_yaw_weight(prior, context_key, scene_type)

    # Pitch and roll remain useful for visible geometry; yaw is modulated by context entropy.
    s = 2.0 * pitch_logp + yaw_weight * yaw_logp + roll_logp

    stats = get_scale_stats(prior, int(cand["y_bin"]), context_key, scene_type)
    log_area = math.log(max(float(cand["area_frac"]), EPS))
    s += 0.7 * gaussian_logpdf(
        log_area,
        float(stats["log_area_mean"]),
        max(float(stats["log_area_std"]), 0.15),
    )

    ybin_hist = get_ybin_given_pose(prior, pose_key(cand), context_key, scene_type)
    s += 0.5 * log_prob_lookup(ybin_hist, int(cand["y_bin"]))
    return float(s)


def circular_bin_distance(a: int, b: int, n: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, n - d)


def pose_kernel_score(cand: Dict, obs: Dict, num_yaw_bins: int, sigma_pitch: float, sigma_yaw: float, sigma_roll: float) -> float:
    dp = abs(int(cand["pitch_bin"]) - int(obs["pitch_bin"]))
    dy = circular_bin_distance(int(cand["yaw_bin"]), int(obs["yaw_bin"]), num_yaw_bins)
    dr = abs(int(cand["roll_bin"]) - int(obs["roll_bin"]))
    return math.exp(
        -0.5 * ((dp / sigma_pitch) ** 2 + (dy / sigma_yaw) ** 2 + (dr / sigma_roll) ** 2)
    )


def observed_pose_score(
    query: Dict,
    cand: Dict,
    obs_by_image: Dict[int, List[Dict]],
    num_yaw_bins: int = 18,
    alpha: float = 0.7,
    sigma_pitch: float = 1.5,
    sigma_yaw: float = 2.0,
    sigma_roll: float = 1.5,
) -> float:
    """S_obs: smoothed support under poses already present in the query image.

    This is intentionally NOT nearest-neighbor pose matching. It is a broad kernel-density support
    estimate mixed with a uniform term, so plausible nearby modes are boosted without collapsing top-K.
    """
    image_id = int(query.get("image_id", -1))
    observations = obs_by_image.get(image_id, [])

    # Exclude the query vessel itself from the observed set.
    q_ann = int(query.get("annotation_id", -1))
    observations = [o for o in observations if int(o.get("annotation_id", -2)) != q_ann]

    if not observations:
        return 0.0

    vals = [pose_kernel_score(cand, o, num_yaw_bins, sigma_pitch, sigma_yaw, sigma_roll) for o in observations]
    kde = float(np.mean(vals))

    # Uniform background over coarse joint pose space. Exact size is unimportant because scores are normalized.
    uniform = 1.0 / max(1.0, float(num_yaw_bins * 18 * 18))
    p = (1.0 - alpha) * uniform + alpha * kde
    return math.log(max(p, EPS))


def normalize_scores(scores: List[float]) -> List[float]:
    arr = np.asarray(scores, dtype=np.float32)
    if len(arr) == 0:
        return []
    return ((arr - arr.mean()) / (arr.std() + 1e-6)).tolist()


# -----------------------------
# Embeddings
# -----------------------------

def resolve_crop_path(crop_path: Optional[str], library_root: Path) -> Optional[Path]:
    if not crop_path:
        return None
    p = Path(crop_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    q = library_root / p
    if q.exists():
        return q
    q = library_root / p.name
    if q.exists():
        return q
    # Common case: crop_path is relative to project root or output dir.
    for sub in [
        library_root / "merged_scene_pose_library_out_grounded_sam2" / "crops",
        library_root / "merged_scene_pose_library_out" / "crops",
    ]:
        q = sub / p.name
        if q.exists():
            return q
    return None


def dino_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def compute_dino_embeddings(records: List[Dict], cache_path: Path, library_root: Path, batch_size: int, device: str, model_name: str) -> Tuple[np.ndarray, List[int]]:
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        return obj["embeddings"].numpy().astype(np.float32), list(obj["valid_indices"])

    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
    model.eval()
    tf = dino_transform()

    embs, valid = [], []
    for i in tqdm(range(0, len(records), batch_size), desc=f"DINO embeddings {cache_path.name}"):
        batch = records[i:i + batch_size]
        imgs, idxs = [], []
        for j, r in enumerate(batch):
            p = resolve_crop_path(r.get("crop_path"), library_root)
            if p is None:
                continue
            try:
                imgs.append(tf(Image.open(p).convert("RGB")))
                idxs.append(i + j)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        feats = model(x)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken")
        feats = F.normalize(feats, dim=-1).cpu()
        embs.append(feats)
        valid.extend(idxs)

    if not embs:
        raise RuntimeError("No DINO embeddings computed. Check crop paths.")
    arr = torch.cat(embs, dim=0).numpy().astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": torch.from_numpy(arr), "valid_indices": valid}, cache_path)
    return arr, valid


@torch.no_grad()
def compute_clip_embeddings(records: List[Dict], cache_path: Path, library_root: Path, batch_size: int, device: str, model_id: str) -> Tuple[np.ndarray, List[int]]:
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        return obj["embeddings"].numpy().astype(np.float32), list(obj["valid_indices"])

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    embs, valid = [], []
    for i in tqdm(range(0, len(records), batch_size), desc=f"CLIP embeddings {cache_path.name}"):
        batch = records[i:i + batch_size]
        imgs, idxs = [], []
        for j, r in enumerate(batch):
            p = resolve_crop_path(r.get("crop_path"), library_root)
            if p is None:
                continue
            try:
                imgs.append(Image.open(p).convert("RGB"))
                idxs.append(i + j)
            except Exception:
                continue
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1).cpu()
        embs.append(feats)
        valid.extend(idxs)

    if not embs:
        raise RuntimeError("No CLIP embeddings computed. Check crop paths.")
    arr = torch.cat(embs, dim=0).numpy().astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": torch.from_numpy(arr), "valid_indices": valid}, cache_path)
    return arr, valid


def load_neural_scores(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Expected format: {query_key: {candidate_id: score}}, where query_key is imageId_annotationId."""
    if path is None:
        raise RuntimeError("neural_only/sealect with neural requires --neural_scores_json")
    return load_json(path)


def query_key(q: Dict) -> str:
    return f"{int(q['image_id'])}_{int(q['annotation_id'])}"


# -----------------------------
# Candidate filtering / formatting
# -----------------------------

def is_valid_candidate(query: Dict, cand: Dict) -> bool:
    # Train/test split should already prevent leakage; keep safeguards for reruns/custom splits.
    same_instance = (
        int(query.get("image_id", -1)) == int(cand.get("image_id", -2))
        and int(query.get("annotation_id", -1)) == int(cand.get("annotation_id", -2))
    )
    same_group = str(query.get("sequence_group", "")) == str(cand.get("sequence_group", ""))
    return not same_instance and not same_group


def compact_context(r: Dict) -> Dict:
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


def format_result(query: Dict, selected: List[Dict], scores_by_id: Optional[Dict[str, Dict]] = None) -> Dict:
    topk = []
    for rank, cand in enumerate(selected, 1):
        cid = cand["candidate_id"]
        rec = {
            "rank": rank,
            "candidate_id": cid,
            "dataset": cand.get("dataset"),
            "source": cand.get("source"),
            "sequence_group": cand.get("sequence_group"),
            "image_id": cand.get("image_id"),
            "annotation_id": cand.get("annotation_id"),
            "image_name": cand.get("image_name"),
            "crop_path": cand.get("crop_path"),
            "image_path": cand.get("image_path"),
            "pitch_bin": cand.get("pitch_bin"),
            "yaw_bin": cand.get("yaw_bin"),
            "roll_bin": cand.get("roll_bin"),
            "pitch_deg": cand.get("pitch_deg"),
            "yaw_deg": cand.get("yaw_deg"),
            "roll_deg": cand.get("roll_deg"),
            "pose_family": cand.get("pose_family"),
            "yaw_family": cand.get("yaw_family"),
            "area_frac": cand.get("area_frac"),
            "y_bin": cand.get("y_bin"),
            **compact_context(cand),
        }
        if scores_by_id and cid in scores_by_id:
            rec.update(scores_by_id[cid])
        topk.append(rec)

    return {
        "query": {
            "dataset": query.get("dataset"),
            "source": query.get("source"),
            "sequence_group": query.get("sequence_group"),
            "image_id": query.get("image_id"),
            "annotation_id": query.get("annotation_id"),
            "image_name": query.get("image_name"),
            "center_x_norm": query.get("center_x_norm"),
            "center_y_norm": query.get("center_y_norm"),
            "y_bin": query.get("y_bin"),
            "area_frac": query.get("area_frac"),
            "pitch_bin": query.get("pitch_bin"),
            "yaw_bin": query.get("yaw_bin"),
            "roll_bin": query.get("roll_bin"),
            "pitch_deg": query.get("pitch_deg"),
            "yaw_deg": query.get("yaw_deg"),
            "roll_deg": query.get("roll_deg"),
            "pose_family": query.get("pose_family"),
            "yaw_family": query.get("yaw_family"),
            **compact_context(query),
        },
        "topk": topk,
    }


def greedy_mmr_select(
    valid: List[Dict],
    base_scores: np.ndarray,
    candidate_embs: Optional[np.ndarray],
    top_k: int,
    lambda_d: float,
) -> Tuple[List[int], np.ndarray]:
    """Return selected indices into valid list and final MMR scores at selection time."""
    if top_k <= 0 or len(valid) == 0:
        return [], np.zeros(0, dtype=np.float32)

    if lambda_d <= 0 or candidate_embs is None:
        idxs = np.argsort(-base_scores)[:top_k]
        return idxs.tolist(), base_scores[idxs]

    selected = []
    selected_scores = []
    remaining = set(range(len(valid)))

    # assume normalized embeddings
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: float(base_scores[i]))
            score = float(base_scores[best])
        else:
            sel_emb = candidate_embs[selected]
            best, score = None, -1e18
            for i in remaining:
                sim_to_sel = float((sel_emb @ candidate_embs[i]).max())
                mmr_score = float(base_scores[i]) - lambda_d * sim_to_sel
                if mmr_score > score:
                    best, score = i, mmr_score
        selected.append(best)
        selected_scores.append(score)
        remaining.remove(best)

    return selected, np.asarray(selected_scores, dtype=np.float32)


# -----------------------------
# Main scoring runner
# -----------------------------

def build_embedding_maps(
    method: str,
    train_candidates: List[Dict],
    test_candidates: List[Dict],
    args,
) -> Tuple[Optional[Dict[Tuple[int, int], np.ndarray]], Optional[Dict[Tuple[int, int], np.ndarray]]]:
    if method not in {"dino", "clip", "sealect_dino", "sealect_clip"}:
        return None, None

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if method in {"dino", "sealect_dino"}:
        train_emb, train_valid = compute_dino_embeddings(
            train_candidates,
            args.cache_dir / "train_dino_embeddings.pt",
            args.library_root,
            args.batch_size,
            args.device,
            args.dino_model,
        )
        test_emb, test_valid = compute_dino_embeddings(
            test_candidates,
            args.cache_dir / "test_dino_embeddings.pt",
            args.library_root,
            args.batch_size,
            args.device,
            args.dino_model,
        )
    else:
        train_emb, train_valid = compute_clip_embeddings(
            train_candidates,
            args.cache_dir / "train_clip_embeddings.pt",
            args.library_root,
            args.batch_size,
            args.device,
            args.clip_model,
        )
        test_emb, test_valid = compute_clip_embeddings(
            test_candidates,
            args.cache_dir / "test_clip_embeddings.pt",
            args.library_root,
            args.batch_size,
            args.device,
            args.clip_model,
        )

    train_emb = l2_normalize(train_emb)
    test_emb = l2_normalize(test_emb)

    train_map = {key_for_record(train_candidates[i]): train_emb[j] for j, i in enumerate(train_valid)}
    test_map = {key_for_record(test_candidates[i]): test_emb[j] for j, i in enumerate(test_valid)}
    return train_map, test_map


def run_method(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train_candidates = load_json(args.train_candidates)
    test_queries = load_json(args.test_queries)
    test_candidates = load_json(args.test_candidates)
    prior = load_json(args.prior_json) if args.prior_json and Path(args.prior_json).exists() else {}

    print(f"method:           {args.method}")
    print(f"train candidates: {len(train_candidates)}")
    print(f"test queries:     {len(test_queries)}")
    print(f"test candidates:  {len(test_candidates)}")
    print(f"top_k:            {args.top_k}")

    test_candidate_by_key = {key_for_record(c): c for c in test_candidates}
    obs_by_image = defaultdict(list)
    for q in test_queries:
        obs_by_image[int(q["image_id"])].append(q)

    emb_train_map, emb_test_map = build_embedding_maps(args.method, train_candidates, test_candidates, args)
    neural_scores = None
    if args.method == "neural_only" or args.method == "sealect":
        neural_scores = load_neural_scores(args.neural_scores_json)

    rng = random.Random(args.random_seed)
    results = []

    for qi, q in enumerate(tqdm(test_queries, desc=f"retrieval {args.method}")):
        valid = [c for c in train_candidates if is_valid_candidate(q, c)]
        if not valid:
            results.append(format_result(q, []))
            continue

        scores_by_component = {}

        if args.method == "random":
            selected = rng.sample(valid, min(args.top_k, len(valid)))
            results.append(format_result(q, selected))
            continue

        # Component scores, normalized within each query for stable lambda behavior.
        context_raw = np.zeros(len(valid), dtype=np.float32)
        obs_raw = np.zeros(len(valid), dtype=np.float32)
        neural_raw = np.zeros(len(valid), dtype=np.float32)

        if args.lambda_c != 0 or args.method == "prior_only":
            context_raw = np.asarray([context_prior_score(q, c, prior) for c in valid], dtype=np.float32)

        if args.lambda_o != 0:
            obs_raw = np.asarray([
                observed_pose_score(
                    q, c, obs_by_image,
                    num_yaw_bins=args.num_yaw_bins,
                    alpha=args.obs_alpha,
                    sigma_pitch=args.obs_sigma_pitch,
                    sigma_yaw=args.obs_sigma_yaw,
                    sigma_roll=args.obs_sigma_roll,
                )
                for c in valid
            ], dtype=np.float32)

        if args.method in {"dino", "clip", "sealect_dino", "sealect_clip"}:
            q_record = test_candidate_by_key.get(key_for_record(q), q)
            q_emb = emb_test_map.get(key_for_record(q_record))
            # fallback key directly from query if a matching test candidate record is missing
            if q_emb is None:
                q_emb = emb_test_map.get(key_for_record(q))
            if q_emb is None:
                # If query crop missing, leave neural score zero and rely on priors.
                neural_raw = np.zeros(len(valid), dtype=np.float32)
            else:
                cand_embs = []
                missing = []
                for i, c in enumerate(valid):
                    e = emb_train_map.get(key_for_record(c))
                    if e is None:
                        missing.append(i)
                        e = np.zeros_like(q_emb)
                    cand_embs.append(e)
                cand_embs = np.stack(cand_embs, axis=0).astype(np.float32)
                neural_raw = cand_embs @ q_emb.astype(np.float32)
        elif args.method in {"neural_only", "sealect"}:
            qk = query_key(q)
            row = neural_scores.get(qk, {}) if neural_scores is not None else {}
            neural_raw = np.asarray([float(row.get(c["candidate_id"], 0.0)) for c in valid], dtype=np.float32)

        neural = np.asarray(normalize_scores(neural_raw.tolist()), dtype=np.float32)
        context = np.asarray(normalize_scores(context_raw.tolist()), dtype=np.float32)
        obs = np.asarray(normalize_scores(obs_raw.tolist()), dtype=np.float32)

        # Method presets.
        if args.method == "prior_only":
            lam_n, lam_c, lam_o = 0.0, 1.0, 0.0
        elif args.method in {"dino", "clip", "neural_only"}:
            lam_n, lam_c, lam_o = 1.0, 0.0, 0.0
        elif args.method in {"sealect", "sealect_dino", "sealect_clip"}:
            lam_n, lam_c, lam_o = args.lambda_n, args.lambda_c, args.lambda_o
        else:
            raise ValueError(f"Unknown method: {args.method}")

        base = lam_n * neural + lam_c * context + lam_o * obs

        # Diversity/MMR uses same visual embedding space when available.
        mmr_embs = None
        if args.lambda_d > 0 and emb_train_map is not None:
            tmp = []
            for c in valid:
                e = emb_train_map.get(key_for_record(c))
                if e is None:
                    # Missing embeddings are assigned zero; they naturally receive low diversity confidence.
                    e = np.zeros_like(next(iter(emb_train_map.values())))
                tmp.append(e)
            mmr_embs = l2_normalize(np.stack(tmp, axis=0).astype(np.float32))

        selected_idxs, selected_scores = greedy_mmr_select(
            valid,
            base,
            mmr_embs,
            args.top_k,
            args.lambda_d,
        )
        selected = [valid[i] for i in selected_idxs]

        scores_by_id = {}
        for local_rank, i in enumerate(selected_idxs):
            c = valid[i]
            scores_by_id[c["candidate_id"]] = {
                "neural_score_raw": float(neural_raw[i]),
                "context_score_raw": float(context_raw[i]),
                "observed_score_raw": float(obs_raw[i]),
                "neural_score_used": float(neural[i]),
                "context_score_used": float(context[i]),
                "observed_score_used": float(obs[i]),
                "base_score": float(base[i]),
                "final_score": float(selected_scores[local_rank]),
                "lambda_n": float(lam_n),
                "lambda_c": float(lam_c),
                "lambda_o": float(lam_o),
                "lambda_d": float(args.lambda_d),
            }

        results.append(format_result(q, selected, scores_by_id))

    run_name = args.run_name or args.method
    out_path = args.out_dir / run_name / f"topk_{args.top_k}" / "retrieval_results.json"
    save_json(results, out_path)
    config_path = args.out_dir / run_name / f"topk_{args.top_k}" / "config.json"
    # save_json(vars(args) | {"out_path": str(out_path)}, config_path)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }
    config["out_path"] = str(out_path)
    save_json(config, config_path)
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=[
        "random", "prior_only", "dino", "clip", "neural_only", "sealect", "sealect_dino", "sealect_clip"
    ])
    p.add_argument("--run_name", default=None, help="Output subdirectory name; useful for lambda ablations.")
    p.add_argument("--top_k", type=int, default=100)

    p.add_argument("--train_candidates", type=Path, default=Path("splits_dino_decorrelated_kbalanced_grounded_sam2/train_candidate_library.json"))
    p.add_argument("--test_candidates", type=Path, default=Path("splits_dino_decorrelated_kbalanced_grounded_sam2/test_candidate_library.json"))
    p.add_argument("--test_queries", type=Path, default=Path("splits_dino_decorrelated_kbalanced_grounded_sam2/test_scene_object_pairs.json"))
    p.add_argument("--prior_json", type=Path, default=Path("scene_pose_prior_train_grounded_sam2.json"))
    p.add_argument("--out_dir", type=Path, default=Path("retrieval_results_grounded_sam2"))
    p.add_argument("--library_root", type=Path, default=Path("."))
    p.add_argument("--cache_dir", type=Path, default=Path("retrieval_results_grounded_sam2/cache"))

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dino_model", default="dinov2_vits14")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--neural_scores_json", type=Path, default=None)

    p.add_argument("--lambda_n", type=float, default=1.0)
    p.add_argument("--lambda_c", type=float, default=1.0)
    p.add_argument("--lambda_o", type=float, default=0.0)
    p.add_argument("--lambda_d", type=float, default=0.0)

    p.add_argument("--num_yaw_bins", type=int, default=18)
    p.add_argument("--obs_alpha", type=float, default=0.7)
    p.add_argument("--obs_sigma_pitch", type=float, default=1.5)
    p.add_argument("--obs_sigma_yaw", type=float, default=2.0)
    p.add_argument("--obs_sigma_roll", type=float, default=1.5)

    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_method(parse_args())