#!/usr/bin/env python3
"""
train_sealect_neural_retriever_grounded_sam2_dino.py

Train SEALECT q/k retriever using cached DINOv2 crop embeddings appended to
candidate metadata. This uses option B: do not rewrite JSONs; load the DINO
cache directly and align it by (image_id, annotation_id).
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_ROOT = Path("/path/to/sealect_root/experiments/dataset_decide")
SPLIT_DIR = DATA_ROOT / "splits_dino_decorrelated_kbalanced_grounded_sam2"
MERGED_DIR = DATA_ROOT / "merged_scene_pose_library_out_grounded_sam2"

TRAIN_PAIRS = SPLIT_DIR / "train_scene_object_pairs.json"
TRAIN_CANDIDATES = SPLIT_DIR / "train_candidate_library.json"
DINO_CACHE = SPLIT_DIR / "cache" / "candidate_dino_embeddings.pt"

OUT_DIR = Path("exp1_retrieval_quality/sealect_neural_dino")
OUT_MODEL = OUT_DIR / "sealect_neural_retriever_dino.pt"
OUT_CONFIG = OUT_DIR / "config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMP = 0.07
EMB_DIM = 128
NUM_WORKERS = 0
FALLBACK_EMBED_DIM = 256
EPS = 1e-8

# Important: disable initially. Previous model showed sequence/context collapse.
USE_CONTEXT_SOFT_POSITIVES = False
SOFT_POS_CONTEXT_WEIGHT = 0.20

VIEWPOINTS = [
    "nadir_aerial", "oblique_aerial", "horizon_view",
    "low_oblique_or_shore", "unknown_viewpoint",
]
LAYOUTS = [
    "open_water", "open_water_uncertain", "coastal_transition",
    "harbor_port", "water_land_mixed", "non_maritime_or_uncertain",
]
POSE_FAMILIES = [
    "nadir__front_back__level", "nadir__side__level", "nadir__oblique__level",
    "level__front_back__level", "level__side__level", "level__oblique__level",
    "level__front_back__rolled", "level__side__rolled", "level__oblique__rolled",
    "steep_up__front_back__level", "steep_up__side__level", "steep_up__oblique__level",
]
YAW_FAMILIES = ["front_back", "side", "oblique"]


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

        # Restore target-local query geometry.
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
    emb = r.get("embedding", [])
    if not emb:
        emb = [0.0] * FALLBACK_EMBED_DIM
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


def load_dino_cache_by_key() -> Dict[Tuple[int, int], np.ndarray]:
    if not DINO_CACHE.exists():
        raise FileNotFoundError(f"Missing DINO cache: {DINO_CACHE}")
    obj = torch.load(DINO_CACHE, map_location="cpu")
    emb = obj["embeddings"].float().numpy().astype(np.float32)
    valid_indices = list(obj["valid_indices"])

    scene_pairs = load_json(MERGED_DIR / "scene_object_pairs.json")
    all_candidates = load_json(MERGED_DIR / "candidate_library.json")
    cand_by_key = {key_for_record(c): c for c in all_candidates}
    paired = [r for r in scene_pairs if key_for_record(r) in cand_by_key]
    candidates_paired = [cand_by_key[key_for_record(r)] for r in paired]

    if max(valid_indices) >= len(candidates_paired):
        raise RuntimeError("DINO cache valid_indices do not align with merged candidate list.")

    dino_by_key = {}
    for j, raw_idx in enumerate(valid_indices):
        dino_by_key[key_for_record(candidates_paired[raw_idx])] = emb[j]

    print(f"Loaded DINO cache: {DINO_CACHE}")
    print(f"DINO vectors: {len(dino_by_key)} dim={emb.shape[1]}")
    return dino_by_key


class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict], candidates: List[Dict], dino_by_key: Dict[Tuple[int, int], np.ndarray]):
        cand_index = {key_for_record(c): c for c in candidates}
        self.rows = []
        self.dino_by_key = dino_by_key
        missing = 0
        for p in pairs:
            k = key_for_record(p)
            c = cand_index.get(k)
            if c is None:
                continue
            if k not in dino_by_key:
                missing += 1
                continue
            self.rows.append((p, c))
        if not self.rows:
            raise RuntimeError("No train pairs matched candidates with DINO vectors.")
        if missing:
            print(f"Warning: skipped {missing} train pairs without DINO vectors.")

    def __len__(self):
        return len(self.rows)

    def cand_features(self, c: Dict) -> np.ndarray:
        dino = self.dino_by_key[key_for_record(c)]
        dino = dino / (np.linalg.norm(dino) + EPS)
        x = base_candidate_features(c)
        x.extend(float(v) for v in dino)
        return np.asarray(x, dtype=np.float32)

    def __getitem__(self, idx):
        s, c = self.rows[idx]
        return torch.from_numpy(scene_features(s)), torch.from_numpy(self.cand_features(c)), str(s.get("context_key", ""))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.LayerNorm(512), nn.Dropout(0.05),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.LayerNorm(256),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def soft_contrastive_loss(q: torch.Tensor, k: torch.Tensor, context_keys: List[str]) -> torch.Tensor:
    logits = q @ k.T / TEMP
    b = logits.shape[0]
    labels = torch.eye(b, device=logits.device)
    if USE_CONTEXT_SOFT_POSITIVES:
        ctx = np.asarray(context_keys, dtype=object)
        same = torch.from_numpy((ctx[:, None] == ctx[None, :]).astype(np.float32)).to(logits.device)
        same.fill_diagonal_(0.0)
        if same.sum() > 0:
            same = same / same.sum(dim=1, keepdim=True).clamp_min(1.0)
            labels = (1.0 - SOFT_POS_CONTEXT_WEIGHT) * labels + SOFT_POS_CONTEXT_WEIGHT * same
            labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(EPS)
    q2k = -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    k2q = -(labels.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (q2k + k2q)


def collate_batch(batch):
    s, c, ctx = zip(*batch)
    return torch.stack(s), torch.stack(c), list(ctx)


def make_config():
    return {
        "data_root": str(DATA_ROOT), "split_dir": str(SPLIT_DIR), "train_pairs": str(TRAIN_PAIRS),
        "train_candidates": str(TRAIN_CANDIDATES), "dino_cache": str(DINO_CACHE),
        "out_model": str(OUT_MODEL), "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "lr": LR, "temp": TEMP, "emb_dim": EMB_DIM,
        "use_context_soft_positives": USE_CONTEXT_SOFT_POSITIVES,
        "soft_pos_context_weight": SOFT_POS_CONTEXT_WEIGHT, "device": DEVICE, "seed": SEED,
    }


def train_model():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pairs = load_json(TRAIN_PAIRS)
    candidates = load_json(TRAIN_CANDIDATES)
    dino_by_key = load_dino_cache_by_key()
    dataset = PairDataset(pairs, candidates, dino_by_key)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=NUM_WORKERS, collate_fn=collate_batch)
    s0, c0, _ = dataset[0]
    scene_enc = MLP(s0.numel(), EMB_DIM).to(DEVICE)
    cand_enc = MLP(c0.numel(), EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(list(scene_enc.parameters()) + list(cand_enc.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"Training SEALECT+DINO neural retriever on {len(dataset)} pairs")
    print(f"scene_dim={s0.numel()} cand_dim={c0.numel()} emb_dim={EMB_DIM} device={DEVICE}")
    print(f"context_soft_positives={USE_CONTEXT_SOFT_POSITIVES}")
    for ep in range(EPOCHS):
        scene_enc.train(); cand_enc.train(); losses = []
        for s, c, ctx in tqdm(loader, desc=f"train epoch {ep + 1}/{EPOCHS}"):
            s = s.to(DEVICE).float(); c = c.to(DEVICE).float()
            q = scene_enc(s); k = cand_enc(c)
            loss = soft_contrastive_loss(q, k, ctx)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {ep + 1}: loss={np.mean(losses):.4f}")
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "scene_enc": scene_enc.state_dict(), "cand_enc": cand_enc.state_dict(),
        "scene_dim": s0.numel(), "cand_dim": c0.numel(), "emb_dim": EMB_DIM,
        "viewpoints": VIEWPOINTS, "layouts": LAYOUTS, "pose_families": POSE_FAMILIES,
        "yaw_families": YAW_FAMILIES, "uses_cached_dino": True,
        "dino_cache": str(DINO_CACHE), "config": make_config(),
    }, OUT_MODEL)
    print(f"Saved model: {OUT_MODEL}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(make_config(), OUT_CONFIG)
    train_model()


if __name__ == "__main__":
    main()