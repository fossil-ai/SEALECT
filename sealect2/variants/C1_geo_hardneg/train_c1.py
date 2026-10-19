"""
train_c1.py

C1 - geometry-aware hard-negative training. Train a NEW neural retriever with the
same in-batch contrastive loss as v1 PLUS a margin ranking term over
geometry hard negatives: candidates that are visually attractive (high v1 S_neural
for the scene) but geometrically implausible (low pose-prior score for the scene).

    N_geo(q) = { j : S_n(j) high  AND  S_c(j) low }
    L_hard   = mean_j-  max(0, m + S_n(q, j-) - S_n(q, j+))
    L        = L_contrastive + beta * L_hard

The original v1 trainer/checkpoint is untouched (it is the baseline). We reuse the
frozen v1 model only to MINE the hard negatives, then train from scratch.
Outputs: variants/C1_geo_hardneg/c1_model.pt
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import legacy_bridge as lb

HERE = Path(__file__).resolve().parent
HARDNEG_CACHE = lb.CACHE_DIR / "c1_hard_negatives.pt"
C1_MODEL = HERE / "c1_model.pt"

BETA = 0.5
MARGIN = 0.2
K_HARD = 8
TOP_M = 200
EPOCHS = 30
BATCH = 256
LR = 1e-3


def _load_trainer():
    f = lb.EXP1_DIR / "train_sealect_neural_retriever_grounded_sam2.py"
    spec = importlib.util.spec_from_file_location("v1_trainer", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pose_loghists(legacy, prior, q):
    """[18]x3 log-prob arrays for the query's context (scene fallback)."""
    ck, st = q.get("context_key"), q.get("scene_type")

    def getd(section):
        if ck is not None and prior.get(section, {}).get(str(ck)) is not None:
            return prior[section][str(ck)]
        s2 = section.replace("context_", "scene_")
        if st is not None and prior.get(s2, {}).get(str(st)) is not None:
            return prior[s2][str(st)]
        return {}
    return (legacy.log_prob_hist_array(getd("context_pitch_hist")),
            legacy.log_prob_hist_array(getd("context_yaw_hist")),
            legacy.log_prob_hist_array(getd("context_roll_hist")))


def mine_hard_negatives(use_cache=True):
    if use_cache and HARDNEG_CACHE.exists():
        blob = torch.load(HARDNEG_CACHE, map_location="cpu")
        print(f"[C1] loaded cached hard negatives: {HARDNEG_CACHE}")
        return blob["hard"], blob["pair_keys"]

    legacy = lb.load_legacy()
    trainer = _load_trainer()
    device = legacy.DEVICE
    candidates = legacy.load_json(legacy.TRAIN_CANDIDATES)
    train_pairs = legacy.load_json(legacy.SPLIT_DIR / "train_scene_object_pairs.json")
    prior = legacy.load_json(legacy.PRIOR_JSON)
    dino = legacy.load_dino_cache_by_key()

    scene_enc, cand_enc = legacy.load_model()  # frozen v1 model, for mining only
    with torch.no_grad():
        cand_embs = legacy.encode_candidates(cand_enc, candidates, dino)   # [Ncand,emb]
        q_embs = legacy.encode_queries(scene_enc, train_pairs)             # [Npair,emb]

    c_pitch = torch.tensor([int(c.get("pitch_bin", 0)) for c in candidates]).clamp(0, 17)
    c_yaw = torch.tensor([int(c.get("yaw_bin", 0)) for c in candidates]).clamp(0, 17)
    c_roll = torch.tensor([int(c.get("roll_bin", 0)) for c in candidates]).clamp(0, 17)

    Np = len(train_pairs)
    hard = np.zeros((Np, K_HARD), dtype=np.int64)
    for s in range(0, Np, 512):
        e = min(s + 512, Np)
        Sn = q_embs[s:e] @ cand_embs.T                     # [B,Ncand]
        topv, topi = torch.topk(Sn, TOP_M, dim=1)          # visually attractive
        topi_cpu = topi.cpu()
        for r, qi in enumerate(range(s, e)):
            lp_p, lp_y, lp_r = _pose_loghists(legacy, prior, train_pairs[qi])
            idx = topi_cpu[r]
            sc = (torch.from_numpy(lp_p)[c_pitch[idx]]
                  + torch.from_numpy(lp_y)[c_yaw[idx]]
                  + torch.from_numpy(lp_r)[c_roll[idx]])   # pose-prior score of top-M
            worst = torch.topk(-sc, K_HARD).indices          # lowest prior score = implausible
            hard[qi] = idx[worst].numpy()
    pair_keys = [legacy.key_for_record(p) for p in train_pairs]
    torch.save({"hard": hard, "pair_keys": pair_keys}, HARDNEG_CACHE)
    print(f"[C1] mined + cached hard negatives -> {HARDNEG_CACHE}")
    return hard, pair_keys


def _cand_feature_matrix(trainer, candidates, dino):
    rows = []
    for c in candidates:
        d = dino[trainer.key_for_record(c)]
        d = d / (np.linalg.norm(d) + 1e-8)
        x = trainer.base_candidate_features(c)
        x.extend(float(v) for v in d)
        rows.append(np.asarray(x, dtype=np.float32))
    return torch.from_numpy(np.stack(rows))


class C1PairDataset(Dataset):
    """Precomputed feature matrices -> __getitem__ is pure tensor indexing (fast)."""

    def __init__(self, trainer, pairs, candidates, dino, hard):
        cand_index = {trainer.key_for_record(c): i for i, c in enumerate(candidates)}
        self.cand_mat = _cand_feature_matrix(trainer, candidates, dino)     # [Ncand,cand_dim]
        scene_rows, self.rows = [], []
        for pi, p in enumerate(pairs):
            k = trainer.key_for_record(p)
            ci = cand_index.get(k)
            if ci is None or k not in dino:
                continue
            self.rows.append((len(scene_rows), ci, torch.as_tensor(hard[pi], dtype=torch.long)))
            scene_rows.append(trainer.scene_features(p))
        self.scene_mat = torch.from_numpy(np.stack(scene_rows))             # [Npair,scene_dim]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        si, ci, hidx = self.rows[i]
        return self.scene_mat[si], self.cand_mat[ci], self.cand_mat[hidx]


def train(use_cache=True):
    if use_cache and C1_MODEL.exists():
        print(f"[C1] using existing checkpoint {C1_MODEL}")
        return C1_MODEL
    legacy = lb.load_legacy()
    trainer = _load_trainer()
    device = legacy.DEVICE
    candidates = legacy.load_json(legacy.TRAIN_CANDIDATES)
    pairs = legacy.load_json(legacy.SPLIT_DIR / "train_scene_object_pairs.json")
    dino = legacy.load_dino_cache_by_key()
    hard, _ = mine_hard_negatives(use_cache=use_cache)

    ds = C1PairDataset(trainer, pairs, candidates, dino, hard)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    s0, p0, h0 = ds[0]
    scene_enc = trainer.MLP(s0.numel(), trainer.EMB_DIM).to(device)
    cand_enc = trainer.MLP(p0.numel(), trainer.EMB_DIM).to(device)
    opt = torch.optim.AdamW(list(scene_enc.parameters()) + list(cand_enc.parameters()),
                            lr=LR, weight_decay=1e-4)
    print(f"[C1] train pairs={len(ds)} scene_dim={s0.numel()} cand_dim={p0.numel()} beta={BETA} margin={MARGIN} K={K_HARD}")
    t0 = time.time()
    for ep in range(EPOCHS):
        scene_enc.train(); cand_enc.train(); Lc, Lh = [], []
        for s, pos, hn in loader:
            s, pos, hn = s.to(device).float(), pos.to(device).float(), hn.to(device).float()
            q = scene_enc(s); kpos = cand_enc(pos)
            B, K, D = hn.shape
            khn = cand_enc(hn.reshape(B * K, D)).reshape(B, K, -1)
            loss_c = trainer.soft_contrastive_loss(q, kpos, [""] * B)
            sn_pos = (q * kpos).sum(-1, keepdim=True)                 # [B,1]
            sn_hard = (q.unsqueeze(1) * khn).sum(-1)                   # [B,K]
            loss_h = F.relu(MARGIN + sn_hard - sn_pos).mean()
            loss = loss_c + BETA * loss_h
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            Lc.append(float(loss_c.item())); Lh.append(float(loss_h.item()))
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"    [C1] epoch {ep+1}/{EPOCHS} L_contrastive={np.mean(Lc):.4f} L_hard={np.mean(Lh):.4f}")
    torch.save({"scene_enc": scene_enc.state_dict(), "cand_enc": cand_enc.state_dict(),
                "scene_dim": s0.numel(), "cand_dim": p0.numel(), "emb_dim": trainer.EMB_DIM,
                "train_time_s": round(time.time() - t0, 1),
                "beta": BETA, "margin": MARGIN, "k_hard": K_HARD}, C1_MODEL)
    print(f"[C1] saved {C1_MODEL}  ({time.time()-t0:.1f}s)")
    return C1_MODEL


if __name__ == "__main__":
    train()
