#!/usr/bin/env python3
"""
GALA-style discriminative geometry-aware retrieval baseline for SEALECT.

Motivation: reviewers will ask why SEALECT is not compared against a learned
discriminative compatibility scorer (GALA / DiscoFOS family). Those methods need
compatibility-labeled foreground-background pairs that maritime lacks, but we can
build a fair adapted version on the SAME (scene, geometry) inputs SEALECT's prior
uses: an MLP that scores compatibility of a scene with a candidate vessel's pose,
placement, and scale, trained discriminatively on real vs. mismatched pairs.

We then run it through the IDENTICAL retrieval + relevance metric used for Table 1
(summarize_exp1_retrieval_results.relevance) and compare to the existing SEALECT
and baseline result files.
"""
import json, math, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn

ROOT = Path("/path/to/sealect_root")
SP = ROOT / "experiments/dataset_decide/splits_dino_decorrelated_kbalanced_grounded_sam2"
EXP = ROOT / "experiments/experiment1/exp1_retrieval_quality"
OUT = ROOT / "experiments/experiment1/gala_baseline"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- metric (copied verbatim from summarize_exp1_retrieval_results.py) ----------
PITCH_TOL, YAW_TOL, YBIN_TOL, AREA_LOG_TOL, RECALL_THRESHOLD = 0, 1, 0, 0.30, 4
EPS = 1e-8
def log_area(x): return math.log(max(float(x), EPS))
def circ(a, b, n=18): d = abs(int(a) - int(b)); return min(d, n - d)
def relevance(q, c):
    s = 0
    if abs(int(q["pitch_bin"]) - int(c["pitch_bin"])) <= PITCH_TOL: s += 1
    if circ(int(q["yaw_bin"]), int(c["yaw_bin"])) <= YAW_TOL: s += 1
    if abs(int(q["y_bin"]) - int(c["y_bin"])) <= YBIN_TOL: s += 1
    if abs(log_area(q["area_frac"]) - log_area(c["area_frac"])) <= AREA_LOG_TOL: s += 1
    return s
def dcg(xs): return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(xs))
def ndcg(xs):
    ideal = sorted(xs, reverse=True); d = dcg(ideal); return dcg(xs) / d if d > 0 else 0.0
def summarize(results, k):
    rec, nd, ps = [], [], []
    for row in results:
        q = row["query"]; cands = row["topk"][:k]
        if not cands: continue
        rels = [relevance(q, c) for c in cands]
        rec.append(1.0 if max(rels) >= RECALL_THRESHOLD else 0.0)
        nd.append(ndcg(rels)); ps.append(sum(rels) / max(1, len(rels)))
    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"Recall": m(rec), "NDCG": m(nd), "PS": m(ps), "n": len(rec)}

def is_valid(q, c):
    same = int(q.get("image_id", -1)) == int(c.get("image_id", -2)) and \
           int(q.get("annotation_id", -1)) == int(c.get("annotation_id", -2))
    grp = str(q.get("sequence_group", "")) == str(c.get("sequence_group", ""))
    return not same and not grp

# ---------- features ----------
SCENE_KEYS = ["water_frac", "sky_frac", "land_frac", "other_frac", "horizon_y_norm",
              "water_row_mean", "water_row_std", "sky_row_mean", "sky_row_std",
              "land_row_mean", "land_row_std"]
def scene_vec(r):
    return [float(r.get(k) or 0.0) for k in SCENE_KEYS]
def geom_vec(r, Np, Ny, Nr):
    def sc(b, N): a = 2 * math.pi * int(b) / max(1, N); return [math.sin(a), math.cos(a)]
    return (sc(r["pitch_bin"], Np) + sc(r["yaw_bin"], Ny) + sc(r["roll_bin"], Nr) +
            [int(r["y_bin"]) / 8.0, log_area(r["area_frac"]),
             math.log(max(float(r.get("aspect_ratio", 1.0)), 1e-6)),
             float(r.get("center_x_norm") or 0.5), float(r.get("center_y_norm") or 0.5)])

def main():
    train = json.load(open(SP / "train_scene_object_pairs.json"))
    queries = json.load(open(SP / "test_scene_object_pairs.json"))
    cands = json.load(open(SP / "train_candidate_library.json"))
    print(f"train {len(train)}  queries {len(queries)}  candidates {len(cands)}", flush=True)

    Np = max(int(r["pitch_bin"]) for r in train) + 1
    Ny = 18
    Nr = max(int(r["roll_bin"]) for r in train) + 1

    S = np.asarray([scene_vec(r) for r in train], dtype=np.float32)
    G = np.asarray([geom_vec(r, Np, Ny, Nr) for r in train], dtype=np.float32)
    smu, ssd = S.mean(0), S.std(0) + 1e-6
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    S = (S - smu) / ssd; G = (G - gmu) / gsd
    St = torch.tensor(S, device=DEVICE); Gt = torch.tensor(G, device=DEVICE)
    ds, dg = S.shape[1], G.shape[1]

    model = nn.Sequential(
        nn.Linear(ds + dg, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    N = St.shape[0]; bs = 1024
    print("training GALA-style compatibility scorer...", flush=True)
    for ep in range(8):
        model.train(); perm = torch.randperm(N, device=DEVICE); tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]; s = St[idx]; gpos = Gt[idx]
            neg = idx[torch.randperm(len(idx), device=DEVICE)]  # mismatched geometry
            gneg = Gt[neg]
            x = torch.cat([torch.cat([s, gpos], 1), torch.cat([s, gneg], 1)], 0)
            y = torch.cat([torch.ones(len(idx), device=DEVICE), torch.zeros(len(idx), device=DEVICE)])
            opt.zero_grad(); loss = bce(model(x).squeeze(1), y); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {ep+1}/8  loss {tot/N:.4f}", flush=True)

    # ---- retrieval: score all valid candidates per query ----
    model.eval()
    Cg = np.asarray([geom_vec(c, Np, Ny, Nr) for c in cands], dtype=np.float32)
    Cg = (Cg - gmu) / gsd
    Cg_t = torch.tensor(Cg, device=DEVICE)
    cgroup = np.asarray([str(c.get("sequence_group", "")) for c in cands])
    cimg = np.asarray([int(c.get("image_id", -2)) for c in cands])
    cann = np.asarray([int(c.get("annotation_id", -2)) for c in cands])
    keep_fields = ["candidate_id", "image_id", "annotation_id", "pitch_bin", "yaw_bin",
                   "roll_bin", "y_bin", "area_frac", "sequence_group", "context_key", "viewpoint_type"]
    cand_slim = [{k: c.get(k) for k in keep_fields} for c in cands]

    results = []
    with torch.no_grad():
        for qi, q in enumerate(queries):
            sv = (np.asarray(scene_vec(q), np.float32) - smu) / ssd
            svt = torch.tensor(sv, device=DEVICE).unsqueeze(0).expand(Cg_t.shape[0], -1)
            score = model(torch.cat([svt, Cg_t], 1)).squeeze(1).cpu().numpy()
            invalid = (cgroup == str(q.get("sequence_group", ""))) | \
                      ((cimg == int(q.get("image_id", -1))) & (cann == int(q.get("annotation_id", -1))))
            score[invalid] = -1e9
            top = np.argpartition(-score, 100)[:100]
            top = top[np.argsort(-score[top])]
            qslim = {k: q.get(k) for k in ["pitch_bin", "yaw_bin", "y_bin", "area_frac",
                                           "context_key", "viewpoint_type", "image_id", "annotation_id"]}
            results.append({"query": qslim, "topk": [cand_slim[j] for j in top]})
            if (qi + 1) % 1000 == 0: print(f"  retrieved {qi+1}/{len(queries)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / "retrieval_results.json", "w"))

    # ---- compare against existing SEALECT / baselines with the same metric ----
    def load_existing(name, k):
        p = EXP / f"{name}_k{k}" / f"topk_{k}" / "retrieval_results.json"
        return json.load(open(p)) if p.exists() else None

    print("\n RETRIEVAL QUALITY: GALA baseline vs SEALECT (same relevance metric)")
    methods = [("random", "random"), ("clip", "clip"), ("dino", "dino"),
               ("prior_only", "prior_only"), ("neural_only", "neural_only_dinofeat"),
               ("SEALECT", "sealect_dinofeat")]
    for k in (20, 100):
        print(f"\n--- K={k} ---")
        print(f"{'method':16s} {'Recall':>8s} {'NDCG':>8s} {'PS':>8s}")
        for disp, name in methods:
            res = load_existing(name, k)
            if res is None: continue
            m = summarize(res, k)
            print(f"{disp:16s} {m['Recall']:8.3f} {m['NDCG']:8.3f} {m['PS']:8.3f}")
        mg = summarize(results, k)
        print(f"{'GALA (ours-impl)':16s} {mg['Recall']:8.3f} {mg['NDCG']:8.3f} {mg['PS']:8.3f}   <-- discriminative baseline")
    json.dump({"Np": Np, "Ny": Ny, "Nr": Nr}, open(OUT / "meta.json", "w"))
    print("\nwrote", OUT / "retrieval_results.json")

if __name__ == "__main__":
    main()
