#!/usr/bin/env python3
"""
Evaluate DiscoFOS (ICCV 2023) as a maritime foreground-object-search baseline on the
SEALECT Exp 1 retrieval benchmark. Same candidate pool, same validity filter, same
graded-relevance metric as every other Table 1 row.

DiscoFOS is spotted the target box (its architecture requires it), matching the
placement information SEALECT's scene encoder already receives. Ranking is by cosine
similarity of the two-tower background and foreground embeddings.

Usage:
  python experiments/experiment1/run_discofos_eval.py --student <ckpt> --tag zeroshot
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = Path("/path/to/sealect_root")
sys.path.insert(0, str(REPO / "experiments/experiment1"))
from discofos_common import (  # noqa: E402
    load_student, make_bg_inputs, make_fg_input, IMAGE_SIZE, CKPT_STUDENT,
)

DD = REPO / "experiments/dataset_decide"
SP = DD / "splits_dino_decorrelated_kbalanced_grounded_sam2"
ML = DD / "merged_scene_pose_library_out_grounded_sam2"
OUT = REPO / "experiments/experiment1/exp1_retrieval_quality"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8


# ---- metric (identical to run_gala_swap_test_neural.py) ----
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


def load_json(p): return json.load(open(p))
def key(r): return (int(r["image_id"]), int(r["annotation_id"]))


class FgDataset(Dataset):
    def __init__(self, crop_paths): self.paths = crop_paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return make_fg_input(self.paths[i])


class BgDataset(Dataset):
    def __init__(self, scene_paths, norm_boxes): self.paths = scene_paths; self.boxes = norm_boxes
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        bg_t, qbox, cbox = make_bg_inputs(self.paths[i], self.boxes[i])
        return bg_t, qbox, cbox


@torch.no_grad()
def encode_fgs(model, crop_paths, bs=128, workers=8):
    loader = DataLoader(FgDataset(crop_paths), batch_size=bs, shuffle=False, num_workers=workers)
    out = []
    for xb in tqdm(loader, desc="encode candidate crops"):
        out.append(F.normalize(model.forward_fg_encoder(xb.to(DEVICE)), dim=1))
    return torch.cat(out, 0)


@torch.no_grad()
def encode_bgs(model, scene_paths, norm_boxes, bs=64, workers=8):
    loader = DataLoader(BgDataset(scene_paths, norm_boxes), batch_size=bs, shuffle=False, num_workers=workers)
    out = []
    for bg_t, qbox, cbox in tqdm(loader, desc="encode query backgrounds"):
        out.append(F.normalize(model.forward_bg_encoder(bg_t.to(DEVICE), cbox.to(DEVICE)), dim=1))
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=CKPT_STUDENT)
    ap.add_argument("--tag", default="zeroshot")
    args = ap.parse_args()

    queries = load_json(SP / "test_scene_object_pairs.json")
    cands = load_json(SP / "train_candidate_library.json")
    merged = load_json(ML / "candidate_library.json")
    ctx = load_json(ML / "image_contexts.json")
    mby = {key(m): m for m in merged}
    cby = {int(c["image_id"]): c for c in ctx}
    print(f"queries {len(queries)}  candidates {len(cands)}", flush=True)

    # candidate crop paths (foregrounds)
    crop_paths = []
    for c in cands:
        m = mby[key(c)]
        crop_paths.append(str(DD / m["crop_path"]))
    # query scene image + normalized target box (backgrounds)
    scene_paths, norm_boxes = [], []
    for q in queries:
        m = mby[key(q)]; c = cby[int(q["image_id"])]
        bb = m["bbox_xyxy"]; W, H = float(c["width"]), float(c["height"])
        scene_paths.append(c["image_path"])
        norm_boxes.append([bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H])

    model = load_student(DEVICE, ckpt=args.student)
    fg_embs = encode_fgs(model, crop_paths)     # (Nc,512) normalized, on GPU
    bg_embs = encode_bgs(model, scene_paths, norm_boxes)  # (Nq,512)
    print("fg_embs", tuple(fg_embs.shape), "bg_embs", tuple(bg_embs.shape), flush=True)

    # validity arrays + slim dicts
    cimg = np.asarray([int(c.get("image_id", -2)) for c in cands])
    cann = np.asarray([int(c.get("annotation_id", -2)) for c in cands])
    cgrp = np.asarray([str(c.get("sequence_group", "")) for c in cands])
    keep = ["pitch_bin", "yaw_bin", "y_bin", "area_frac"]
    cslim = [{k: c.get(k) for k in keep} for c in cands]

    results = []
    with torch.no_grad():
        for qi, q in enumerate(tqdm(queries, desc="rank")):
            score = (fg_embs @ bg_embs[qi]).cpu().numpy().astype(np.float32)  # cosine
            invalid = (cgrp == str(q.get("sequence_group", ""))) | \
                      ((cimg == int(q.get("image_id", -1))) & (cann == int(q.get("annotation_id", -1))))
            score[invalid] = -1e9
            top = np.argpartition(-score, 100)[:100]; top = top[np.argsort(-score[top])]
            results.append({"query": {k: q.get(k) for k in keep}, "topk": [cslim[j] for j in top]})

    outdir = OUT / f"discofos_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outdir / "retrieval_results.json", "w"))

    print(f"\n DiscoFOS ({args.tag}) on SEALECT Exp 1 retrieval")
    print(f"{'K':>4s} {'Recall':>8s} {'NDCG':>8s} {'PS':>8s}")
    summ = {}
    for k in (20, 100):
        m = summarize(results, k); summ[k] = m
        print(f"{k:>4d} {m['Recall']:8.3f} {m['NDCG']:8.3f} {m['PS']:8.3f}")
    json.dump(summ, open(outdir / "summary.json", "w"), indent=2)
    print("saved to", outdir, flush=True)


if __name__ == "__main__":
    main()
