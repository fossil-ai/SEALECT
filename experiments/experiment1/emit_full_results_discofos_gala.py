#!/usr/bin/env python3
"""
Emit full-format retrieval_results.json for the DiscoFOS and GALA methods so the
existing Exp 2 (distributional) and Exp 3 (plausibility) scripts can score them
alongside the paper's rows. Writes to the same layout the other methods use:

  exp1_retrieval_quality/<name>_k20/topk_20/retrieval_results.json
  exp1_retrieval_quality/<name>_k100/topk_100/retrieval_results.json

Methods:
  discofos_ft   fine-tuned DiscoFOS (epoch3, anti-collapse), box given
  neural_gala   SEALECT neural score + learned GALA geometry (the swap arm)
  gala_only     GALA compatibility alone
Records carry the full pose/context fields the Table 2/3 scripts read.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path("/path/to/sealect_root")
sys.path.insert(0, str(REPO / "experiments/experiment1"))
import run_gala_swap_test_neural as G          # SEALECT neural + prior + GALA pipeline
import discofos_common as DC                    # DiscoFOS two-tower
from run_discofos_eval import encode_fgs, encode_bgs  # reuse encoders

DD = REPO / "experiments/dataset_decide"
SP = DD / "splits_dino_decorrelated_kbalanced_grounded_sam2"
ML = DD / "merged_scene_pose_library_out_grounded_sam2"
OUT = REPO / "experiments/experiment1/exp1_retrieval_quality"
DISCOFOS_CKPT = REPO / "experiments/baselines/DiscoFOS/checkpoints/maritime_ec/epoch3.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QKEEP = ["pitch_bin", "yaw_bin", "roll_bin", "y_bin", "area_frac", "context_key",
         "scene_type", "viewpoint_type", "sequence_group", "image_id", "annotation_id"]
CKEEP = ["pitch_bin", "yaw_bin", "roll_bin", "y_bin", "area_frac", "pose_family",
         "yaw_family", "sequence_group", "candidate_id", "context_key", "viewpoint_type"]


def load_json(p): return json.load(open(p))
def key(r): return (int(r["image_id"]), int(r["annotation_id"]))


def write_method(name, top_idx, queries, cslim):
    """top_idx: (Nq, 100) int array of ranked candidate indices per query."""
    qslim = [{k: q.get(k) for k in QKEEP} for q in queries]
    for K in (20, 100):
        rows = [{"query": qslim[qi], "topk": [cslim[j] for j in top_idx[qi][:K]]}
                for qi in range(len(queries))]
        d = OUT / f"{name}_k{K}" / f"topk_{K}"
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(d / "retrieval_results.json", "w"))
    print(f"wrote {name}_k20 and {name}_k100", flush=True)


def main():
    queries = load_json(SP / "test_scene_object_pairs.json")
    cands = load_json(SP / "train_candidate_library.json")
    cslim = [{k: c.get(k) for k in CKEEP} for c in cands]
    cimg = np.asarray([int(c.get("image_id", -2)) for c in cands])
    cann = np.asarray([int(c.get("annotation_id", -2)) for c in cands])
    cgrp = np.asarray([str(c.get("sequence_group", "")) for c in cands])

    def invalid_mask(q):
        return (cgrp == str(q.get("sequence_group", ""))) | \
               ((cimg == int(q.get("image_id", -1))) & (cann == int(q.get("annotation_id", -1))))

    def rank_from_scores(score_of_query):
        top = np.empty((len(queries), 100), dtype=np.int64)
        for qi, q in enumerate(queries):
            sc = score_of_query(qi, q).astype(np.float32)
            sc[invalid_mask(q)] = -1e9
            t = np.argpartition(-sc, 100)[:100]
            top[qi] = t[np.argsort(-sc[t])]
        return top

    # ---------- DiscoFOS fine-tuned ----------
    print("=== DiscoFOS fine-tuned ===", flush=True)
    merged = load_json(ML / "candidate_library.json"); ctx = load_json(ML / "image_contexts.json")
    mby = {key(m): m for m in merged}; cby = {int(c["image_id"]): c for c in ctx}
    crop_paths = [str(DD / mby[key(c)]["crop_path"]) for c in cands]
    scene_paths, norm_boxes = [], []
    for q in queries:
        m = mby[key(q)]; c = cby[int(q["image_id"])]; bb = m["bbox_xyxy"]; W = float(c["width"]); H = float(c["height"])
        scene_paths.append(c["image_path"]); norm_boxes.append([bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H])
    model = DC.load_student(DEVICE, ckpt=str(DISCOFOS_CKPT))
    fg_embs = encode_fgs(model, crop_paths); bg_embs = encode_bgs(model, scene_paths, norm_boxes)
    top = rank_from_scores(lambda qi, q: (fg_embs @ bg_embs[qi]).cpu().numpy())
    write_method("discofos_ft", top, queries, cslim)
    del model, fg_embs, bg_embs; torch.cuda.empty_cache()

    # ---------- SEALECT neural + GALA / GALA-only ----------
    print("=== neural + GALA / GALA-only ===", flush=True)
    train = load_json(SP / "train_scene_object_pairs.json")
    prior = load_json(G.PRIOR_JSON)
    Np = max(int(r["pitch_bin"]) for r in train) + 1; Ny = 18; Nr = max(int(r["roll_bin"]) for r in train) + 1
    scene_enc, cand_enc = G.load_model()
    dino_by_key = G.load_dino_cache_by_key()
    with torch.no_grad():
        cand_embs = G.encode_candidates(cand_enc, cands, dino_by_key)
        q_embs = G.encode_queries(scene_enc, queries)
    gala_model, (gsmu, gssd, ggmu, ggsd) = G.train_gala(train, Np, Ny, Nr)
    Cg = torch.tensor((np.asarray([G.gala_geom_vec(c, Np, Ny, Nr) for c in cands], np.float32) - ggmu) / ggsd, device=DEVICE)

    def neural_of(qi): return (cand_embs @ q_embs[qi]).cpu().numpy().astype(np.float32)
    Nc = cand_embs.shape[0]

    def gala_of(q):
        sv = torch.tensor((np.asarray(G.gala_scene_vec(q), np.float32) - gsmu) / gssd, device=DEVICE).unsqueeze(0).expand(Nc, -1)
        with torch.no_grad():
            return gala_model(torch.cat([sv, Cg], 1)).squeeze(1).cpu().numpy().astype(np.float32)

    top_ng = rank_from_scores(lambda qi, q: G.znorm(neural_of(qi)) + G.znorm(gala_of(q)))
    write_method("neural_gala", top_ng, queries, cslim)
    top_g = rank_from_scores(lambda qi, q: G.znorm(gala_of(q)))
    write_method("gala_only", top_g, queries, cslim)
    print("done", flush=True)


if __name__ == "__main__":
    main()
