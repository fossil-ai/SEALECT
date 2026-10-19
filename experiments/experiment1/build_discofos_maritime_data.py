#!/usr/bin/env python3
"""
Build DiscoFOS-format training data from the SEALECT maritime library.

Layout produced under <DiscoFOS>/data/train:
  bg/<id_ann>.jpg     symlink to the full scene image (background)
  fg/<id_ann>.jpg     symlink to the vessel crop (foreground)
  labels/train_maritime.json   {name: {bbox:[x1,y1,x2,y2] norm, similars:[], dissimilars:[names], class:'vessel'}}
  labels/category.json         {'vessel': {'images':[names]}}

Positive foreground for a background = the crop with the same name (DiscoFOS
convention when num_positive=1: the positive fg file is named like the bg).
Negatives = random other vessels (mirrors SEALECT's contrastive negatives).
Degenerate tiny boxes (sub-2px at 224) are dropped from TRAINING only; eval is
unaffected.
"""
import os
import json
import random
from pathlib import Path

random.seed(42)
REPO = Path("/path/to/sealect_root")
DD = REPO / "experiments/dataset_decide"
SP = DD / "splits_dino_decorrelated_kbalanced_grounded_sam2"
ML = DD / "merged_scene_pose_library_out_grounded_sam2"
DISCOFOS = REPO / "experiments/baselines/DiscoFOS"
TRAIN_ROOT = DISCOFOS / "data/train"

N_NEG_POOL = 20      # random negatives stored per bg; loader cycles 10/epoch
IMG = 224


def key(r): return (int(r["image_id"]), int(r["annotation_id"]))


def main():
    pairs = json.load(open(SP / "train_scene_object_pairs.json"))
    merged = json.load(open(ML / "candidate_library.json"))
    ctx = json.load(open(ML / "image_contexts.json"))
    mby = {key(m): m for m in merged}
    cby = {int(c["image_id"]): c for c in ctx}
    print(f"train pairs: {len(pairs)}")

    (TRAIN_ROOT / "bg").mkdir(parents=True, exist_ok=True)
    (TRAIN_ROOT / "fg").mkdir(parents=True, exist_ok=True)
    (TRAIN_ROOT / "labels").mkdir(parents=True, exist_ok=True)

    anno = {}
    names, seq_of = [], {}
    dropped = 0
    for p in pairs:
        k = key(p)
        m = mby.get(k); c = cby.get(int(p["image_id"]))
        if m is None or c is None:
            dropped += 1; continue
        bb = m.get("bbox_xyxy"); W = float(c["width"]); H = float(c["height"])
        scene = c["image_path"]; crop = str(DD / m["crop_path"])
        if not bb or not os.path.exists(scene) or not os.path.exists(crop):
            dropped += 1; continue
        nb = [bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H]
        # drop degenerate boxes (would break cv2.resize in the compositing dataloader)
        tw = int(nb[2] * IMG) - int(nb[0] * IMG); th = int(nb[3] * IMG) - int(nb[1] * IMG)
        ow = int(bb[2]) - int(bb[0]); oh = int(bb[3]) - int(bb[1])
        if tw < 2 or th < 2 or ow < 2 or oh < 2:
            dropped += 1; continue
        name = f"{int(p['image_id'])}_{int(p['annotation_id'])}.jpg"
        # symlinks (relative-safe: use absolute targets)
        for folder, target in (("bg", scene), ("fg", crop)):
            link = TRAIN_ROOT / folder / name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(target, link)
        anno[name] = {"bbox": [float(x) for x in nb], "similars": [], "dissimilars": [], "class": "vessel"}
        names.append(name); seq_of[name] = str(p.get("sequence_group", ""))

    # negatives: random other vessels, preferring a different sequence_group
    print(f"built {len(names)} bg/fg pairs (dropped {dropped}); sampling negatives...")
    all_names = names
    for name in names:
        sg = seq_of[name]
        negs = []
        tries = 0
        while len(negs) < N_NEG_POOL and tries < N_NEG_POOL * 5:
            cand = random.choice(all_names)
            tries += 1
            if cand == name or seq_of[cand] == sg:
                continue
            negs.append(cand)
        if not negs:  # fallback
            negs = [random.choice(all_names) for _ in range(N_NEG_POOL)]
        anno[name]["dissimilars"] = negs

    json.dump(anno, open(TRAIN_ROOT / "labels/train_maritime.json", "w"))
    json.dump({"vessel": {"images": names}}, open(TRAIN_ROOT / "labels/category.json", "w"))
    print(f"wrote labels/train_maritime.json ({len(anno)} entries) and category.json")
    print("bg dir:", TRAIN_ROOT / "bg", "fg dir:", TRAIN_ROOT / "fg")


if __name__ == "__main__":
    main()
