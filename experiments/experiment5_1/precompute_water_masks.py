#!/usr/bin/env python3
"""
Precompute per-scene WATER masks for BootPlace placement QC (the water gate in
bootplace/BOOTPLACE/test_v2.py:select_maritime_box).

Runs an ADE20K SegFormer semantic segmenter (via transformers) over each background
scene and unions the water-family classes (water / sea / ocean / river / lake) into a
binary mask. test_v2.py loads <data_path>/test/water_masks/<stem>.png and requires a
placement box to be >= 50% water.

Usage:
  python3 precompute_water_masks.py \
      --images-dir /path/to/<data_path>/test/backgrounds_single \
      --out-dir    /path/to/<data_path>/test/water_masks
  [--model nvidia/segformer-b4-finetuned-ade-512-512] [--device cuda]

Backgrounds are identical across methods, so you can compute once and copy/symlink the
resulting water_masks/ into each method's request dir.

Output: <out-dir>/<image_stem>.png  (L-mode, 255 = water, 0 = non-water)
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

WATER_TERMS = {"water", "sea", "ocean", "river", "lake"}


def water_class_ids(id2label):
    """ADE20K label ids whose name contains a water-family token (robust to indexing)."""
    ids = []
    for i, lab in id2label.items():
        toks = {t.strip().lower() for t in str(lab).replace(";", ",").split(",")}
        if toks & WATER_TERMS:
            ids.append(int(i))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="dir of background scene images")
    ap.add_argument("--out-dir", required=True, help="dir to write <stem>.png water masks")
    ap.add_argument("--model", default="nvidia/segformer-b4-finetuned-ade-512-512")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", default="jpg,jpeg,png")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    proc = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(args.device).eval()
    wids = water_class_ids(model.config.id2label)
    if not wids:
        raise SystemExit("No water-family classes found in the model's label set.")
    print("water class ids:", {i: model.config.id2label[i] for i in wids})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for e in args.exts.split(","):
        files += glob.glob(os.path.join(args.images_dir, f"*.{e.strip()}"))
    files = sorted(set(files))
    print(f"{len(files)} images -> {out}")

    for n, fp in enumerate(files, 1):
        img = Image.open(fp).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, C, h', w')
        up = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        )
        seg = up.argmax(dim=1)[0].cpu().numpy()
        water = (np.isin(seg, wids).astype(np.uint8) * 255)
        Image.fromarray(water, mode="L").save(out / (Path(fp).stem + ".png"))
        if n % 25 == 0 or n == len(files):
            print(f"  {n}/{len(files)}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
