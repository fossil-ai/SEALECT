#!/usr/bin/env python3
"""
Experiment A -- independent-detector plausibility.

Run the FROZEN maritime RF-DETR (ONNX) on each retrieval method's rendered
composites and check whether it detects a Watercraft at the inserted-vessel box.
The detector is a different model that never saw SEALECT's pose priors, so this
is a prior-independent measure of insertion plausibility.

Runs in the PaddleDetection venv (onnxruntime). Reports, per method:
  - recall@0.5 : fraction of composites where the inserted vessel is detected
                 (a Watercraft det with IoU>=0.5 to the paste box)
  - mean_conf  : mean detector confidence of the matched detection
  - mean_bestIoU
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path("/path/to/sealect_root/experiments/experiment5_1")
ONNX = Path("/path/to/PaddleDetection/DEAD40/script/"
            "rfdetr_dead40_maritime_land_eo/inference_model.onnx")
METHODS = ["sealect", "dino", "clip", "random", "prior_only", "dino_prior"]
KS = ["k20", "k100"]
WATERCRAFT_CLS = 2
CONF_THRESH = 0.25
IOU_MATCH = 0.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (576, 576)).astype(np.float32) / 255.0
    r = (r - MEAN) / STD
    return np.transpose(r, (2, 0, 1))[None].astype(np.float32)


def detect(sess, inp_name, img_bgr):
    H, W = img_bgr.shape[:2]
    dets, labels = sess.run(None, {inp_name: preprocess(img_bgr)})
    dets, labels = dets[0], labels[0]                       # [300,4], [300,91]
    scores = sigmoid(labels)
    cls = scores.argmax(1)
    conf = scores.max(1)
    out = []
    for i in range(dets.shape[0]):
        if conf[i] < CONF_THRESH:
            continue
        cx, cy, w, h = dets[i]
        x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        out.append((int(cls[i]), float(conf[i]), [x1, y1, x2, y2]))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_image(method, K, entry):
    fn = entry.get("file_name")
    cand = [ROOT / "diffharmony_outputs" / K / method / fn,          # harmonized final
            Path(entry.get("save_path", "")),                         # metadata's own composite
            ROOT / "bootplace_outputs_clean" / K / method / "composites" / fn]
    for p in cand:
        if p and Path(p).exists():
            return Path(p), ("harmonized" if "diffharmony" in str(p) else "composite")
    return None, None


def box_of(entry):
    for k in ("paste_bbox_xyxy", "target_box_xyxy"):
        if entry.get(k):
            return [float(v) for v in entry[k]]
    return None


def main(limit=None):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(ONNX), so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cls_hist = defaultdict(int)
    rows = {}

    for K in KS:
        for method in METHODS:
            mp = ROOT / "bootplace_outputs_clean" / K / method / "placement_metadata.json"
            if not mp.exists():
                continue
            meta = json.load(open(mp))
            if limit:
                meta = meta[:limit]
            n, hit, confs, ious, src_used = 0, 0, [], [], defaultdict(int)
            for e in meta:
                box = box_of(e)
                img_path, src = resolve_image(method, K, e)
                if box is None or img_path is None:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                n += 1
                src_used[src] += 1
                best_iou, best_conf = 0.0, 0.0
                for c, cf, b in detect(sess, inp, img):
                    cls_hist[c] += 1
                    if c != WATERCRAFT_CLS:
                        continue
                    j = iou(b, box)
                    if j > best_iou:
                        best_iou, best_conf = j, cf
                ious.append(best_iou)
                if best_iou >= IOU_MATCH:
                    hit += 1
                    confs.append(best_conf)
            if n:
                rows[f"{method}/{K}"] = dict(
                    n=n, recall=hit / n,
                    mean_conf=float(np.mean(confs)) if confs else 0.0,
                    mean_bestIoU=float(np.mean(ious)),
                    src=dict(src_used))
                print(f"{method:11s} {K}: n={n:3d}  recall@.5={hit/n:.3f}  "
                      f"mean_conf={np.mean(confs) if confs else 0:.3f}  "
                      f"mean_bestIoU={np.mean(ious):.3f}  src={dict(src_used)}")

    print("\n[sanity] detected-class histogram (2=Watercraft expected dominant):",
          dict(sorted(cls_hist.items(), key=lambda x: -x[1])[:6]))
    out = Path("/path/to/sealect_root/sealect2/downstream/expA_detector_plausibility.json")
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    import sys
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
