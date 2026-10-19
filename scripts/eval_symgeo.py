#!/usr/bin/env python3
"""Symmetry-aware geodesic error for the canonical pose estimator.

geo_sym = min( geo(R_pred, R_gt), geo(R_pred, Ry(pi) @ R_gt) )  -- the metric
consistent with the paper's L_sym (forward/backward 180-deg heading ambiguity).
Reuses validat3's helpers + config and pose3's model. Read-only; prints CSV.
"""
import os, math, glob
from collections import defaultdict

import numpy as np
import torch

import validat3 as V
from pose3 import PoseNet, cfg

# Match the canonical v12sym_stable checkpoint's bin heads (18/24/18).
cfg.PITCH_BINS, cfg.YAW_BINS, cfg.ROLL_BINS = 18, 24, 18

DEVICE = V.DEVICE
print(f"[symgeo] device={DEVICE} ckpt={V.CKPT_PATH}")

gt_map = V.load_gt_map(V.VAL_ANN_DIR)
model = PoseNet(cfg.SEGFORMER_ENCODER_CKPT, cfg.SEGFORMER_UNFREEZE_LAST_N).to(DEVICE)
model.eval()
with torch.no_grad():                                    # probe to build lazy layers
    _ = model(torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=DEVICE))
model.load_state_dict(torch.load(V.CKPT_PATH, map_location=DEVICE), strict=True)
model.eval()

# Ry(pi): 180-deg rotation about the yaw axis, same Euler convention as the GT.
Rypi = V.euler_to_rotmat(torch.zeros(1), torch.tensor([math.pi]), torch.zeros(1))[0].to(DEVICE).float()

groups = defaultdict(list); allrows = []
with torch.no_grad():
    for ip in sorted(glob.glob(os.path.join(V.VAL_IMG_DIR, "*.jpg"))):
        name = os.path.basename(ip)
        if name not in gt_map:
            continue
        ds = V.parse_dataset_key(name)
        x = V.preprocess_image(ip, V.IMG_SIZE).unsqueeze(0).to(DEVICE)
        R_gt = gt_map[name]["R"].unsqueeze(0).to(DEVICE).float()
        R_pred = model(x)[0].float()
        geo_raw = V.rad2deg(V.geodesic_loss(R_pred, R_gt))[0].item()
        R_gt_flip = (Rypi @ R_gt[0]).unsqueeze(0)
        geo_flip = V.rad2deg(V.geodesic_loss(R_pred, R_gt_flip))[0].item()
        geo_sym = min(geo_raw, geo_flip)
        row = {"ds": ds, "geo_raw": geo_raw, "geo_sym": geo_sym}
        groups[ds].append(row); allrows.append(row)

def agg(rows):
    gs = np.array([r["geo_sym"] for r in rows]); gr = np.array([r["geo_raw"] for r in rows])
    return (len(rows), gs.mean(), np.median(gs), np.percentile(gs, 90), gr.mean(), np.median(gr))

print("dataset,n,geosym_mean,geosym_med,geosym_p90,georaw_mean,georaw_med")
for ds in sorted(groups):
    n, m, md, p90, rm, rmd = agg(groups[ds])
    print(f"{ds},{n},{m:.2f},{md:.2f},{p90:.2f},{rm:.2f},{rmd:.2f}")
n, m, md, p90, rm, rmd = agg(allrows)
print(f"__OVERALL__,{n},{m:.2f},{md:.2f},{p90:.2f},{rm:.2f},{rmd:.2f}")
