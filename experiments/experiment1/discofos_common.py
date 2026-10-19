#!/usr/bin/env python3
"""
Shared helpers for the DiscoFOS (ICCV 2023) maritime baseline.

DiscoFOS = "Foreground Object Search by Distilling Composite Image Feature"
(Zhang, Sui, Niu). We reuse the official architecture (network.networks.StudentModel,
SingleScaleD teacher) and its exact image preprocessing, adapted to the SEALECT
maritime scene/candidate records. Retrieval ranks candidates by cosine similarity
of the L2-normalized background and foreground embeddings from the two-tower student
(the space the triplet loss optimizes).
"""
import os
import sys
import math
from types import SimpleNamespace

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = "/path/to/sealect_root"
DISCOFOS_DIR = os.path.join(REPO, "experiments/baselines/DiscoFOS")
if DISCOFOS_DIR not in sys.path:
    sys.path.insert(0, DISCOFOS_DIR)

from network.networks import StudentModel, SingleScaleD  # noqa: E402

# S-FOSD config (config/config_sfosd.yaml), the setting the released sd.pth was trained under.
IMAGE_SIZE = 224
FILL_PIXEL = 128
PAD_PIXEL = 255
IMAGE_NET_MEAN = [0.5, 0.5, 0.5]
IMAGE_NET_STD = [0.5, 0.5, 0.5]

CKPT_STUDENT = os.path.join(DISCOFOS_DIR, "checkpoints/sd.pth")
CKPT_TEACHER = os.path.join(DISCOFOS_DIR, "checkpoints/teacher/teacher_sfosd.pth")


def student_cfg():
    return SimpleNamespace(
        backbone="vgg19", feature_dim=512, encoder_feature="gap",
        encoder_classify=False, distill_type="roiconcat", triplet_margin=0.1,
    )


_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])
_tf_resize = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])


def load_student(device, ckpt=CKPT_STUDENT, loadweights=False):
    model = StudentModel(student_cfg(), loadweights=loadweights).to(device)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # triplet_loss has no params; distill/classifier heads may be unused at eval.
        missing = [k for k in missing if not k.startswith("triplet_loss")]
        if missing:
            print(f"[discofos] missing keys ({len(missing)}): {missing[:6]}")
        if unexpected:
            print(f"[discofos] unexpected keys ({len(unexpected)}): {unexpected[:6]}")
    return model.eval()


def load_teacher(device, ckpt=CKPT_TEACHER):
    model = SingleScaleD(loadweights=False).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    model.load_state_dict(sd, strict=True)
    return model.eval()


# ---- preprocessing (mirrors dataset.datasets.TestDataset) ----
def _normbox2global(norm_box, w, h):
    x1, y1, x2, y2 = norm_box
    return [max(int(x1 * w), 0), max(int(y1 * h), 0), min(int(x2 * w), w), min(int(y2 * h), h)]


def _pad_to_square(img, pad_pixel=PAD_PIXEL):
    h, w = img.shape[:2]
    if h == w:
        return img
    if w > h:
        pad_w, pad_h = 0, w - h
    else:
        pad_w, pad_h = h - w, 0
    py1, py2 = pad_h // 2, pad_h - pad_h // 2
    px1, px2 = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        return np.pad(img, ((py1, py2), (px1, px2), (0, 0)), "constant", constant_values=pad_pixel)
    return np.pad(img, ((py1, py2), (px1, px2)), "constant", constant_values=pad_pixel)


def make_bg_inputs(scene_path, norm_box):
    """Return (bg_tensor[C,H,W], query_box[4], crop_box[4]) exactly like TestDataset."""
    bg_im = Image.open(scene_path).convert("RGB")
    # fill the target box with FILL_PIXEL so the model cannot peek at the true object
    bw, bh = bg_im.size
    gx1, gy1, gx2, gy2 = _normbox2global(norm_box, bw, bh)
    arr = np.asarray(bg_im).copy()
    arr[gy1:gy2, gx1:gx2] = FILL_PIXEL
    bg_im = Image.fromarray(arr)
    bg_t = _tf_resize(bg_im)

    bg_ori_w, bg_ori_h = bg_im.size
    ori_x1, ori_y1, ori_x2, ori_y2 = _normbox2global(norm_box, bg_ori_w, bg_ori_h)
    ori_tw, ori_th = ori_x2 - ori_x1, ori_y2 - ori_y1

    query_box = _normbox2global(norm_box, IMAGE_SIZE, IMAGE_SIZE)
    add_w = int(ori_tw * (math.sqrt(2) - 1) / 2)
    add_h = int(ori_th * (math.sqrt(2) - 1) / 2)
    new_x1 = max(0, ori_x1 - add_w); new_x2 = min(bg_ori_w, ori_x2 + add_w)
    new_y1 = max(0, ori_y1 - add_h); new_y2 = min(bg_ori_h, ori_y2 + add_h)
    new_norm = [new_x1 / bg_ori_w, new_y1 / bg_ori_h, new_x2 / bg_ori_w, new_y2 / bg_ori_h]
    crop_box = _normbox2global(new_norm, IMAGE_SIZE, IMAGE_SIZE)
    return bg_t, torch.tensor(query_box).float(), torch.tensor(crop_box).float()


def make_fg_input(crop_path):
    """Return fg_tensor[C,H,W] exactly like TestDataset (pad to square, resize, normalize)."""
    fg = np.asarray(Image.open(crop_path).convert("RGB"))
    pad_fg = Image.fromarray(_pad_to_square(fg.copy(), PAD_PIXEL))
    return _tf_resize(pad_fg)


@torch.no_grad()
def encode_bg(model, bg_t, crop_box, device):
    return F.normalize(model.forward_bg_encoder(bg_t.to(device), crop_box.to(device)), dim=1)


@torch.no_grad()
def encode_fg(model, fg_t, device):
    return F.normalize(model.forward_fg_encoder(fg_t.to(device)), dim=1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    student = load_student(device)
    teacher = load_teacher(device)
    print("student + teacher loaded ok")
    bg = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    fg = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    cbox = torch.tensor([[10, 10, 100, 100], [20, 20, 120, 120]]).float()
    be = encode_bg(student, bg, cbox, device)
    fe = encode_fg(student, fg, device)
    print("bg_emb:", tuple(be.shape), "fg_emb:", tuple(fe.shape))
    print("cos(bg,fg) sample:", (be * fe).sum(1).tolist())
