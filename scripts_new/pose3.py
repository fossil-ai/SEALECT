#!/usr/bin/env python3
"""
PoseNet v5 — Dual-Stream Pose Estimator (CNN + SegFormer) — STABLE + CLEAN VIZ


- No phases. Single objective from step 1.
- SegFormer forward forced FP32.
- AMP (BF16 preferred) for the rest, but auto-disables permanently on ANY non-finite grads/loss.
- Removes Euler loss entirely (common source of exploding/discontinuous grads).
- Replaces custom BatchNorm layers (in our fusion/proj/align blocks) with GroupNorm for small-batch stability.
- Visualization: ONLY draw triad axes for GT and Pred.
  Also fixes the typical “GT is backwards” issue by using inverse rotation (R^T) for display.
  (If your GT is already camera->object, set VIZ_USE_R_TRANSPOSE=False.)

NOTE:
- We assume your GT Euler->R is object->camera (or world->camera) style.
  The viz usually wants camera->object (inverse), hence transpose.
"""

import os, json, math, glob, random, warnings
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import SegformerModel

warnings.filterwarnings("ignore", category=FutureWarning)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class Cfg:
    BASE_DIR: str = "/path/to/sealect_root/data"
    ANNOTATION_DIR: str = ""
    IMAGE_DIR: str = ""
    VAL_ANNOTATION_DIR: str = "/path/to/sealect_root/unified/val_annotations"
    VAL_IMAGE_DIR: str = "/path/to/sealect_root/unified/val_images"
    SEGFORMER_ENCODER_CKPT: str = (
        "/path/to/sealect_root/segment1/segformer_boat/segformer_encoder.pt"
    )

    IMG_SIZE: int = 512
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    EPOCHS: int = 50

    # Safer defaults
    LR_CNN: float = 2e-4
    LR_HEAD: float = 1e-4
    LR_ENCODER: float = 1e-6
    WEIGHT_DECAY: float = 1e-3
    GRAD_CLIP_NORM: float = 0.5

    # AMP: prefer BF16 if supported
    USE_AMP: bool = True
    PREFER_BF16: bool = True

    EMA_DECAY: float = 0.999
    WARMUP_RATIO: float = 0.05

    # Augmentation
    COLOR_JITTER: Tuple = (0.3, 0.3, 0.2, 0.08)
    RANDOM_ERASING_P: float = 0.15
    AFFINE_TRANSLATE: float = 0.08
    AFFINE_SCALE: Tuple = (0.90, 1.10)
    GAUSSIAN_BLUR_P: float = 0.12

    # Bins
    # PITCH_BINS: int = 18
    # YAW_BINS: int = 18
    # ROLL_BINS: int = 18
    # LABEL_SMOOTHING: float = 0.05
    #
    # # Loss weights (Euler disabled)
    # BIN_W: float = 2.0
    # SIXD_W: float = 0.35
    # GEODESIC_W: float = 1.0
    EULER_W: float = 0.0  # intentionally unused

    PITCH_BINS = 12
    YAW_BINS = 18
    ROLL_BINS = 12
    BIN_W = 2.5
    GEODESIC_W = 1.0
    SIXD_W = 0.25
    LABEL_SMOOTHING = 0.08

    # Encoder
    SEGFORMER_UNFREEZE_LAST_N: int = 4

    OUT_DIR: str = "imback1"
    VIZ_EVERY_EPOCH: int = 5
    VIZ_SAMPLES: int = 50
    FINAL_VIZ_ALL: bool = True
    SEED: int = 42

    # Stability
    FREEZE_RESNET_BN_STATS: bool = True

    # Bound 6D head output
    ROT6D_TANH_SCALE: float = 2.0

    # Visualization:
    # Most “GT facing opposite” issues are because the stored R is object->camera,
    # but the drawn axes should use camera->object. Using R^T fixes that.
    VIZ_USE_R_TRANSPOSE: bool = True

    def __post_init__(self):
        self.ANNOTATION_DIR = os.path.join(self.BASE_DIR, "annotations1")
        self.IMAGE_DIR = os.path.join(self.BASE_DIR, "images")


cfg = Cfg()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       ROTATION MATH                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def deg2rad(x: float) -> float:
    return x * (math.pi / 180.0)

def rad2deg(x: torch.Tensor) -> torch.Tensor:
    return x * (180.0 / math.pi)

def euler_to_rotmat(p: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    cr, sr = torch.cos(r), torch.sin(r)
    z = torch.zeros_like(cp)
    o = torch.ones_like(cp)

    Rx = torch.stack([o,z,z, z,cp,-sp, z,sp,cp], -1).reshape(-1,3,3)
    Ry = torch.stack([cy,z,sy, z,o,z, -sy,z,cy], -1).reshape(-1,3,3)
    Rz = torch.stack([cr,-sr,z, sr,cr,z, z,z,o], -1).reshape(-1,3,3)
    return Ry @ Rx @ Rz

def rotmat_to_euler(R: torch.Tensor, eps: float = 1e-6):
    x = (-R[:,1,2]).clamp(-1 + eps, 1 - eps)
    pitch = torch.asin(x)
    yaw   = torch.atan2(R[:,0,2], R[:,2,2])
    roll  = torch.atan2(R[:,1,0], R[:,1,1])
    return pitch, yaw, roll

def sixd_to_rotmat(sixd: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a1, a2 = sixd[:, :3], sixd[:, 3:]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def geodesic_loss(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    Rd = R1.transpose(-1,-2) @ R2
    tr = Rd[:,0,0] + Rd[:,1,1] + Rd[:,2,2]
    c = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    return torch.acos(c)

def euler_bin_id(angle_rad: torch.Tensor, num_bins: int) -> torch.Tensor:
    t = (angle_rad + math.pi) / (2 * math.pi)
    return torch.floor(t * num_bins).long().clamp(0, num_bins - 1)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          DATASET                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class BoatPoseDataset(Dataset):
    def __init__(self, ann_dir: str, img_dir: str, img_size: int, augment: bool = False):
        self.files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        assert len(self.files) > 0, f"No annotations in {ann_dir}"
        self.img_dir = img_dir
        self.augment = augment
        self.img_size = img_size
        self.normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        self.jitter = transforms.ColorJitter(*cfg.COLOR_JITTER)
        self.erasing = transforms.RandomErasing(p=cfg.RANDOM_ERASING_P, scale=(0.02,0.10))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with open(self.files[idx]) as f:
            data = json.load(f)

        pose = data["annotations"][0]["pose"]
        p_d = float(pose["pitch"])
        y_d = float(pose["yaw"])
        r_d = float(pose["roll"])

        name = data["imageFileName"]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        if self.augment:
            img = self._augment(img)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = transforms.ToTensor()(img)
        img_t = self.normalize(img_t)
        if self.augment:
            img_t = self.erasing(img_t)

        p_r = torch.tensor(deg2rad(p_d), dtype=torch.float32)
        y_r = torch.tensor(deg2rad(y_d), dtype=torch.float32)
        r_r = torch.tensor(deg2rad(r_d), dtype=torch.float32)

        R = euler_to_rotmat(p_r.unsqueeze(0), y_r.unsqueeze(0), r_r.unsqueeze(0)).squeeze(0)

        euler_deg = torch.tensor([p_d, y_d, r_d], dtype=torch.float32)
        bins = torch.tensor([
            euler_bin_id(p_r, cfg.PITCH_BINS).item(),
            euler_bin_id(y_r, cfg.YAW_BINS).item(),
            euler_bin_id(r_r, cfg.ROLL_BINS).item(),
        ], dtype=torch.long)

        return img_t, R, euler_deg, bins, name

    def _augment(self, img: Image.Image):
        w, h = img.size
        if random.random() < 0.4:
            img = self.jitter(img)
        if random.random() < cfg.GAUSSIAN_BLUR_P:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        if random.random() < 0.35:
            dx = random.uniform(-cfg.AFFINE_TRANSLATE, cfg.AFFINE_TRANSLATE) * w
            dy = random.uniform(-cfg.AFFINE_TRANSLATE, cfg.AFFINE_TRANSLATE) * h
            s = random.uniform(*cfg.AFFINE_SCALE)
            cx, cy = w / 2, h / 2
            img = img.transform(
                (w, h),
                Image.AFFINE,
                (s, 0, cx * (1 - s) + dx, 0, s, cy * (1 - s) + dy),
                resample=Image.BILINEAR,
            )
        return img


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     NORMALIZATION HELPERS                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def GN(c: int, g: int = 32) -> nn.GroupNorm:
    # stable for tiny batches, works fine with BF16/FP16
    return nn.GroupNorm(num_groups=min(g, c), num_channels=c)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     CNN STREAM (Geometry)                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class CoordConv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, xx, yy], dim=1)

class CNNStream(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        self.coord = CoordConv()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            self.conv1.weight[:, 3:] = 0.0

        # keep backbone BN, but we'll freeze stats in train loop
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # custom blocks: use GN instead of BN
        self.proj2 = nn.Sequential(nn.Conv2d(128, out_dim, 1), GN(out_dim), nn.GELU())
        self.proj3 = nn.Sequential(nn.Conv2d(256, out_dim, 1), GN(out_dim), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(512, out_dim, 1), GN(out_dim), nn.GELU())
        self.fuse  = nn.Sequential(nn.Conv2d(out_dim * 3, out_dim, 1), GN(out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord(x)
        x = self.conv1(x)
        x = self.bn1(x)      # frozen stats during training
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        th, tw = f3.shape[2:]
        p2 = F.interpolate(self.proj2(f2), (th, tw), mode="bilinear", align_corners=False)
        p3 = self.proj3(f3)
        p4 = F.interpolate(self.proj4(f4), (th, tw), mode="bilinear", align_corners=False)

        return self.fuse(torch.cat([p2, p3, p4], dim=1))


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  SEGFORMER STREAM (Semantics)                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class SegFormerStream(nn.Module):
    def __init__(self, encoder_ckpt: str, unfreeze_last_n: int, out_dim: int = 512):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        if os.path.isfile(encoder_ckpt):
            sd = torch.load(encoder_ckpt, map_location="cpu")
            m, u = self.encoder.load_state_dict(sd, strict=False)
            print(f"SegFormer ckpt: {len(m)} missing, {len(u)} unexpected")
        else:
            print("WARNING: SegFormer ckpt not found")

        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze last N blocks + layer norms
        blocks = self.encoder.encoder.block
        for i in range(max(0, len(blocks) - unfreeze_last_n), len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
        for ln in self.encoder.encoder.layer_norm:
            for p in ln.parameters():
                p.requires_grad = True

        self.out_dim = out_dim
        self._built = False
        self.projs = None
        self.combine = None

    def _build_fusion(self, channel_sizes, device):
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, self.out_dim, 1), GN(self.out_dim), nn.GELU())
            for c in channel_sizes
        ])
        self.combine = nn.Sequential(
            nn.Conv2d(self.out_dim * len(channel_sizes), self.out_dim, 1),
            GN(self.out_dim),
            nn.GELU(),
        )
        self.projs.to(device)
        self.combine.to(device)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force SegFormer in FP32 (autocast disabled)
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.encoder(pixel_values=x.float(), output_hidden_states=True, return_dict=True)

        feats = []
        for hs in out.hidden_states[1:]:
            if hs.dim() == 3:
                B, N, C = hs.shape
                h = w = int(math.sqrt(N))
                hs = hs.transpose(1, 2).reshape(B, C, h, w)
            feats.append(hs)

        if not self._built:
            self._build_fusion([f.shape[1] for f in feats], feats[0].device)

        th, tw = feats[-1].shape[2:]
        proj = []
        for f, p in zip(feats, self.projs):
            o = p(f)
            if o.shape[2:] != (th, tw):
                o = F.interpolate(o, (th, tw), mode="bilinear", align_corners=False)
            proj.append(o)

        return self.combine(torch.cat(proj, dim=1))


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      SPATIAL POOLING                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class SpatialAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.attn(x).flatten(2), dim=-1)          # [B,1,HW]
        return (x.flatten(2) * w).sum(-1)                       # [B,C]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       DUAL-STREAM MODEL                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class PoseNet(nn.Module):
    def __init__(self, encoder_ckpt: str, segformer_unfreeze: int):
        super().__init__()
        feat_dim = 512

        self.cnn = CNNStream(out_dim=feat_dim)
        self.seg = SegFormerStream(encoder_ckpt, segformer_unfreeze, out_dim=feat_dim)

        # custom blocks: GN not BN
        self.align_conv = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1),
            GN(feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            GN(feat_dim),
            nn.GELU(),
        )

        self.attn_pool = SpatialAttentionPool(feat_dim)
        pool_dim = feat_dim * 2

        self.trunk = nn.Sequential(
            nn.Linear(pool_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.head_pitch_bin = nn.Linear(256, cfg.PITCH_BINS)
        self.head_yaw_bin   = nn.Linear(256, cfg.YAW_BINS)
        self.head_roll_bin  = nn.Linear(256, cfg.ROLL_BINS)

        self.head_6d = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor):
        cnn_feat = self.cnn(x)
        seg_feat = self.seg(x)

        th, tw = cnn_feat.shape[2:]
        if seg_feat.shape[2:] != (th, tw):
            seg_feat = F.interpolate(seg_feat, (th, tw), mode="bilinear", align_corners=False)

        fused = self.align_conv(torch.cat([cnn_feat, seg_feat], dim=1))

        z_attn = self.attn_pool(fused)
        z_gap = fused.mean(dim=[2, 3])
        z = torch.cat([z_attn, z_gap], dim=-1)

        z = self.trunk(z)

        p_logits = self.head_pitch_bin(z)
        y_logits = self.head_yaw_bin(z)
        r_logits = self.head_roll_bin(z)

        # Bound 6D output to prevent inf -> normalize -> NaN
        rot6d = cfg.ROT6D_TANH_SCALE * torch.tanh(self.head_6d(z))
        R_pred = sixd_to_rotmat(rot6d)

        return R_pred, p_logits, y_logits, r_logits


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          LOSSES                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# def compute_loss(R_pred, p_logits, y_logits, r_logits,
#                  R_gt, euler_gt_deg, bin_gt):
#     # CE in FP32
#     p_logits = p_logits.float()
#     y_logits = y_logits.float()
#     r_logits = r_logits.float()
#
#     ls = cfg.LABEL_SMOOTHING
#     l_bp = F.cross_entropy(p_logits, bin_gt[:, 0], label_smoothing=ls)
#     l_by = F.cross_entropy(y_logits, bin_gt[:, 1], label_smoothing=ls)
#     l_br = F.cross_entropy(r_logits, bin_gt[:, 2], label_smoothing=ls)
#     l_bin = (l_bp + l_by + l_br) / 3.0
#
#     # Rotation losses FP32
#     R_pred_f = R_pred.float()
#     R_gt_f   = R_gt.float()
#
#     geo = geodesic_loss(R_pred_f, R_gt_f)
#     l_geo = geo.mean()
#     l_6d  = ((R_pred_f - R_gt_f) ** 2).sum(dim=[1, 2]).mean()
#
#     total = (
#         cfg.BIN_W * l_bin
#         + cfg.GEODESIC_W * l_geo
#         + cfg.SIXD_W * l_6d
#     ).float()
#
#     return total, {
#         "total": total.detach(),
#         "bin_ce": l_bin.detach(),
#         "geo": l_geo.detach(),
#         "sixd": l_6d.detach(),
#         "euler": torch.tensor(0.0, device=total.device),
#         "geo_deg": rad2deg(geo.detach()).mean(),
#     }

def compute_loss(R_pred, p_logits, y_logits, r_logits,
                 R_gt, euler_gt_deg, bin_gt):
    # CE in FP32
    p_logits = p_logits.float()
    y_logits = y_logits.float()
    r_logits = r_logits.float()

    ls = cfg.LABEL_SMOOTHING

    # Pitch/Roll bins: normal CE
    l_bp = F.cross_entropy(p_logits, bin_gt[:, 0], label_smoothing=ls)
    l_br = F.cross_entropy(r_logits, bin_gt[:, 2], label_smoothing=ls)

    # ---- Yaw bin symmetry: allow yaw or yaw+pi ----
    yaw_bin0 = bin_gt[:, 1]
    yaw_bin1 = yaw_bin_sym_from_euler_deg(euler_gt_deg, cfg.YAW_BINS).to(yaw_bin0.device)

    l_by0 = F.cross_entropy(y_logits, yaw_bin0, label_smoothing=ls, reduction="none")  # [B]
    l_by1 = F.cross_entropy(y_logits, yaw_bin1, label_smoothing=ls, reduction="none")  # [B]
    l_by = torch.minimum(l_by0, l_by1).mean()

    l_bin = (l_bp + l_by + l_br) / 3.0

    # Rotation losses FP32
    R_pred_f = R_pred.float()
    R_gt_f   = R_gt.float()

    # ---- Rotation symmetry: compare to R_gt and Ry(pi)@R_gt ----
    B = R_gt_f.shape[0]
    Rflip = Ry_pi(B, R_gt_f.device, R_gt_f.dtype)         # [B,3,3]
    R_gt_sym = Rflip @ R_gt_f                              # [B,3,3]   (because your R = Ry @ Rx @ Rz)

    geo0 = geodesic_loss(R_pred_f, R_gt_f)                 # [B]
    geo1 = geodesic_loss(R_pred_f, R_gt_sym)               # [B]
    geo  = torch.minimum(geo0, geo1)
    l_geo = geo.mean()

    l6d0 = ((R_pred_f - R_gt_f) ** 2).sum(dim=[1, 2])      # [B]
    l6d1 = ((R_pred_f - R_gt_sym) ** 2).sum(dim=[1, 2])    # [B]
    l_6d = torch.minimum(l6d0, l6d1).mean()

    total = (
        cfg.BIN_W * l_bin
        + cfg.GEODESIC_W * l_geo
        + cfg.SIXD_W * l_6d
    ).float()

    return total, {
        "total": total.detach(),
        "bin_ce": l_bin.detach(),
        "geo": l_geo.detach(),
        "sixd": l_6d.detach(),
        "euler": torch.tensor(0.0, device=total.device),
        "geo_deg": rad2deg(geo.detach()).mean(),
    }
# ╔══════════════════════════════════════════════════════════════════╗
# ║                            EMA                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        bk = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                bk[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return bk

    @torch.no_grad()
    def restore(self, model: nn.Module, bk):
        for n, p in model.named_parameters():
            if n in bk:
                p.data.copy_(bk[n])


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      EVALUATION                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    keys = ["total","bin_ce","geo","sixd","euler","geo_deg",
            "pitch_err","yaw_err","roll_err",
            "pitch_acc","yaw_acc","roll_acc"]
    accum = {k: 0.0 for k in keys}
    n = 0

    for imgs, R_gt, euler_gt, bins_gt, _ in loader:
        imgs     = imgs.to(DEVICE, non_blocking=True)
        R_gt     = R_gt.to(DEVICE, non_blocking=True)
        euler_gt = euler_gt.to(DEVICE, non_blocking=True)
        bins_gt  = bins_gt.to(DEVICE, non_blocking=True)

        R_pred, pl, yl, rl = model(imgs)
        _, losses = compute_loss(R_pred, pl, yl, rl, R_gt, euler_gt, bins_gt)

        for k in ["total","bin_ce","geo","sixd","euler","geo_deg"]:
            accum[k] += losses[k].item()

        R_pred_f = R_pred.float()
        pp, yp, rp = rotmat_to_euler(R_pred_f)
        pd = torch.stack([rad2deg(pp), rad2deg(yp), rad2deg(rp)], -1)
        df = ((pd - euler_gt + 180) % 360 - 180).abs().mean(0)
        accum["pitch_err"] += df[0].item()

        # accum["yaw_err"]   += df[1].item()
        yaw = ((pd[:, 1] - euler_gt[:, 1] + 180) % 360 - 180).abs()
        yaw_sym = torch.minimum(yaw, (yaw - 180.0).abs())
        accum["yaw_err"] += yaw_sym.mean().item()

        accum["roll_err"]  += df[2].item()

        accum["pitch_acc"] += (pl.argmax(1) == bins_gt[:,0]).float().mean().item()
        accum["yaw_acc"]   += (yl.argmax(1) == bins_gt[:,1]).float().mean().item()
        accum["roll_acc"]  += (rl.argmax(1) == bins_gt[:,2]).float().mean().item()

        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      VISUALIZATION                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def _viz_rot(R_np: np.ndarray) -> np.ndarray:
    # Typical fix: show inverse (camera->object) instead of stored (object->camera)
    if cfg.VIZ_USE_R_TRANSPOSE:
        return R_np.T
    return R_np

def draw_triad_axes(img: np.ndarray, R_np: np.ndarray, center, length: int = 80, thickness: int = 2):
    """
    Draw ONLY triad axes (X red, Y green, Z blue).
    """
    Rv = _viz_rot(R_np)

    axes = np.eye(3, dtype=np.float32) * float(length)
    rotated = Rv @ axes

    colors = [(0,0,255), (0,255,0), (255,0,0)]  # X,Y,Z in BGR
    cx, cy = int(center[0]), int(center[1])

    for i in range(3):
        dx = float(rotated[0, i])
        dy = float(rotated[1, i])

        end = (int(cx + dx), int(cy - dy))
        cv2.arrowedLine(img, (cx, cy), end, colors[i], thickness, tipLength=0.2)

@torch.no_grad()
def visualize(model: nn.Module, dataset: BoatPoseDataset, out_dir: str, n: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idxs = list(range(len(dataset)))
    if n is not None and n < len(dataset):
        idxs = random.sample(idxs, n)

    for j, idx in enumerate(idxs):
        img_t, R_gt_t, euler_gt, _, name = dataset[idx]

        R_pred, _, _, _ = model(img_t.unsqueeze(0).to(DEVICE))
        R_pred_f = R_pred.float()
        R_pred_np = R_pred_f[0].cpu().numpy()
        R_gt_np = R_gt_t.numpy()

        if not np.isfinite(R_pred_np).all():
            continue

        # metrics text (pred euler only for display)
        pp, yp, rp = rotmat_to_euler(R_pred_f.cpu())
        pred_deg = torch.tensor([rad2deg(pp), rad2deg(yp), rad2deg(rp)]).squeeze()
        diff = ((pred_deg - euler_gt + 180) % 360 - 180).abs()
        geo_err = rad2deg(geodesic_loss(R_pred_f.cpu(), R_gt_t.unsqueeze(0).float())).item()

        img = cv2.imread(os.path.join(dataset.img_dir, name))
        if img is None:
            continue

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # ONLY two triads: GT and Pred
        draw_triad_axes(img, R_gt_np,   (cx - 90, cy), length=70, thickness=2)
        cv2.putText(img, "GT",   (cx - 115, cy - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        draw_triad_axes(img, R_pred_np, (cx + 90, cy), length=70, thickness=2)
        cv2.putText(img, "Pred", (cx + 55,  cy - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        gt_e = euler_gt.numpy()
        pr_e = pred_deg.numpy()
        d = diff.numpy()

        txts = [
            f"Geo: {geo_err:.1f}deg",
            f"GT   P:{gt_e[0]:.1f} Y:{gt_e[1]:.1f} R:{gt_e[2]:.1f}",
            f"Pred P:{pr_e[0]:.1f} Y:{pr_e[1]:.1f} R:{pr_e[2]:.1f}",
            f"Err  P:{d[0]:.1f} Y:{d[1]:.1f} R:{d[2]:.1f}",
        ]
        for ti, t in enumerate(txts):
            y0 = 28 + ti * 28
            cv2.putText(img, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 3)
            cv2.putText(img, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)

        cv2.imwrite(os.path.join(out_dir, f"{j:04d}_{os.path.splitext(name)[0]}.jpg"), img)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       METRIC PLOTS                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def plot_metrics(history, out_path: str):
    ep = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(20, 17))
    fig.suptitle("PoseNet — Dual Stream (CNN + SegFormer) — Stable", fontsize=16, fontweight="bold")

    def _p(ax, tk, vk, title, yl="", best=False):
        ax.plot(list(ep), history[tk], "b-", label="Train", lw=1.5, alpha=0.8)
        ax.plot(list(ep), history[vk], "r-", label="Val", lw=1.5)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
        if best and history[vk]:
            bi = int(np.argmin(history[vk]))
            ax.annotate(f"{history[vk][bi]:.1f}° @{bi+1}",
                        (bi+1, history[vk][bi]), fontsize=9, color="green", fontweight="bold")

    _p(axes[0,0], "train_total", "val_total", "Total Loss")
    _p(axes[0,1], "train_bin_ce", "val_bin_ce", "Bin CE Loss")
    _p(axes[0,2], "train_geo_deg", "val_geo_deg", "Geodesic Error (°)", "°", True)

    _p(axes[1,0], "train_pitch_err", "val_pitch_err", "Pitch Error (°)", "°", True)
    _p(axes[1,1], "train_yaw_err", "val_yaw_err", "Yaw Error (°)", "°", True)
    _p(axes[1,2], "train_roll_err", "val_roll_err", "Roll Error (°)", "°", True)

    _p(axes[2,0], "train_pitch_acc", "val_pitch_acc", "Pitch Bin Acc")
    _p(axes[2,1], "train_yaw_acc", "val_yaw_acc", "Yaw Bin Acc")
    _p(axes[2,2], "train_roll_acc", "val_roll_acc", "Roll Bin Acc")

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_distribution(dataset: BoatPoseDataset, out_path: str):
    ps, ys, rs = [], [], []
    for i in range(len(dataset)):
        with open(dataset.files[i]) as f:
            d = json.load(f)
        p = d["annotations"][0]["pose"]
        ps.append(p["pitch"])
        ys.append(p["yaw"])
        rs.append(p["roll"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, v, nm in zip(axes, [ps, ys, rs], ["Pitch", "Yaw", "Roll"]):
        ax.hist(v, bins=36, edgecolor="black", alpha=0.7)
        ax.set_title(f"{nm} Distribution")
        ax.set_xlabel("°")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       TRAINING LOOP                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def make_optimizer(model: nn.Module):
    cnn_p, seg_p, head_p = [], [], []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if nm.startswith("cnn."):
            cnn_p.append(p)
        elif nm.startswith("seg."):
            seg_p.append(p)
        else:
            head_p.append(p)

    return torch.optim.AdamW([
        {"params": cnn_p,  "lr": cfg.LR_CNN,     "name": "cnn"},
        {"params": seg_p,  "lr": cfg.LR_ENCODER, "name": "seg"},
        {"params": head_p, "lr": cfg.LR_HEAD,    "name": "head"},
    ], weight_decay=cfg.WEIGHT_DECAY)


def Ry_pi(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Rotation of +pi about Y axis
    # [[-1,0,0],
    #  [ 0,1,0],
    #  [ 0,0,-1]]
    R = torch.tensor([[-1., 0., 0.],
                      [ 0., 1., 0.],
                      [ 0., 0.,-1.]], device=device, dtype=dtype)
    return R.unsqueeze(0).expand(batch, 3, 3)

def yaw_bin_sym_from_euler_deg(euler_gt_deg: torch.Tensor, num_bins: int) -> torch.Tensor:
    # euler_gt_deg: [B,3] in degrees
    yaw_rad = torch.deg2rad(euler_gt_deg[:, 1].float())
    yaw_sym = yaw_rad + math.pi
    return euler_bin_id(yaw_sym, num_bins)  # [B]

def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))

def freeze_resnet_bn_stats(model: nn.Module):
    # Freeze only BatchNorm2d (ResNet backbone). GroupNorm is unaffected.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def _finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def _check_forward_finite(R_pred, pl, yl, rl):
    if not _finite_tensor(pl): return False, "p_logits"
    if not _finite_tensor(yl): return False, "y_logits"
    if not _finite_tensor(rl): return False, "r_logits"
    if not _finite_tensor(R_pred): return False, "R_pred"
    return True, ""

def _pick_autocast_dtype() -> torch.dtype:
    if DEVICE != "cuda":
        return torch.float32
    if cfg.PREFER_BF16:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16

def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
          train_ds: BoatPoseDataset, val_ds: BoatPoseDataset):

    out = cfg.OUT_DIR
    ckpt_dir = os.path.join(out, "checkpoints")
    viz_dir  = os.path.join(out, "viz")
    plot_dir = os.path.join(out, "plots")
    for d in [ckpt_dir, viz_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    plot_distribution(train_ds, os.path.join(plot_dir, "train_dist.png"))
    plot_distribution(val_ds, os.path.join(plot_dir, "val_dist.png"))

    opt = make_optimizer(model)

    amp_enabled = (DEVICE == "cuda" and cfg.USE_AMP)
    autocast_dtype = _pick_autocast_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and autocast_dtype == torch.float16))

    ema = EMA(model, cfg.EMA_DECAY)

    total_steps = cfg.EPOCHS * len(train_dl)
    warmup_steps = int(cfg.WARMUP_RATIO * total_steps)
    global_step = 0

    mk = ["total","bin_ce","geo","sixd","euler","geo_deg",
          "pitch_err","yaw_err","roll_err",
          "pitch_acc","yaw_acc","roll_acc"]
    history = {f"{s}_{k}": [] for s in ["train","val"] for k in mk}

    best_geo = float("inf")
    best_yaw = float("inf")

    print(f"[AMP] enabled={amp_enabled}  autocast_dtype={str(autocast_dtype).replace('torch.', '')}  "
          f"gradscaler={'ON' if scaler.is_enabled() else 'OFF'}")

    for epoch in range(cfg.EPOCHS):
        model.train()
        if cfg.FREEZE_RESNET_BN_STATS:
            freeze_resnet_bn_stats(model)

        run = {k: 0.0 for k in mk}
        nb = 0

        for imgs, R_gt, euler_gt, bins_gt, _ in train_dl:
            imgs     = imgs.to(DEVICE, non_blocking=True)
            R_gt     = R_gt.to(DEVICE, non_blocking=True)
            euler_gt = euler_gt.to(DEVICE, non_blocking=True)
            bins_gt  = bins_gt.to(DEVICE, non_blocking=True)

            # cosine LR per step
            base_lrs = [cfg.LR_CNN, cfg.LR_ENCODER, cfg.LR_HEAD]
            for pg, blr in zip(opt.param_groups, base_lrs):
                pg["lr"] = cosine_lr(global_step, total_steps, warmup_steps, blr)

            opt.zero_grad(set_to_none=True)

            # forward
            if DEVICE == "cuda" and amp_enabled:
                with torch.amp.autocast("cuda", enabled=True, dtype=autocast_dtype):
                    R_pred, pl, yl, rl = model(imgs)
                ok, bad = _check_forward_finite(R_pred, pl, yl, rl)
                if not ok:
                    print(f"[NON-FINITE] forward={bad} at epoch={epoch+1} step={global_step} -> skipping batch")
                    global_step += 1
                    continue
                loss, losses = compute_loss(R_pred, pl, yl, rl, R_gt, euler_gt, bins_gt)
            else:
                R_pred, pl, yl, rl = model(imgs)
                ok, bad = _check_forward_finite(R_pred, pl, yl, rl)
                if not ok:
                    print(f"[NON-FINITE] forward={bad} at epoch={epoch+1} step={global_step} -> skipping batch")
                    global_step += 1
                    continue
                loss, losses = compute_loss(R_pred, pl, yl, rl, R_gt, euler_gt, bins_gt)

            if not torch.isfinite(loss):
                print(f"[NON-FINITE] loss at epoch={epoch+1} step={global_step} -> skipping batch")
                if amp_enabled:
                    print("[AMP] Disabling AMP permanently due to non-finite loss; continuing in FP32.")
                    amp_enabled = False
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                global_step += 1
                continue

            # backward/step
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

                grads_finite = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break

                if not grads_finite:
                    print(f"[NON-FINITE] grads at epoch={epoch+1} step={global_step} -> skipping optimizer step")
                    if amp_enabled:
                        print("[AMP] Disabling AMP permanently due to non-finite grads; continuing in FP32.")
                        amp_enabled = False
                        scaler = torch.amp.GradScaler("cuda", enabled=False)
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

                grads_finite = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break

                if not grads_finite:
                    print(f"[NON-FINITE] grads at epoch={epoch+1} step={global_step} -> skipping optimizer step")
                    if amp_enabled:
                        print("[AMP] Disabling AMP permanently due to non-finite grads; continuing in FP32.")
                        amp_enabled = False
                        scaler = torch.amp.GradScaler("cuda", enabled=False)
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                opt.step()

            ema.update(model)

            for k in ["total","bin_ce","geo","sixd","euler","geo_deg"]:
                run[k] += losses[k].item()

            with torch.no_grad():
                R_pred_f = R_pred.float()
                pp, yp, rp = rotmat_to_euler(R_pred_f)
                pd = torch.stack([rad2deg(pp), rad2deg(yp), rad2deg(rp)], -1)
                df = ((pd - euler_gt + 180) % 360 - 180).abs().mean(0)
                run["pitch_err"] += df[0].item()
                # run["yaw_err"]   += df[1].item()

                yaw = ((pd[:, 1] - euler_gt[:, 1] + 180) % 360 - 180).abs()
                yaw_sym = torch.minimum(yaw, (yaw - 180.0).abs())
                run["yaw_err"] += yaw_sym.mean().item()

                run["roll_err"]  += df[2].item()

                run["pitch_acc"] += (pl.argmax(1) == bins_gt[:,0]).float().mean().item()
                run["yaw_acc"]   += (yl.argmax(1) == bins_gt[:,1]).float().mean().item()
                run["roll_acc"]  += (rl.argmax(1) == bins_gt[:,2]).float().mean().item()

            nb += 1
            global_step += 1

        for k in run:
            run[k] /= max(nb, 1)
        for k in mk:
            history[f"train_{k}"].append(run[k])

        bk = ema.apply(model)
        val = evaluate(model, val_dl)
        ema.restore(model, bk)
        for k in mk:
            history[f"val_{k}"].append(val[k])

        lr_c = opt.param_groups[0]["lr"]
        lr_s = opt.param_groups[1]["lr"]
        lr_h = opt.param_groups[2]["lr"]

        print(f"\n{'='*115}")
        print(f"Epoch {epoch+1:03d}/{cfg.EPOCHS}  [Train]  "
              f"LR cnn={lr_c:.2e} seg={lr_s:.2e} head={lr_h:.2e}")
        print(f"  TRAIN | loss={run['total']:.4f}  bin={run['bin_ce']:.4f}  "
              f"geo={run['geo']:.4f}  6d={run['sixd']:.4f}")
        print(f"        | geo_err={run['geo_deg']:.1f}°  "
              f"P={run['pitch_err']:.1f}° Y={run['yaw_err']:.1f}° R={run['roll_err']:.1f}°")
        print(f"        | bin_acc P={run['pitch_acc']:.3f}  Y={run['yaw_acc']:.3f}  R={run['roll_acc']:.3f}")
        print(f"  VAL   | loss={val['total']:.4f}  bin={val['bin_ce']:.4f}  "
              f"geo={val['geo']:.4f}  6d={val['sixd']:.4f}")
        print(f"        | geo_err={val['geo_deg']:.1f}°  "
              f"P={val['pitch_err']:.1f}° Y={val['yaw_err']:.1f}° R={val['roll_err']:.1f}°")
        print(f"        | bin_acc P={val['pitch_acc']:.3f}  Y={val['yaw_acc']:.3f}  R={val['roll_acc']:.3f}")

        if val["geo_deg"] < best_geo:
            best_geo = val["geo_deg"]
            bk2 = ema.apply(model)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_geo.pt"))
            ema.restore(model, bk2)
            print(f"   BEST GEO: {best_geo:.1f}°")

        if val["yaw_err"] < best_yaw:
            best_yaw = val["yaw_err"]
            bk2 = ema.apply(model)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_yaw.pt"))
            ema.restore(model, bk2)
            print(f"   BEST YAW: {best_yaw:.1f}°")

        with open(os.path.join(out, "best.txt"), "w") as f:
            f.write(f"geo={best_geo:.4f}\nyaw={best_yaw:.4f}\n")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        plot_metrics(history, os.path.join(plot_dir, "metrics.png"))

        if (epoch + 1) % cfg.VIZ_EVERY_EPOCH == 0:
            bk2 = ema.apply(model)
            visualize(model, val_ds, os.path.join(viz_dir, f"ep_{epoch+1:03d}"), cfg.VIZ_SAMPLES)
            ema.restore(model, bk2)

    if cfg.FINAL_VIZ_ALL:
        print("\nFinal viz...")
        bp = os.path.join(ckpt_dir, "best_geo.pt")
        if os.path.isfile(bp):
            model.load_state_dict(torch.load(bp, map_location=DEVICE))
        visualize(model, val_ds, os.path.join(viz_dir, "final"), n=None)

    print(f"\nDone. Best geo={best_geo:.1f}° yaw={best_yaw:.1f}°    {out}")


def main():
    seed_all(cfg.SEED)
    print(f"PoseNet — Dual Stream | {DEVICE}")

    train_ds = BoatPoseDataset(cfg.ANNOTATION_DIR, cfg.IMAGE_DIR, cfg.IMG_SIZE, augment=True)
    val_ds   = BoatPoseDataset(cfg.VAL_ANNOTATION_DIR, cfg.VAL_IMAGE_DIR, cfg.IMG_SIZE, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS if os.name != "nt" else 0,
        pin_memory=cfg.PIN_MEMORY and DEVICE == "cuda",
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS if os.name != "nt" else 0,
        pin_memory=cfg.PIN_MEMORY and DEVICE == "cuda",
        drop_last=False
    )

    model = PoseNet(cfg.SEGFORMER_ENCODER_CKPT, cfg.SEGFORMER_UNFREEZE_LAST_N).to(DEVICE)

    # Probe to init lazy SegFormer fusion
    with torch.no_grad():
        _ = model(torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=DEVICE))
    print("Model initialized.")

    cnn_p = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith("cnn."))
    seg_p = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith("seg."))
    head_p = sum(p.numel() for n,p in model.named_parameters()
                 if p.requires_grad and not n.startswith("cnn.") and not n.startswith("seg."))
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable: CNN={cnn_p/1e6:.1f}M  SegFormer={seg_p/1e6:.1f}M  Heads={head_p/1e6:.1f}M  |  Total={total_p/1e6:.1f}M")

    train(model, train_dl, val_dl, train_ds, val_ds)


if __name__ == "__main__":
    main()