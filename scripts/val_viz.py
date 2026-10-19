import os
import json
import glob
import math

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import SegformerForSemanticSegmentation

# 
# CONFIG
# 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CKPT = "/path/to/sealect_root/scripts/checkpoints/pose_net_weights.pt"
ANN_DIR = "/path/to/sealect_root/unified/val_annotations"
IMG_DIR = "/path/to/sealect_root/unified/val_images"
OUT_DIR = "validation_outputs"
BATCH_SIZE = 1

os.makedirs(OUT_DIR, exist_ok=True)

# 
# UTILITY FUNCTIONS
# 

def unity_euler_to_matrix(pitch, yaw, roll):
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)

    Rx = torch.tensor([[1,0,0],[0,cp,-sp],[0,sp,cp]], device=pitch.device)
    Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=pitch.device)
    Rz = torch.tensor([[cr,-sr,0],[sr,cr,0],[0,0,1]], device=pitch.device)

    return Ry @ Rx @ Rz

def forward_to_yaw_pitch(forward):
    fx, fy, fz = forward
    yaw = math.degrees(math.atan2(fx, -fz))
    pitch = math.degrees(math.asin(np.clip(fy, -1.0, 1.0)))
    return yaw, pitch

def project_forward(forward):
    fx, fy, _ = forward
    d2 = np.array([fx, -fy])
    return d2, np.linalg.norm(d2)

def draw_arrow(img, center, d, mag, color):
    if mag < 1e-3:
        return
    d = d / (np.linalg.norm(d) + 1e-6)
    end = (int(center[0] + d[0] * 90), int(center[1] + d[1] * 90))
    cv2.arrowedLine(
        img,
        center,
        end,
        color,
        thickness=1,        #  thinner arrows
        tipLength=0.25
    )

def draw_pose_text(img, gt_fwd, pred_fwd):
    gt_yaw, gt_pitch = forward_to_yaw_pitch(gt_fwd)
    pr_yaw, pr_pitch = forward_to_yaw_pitch(pred_fwd)

    lines = [
        f"GT   yaw: {gt_yaw:6.1f}  pitch: {gt_pitch:6.1f}  roll: N/A",
        f"PRED yaw: {pr_yaw:6.1f}  pitch: {pr_pitch:6.1f}  roll: N/A",
    ]

    y = 30
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (10, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            text,
            (10, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

# 
# DATASET
# 

class BoatPoseDataset(Dataset):
    def __init__(self, ann_dir, img_dir):
        self.files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx]) as f:
            data = json.load(f)

        pose = data["annotations"][0]["pose"]
        pitch = torch.tensor(math.radians(pose["pitch"]), device=DEVICE)
        yaw   = torch.tensor(math.radians(pose["yaw"]), device=DEVICE)
        roll  = torch.tensor(math.radians(pose["roll"]), device=DEVICE)

        R = unity_euler_to_matrix(pitch, yaw, roll)
        forward = F.normalize(R @ torch.tensor([0.,0.,-1.], device=DEVICE), dim=0)

        img = Image.open(os.path.join(self.img_dir, data["imageFileName"])).convert("RGB")
        img_t = self.transform(img)

        return img_t, forward, data["imageFileName"]

# 
# MODEL
# 

class GeometricStem(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128,5,stride=2,padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,dim,3,stride=2,padding=1),
            nn.BatchNorm2d(dim), nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x.flatten(2).transpose(1,2)

class PoseNet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=1,
            ignore_mismatched_sizes=True
        )

        self.encoder = segformer.segformer
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.geo = GeometricStem(dim)
        self.pose_token = nn.Parameter(torch.randn(1,1,dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,3)
        )

    def forward(self, x):
        enc = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        feat = enc.hidden_states[-1].flatten(2).transpose(1,2)
        geo  = self.geo(x)
        tokens = torch.cat([feat, geo], dim=1)

        B = x.size(0)
        pose_token = self.pose_token.expand(B,-1,-1)
        pose_feat,_ = self.cross_attn(pose_token, tokens, tokens)
        out = self.head(pose_feat.squeeze(1))
        return F.normalize(out, dim=1)

# 
# VALIDATION
# 

def validate(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    for imgs, fwd_gt, names in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)

        for i in range(imgs.size(0)):
            # pred = preds[i].cpu().numpy()
            pred = preds[i].detach().cpu().numpy()

            gt = fwd_gt[i].cpu().numpy()

            img_path = os.path.join(dataset.img_dir, names[i])
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            draw_arrow(img, center, *project_forward(gt), (0,255,255))
            draw_arrow(img, center, *project_forward(pred), (0,0,255))
            draw_pose_text(img, gt, pred)

            out_path = os.path.join(OUT_DIR, names[i])
            cv2.imwrite(out_path, img)
            print(f"Saved: {out_path}")

# 
# MAIN
# 

def main():
    dataset = BoatPoseDataset(ANN_DIR, IMG_DIR)

    model = PoseNet().to(DEVICE)
    state = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    validate(model, dataset)

if __name__ == "__main__":
    main()
