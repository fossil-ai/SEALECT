import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
from tqdm import tqdm  # <-- added for progress bar

# 
# CONFIG
# 

IMAGE_ROOT = "/path/to/project/ProjectDataset/images"
MASK_ROOT  = "/path/to/sealect_root/segment1/boat_masks"

OUT_DIR = "/path/to/sealect_root/segment1/segformer_boat"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

VALID_EXTS = (".jpg", ".jpeg", ".png")

# 
# DATASET
# 

class BoatSegDataset(Dataset):
    def __init__(self, image_root, mask_root):
        self.samples = []

        # Iterate over masks and match to images
        for mask_path in glob.glob(os.path.join(mask_root, "**/*.npy"), recursive=True):
            rel = os.path.relpath(mask_path, mask_root)
            img_path = None
            # Check common extensions
            for ext in VALID_EXTS:
                candidate = os.path.join(image_root, os.path.splitext(rel)[0] + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # Load mask and skip empty ones
            mask = np.load(mask_path)
            if mask.sum() == 0:
                continue

            self.samples.append((img_path, mask_path))

        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST)
        ])

        print(f"[Dataset] Loaded {len(self.samples)} valid segmentation samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path).astype(np.uint8)

        # Convert mask to PIL and resize
        mask = Image.fromarray(mask)
        mask = self.mask_tf(mask)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()  # single channel

        image = self.img_tf(image)

        return image, mask

# 
# LOSSES
# 

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return 1 - (num / den).mean()

# 
# MODEL
# 

def build_segformer():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    return model

# 
# TRAIN LOOP (with progress bar)
# 

def train():
    dataset = BoatSegDataset(IMAGE_ROOT, MASK_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model = build_segformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            out = model(pixel_values=imgs).logits
            out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = bce(out, masks) + dice_loss(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1:03d}] Avg Loss: {avg_loss:.4f}")

        # Optional: save intermediate checkpoints
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(OUT_DIR, f"segformer_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")

    # 
    # FINAL SAVE
    # 

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "segformer_full.pt"))
    torch.save(model.segformer.state_dict(), os.path.join(OUT_DIR, "segformer_encoder.pt"))

    print(" Training complete")
    print(" Saved:")
    print("  - Full segmentation model: segformer_full.pt")
    print("  - Encoder-only backbone: segformer_encoder.pt (ready for pose fine-tuning)")

# 
# ENTRY
# 

if __name__ == "__main__":
    train()
