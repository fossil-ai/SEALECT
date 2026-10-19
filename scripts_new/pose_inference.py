#!/usr/bin/env python3
import os
import json
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


# 
# EDIT THESE
# 
TRAIN_MODULE_NAME = "pose3"   # the script/module you used to train
CHECKPOINT_PATH = "/path/to/sealect_root/scripts_new/pose_model/best_yaw.pt"
# 


@dataclass
class PosePrediction:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float

    pitch_bin: int
    yaw_bin: int
    roll_bin: int

    pitch_conf: float
    yaw_conf: float
    roll_conf: float
    pose_confidence: float

    rotation_matrix: np.ndarray
    embedding: np.ndarray


class PoseNetEstimator:
    """
    Thin wrapper around your trained PoseNet model.

    Usage:
        pose_net = PoseNetEstimator()
        pred = pose_net.predict_image("/path/to/img.jpg")
        print(pred.yaw_bin, pred.pose_confidence)
    """

    def __init__(
        self,
        train_module_name: str = TRAIN_MODULE_NAME,
        checkpoint_path: str = CHECKPOINT_PATH,
        device: Optional[str] = None,
    ):
        self.train_mod = importlib.import_module(train_module_name)
        self.cfg = self.train_mod.cfg

        if device is None:
            self.device = self.train_mod.DEVICE
        else:
            self.device = device

        self.model = self.train_mod.PoseNet(
            self.cfg.SEGFORMER_ENCODER_CKPT,
            self.cfg.SEGFORMER_UNFREEZE_LAST_N
        ).to(self.device)

        # IMPORTANT: lazy-build SegFormer fusion layers before loading checkpoint
        with torch.no_grad():
            _ = self.model(
                torch.randn(1, 3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE, device=self.device)
            )

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

        self._hooked = {}
        self._hook = self.model.trunk.register_forward_hook(self._save_trunk)

    def _save_trunk(self, module, inputs, output):
        self._hooked["trunk"] = output.detach()

    def close(self):
        if hasattr(self, "_hook") and self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        return img

    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        img = self._load_image(image)
        img = img.resize((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE), Image.BILINEAR)
        img_t = transforms.ToTensor()(img)
        img_t = self.normalize(img_t)
        return img_t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_image(self, image: Union[str, Image.Image]) -> PosePrediction:
        x = self._preprocess(image)

        R_pred, p_logits, y_logits, r_logits = self.model(x)

        p_probs = torch.softmax(p_logits, dim=1)[0]
        y_probs = torch.softmax(y_logits, dim=1)[0]
        r_probs = torch.softmax(r_logits, dim=1)[0]

        pitch_bin = int(torch.argmax(p_probs).item())
        yaw_bin = int(torch.argmax(y_probs).item())
        roll_bin = int(torch.argmax(r_probs).item())

        pitch_conf = float(torch.max(p_probs).item())
        yaw_conf = float(torch.max(y_probs).item())
        roll_conf = float(torch.max(r_probs).item())

        # conservative aggregate confidence
        pose_confidence = min(pitch_conf, yaw_conf, roll_conf)

        R_pred_f = R_pred.float().cpu()
        pitch_r, yaw_r, roll_r = self.train_mod.rotmat_to_euler(R_pred_f)

        pitch_deg = float(self.train_mod.rad2deg(pitch_r)[0].item())
        yaw_deg = float(self.train_mod.rad2deg(yaw_r)[0].item())
        roll_deg = float(self.train_mod.rad2deg(roll_r)[0].item())

        embedding = self._hooked["trunk"][0].float().cpu().numpy()
        rotation_matrix = R_pred_f[0].numpy()

        return PosePrediction(
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,

            pitch_bin=pitch_bin,
            yaw_bin=yaw_bin,
            roll_bin=roll_bin,

            pitch_conf=pitch_conf,
            yaw_conf=yaw_conf,
            roll_conf=roll_conf,
            pose_confidence=pose_confidence,

            rotation_matrix=rotation_matrix,
            embedding=embedding,
        )

    @torch.no_grad()
    def predict_to_dict(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        pred = self.predict_image(image)
        return {
            "pitch_deg": pred.pitch_deg,
            "yaw_deg": pred.yaw_deg,
            "roll_deg": pred.roll_deg,

            "pitch_bin": pred.pitch_bin,
            "yaw_bin": pred.yaw_bin,
            "roll_bin": pred.roll_bin,

            "pitch_conf": pred.pitch_conf,
            "yaw_conf": pred.yaw_conf,
            "roll_conf": pred.roll_conf,
            "pose_confidence": pred.pose_confidence,

            "rotation_matrix": pred.rotation_matrix.tolist(),
            "embedding": pred.embedding.tolist(),
        }


if __name__ == "__main__":
    pose_net = PoseNetEstimator()
    out = pose_net.predict_to_dict("/path/to/sealect_root/KOLOMVERSE/images/0000005819_3234_1341_3418_1534.jpg")
    print(json.dumps(out, indent=2)[:1200])