#!/usr/bin/env python3
"""
build_merged_scene_pose_library_grounded_sam2.py

Unified routine for a merged COCO-style maritime dataset, using Grounding DINO +
SAM2 for image-level scene context instead of ADE20K SegFormer.

Expected structure:
    merged/
        images/
        annotations_fixed.json

Key assumptions:
- category_id/name for vessels is WATERCRAFT_CATEGORY_NAME
- each annotation may have a "source" field containing the dataset name
- all outputs are written into OUT_DIR

Outputs:
    OUT_DIR/
        image_contexts.json
        candidate_library.json
        scene_object_pairs.json
        summary.json
        crops/                         optional
        segmentation_debug/            optional side-by-side context panels

New scene-context fields:
- water_frac, sky_frac, land_frac, other_frac, vessel_ignore_frac
- horizon_y_norm
- vertical_profile
- scene_layout: open_water | coastal_transition | harbor_port | water_land_mixed | non_maritime_or_uncertain
- viewpoint_type: nadir_aerial | oblique_aerial | horizon_view | low_oblique_or_shore | unknown_viewpoint
- context_key: viewpoint__layout__water_bucket__sky_bucket__land_bucket__hz_bucket

Install:
    pip install torch torchvision pillow numpy tqdm transformers accelerate
    pip install git+https://github.com/facebookresearch/sam2.git

Notes:
- Grounding DINO proposes text-grounded boxes for water/sky/land/vessel phrases.
- SAM2 converts boxes into masks.
- Unlabeled pixels are filled with a conservative maritime HSV/position heuristic.
- Vessel masks and COCO vessel boxes are excluded from scene fractions by default.
"""

import os
import re
import csv
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from pose_inference import PoseNetEstimator

# 
# EDIT THESE
# 
MERGED_ROOT = "/path/to/datasets/data/merged"
COCO_JSON = os.path.join(MERGED_ROOT, "annotations_fixed.json")
IMAGE_DIR = os.path.join(MERGED_ROOT, "images")
OUT_DIR = "merged_scene_pose_library_out_grounded_sam2"

WATERCRAFT_CATEGORY_NAME = "watercraft"

MIN_BOX_SIZE = 24
BOX_EXPAND_RATIO = 0.10
SAVE_CROPS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force fp32 for Grounding DINO to avoid fp16/fp32 mismatch across transformers versions.
GROUNDING_DTYPE = torch.float32

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"

# Resize longest side before Grounding/SAM2. 1024 is a good balance.
SCENE_WORKING_SIZE = 768

# Grounding thresholds. Lower if water/sky are missed; raise if false positives dominate.
BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.18

# Limit processing to a manageable subset.
MAX_IMAGES_TO_PROCESS = None
RANDOM_SUBSET = True
RANDOM_SEED = 42

SAVE_SEGMENTATION_DEBUG = True
MAX_DEBUG_VIS = 100

# If True, COCO vessel boxes are marked as vessel/ignore before fractions.
MASK_OUT_VESSEL_BOXES_FOR_CONTEXT = True
IGNORE_VESSELS_IN_SCENE_FRACTIONS = True
FILL_UNLABELED_WITH_HEURISTIC = True

Y_BINS = 8
BANDS = 16

# Text prompts. Keep periods in final prompt; Grounding DINO expects dot-separated phrases.
PROMPTS = {
    "water": [
        "water", "sea", "ocean", "river", "lake", "water surface", "open water"
    ],
    "sky": [
        "sky", "cloudy sky", "horizon sky"
    ],
    "land": [
        "land", "shore", "shoreline", "coast", "coastline", "beach", "sand",
        "dock", "pier", "harbor", "port", "building", "crane", "trees", "forest",
        "road", "bridge", "ground"
    ],
    "vessel": [
        "boat", "ship", "vessel", "watercraft", "sailboat"
    ],
}

COLORS = {
    "water": (0, 80, 255),
    "sky": (0, 220, 255),
    "land": (0, 220, 0),
    "other": (70, 70, 70),
    "vessel": (255, 90, 0),
}

LABEL_PRIORITY = ["other", "sky", "water", "land", "vessel"]
# 


# 
# Utility
# 

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def infer_sequence_group(dataset_name: str, image_name: str) -> str:
    stem = os.path.splitext(os.path.basename(image_name))[0]
    stem = re.sub(r"(_frame)?[_-]?\d{4,}$", "", stem)
    stem = re.sub(r"[_-]\d{6,}$", "", stem)
    return f"{dataset_name}::{stem}"


def expand_box_xyxy(box, width, height, ratio=0.10):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * ratio
    dy = bh * ratio
    x1 = max(0, int(round(x1 - dx)))
    y1 = max(0, int(round(y1 - dy)))
    x2 = min(width, int(round(x2 + dx)))
    y2 = min(height, int(round(y2 + dy)))
    return [x1, y1, x2, y2]


def crop_pil(img: Image.Image, box_xyxy):
    return img.crop(tuple(box_xyxy))


def center_and_area_frac(box_xyxy, width, height):
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    area_frac = (bw * bh) / float(width * height)
    aspect_ratio = bw / max(1.0, bh)
    return float(cx), float(cy), float(area_frac), float(aspect_ratio)


def y_bin_from_center(center_y_norm: float, num_bins: int = Y_BINS) -> int:
    center_y_norm = max(0.0, min(1.0, float(center_y_norm)))
    idx = int(math.floor(center_y_norm * num_bins))
    return min(num_bins - 1, max(0, idx))


def resize_for_model(img: Image.Image, working_size: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = working_size / max(w, h)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR), scale


def load_font(size: int = 22):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# 
# Scene context definitions
# 

@dataclass
class BackgroundStats:
    dataset: str
    image_id: int
    image_name: str
    image_path: str
    width: int
    height: int

    water_frac: float
    sky_frac: float
    land_frac: float
    other_frac: float
    vessel_ignore_frac: float

    horizon_y_norm: float
    horizon_confidence: float

    water_row_mean: float
    water_row_std: float
    sky_row_mean: float
    sky_row_std: float
    land_row_mean: float
    land_row_std: float

    vertical_profile: Dict[str, List[float]]

    scene_layout: str
    viewpoint_type: str
    context_key: str

    # Backward-compatible field for old prior scripts.
    # Prefer scene_layout/viewpoint_type going forward.
    scene_type: str

    grounding_counts: Dict[str, int]
    grounding_total_detections: int


def vertical_profile(mask_probs: Dict[str, np.ndarray], bands: int = BANDS) -> Dict[str, List[float]]:
    h, _ = mask_probs["water"].shape
    edges = np.linspace(0, h, bands + 1, dtype=int)
    out = {k: [] for k in ["water", "sky", "land", "other", "vessel"]}
    for i in range(bands):
        y0, y1 = edges[i], edges[i + 1]
        for k in out.keys():
            band_mean = float(mask_probs[k][y0:y1].mean()) if y1 > y0 else 0.0
            out[k].append(band_mean)
    return out


def weighted_row_stats(prob_map: np.ndarray) -> Tuple[float, float]:
    h, _ = prob_map.shape
    row_mass = prob_map.mean(axis=1)
    total = row_mass.sum()
    if total <= 1e-12:
        return 0.0, 0.0
    row_mass = row_mass / total
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    mu = float((y * row_mass).sum())
    var = float(((y - mu) ** 2 * row_mass).sum())
    return mu, math.sqrt(max(var, 0.0))


def estimate_horizon_from_maps(maps: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Estimate the vertical sky/non-sky transition.
    Returns (horizon_y_norm, confidence). Confidence is low for no-sky aerial scenes.
    """
    sky_row = maps["sky"].mean(axis=1)
    water_row = maps["water"].mean(axis=1)
    land_row = maps["land"].mean(axis=1)
    nonsky_row = water_row + land_row

    h = len(sky_row)
    if h <= 1:
        return 0.5, 0.0

    # If almost no sky exists, there is no reliable horizon.
    if float(maps["sky"].mean()) < 0.03:
        return 0.0, 0.0

    kernel = np.ones(9, dtype=np.float32) / 9.0
    sky_s = np.convolve(sky_row, kernel, mode="same")
    nonsky_s = np.convolve(nonsky_row, kernel, mode="same")

    # Horizon where sky and non-sky are balanced, favoring strong gradients.
    balance = np.abs(sky_s - nonsky_s)
    grad = np.abs(np.gradient(sky_s - nonsky_s))
    score = balance - 0.25 * grad
    idx = int(np.argmin(score))

    # Confidence based on top-sky and bottom-nonsky separation.
    top_sky = float(sky_row[: max(1, h // 3)].mean())
    bottom_nonsky = float(nonsky_row[(2 * h) // 3 :].mean())
    conf = max(0.0, min(1.0, 0.5 * (top_sky + bottom_nonsky)))
    return float(idx / max(1, h - 1)), conf


def classify_scene_layout(water: float, sky: float, land: float, other: float) -> str:
    """
    Layout = composition/geography, not camera viewpoint.
    Keep this secondary; viewpoint drives vessel top-vs-side appearance more directly.
    """
    if water >= 0.80 and land < 0.08:
        return "open_water"
    if land >= 0.35 and water >= 0.12:
        return "harbor_port"
    if water >= 0.30 and land >= 0.08:
        return "coastal_transition"
    if water >= 0.15 and land >= 0.03:
        return "water_land_mixed"
    if water >= 0.50:
        return "open_water_uncertain"
    return "non_maritime_or_uncertain"


def classify_viewpoint_type(
    water: float,
    sky: float,
    land: float,
    horizon_y: float,
    horizon_conf: float,
    vp: Dict[str, List[float]],
) -> str:
    """
    Viewpoint = primary context for vessel appearance.
    This is inferred from sky presence, horizon reliability, and vertical layout.

    Definitions:
    - nadir_aerial: little/no sky, mostly water/land viewed from above. Top-exposed vessels likely.
    - oblique_aerial: little/no sky but significant land/water geography visible. Still top/roof-biased.
    - horizon_view: clear sky over water/land with a reliable horizon. Side-view vessels likely.
    - low_oblique_or_shore: sky exists but horizon is weak/occluded, often shore/harbor/onboard.
    - unknown_viewpoint: ambiguous or non-maritime.
    """
    top_sky = float(np.mean(vp["sky"][: max(1, len(vp["sky"]) // 3)]))
    bottom_water = float(np.mean(vp["water"][(2 * len(vp["water"])) // 3 :]))
    mid_land = float(np.mean(vp["land"][len(vp["land"]) // 3 : (2 * len(vp["land"])) // 3]))

    # No/near-no sky: likely aerial or high oblique. Land does not invalidate aerial.
    if sky < 0.04 and water >= 0.45:
        if land < 0.12:
            return "nadir_aerial"
        return "oblique_aerial"

    # Clear sky + lower water/land: normal horizon-style maritime camera.
    if sky >= 0.12 and horizon_conf >= 0.20 and 0.08 <= horizon_y <= 0.75:
        if bottom_water >= 0.20 or water >= 0.20:
            return "horizon_view"

    # Sky exists but geometry is not clean; common for harbor/onshore/occluded scenes.
    if sky >= 0.06 and (water >= 0.12 or land >= 0.12):
        return "low_oblique_or_shore"

    # No sky but not enough water for a confident aerial maritime label.
    if sky < 0.06 and land >= 0.20 and water >= 0.15:
        return "oblique_aerial"

    return "unknown_viewpoint"


def bucket(x: float, edges=(0.10, 0.30, 0.60, 0.85)) -> str:
    if x < edges[0]:
        return "very_low"
    if x < edges[1]:
        return "low"
    if x < edges[2]:
        return "medium"
    if x < edges[3]:
        return "high"
    return "very_high"


def horizon_bucket(h: float, conf: float) -> str:
    if conf < 0.10:
        return "none"
    if h < 0.25:
        return "top"
    if h < 0.55:
        return "mid"
    if h < 0.80:
        return "low"
    return "bottom"


def make_context_key(viewpoint_type: str, scene_layout: str, water: float, sky: float, land: float, horizon_y: float, horizon_conf: float) -> str:
    return (
        f"vp_{viewpoint_type}__layout_{scene_layout}__"
        f"water_{bucket(water)}__sky_{bucket(sky)}__land_{bucket(land)}__"
        f"hz_{horizon_bucket(horizon_y, horizon_conf)}"
    )


def legacy_scene_type(viewpoint_type: str, scene_layout: str) -> str:
    """Backward-compatible scene_type. Prefer new fields in new prior scripts."""
    if viewpoint_type == "nadir_aerial" and scene_layout.startswith("open_water"):
        return "aerial_open_water"
    if scene_layout == "open_water":
        return "open_water"
    if scene_layout in ("coastal_transition", "water_land_mixed"):
        return "shoreline"
    if scene_layout == "harbor_port":
        return "harbor_or_coastal"
    return "mixed_maritime"


# 
# Grounding DINO + SAM2 scene segmenter
# 

class GroundedSAM2SceneSegmenter:
    def __init__(self):
        print(f"Loading Grounding DINO: {GROUNDING_MODEL_ID}")
        self.processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_MODEL_ID,
            torch_dtype=GROUNDING_DTYPE,
        ).to(DEVICE)
        self.grounding_model.eval()

        print(f"Loading SAM2: {SAM2_MODEL_ID}")
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise RuntimeError(
                "Could not import SAM2ImagePredictor. Install SAM2 first:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                f"Original import error: {repr(e)}"
            )
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=DEVICE)
        self.prompt = self._make_prompt()

    def _make_prompt(self) -> str:
        phrases = []
        for words in PROMPTS.values():
            phrases.extend(words)
        return ". ".join(phrases) + "."

    def phrase_to_label(self, text: str) -> str:
        t = text.lower().strip().replace(".", "")
        best_label = "other"
        best_len = -1
        for label, phrases in PROMPTS.items():
            for p in phrases:
                p_low = p.lower()
                if p_low in t and len(p_low) > best_len:
                    best_label = label
                    best_len = len(p_low)
        return best_label

    @torch.no_grad()
    def detect_boxes(self, img: Image.Image) -> Tuple[np.ndarray, List[str], np.ndarray]:
        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(DEVICE)

        # Version-safe dtype alignment. Some transformers builds load fp16 under CUDA.
        model_dtype = next(self.grounding_model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        outputs = self.grounding_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]], device=DEVICE)

        if hasattr(self.processor, "post_process_grounded_object_detection"):
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                # Older/newer HF signature uses threshold instead of box_threshold.
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=target_sizes,
                )[0]
        else:
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=BOX_THRESHOLD,
                target_sizes=target_sizes,
            )[0]

        boxes = results.get("boxes", torch.empty((0, 4), device=DEVICE)).detach().cpu().numpy()
        scores = results.get("scores", torch.empty((0,), device=DEVICE)).detach().cpu().numpy()
        raw_labels = results.get("labels", [])

        labels = []
        for lab in raw_labels:
            labels.append(self.phrase_to_label(str(lab)) if not isinstance(lab, int) else "other")

        return boxes.astype(np.float32), labels, scores.astype(np.float32)

    @torch.no_grad()
    def sam2_masks_from_boxes(self, img: Image.Image, boxes_xyxy: np.ndarray) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            return np.zeros((0, img.height, img.width), dtype=bool)

        rgb = np.array(img.convert("RGB"))
        self.sam2_predictor.set_image(rgb)
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks[None]
        return masks.astype(bool)

    def rgb_to_hsv_np(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = rgb.astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = np.max(arr, axis=-1)
        mn = np.min(arr, axis=-1)
        diff = mx - mn
        h = np.zeros_like(mx)
        mask = diff > 1e-6
        rmax = mask & (mx == r)
        gmax = mask & (mx == g)
        bmax = mask & (mx == b)
        h[rmax] = ((g[rmax] - b[rmax]) / diff[rmax]) % 6.0
        h[gmax] = ((b[gmax] - r[gmax]) / diff[gmax]) + 2.0
        h[bmax] = ((r[bmax] - g[bmax]) / diff[bmax]) + 4.0
        h = h / 6.0
        s = np.where(mx <= 1e-6, 0.0, diff / np.maximum(mx, 1e-6))
        v = mx
        return h, s, v

    def heuristic_label_map(self, rgb: np.ndarray) -> np.ndarray:
        """0 water, 1 sky, 2 land, 3 other. Conservative fallback for unlabeled pixels."""
        h, s, v = self.rgb_to_hsv_np(rgb)
        H, _ = h.shape
        y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

        blue_cyan = (h >= 0.42) & (h <= 0.72)
        green_cyan = (h >= 0.28) & (h < 0.42)

        sky = blue_cyan & (v > 0.38) & (s < 0.55) & (y < 0.58)
        water = ((blue_cyan | green_cyan) & (v > 0.08)) & ~sky
        water |= blue_cyan & (v > 0.06) & (s > 0.08)

        brown_yellow = (h >= 0.04) & (h <= 0.22) & (s > 0.10) & (v > 0.12)
        vegetation = (h >= 0.20) & (h <= 0.42) & (s > 0.18) & (v > 0.12)
        gray_bright = (s < 0.20) & (v > 0.28)
        land = (brown_yellow | vegetation | gray_bright) & ~(water | sky)

        label = np.full((H, rgb.shape[1]), 3, dtype=np.uint8)
        label[water] = 0
        label[sky] = 1
        label[land] = 2
        return label

    def merge_masks(
        self,
        img: Image.Image,
        labels: List[str],
        scores: np.ndarray,
        masks: np.ndarray,
        suppress_boxes: Optional[List[List[float]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int], int]:
        H, W = img.height, img.width
        label_map = np.full((H, W), 255, dtype=np.uint8)
        label_to_id = {"water": 0, "sky": 1, "land": 2, "other": 3, "vessel": 4}

        detections = []
        for i, lab in enumerate(labels):
            if i >= len(masks):
                continue
            detections.append((lab, float(scores[i]) if i < len(scores) else 0.0, masks[i]))

        priority_rank = {lab: idx for idx, lab in enumerate(LABEL_PRIORITY)}
        detections.sort(key=lambda x: (priority_rank.get(x[0], 0), x[1]))

        counts = {"water": 0, "sky": 0, "land": 0, "other": 0, "vessel": 0}
        for lab, score, mask in detections:
            lab = lab if lab in label_to_id else "other"
            if mask.shape != (H, W) or mask.sum() == 0:
                continue
            label_map[mask] = label_to_id[lab]
            counts[lab] += 1

        if FILL_UNLABELED_WITH_HEURISTIC:
            fallback = self.heuristic_label_map(np.array(img.convert("RGB")))
            unlabeled = label_map == 255
            label_map[unlabeled] = fallback[unlabeled]
        else:
            label_map[label_map == 255] = label_to_id["other"]

        # Force annotated vessel boxes into ignore, so foreground does not corrupt context.
        if suppress_boxes and MASK_OUT_VESSEL_BOXES_FOR_CONTEXT:
            for box in suppress_boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 > x1 and y2 > y1:
                    label_map[y1:y2, x1:x2] = label_to_id["vessel"]

        maps = {
            "water": (label_map == 0).astype(np.float32),
            "sky": (label_map == 1).astype(np.float32),
            "land": (label_map == 2).astype(np.float32),
            "other": (label_map == 3).astype(np.float32),
            "vessel": (label_map == 4).astype(np.float32),
        }
        return maps, counts, len(detections)

    def compute_fractions(self, maps: Dict[str, np.ndarray]) -> Dict[str, float]:
        if IGNORE_VESSELS_IN_SCENE_FRACTIONS:
            valid = 1.0 - maps["vessel"]
            denom = float(valid.sum())
            if denom < 1.0:
                valid = np.ones_like(valid)
                denom = float(valid.size)
        else:
            valid = np.ones_like(maps["water"])
            denom = float(valid.size)

        out = {}
        for k in ["water", "sky", "land", "other"]:
            out[k] = float((maps[k] * valid).sum() / max(1.0, denom))
        out["vessel_ignore"] = float(maps["vessel"].mean())
        return out

    def predict_context(self, image: Image.Image, suppress_boxes: Optional[List[List[float]]] = None):
        img_small, scale = resize_for_model(image, SCENE_WORKING_SIZE)

        # Scale suppress boxes into small-image coordinates.
        small_boxes = None
        if suppress_boxes:
            small_boxes = [[v * scale for v in box] for box in suppress_boxes]

        boxes, labels, scores = self.detect_boxes(img_small)
        masks = self.sam2_masks_from_boxes(img_small, boxes)
        maps, counts, total = self.merge_masks(img_small, labels, scores, masks, small_boxes)
        fracs = self.compute_fractions(maps)
        return maps, fracs, counts, total, img_small


# 
# Background extraction
# 

def extract_background_stats_for_image(
    dataset_name: str,
    image_id: int,
    image_name: str,
    image_path: str,
    image: Image.Image,
    segmenter: GroundedSAM2SceneSegmenter,
    suppress_boxes: Optional[List[List[float]]] = None,
) -> Tuple[BackgroundStats, Dict[str, np.ndarray], Image.Image]:
    maps, fracs, counts, total_detections, context_img = segmenter.predict_context(image, suppress_boxes)

    water_frac = fracs["water"]
    sky_frac = fracs["sky"]
    land_frac = fracs["land"]
    other_frac = fracs["other"]
    vessel_ignore_frac = fracs["vessel_ignore"]

    horizon_y_norm, horizon_confidence = estimate_horizon_from_maps(maps)
    water_row_mean, water_row_std = weighted_row_stats(maps["water"])
    sky_row_mean, sky_row_std = weighted_row_stats(maps["sky"])
    land_row_mean, land_row_std = weighted_row_stats(maps["land"])
    vp = vertical_profile(maps, bands=BANDS)

    scene_layout = classify_scene_layout(water_frac, sky_frac, land_frac, other_frac)
    viewpoint_type = classify_viewpoint_type(
        water_frac, sky_frac, land_frac, horizon_y_norm, horizon_confidence, vp
    )
    context_key = make_context_key(
        viewpoint_type, scene_layout, water_frac, sky_frac, land_frac, horizon_y_norm, horizon_confidence
    )
    scene_type = legacy_scene_type(viewpoint_type, scene_layout)

    stats = BackgroundStats(
        dataset=dataset_name,
        image_id=int(image_id),
        image_name=image_name,
        image_path=image_path,
        width=image.width,
        height=image.height,
        water_frac=water_frac,
        sky_frac=sky_frac,
        land_frac=land_frac,
        other_frac=other_frac,
        vessel_ignore_frac=vessel_ignore_frac,
        horizon_y_norm=horizon_y_norm,
        horizon_confidence=horizon_confidence,
        water_row_mean=water_row_mean,
        water_row_std=water_row_std,
        sky_row_mean=sky_row_mean,
        sky_row_std=sky_row_std,
        land_row_mean=land_row_mean,
        land_row_std=land_row_std,
        vertical_profile=vp,
        scene_layout=scene_layout,
        viewpoint_type=viewpoint_type,
        context_key=context_key,
        scene_type=scene_type,
        grounding_counts=counts,
        grounding_total_detections=total_detections,
    )
    return stats, maps, context_img


# 
# Visualization
# 

def make_rgb_mask(maps: Dict[str, np.ndarray]) -> Image.Image:
    H, W = maps["water"].shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for label, color in COLORS.items():
        if label not in maps:
            continue
        m = maps[label][..., None]
        rgb = np.where(m > 0.5, np.array(color, dtype=np.float32), rgb)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def blend_overlay(base: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = base.convert("RGB")
    mask_big = mask_rgb.convert("RGB").resize(base.size, Image.NEAREST)
    return Image.blend(base, mask_big, alpha)


def add_title(im: Image.Image, title: str, font) -> Image.Image:
    pad_h = 34
    canvas = Image.new("RGB", (im.width, im.height + pad_h), (255, 255, 255))
    canvas.paste(im.convert("RGB"), (0, pad_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    return canvas


def make_stats_panel(size: Tuple[int, int], stats: BackgroundStats, font, small_font) -> Image.Image:
    w, h = size
    panel = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    y = 18
    draw.text((18, y), "Grounded-SAM2 Context", fill=(0, 0, 0), font=font)
    y += 42
    draw.text((18, y), f"viewpoint: {stats.viewpoint_type}", fill=(0, 0, 0), font=font)
    y += 36
    draw.text((18, y), f"layout:    {stats.scene_layout}", fill=(0, 0, 0), font=font)
    y += 36
    draw.text((18, y), f"context: {stats.context_key[:58]}", fill=(40, 40, 40), font=small_font)
    y += 40

    rows = [
        ("water", stats.water_frac),
        ("sky", stats.sky_frac),
        ("land", stats.land_frac),
        ("other", stats.other_frac),
        ("vessel_ignore", stats.vessel_ignore_frac),
        ("horizon_y", stats.horizon_y_norm),
        ("horizon_conf", stats.horizon_confidence),
    ]
    for name, val in rows:
        draw.text((34, y), f"{name:<14}: {val:6.3f}", fill=(0, 0, 0), font=font)
        y += 34

    y += 10
    draw.text((18, y), f"detections: {stats.grounding_total_detections} {stats.grounding_counts}", fill=(40, 40, 40), font=small_font)

    draw.text((18, h - 112), "Legend:", fill=(0, 0, 0), font=small_font)
    draw.text((18, h - 86), "blue=water, cyan=sky, green=land", fill=(0, 0, 0), font=small_font)
    draw.text((18, h - 60), "orange=vessel ignored, gray=other", fill=(0, 0, 0), font=small_font)
    return panel


def make_side_by_side_debug(original_img: Image.Image, context_img: Image.Image, mask_rgb: Image.Image, overlay_img: Image.Image, stats: BackgroundStats) -> Image.Image:
    title_font = load_font(22)
    font = load_font(22)
    small_font = load_font(16)

    thumb_w = 520
    thumb_h = int(round(thumb_w * original_img.height / max(1, original_img.width)))
    original_t = ImageOps.contain(original_img.convert("RGB"), (thumb_w, thumb_h))
    context_t = ImageOps.contain(context_img.convert("RGB"), (thumb_w, thumb_h))
    mask_t = ImageOps.contain(mask_rgb.resize(original_img.size, Image.NEAREST).convert("RGB"), (thumb_w, thumb_h))
    overlay_t = ImageOps.contain(overlay_img.convert("RGB"), (thumb_w, thumb_h))
    stats_t = ImageOps.contain(make_stats_panel(original_img.size, stats, font, small_font), (thumb_w, thumb_h))

    imgs = [
        add_title(original_t, "Original", title_font),
        add_title(context_t, "Context Input", title_font),
        add_title(mask_t, "Scene Labels", title_font),
        add_title(overlay_t, "Overlay", title_font),
        add_title(stats_t, "Stats", title_font),
    ]

    target_h = max(im.height for im in imgs)
    imgs = [ImageOps.pad(im, (im.width, target_h), color=(255, 255, 255)) for im in imgs]
    canvas = Image.new("RGB", (sum(im.width for im in imgs), target_h), (245, 245, 245))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


# 
# Pose helpers
# 

def coarse_pitch_family(pitch_deg: float) -> str:
    if pitch_deg < -60:
        return "nadir"
    if pitch_deg > 40:
        return "steep_up"
    return "level"


def coarse_yaw_family(yaw_deg: float) -> str:
    ay = abs(yaw_deg)
    if ay < 30:
        return "front_back"
    if 60 <= ay <= 120:
        return "side"
    return "oblique"


def coarse_roll_family(roll_deg: float) -> str:
    return "rolled" if abs(roll_deg) > 25 else "level"


def coarse_pose_family(pitch_deg: float, yaw_deg: float, roll_deg: float) -> str:
    return (
        f"{coarse_pitch_family(pitch_deg)}__"
        f"{coarse_yaw_family(yaw_deg)}__"
        f"{coarse_roll_family(roll_deg)}"
    )


# 
# Main
# 

def main():
    ensure_dir(OUT_DIR)
    if SAVE_CROPS:
        ensure_dir(os.path.join(OUT_DIR, "crops"))
    if SAVE_SEGMENTATION_DEBUG:
        ensure_dir(os.path.join(OUT_DIR, "segmentation_debug"))

    coco = load_json(COCO_JSON)
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    cat_id_to_name = {c["id"]: c["name"].lower() for c in cats}
    watercraft_cat_ids = {
        cid for cid, name in cat_id_to_name.items()
        if name == WATERCRAFT_CATEGORY_NAME.lower()
    }
    if not watercraft_cat_ids:
        raise RuntimeError(f"Could not find category '{WATERCRAFT_CATEGORY_NAME}' in categories")

    print("Using category IDs:", watercraft_cat_ids)
    print("Matched names:", [cat_id_to_name[cid] for cid in sorted(watercraft_cat_ids)])

    image_id_to_info = {im["id"]: im for im in images}
    anns_by_image = defaultdict(list)
    for ann in anns:
        if ann["category_id"] in watercraft_cat_ids:
            anns_by_image[ann["image_id"]].append(ann)

    all_image_ids = list(anns_by_image.keys())
    if MAX_IMAGES_TO_PROCESS is not None and MAX_IMAGES_TO_PROCESS < len(all_image_ids):
        random.seed(RANDOM_SEED)
        if RANDOM_SUBSET:
            selected_image_ids = random.sample(all_image_ids, MAX_IMAGES_TO_PROCESS)
        else:
            selected_image_ids = all_image_ids[:MAX_IMAGES_TO_PROCESS]
    else:
        selected_image_ids = all_image_ids

    print(f"Total images with vessels in JSON: {len(all_image_ids)}")
    print(f"Images selected for processing:   {len(selected_image_ids)}")
    print(f"Scene context: Grounding DINO + SAM2")

    image_contexts: List[Dict] = []
    candidate_library: List[Dict] = []
    scene_object_pairs: List[Dict] = []

    global_summary = {
        "num_images_with_vessels_in_json": len(all_image_ids),
        "num_images_selected": len(selected_image_ids),
        "num_images_processed": 0,
        "num_candidates": 0,
        "num_skipped": 0,
        "datasets": {},
        "category_ids": sorted(list(watercraft_cat_ids)),
        "category_name": WATERCRAFT_CATEGORY_NAME,
        "max_images_to_process": MAX_IMAGES_TO_PROCESS,
        "random_subset": RANDOM_SUBSET,
        "random_seed": RANDOM_SEED,
        "scene_context_model": "grounding_dino_sam2",
        "grounding_model_id": GROUNDING_MODEL_ID,
        "sam2_model_id": SAM2_MODEL_ID,
        "scene_working_size": SCENE_WORKING_SIZE,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "context_fields": ["viewpoint_type", "scene_layout", "context_key"],
    }

    segmenter = GroundedSAM2SceneSegmenter()
    pose_net = PoseNetEstimator()

    dataset_counter_images = Counter()
    dataset_counter_candidates = Counter()
    dataset_counter_skipped = Counter()
    viewpoint_counter = Counter()
    layout_counter = Counter()

    debug_saved = 0
    progress = tqdm(selected_image_ids, desc="Processing images", unit="img")

    for image_id in progress:
        vessel_anns = anns_by_image[image_id]
        info = image_id_to_info[image_id]
        file_name = info["file_name"]
        image_path = os.path.join(IMAGE_DIR, file_name)

        if not os.path.exists(image_path):
            tqdm.write(f"Missing image: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        W, H = img.size

        dataset_name = safe_name(vessel_anns[0].get("source", "unknown"))
        sequence_group = infer_sequence_group(dataset_name, file_name)

        suppress_boxes = [xywh_to_xyxy(ann["bbox"]) for ann in vessel_anns]
        bg_stats, maps, context_img = extract_background_stats_for_image(
            dataset_name=dataset_name,
            image_id=image_id,
            image_name=os.path.basename(file_name),
            image_path=image_path,
            image=img,
            segmenter=segmenter,
            suppress_boxes=suppress_boxes,
        )
        image_contexts.append(asdict(bg_stats))
        viewpoint_counter[bg_stats.viewpoint_type] += 1
        layout_counter[bg_stats.scene_layout] += 1

        if SAVE_SEGMENTATION_DEBUG and debug_saved < MAX_DEBUG_VIS:
            mask_rgb = make_rgb_mask(maps)
            overlay_img = blend_overlay(img, mask_rgb, alpha=0.45)
            panel = make_side_by_side_debug(
                original_img=img,
                context_img=context_img,
                mask_rgb=mask_rgb,
                overlay_img=overlay_img,
                stats=bg_stats,
            )
            dbg_name = f"{safe_name(dataset_name)}__{safe_name(os.path.splitext(os.path.basename(file_name))[0])}.jpg"
            panel.save(os.path.join(OUT_DIR, "segmentation_debug", dbg_name), quality=92)
            debug_saved += 1

        global_summary["num_images_processed"] += 1
        dataset_counter_images[dataset_name] += 1

        for ann in vessel_anns:
            x, y, w, h = ann["bbox"]
            ann_dataset_name = safe_name(ann.get("source", dataset_name))

            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                global_summary["num_skipped"] += 1
                dataset_counter_skipped[ann_dataset_name] += 1
                continue

            box_xyxy = xywh_to_xyxy(ann["bbox"])
            crop_box = expand_box_xyxy(box_xyxy, W, H, BOX_EXPAND_RATIO)
            crop_img = crop_pil(img, crop_box)
            pred = pose_net.predict_to_dict(crop_img)

            cx_norm, cy_norm, area_frac, aspect_ratio = center_and_area_frac(box_xyxy, W, H)
            y_bin = y_bin_from_center(cy_norm, num_bins=Y_BINS)

            crop_path = None
            if SAVE_CROPS:
                crop_name = (
                    f"{safe_name(ann_dataset_name)}__"
                    f"{safe_name(os.path.splitext(os.path.basename(file_name))[0])}"
                    f"__ann{ann['id']}.jpg"
                )
                crop_path = os.path.join(OUT_DIR, "crops", crop_name)
                crop_img.save(crop_path)

            common_context = {
                "scene_type": bg_stats.scene_type,  # legacy
                "scene_layout": bg_stats.scene_layout,
                "viewpoint_type": bg_stats.viewpoint_type,
                "context_key": bg_stats.context_key,
                "water_frac": bg_stats.water_frac,
                "sky_frac": bg_stats.sky_frac,
                "land_frac": bg_stats.land_frac,
                "other_frac": bg_stats.other_frac,
                "vessel_ignore_frac": bg_stats.vessel_ignore_frac,
                "horizon_y_norm": bg_stats.horizon_y_norm,
                "horizon_confidence": bg_stats.horizon_confidence,
            }

            candidate_record = {
                "dataset": ann_dataset_name,
                "candidate_id": f"{ann_dataset_name}_{image_id}_{ann['id']}",
                "image_id": int(image_id),
                "annotation_id": int(ann["id"]),
                "image_name": os.path.basename(file_name),
                "image_path": image_path,
                "crop_path": crop_path,
                "source": ann.get("source", None),
                "category_name": cat_id_to_name[ann["category_id"]],
                "sequence_group": sequence_group,
                "bbox_xywh": [float(v) for v in ann["bbox"]],
                "bbox_xyxy": [float(v) for v in box_xyxy],
                "crop_box_xyxy": [float(v) for v in crop_box],
                "center_x_norm": cx_norm,
                "center_y_norm": cy_norm,
                "y_bin": int(y_bin),
                "area_frac": area_frac,
                "aspect_ratio": aspect_ratio,
                **common_context,
                "pitch_family": coarse_pitch_family(pred["pitch_deg"]),
                "yaw_family": coarse_yaw_family(pred["yaw_deg"]),
                "roll_family": coarse_roll_family(pred["roll_deg"]),
                "pose_family": coarse_pose_family(pred["pitch_deg"], pred["yaw_deg"], pred["roll_deg"]),
                "is_nadir_like": bool(pred["pitch_deg"] < -60.0),
                "is_aerial_scene": bool(bg_stats.viewpoint_type in ["nadir_aerial", "oblique_aerial"]),
                **pred,
            }
            candidate_library.append(candidate_record)

            pair_record = {
                "dataset": ann_dataset_name,
                "image_id": int(image_id),
                "image_name": os.path.basename(file_name),
                "annotation_id": int(ann["id"]),
                "source": ann.get("source", None),
                **common_context,
                "water_row_mean": bg_stats.water_row_mean,
                "water_row_std": bg_stats.water_row_std,
                "sky_row_mean": bg_stats.sky_row_mean,
                "sky_row_std": bg_stats.sky_row_std,
                "land_row_mean": bg_stats.land_row_mean,
                "land_row_std": bg_stats.land_row_std,
                "vertical_profile": bg_stats.vertical_profile,
                "sequence_group": sequence_group,
                "center_x_norm": cx_norm,
                "center_y_norm": cy_norm,
                "y_bin": int(y_bin),
                "area_frac": area_frac,
                "aspect_ratio": aspect_ratio,
                "pitch_bin": pred["pitch_bin"],
                "yaw_bin": pred["yaw_bin"],
                "roll_bin": pred["roll_bin"],
                "pitch_deg": pred["pitch_deg"],
                "yaw_deg": pred["yaw_deg"],
                "roll_deg": pred["roll_deg"],
                "pose_confidence": pred["pose_confidence"],
                "pitch_conf": pred["pitch_conf"],
                "yaw_conf": pred["yaw_conf"],
                "roll_conf": pred["roll_conf"],
                "pitch_family": coarse_pitch_family(pred["pitch_deg"]),
                "yaw_family": coarse_yaw_family(pred["yaw_deg"]),
                "roll_family": coarse_roll_family(pred["roll_deg"]),
                "pose_family": coarse_pose_family(pred["pitch_deg"], pred["yaw_deg"], pred["roll_deg"]),
                "is_nadir_like": bool(pred["pitch_deg"] < -60.0),
                "is_aerial_scene": bool(bg_stats.viewpoint_type in ["nadir_aerial", "oblique_aerial"]),
            }
            scene_object_pairs.append(pair_record)

            global_summary["num_candidates"] += 1
            dataset_counter_candidates[ann_dataset_name] += 1

        progress.set_postfix({
            "dataset": dataset_name,
            "vp": bg_stats.viewpoint_type,
            "layout": bg_stats.scene_layout,
            "cands": global_summary["num_candidates"],
            "debug": debug_saved,
        })

    all_dataset_names = sorted(set(list(dataset_counter_images.keys()) + list(dataset_counter_candidates.keys())))
    for ds in all_dataset_names:
        global_summary["datasets"][ds] = {
            "num_images_processed": int(dataset_counter_images[ds]),
            "num_candidates": int(dataset_counter_candidates[ds]),
            "num_skipped": int(dataset_counter_skipped[ds]),
        }
    global_summary["num_segmentation_debug_saved"] = int(debug_saved)
    global_summary["viewpoint_type_hist"] = dict(viewpoint_counter)
    global_summary["scene_layout_hist"] = dict(layout_counter)

    save_json(image_contexts, os.path.join(OUT_DIR, "image_contexts.json"))
    save_json(candidate_library, os.path.join(OUT_DIR, "candidate_library.json"))
    save_json(scene_object_pairs, os.path.join(OUT_DIR, "scene_object_pairs.json"))
    save_json(global_summary, os.path.join(OUT_DIR, "summary.json"))

    print("\n=== Done ===")
    print(json.dumps(global_summary, indent=2))
    print(f"Saved unified outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
