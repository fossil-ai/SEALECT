#!/usr/bin/env python3
"""
visualize_exp4_qualitative_retrieval.py

Experiment 4: qualitative retrieval visualization across baselines.

This version uses the cleaner PIL thumbnail-grid style from the older
Experiment 4 script, but reads the current Experiment 1 output layout.

Compares:
  Random
  CLIP
  DINO
  S_prior only
  DINO + S_prior
  SEALECT

Runs for both K=20 and K=100 by default.

Expected input layout:

  /path/to/sealect_root/experiments/
    experiment1/
      exp1_retrieval_quality/
        <method_dir>/
          topk_20/
            retrieval_results.json
          topk_100/
            retrieval_results.json

Outputs:

  /path/to/sealect_root/experiments/
    experiment4/
      qualitative_retrieval_grounded_sam2/
        k20/
          panels/
            qual_panel_....
          selected_queries.json
          missing_methods.json
        k100/
          panels/
            qual_panel_....
          selected_queries.json
          missing_methods.json
        qualitative_retrieval_grounded_sam2_summary.json

Important:
  - This script does not recompute retrieval.
  - MMR is display-only and reranks the already-saved top-k pool only to reduce
    duplicate-looking thumbnails in the qualitative panel.
  - Set USE_MMR_DISPLAY_RERANK = False to show the raw top-N exactly.
  - Score labels are method-aware:
      Random: no score
      CLIP/DINO: sim only
      S_prior only: prior/context only
      DINO + S_prior: combined/prior hybrid score
      SEALECT: S/N/C/O trained retriever decomposition
"""

import json
import math
import os
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import matplotlib.pyplot as plt


# 
# CONFIG
# 

ROOT = Path("/path/to/sealect_root/experiments")
DATA_ROOT = ROOT / "dataset_decide"

MEGA_IMAGE_ROOT = Path(
    "/path/to/datasets/"
    "datasets/data/merged/images"
)

# COCO-style annotations beside merged/images, used as an extra source lookup.
# Expected location:
#   /path/to/datasets/merged/annotations_fixed.json
MEGA_ANN_JSON = MEGA_IMAGE_ROOT.parent / "annotations_fixed.json"

SPLIT_DIR = DATA_ROOT / "splits_dino_decorrelated_kbalanced_grounded_sam2"
TEST_PAIRS_JSON = SPLIT_DIR / "test_scene_object_pairs.json"

EXP1_ROOT = ROOT / "experiment1" / "exp1_retrieval_quality"
OUT_ROOT = ROOT / "experiment4" / "qualitative_retrieval_grounded_sam2"

TOP_KS = [20, 100]
NUM_QUERIES = 20
DISPLAY_TOP_K = 6
SEED = 42

# Balance qualitative examples across benchmark datasets using source/dataset keys.
# With NUM_QUERIES=12 and ~6 public sources, this gives roughly 2 per source.
BALANCE_BY_SOURCE = True
MAX_EXAMPLES_PER_SOURCE = 6
PREFER_SOURCE_DIVERSITY_BEFORE_CONTEXT = True

# Display-only MMR. Keeps quantitative retrieval untouched.
USE_MMR_DISPLAY_RERANK = True
MMR_LAMBDA_DIV = 1.0

# Candidate panels should generally show object crops. If False, missing crop_path
# produces a placeholder instead of falling back to the full image.
ALLOW_FULL_IMAGE_FALLBACK_FOR_CANDIDATES = False

# Dataset filtering. RESTRICTED_B is excluded by default because it is private.
# Filtering checks dataset/source metadata, image/crop paths, and annotations_fixed.json.
EXCLUDE_DATASETS = {"RESTRICTED_B"}

# Query selection.
NUM_EXAMPLES_PER_SCENE = 4
MAX_TOTAL_PANELS = NUM_QUERIES
TARGET_SCENE_KEYWORDS = [
    "aerial_open_water",
    "mixed_maritime",
    "shoreline",
    "harbor_or_coastal",
    "nadir",
    "horizon",
    "open_water",
    "coastal",
    "harbor",
]
MANUAL_QUERY_KEYS = []
# Example:
# MANUAL_QUERY_KEYS = [("123", "456"), ("789", "1011")]

# Current method directory candidates.
# Resolver checks each candidate under topk_20 or topk_100.
METHODS = [
    ("Random", ["random_k{K}", "random"]),
    ("CLIP", ["clip_k{K}", "clip"]),
    ("DINO", ["dino_k{K}", "dino"]),
    (r"S_prior only", ["prior_only_k{K}", "prior_only", "sweep_context_only"]),
    (r"DINO + S_prior", ["dino_prior_k{K}", "dino_prior"]),
    ("SEALECT", [
        # "sealect_dinofeat_k{K}",
        # "sealect_dinofeat",
        "sweep_ln1_lc1_lo0p10",
        "sweep_ln1_lc1_lo0p50",
        "sweep_ln1_lc1_lo1p00",
    ]),
]

# Canvas layout.
THUMB_W = 220
THUMB_H = 120
QUERY_W = THUMB_W
QUERY_H = THUMB_H

# Cleaner paper-panel layout:
#   row 0 = query full image + object/vessel crops from that same query image
#   following rows = retrieval methods
# Thumbnail captions are suppressed entirely so image rows connect cleanly.
TOP_PAD = 30
METHOD_LABEL_W = 150
ROW_GAP = 8
COL_GAP = 8
LABEL_H = 0
QUERY_FULL_COLSPAN = 1
QUERY_VESSEL_SLOTS = max(1, DISPLAY_TOP_K - QUERY_FULL_COLSPAN)
PANEL_BG = (255, 255, 255)
TEXT = (20, 20, 20)
MUTED = (70, 70, 70)
LIGHT = (245, 245, 245)
SHOW_NUMERIC_CAPTIONS = False

# Diagnostics for broken/missing image paths.
IMAGE_LOAD_FAILURES = Counter()


def load_font(size: int, bold: bool = False):
    """Best-effort font loading for cleaner paper labels."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


LABEL_FONT = load_font(15, bold=True)
HEADER_FONT = load_font(13, bold=True)
SMALL_FONT = load_font(12, bold=False)


# 
# IO
# 

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_name(s: Any) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def safe_stem(s: Any) -> str:
    """
    Safe filename stem with common image suffixes removed.

    Avoids confusing names like "...jpg.png", which some viewers/file managers
    mishandle even though PIL writes a valid PNG.
    """
    name = safe_name(s)
    for ext in [".jpeg", ".jpg", ".png", ".bmp", ".webp"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name or "image"


def save_panel_image(panel: Image.Image, png_path: Path) -> Dict[str, Any]:
    """
    Save a panel robustly and verify it can be reopened.

    Writes:
      - PNG: lossless paper artifact
      - JPG: compatibility preview for file managers/viewers that dislike huge PNGs
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)
    jpg_path = png_path.with_suffix(".jpg")

    # Force plain RGB and strip any mode/palette oddities.
    panel = panel.convert("RGB")

    panel.save(png_path, format="PNG", optimize=False)
    panel.save(jpg_path, format="JPEG", quality=95, optimize=True)

    info = {
        "png": str(png_path),
        "jpg": str(jpg_path),
        "png_exists": png_path.exists(),
        "jpg_exists": jpg_path.exists(),
        "png_size_bytes": png_path.stat().st_size if png_path.exists() else 0,
        "jpg_size_bytes": jpg_path.stat().st_size if jpg_path.exists() else 0,
    }

    # Verify both files immediately so corrupted/unopenable outputs fail loudly.
    for key, path in [("png", png_path), ("jpg", jpg_path)]:
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                info[f"{key}_format"] = im.format
                info[f"{key}_size"] = list(im.size)
                info[f"{key}_mode"] = im.mode
        except Exception as e:
            raise RuntimeError(f"Saved {key.upper()} panel is not readable: {path} ({type(e).__name__}: {e})")

    return info


def result_path_for(method_dir: str, top_k: int) -> Path:
    return EXP1_ROOT / method_dir / f"topk_{top_k}" / "retrieval_results.json"


def expand_method_name(template: str, top_k: int) -> str:
    return template.replace("{K}", str(top_k))


def resolve_method_dir(candidates: List[str], top_k: int) -> Optional[str]:
    for template in candidates:
        name = expand_method_name(template, top_k)
        if result_path_for(name, top_k).exists():
            return name
    return None


def key_for_record(r: Dict[str, Any]) -> Tuple[str, str]:
    return str(r.get("image_id")), str(r.get("annotation_id"))


def query_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return key_for_record(item.get("query", {}))


def get_dataset_name(r: Dict[str, Any]) -> str:
    # Prefer explicit dataset/source metadata. The user's test_scene_object_pairs.json
    # contains source, and annotations_fixed.json also contains source.
    for key in ["dataset", "source"]:
        val = r.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return ""


def get_source_name(r: Dict[str, Any]) -> str:
    """
    Dataset/source key for balancing benchmark examples.
    Prefer source because test_scene_object_pairs.json uses source to identify
    benchmark provenance; fall back to dataset and then annotation source index.
    """
    for key in ["source", "dataset"]:
        val = r.get(key)
        if val is not None and str(val).strip():
            return str(val)
    recovered = record_source_from_annotation_index(r)
    if recovered:
        return recovered
    return "unknown_source"


def record_source_from_annotation_index(r: Dict[str, Any]) -> str:
    """Recover source from MEGA annotations_fixed.json by id/name/path when metadata is missing."""
    keys = [
        r.get("image_id"),
        r.get("id"),
        r.get("image_name"),
        r.get("file_name"),
        r.get("filename"),
        r.get("img_name"),
        r.get("image_path"),
        r.get("crop_path"),
    ]

    for key in keys:
        if key is None:
            continue
        s = str(key)
        p = Path(s)
        for probe in [s, p.name, p.stem]:
            if probe in ANN_SOURCE_INDEX:
                return ANN_SOURCE_INDEX[probe]

    return ""


def record_has_excluded_source(r: Dict[str, Any]) -> bool:
    """
    Conservative privacy filter:
      1. checks dataset/source fields,
      2. checks annotations_fixed.json source by image id/name,
      3. checks path/name substrings.
    """
    excluded = [x.lower() for x in EXCLUDE_DATASETS]

    explicit_source = get_dataset_name(r).lower()
    indexed_source = record_source_from_annotation_index(r).lower()

    if any(x in explicit_source for x in excluded):
        return True

    if any(x in indexed_source for x in excluded):
        return True

    # Path-based fallback. This catches stale paths or filenames that include RESTRICTED_B.
    for key in [
        "image_path",
        "crop_path",
        "image_name",
        "file_name",
        "filename",
        "img_name",
        "sequence_group",
        "dataset",
        "source",
    ]:
        val = r.get(key)
        if val is None:
            continue
        txt = str(val).lower()
        if any(x in txt for x in excluded):
            return True

    return False


def is_allowed_record(r: Dict[str, Any]) -> bool:
    return not record_has_excluded_source(r)


def build_test_query_lookup() -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not TEST_PAIRS_JSON.exists():
        return {}
    rows = load_json(TEST_PAIRS_JSON)
    return {key_for_record(r): r for r in rows}


def build_image_index(image_root: Path) -> Dict[str, Path]:
    index = {}
    if not image_root.exists():
        return index

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            index[path.name] = path
            index[path.stem] = path
    return index


IMAGE_INDEX = build_image_index(MEGA_IMAGE_ROOT)


def build_annotation_source_index(ann_path: Path) -> Dict[str, str]:
    """
    Build image_id / file_name / image_name -> source map from COCO annotations_fixed.json.

    Supports common COCO forms:
      {
        "images": [{"id": 1, "file_name": "...", "source": "SMD"}],
        "annotations": [...]
      }

    Also tolerates a flat list of image records.
    """
    index = {}
    if not ann_path.exists():
        return index

    try:
        obj = load_json(ann_path)
    except Exception as e:
        print(f"Warning: could not read annotation source file {ann_path}: {e}")
        return index

    if isinstance(obj, dict):
        images = obj.get("images", [])
    elif isinstance(obj, list):
        images = obj
    else:
        images = []

    for im in images:
        if not isinstance(im, dict):
            continue

        src = im.get("source", im.get("dataset", None))
        if src is None:
            continue
        src = str(src)

        for key in [
            im.get("id"),
            im.get("image_id"),
            im.get("file_name"),
            im.get("filename"),
            im.get("image_name"),
            im.get("img_name"),
        ]:
            if key is not None:
                index[str(key)] = src
                pp = Path(str(key))
                index[pp.name] = src
                index[pp.stem] = src

    return index


ANN_SOURCE_INDEX = build_annotation_source_index(MEGA_ANN_JSON)


def build_coco_annotation_indexes(ann_path: Path):
    """Build COCO lookups used for on-the-fly query-vessel cropping."""
    ann_by_id = {}
    anns_by_image_id = defaultdict(list)
    image_by_id = {}

    if not ann_path.exists():
        return ann_by_id, anns_by_image_id, image_by_id

    try:
        obj = load_json(ann_path)
    except Exception as e:
        print(f"Warning: could not read COCO annotation file {ann_path}: {e}")
        return ann_by_id, anns_by_image_id, image_by_id

    if isinstance(obj, dict):
        images = obj.get("images", []) or []
        annotations = obj.get("annotations", []) or []
    else:
        images = []
        annotations = []

    for im in images:
        if not isinstance(im, dict):
            continue
        for key in [im.get("id"), im.get("image_id")]:
            if key is not None:
                image_by_id[str(key)] = im

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        ann_id = ann.get("id", ann.get("annotation_id"))
        image_id = ann.get("image_id")
        if ann_id is not None:
            ann_by_id[str(ann_id)] = ann
        if image_id is not None:
            anns_by_image_id[str(image_id)].append(ann)

    return ann_by_id, anns_by_image_id, image_by_id


COCO_ANN_BY_ID, COCO_ANNS_BY_IMAGE_ID, COCO_IMAGE_BY_ID = build_coco_annotation_indexes(MEGA_ANN_JSON)


def load_method_results(top_k: int):
    loaded = {}
    missing = []

    for display_name, options in METHODS:
        method_dir = resolve_method_dir(options, top_k)
        if method_dir is None:
            missing.append({
                "display_name": display_name,
                "tried": [expand_method_name(x, top_k) for x in options],
                "expected_example": str(result_path_for(expand_method_name(options[0], top_k), top_k)),
            })
            continue

        path = result_path_for(method_dir, top_k)
        rows = load_json(path)
        loaded[display_name] = {
            "method_dir": method_dir,
            "path": str(path),
            "rows": rows,
            "by_query": {query_key(item): item for item in rows},
        }

    return loaded, missing


# 
# PATH / IMAGE HELPERS
# 

def candidate_roots() -> List[Path]:
    return [
        ROOT,
        DATA_ROOT,
        SPLIT_DIR,
        DATA_ROOT / "merged_scene_pose_library_out_grounded_sam2",
        MEGA_IMAGE_ROOT,
        MEGA_IMAGE_ROOT.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]


def resolve_path(path_like: Any) -> Optional[Path]:
    if path_like is None:
        return None

    s = str(path_like).strip()
    if not s or s.lower() == "none":
        return None

    p = Path(s)
    if p.is_absolute() and p.exists():
        return p

    if p.exists():
        return p.resolve()

    if p.name in IMAGE_INDEX:
        return IMAGE_INDEX[p.name]
    if p.stem in IMAGE_INDEX:
        return IMAGE_INDEX[p.stem]

    for root in candidate_roots():
        cand = root / p
        if cand.exists():
            return cand

    parts = p.parts
    for i in range(len(parts)):
        suffix = Path(*parts[i:])
        if suffix.name in IMAGE_INDEX:
            return IMAGE_INDEX[suffix.name]
        if suffix.stem in IMAGE_INDEX:
            return IMAGE_INDEX[suffix.stem]
        for root in candidate_roots():
            cand = root / suffix
            if cand.exists():
                return cand

    return None


def resize_keep_aspect(img: Image.Image, size: Tuple[int, int], fill=(255, 255, 255)) -> Image.Image:
    """
    Force every image/crop to occupy the full thumbnail cell.

    This intentionally does NOT preserve aspect ratio. For the qualitative
    retrieval figure, a slightly stretched vessel is preferable to messy
    letterboxing, inconsistent whitespace, or disconnected-looking rows.
    """
    img = img.convert("RGB")
    target_w, target_h = size

    if img.width <= 0 or img.height <= 0:
        return Image.new("RGB", size, fill)

    return img.resize((target_w, target_h), Image.Resampling.BICUBIC)


def load_image_or_blank(path_like: Any, size: Tuple[int, int], text: str = "missing") -> Image.Image:
    p = resolve_path(path_like)
    if p is not None:
        try:
            img = Image.open(p).convert("RGB")
            return resize_keep_aspect(img, size)
        except Exception as e:
            IMAGE_LOAD_FAILURES[f"open_error::{str(p)}::{type(e).__name__}"] += 1
    else:
        if path_like is not None:
            IMAGE_LOAD_FAILURES[f"resolve_failed::{str(path_like)}"] += 1
        else:
            IMAGE_LOAD_FAILURES["empty_path"] += 1

    img = Image.new("RGB", size, LIGHT)
    draw = ImageDraw.Draw(img)
    wrapped = "\n".join(textwrap.wrap(text, width=22))
    draw.multiline_text((10, max(8, size[1] // 2 - 18)), wrapped, fill=(120, 120, 120), spacing=3)
    return img


def add_label(img: Image.Image, text: str, height: int = 28, bg=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), bg)
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), text[:90], fill=TEXT)
    return canvas


def query_image(
    query: Dict[str, Any],
    test_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    size: Tuple[int, int] = (QUERY_W, QUERY_H),
) -> Image.Image:
    candidates = [
        query.get("image_path"),
        query.get("file_name"),
        query.get("filename"),
        query.get("image_name"),
        query.get("img_name"),
        query.get("crop_path"),
    ]

    original = test_lookup.get(key_for_record(query), {})
    candidates.extend([
        original.get("image_path"),
        original.get("file_name"),
        original.get("filename"),
        original.get("image_name"),
        original.get("img_name"),
        original.get("crop_path"),
    ])

    # COCO fallback: query records may only store image_id / annotation_id.
    coco_img = COCO_IMAGE_BY_ID.get(str(query.get("image_id")), {})
    candidates.extend([
        coco_img.get("file_name"),
        coco_img.get("filename"),
        coco_img.get("image_name"),
        coco_img.get("img_name"),
        coco_img.get("path"),
    ])

    for c in candidates:
        p = resolve_path(c)
        if p is not None:
            return load_image_or_blank(p, size, "query image missing")

    return load_image_or_blank(
        None,
        size,
        f"query image missing\nimage_id={query.get('image_id')}\nann={query.get('annotation_id')}",
    )



def resolve_full_image_path_for_record(
    record: Dict[str, Any],
    test_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Resolve the full image path for a query/candidate record.

    This intentionally ignores crop_path so query-row vessel cells can be
    generated by cropping the full source image using bbox metadata.
    """
    candidates = [
        record.get("image_path"),
        record.get("file_name"),
        record.get("filename"),
        record.get("image_name"),
        record.get("img_name"),
    ]
    if test_lookup is not None:
        original = test_lookup.get(key_for_record(record), {})
        candidates.extend([
            original.get("image_path"),
            original.get("file_name"),
            original.get("filename"),
            original.get("image_name"),
            original.get("img_name"),
        ])
    # COCO fallback: many query records only contain image_id / annotation_id.
    coco_img = COCO_IMAGE_BY_ID.get(str(record.get("image_id")), {})
    candidates.extend([
        coco_img.get("file_name"),
        coco_img.get("filename"),
        coco_img.get("image_name"),
        coco_img.get("img_name"),
        coco_img.get("path"),
    ])

    for c in candidates:
        path = resolve_path(c)
        if path is not None:
            return path
    return None


def open_full_image_for_record(
    record: Dict[str, Any],
    test_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Optional[Image.Image]:
    path = resolve_full_image_path_for_record(record, test_lookup=test_lookup)
    if path is None:
        IMAGE_LOAD_FAILURES[
            f"full_image_resolve_failed::image_id={record.get('image_id')}::ann={record.get('annotation_id')}"
        ] += 1
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        IMAGE_LOAD_FAILURES[f"full_image_open_error::{str(path)}::{type(e).__name__}"] += 1
        return None


def clamp_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = x2 - x1
    bh = y2 - y1
    pad_x = max(2.0, 0.06 * bw)
    pad_y = max(2.0, 0.10 * bh)
    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(1, min(W, round(x2))))
    y2 = int(max(1, min(H, round(y2))))

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def get_bbox_xyxy(record: Dict[str, Any], image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Extract bbox as clamped xyxy pixel coords.

    Supports bbox/box [x,y,w,h], explicit xyxy, scalar xywh/xyxy, and
    normalized versions of each.
    """
    W, H = image_size

    for key in ["bbox_xyxy", "box_xyxy", "xyxy"]:
        val = record.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            x1, y1, x2, y2 = map(float, val[:4])
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H
            return clamp_bbox_xyxy(x1, y1, x2, y2, W, H)

    for keys in [("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax")]:
        if all(record.get(k) is not None for k in keys):
            x1, y1, x2, y2 = [float(record[k]) for k in keys]
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H
            return clamp_bbox_xyxy(x1, y1, x2, y2, W, H)

    for key in ["bbox", "box", "bbox_xywh", "box_xywh"]:
        val = record.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            x, y, w, h = map(float, val[:4])
            if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                x, w = x * W, w * W
                y, h = y * H, h * H
            return clamp_bbox_xyxy(x, y, x + w, y + h, W, H)

    for keys in [("x", "y", "w", "h"), ("x", "y", "width", "height"), ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]:
        if all(record.get(k) is not None for k in keys):
            x, y, w, h = [float(record[k]) for k in keys]
            if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                x, w = x * W, w * W
                y, h = y * H, h * H
            return clamp_bbox_xyxy(x, y, x + w, y + h, W, H)

    for keys in [("cx", "cy", "w", "h"), ("x_center", "y_center", "width", "height"), ("xc", "yc", "bw", "bh")]:
        if all(record.get(k) is not None for k in keys):
            cx, cy, w, h = [float(record[k]) for k in keys]
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
                cx, w = cx * W, w * W
                cy, h = cy * H, h * H
            return clamp_bbox_xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, W, H)

    return None


def query_vessel_image_from_bbox(
    vessel: Dict[str, Any],
    query_full_img: Optional[Image.Image],
    size: Tuple[int, int] = (THUMB_W, THUMB_H),
) -> Image.Image:
    """Render a query-scene vessel crop directly from the full image bbox."""
    if query_full_img is not None:
        bbox = get_bbox_xyxy(vessel, query_full_img.size)
        if bbox is not None:
            return resize_keep_aspect(query_full_img.crop(bbox), size)
        IMAGE_LOAD_FAILURES[
            f"bbox_missing::image_id={vessel.get('image_id')}::ann={vessel.get('annotation_id')}"
        ] += 1

    # Last-resort fallback only; normal query-row crops should come from bbox.
    return candidate_image(vessel)

def full_query_image_path(query: Dict[str, Any], test_lookup: Dict[Tuple[str, str], Dict[str, Any]]) -> Optional[Path]:
    """Resolve the full image path for a query, including COCO image metadata."""
    candidates = [
        query.get("image_path"),
        query.get("file_name"),
        query.get("filename"),
        query.get("image_name"),
        query.get("img_name"),
    ]
    original = test_lookup.get(key_for_record(query), {})
    candidates.extend([
        original.get("image_path"),
        original.get("file_name"),
        original.get("filename"),
        original.get("image_name"),
        original.get("img_name"),
    ])
    coco_img = COCO_IMAGE_BY_ID.get(str(query.get("image_id")), {})
    candidates.extend([
        coco_img.get("file_name"),
        coco_img.get("filename"),
        coco_img.get("image_name"),
        coco_img.get("img_name"),
        coco_img.get("path"),
    ])
    for c in candidates:
        p = resolve_path(c)
        if p is not None:
            return p
    return None


def coco_annotation_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    ann_id = record.get("annotation_id", record.get("id"))
    if ann_id is not None and str(ann_id) in COCO_ANN_BY_ID:
        return COCO_ANN_BY_ID[str(ann_id)]
    return {}


def bbox_xyxy_from_coco_or_record(record: Dict[str, Any], image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Return clipped pixel xyxy bbox from COCO bbox=[x,y,w,h] or common alternatives."""
    W, H = image_size
    ann = coco_annotation_for_record(record)

    raw = None
    fmt = None
    if ann.get("bbox") is not None:
        raw = ann.get("bbox")
        fmt = "xywh"
    elif record.get("bbox") is not None:
        raw = record.get("bbox")
        fmt = "xywh"  # COCO default
    elif record.get("bbox_xywh") is not None:
        raw = record.get("bbox_xywh")
        fmt = "xywh"
    elif record.get("bbox_xyxy") is not None:
        raw = record.get("bbox_xyxy")
        fmt = "xyxy"
    elif all(record.get(k) is not None for k in ["x", "y", "w", "h"]):
        raw = [record.get("x"), record.get("y"), record.get("w"), record.get("h")]
        fmt = "xywh"
    elif all(record.get(k) is not None for k in ["xmin", "ymin", "xmax", "ymax"]):
        raw = [record.get("xmin"), record.get("ymin"), record.get("xmax"), record.get("ymax")]
        fmt = "xyxy"

    if raw is None:
        return None

    try:
        vals = [float(v) for v in raw[:4]]
    except Exception:
        return None

    # Normalized coordinates fallback.
    if max(vals) <= 1.5:
        vals = [vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H]

    if fmt == "xywh":
        x, y, w, h = vals
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = vals

    # Add context so long/thin boats remain visible.
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= 0.10 * bw
    x2 += 0.10 * bw
    y1 -= 0.22 * bh
    y2 += 0.22 * bh

    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(1, min(W, round(x2))))
    y2 = int(max(1, min(H, round(y2))))

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def crop_query_vessel_from_full_image(vessel: Dict[str, Any], full_img: Image.Image, size: Tuple[int, int] = (THUMB_W, THUMB_H)) -> Optional[Image.Image]:
    bbox = bbox_xyxy_from_coco_or_record(vessel, full_img.size)
    if bbox is None:
        return None
    try:
        return resize_keep_aspect(full_img.crop(bbox).convert("RGB"), size)
    except Exception as e:
        IMAGE_LOAD_FAILURES[f"bbox_crop_error::{vessel.get('image_id')}::{vessel.get('annotation_id')}::{type(e).__name__}"] += 1
        return None


def query_vessels_for_image(
    query: Dict[str, Any],
    test_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    max_items: int,
    method_data: Optional[Dict[str, Any]] = None,
    qkey: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return object/vessel records from the same image as the query.

    Priority:
      1. sibling records from test_scene_object_pairs.json;
      2. the query record itself;
      3. candidates in retrieval results that share the query image_id.

    This fixes blank query-vessel cells when the split file only contains the
    selected query annotation and not every sibling annotation in the image.
    """
    image_id = str(query.get("image_id"))
    query_ann = str(query.get("annotation_id"))

    vessels: List[Dict[str, Any]] = []

    # 1) All known vessels from the same source image in the test lookup.
    for r in test_lookup.values():
        if str(r.get("image_id")) == image_id and is_allowed_record(r):
            vessels.append(r)

    # 2) Always include the query record, because it often has crop_path even
    # when the lookup entry is incomplete or the sibling annotations are absent.
    if is_allowed_record(query):
        vessels.append(query)

    # 3) Recover same-image crops from loaded retrieval result payloads.
    if method_data is not None and qkey is not None:
        for md in method_data.values():
            item = md.get("by_query", {}).get(qkey)
            if not item:
                continue
            q = item.get("query", {})
            if str(q.get("image_id")) == image_id and is_allowed_record(q):
                vessels.append(q)
            for cand in item.get("topk", []):
                if str(cand.get("image_id")) == image_id and is_allowed_record(cand):
                    vessels.append(cand)

    # 4) COCO source of truth: all annotations from the same image.
    for ann in COCO_ANNS_BY_IMAGE_ID.get(image_id, []):
        rec = dict(query)
        rec.update({
            "image_id": image_id,
            "annotation_id": ann.get("id", ann.get("annotation_id")),
            "bbox": ann.get("bbox"),
            "area": ann.get("area"),
            "category_id": ann.get("category_id"),
            "_from_coco_annotation": True,
        })
        coco_img = COCO_IMAGE_BY_ID.get(image_id, {})
        for key in ["file_name", "filename", "image_name", "img_name", "path"]:
            if coco_img.get(key) is not None and rec.get(key) is None:
                rec[key] = coco_img.get(key)
        if is_allowed_record(rec):
            vessels.append(rec)

    # De-duplicate by image/annotation/crop identity.
    seen = set()
    deduped = []
    for r in vessels:
        ident = (
            str(r.get("image_id")),
            str(r.get("annotation_id")),
            str(r.get("crop_path")),
            str(r.get("object_crop_path")),
        )
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(r)

    def has_resolvable_crop(r: Dict[str, Any]) -> bool:
        if coco_annotation_for_record(r).get("bbox") is not None or r.get("bbox") is not None:
            return True
        for field in ["crop_path", "object_path", "object_crop_path", "crop_file", "crop_name"]:
            if resolve_path(r.get(field)) is not None:
                return True
        return False

    def sort_key(r: Dict[str, Any]):
        is_query = str(r.get("annotation_id")) == query_ann
        crop_ok = has_resolvable_crop(r)
        area = r.get("area_frac")
        try:
            area_val = float(area)
        except Exception:
            area_val = 0.0
        return (not is_query, not crop_ok, -area_val, str(r.get("annotation_id")))

    deduped = sorted(deduped, key=sort_key)
    return deduped[:max_items]


def candidate_image(cand: Dict[str, Any]) -> Image.Image:
    # Prefer actual object crops. Full-image fallback is optional because it can
    # make qualitative retrieval panels look broken/misleading.
    fields = ["crop_path"]
    if ALLOW_FULL_IMAGE_FALLBACK_FOR_CANDIDATES:
        fields.extend(["image_path", "image_name", "file_name", "filename"])

    for field in fields:
        p = resolve_path(cand.get(field))
        if p is not None:
            return load_image_or_blank(p, (THUMB_W, THUMB_H), "crop missing")

    # Extra fallback: sometimes crop paths are stored under alternate names.
    for field in ["object_path", "object_crop_path", "crop_file", "crop_name"]:
        p = resolve_path(cand.get(field))
        if p is not None:
            return load_image_or_blank(p, (THUMB_W, THUMB_H), "crop missing")

    return load_image_or_blank(
        None,
        (THUMB_W, THUMB_H),
        f"crop missing\nimg={cand.get('image_id')}\nann={cand.get('annotation_id')}",
    )


# 
# HISTOGRAM / LABEL HELPERS
# 

def yaw_family_from_deg(yaw_deg: Any) -> str:
    ay = abs(float(yaw_deg))
    if ay < 30:
        return "front/back"
    if 60 <= ay <= 120:
        return "side"
    return "oblique"


def yaw_family_from_bin(yaw_bin: Any) -> str:
    try:
        b = int(yaw_bin)
    except Exception:
        return "missing"

    # 18 bins over 360 deg, approximately convert bin center to [-180, 180).
    deg = -180 + (b + 0.5) * 20
    return yaw_family_from_deg(deg)


def yaw_token(row: Dict[str, Any]) -> str:
    if row.get("yaw_deg") is not None:
        try:
            return yaw_family_from_deg(row.get("yaw_deg"))
        except Exception:
            pass
    if row.get("yaw_family") is not None:
        y = str(row.get("yaw_family"))
        if y in {"front_back", "front/back"}:
            return "front/back"
        if y in {"side", "oblique"}:
            return y
    return yaw_family_from_bin(row.get("yaw_bin"))


def build_gt_yaw_by_context(test_lookup: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Counter]:
    by_context = defaultdict(Counter)
    for r in test_lookup.values():
        if not is_allowed_record(r):
            continue
        ctx = str(r.get("context_key", "missing"))
        by_context[ctx][yaw_token(r)] += 1
    return by_context


# def make_yaw_hist_image(counter: Counter, size=(HIST_W, HIST_H)) -> Image.Image:
#     labels = ["front/back", "oblique", "side"]
#     vals = np.array([counter.get(k, 0) for k in labels], dtype=float)
#     if vals.sum() > 0:
#         vals = vals / vals.sum()
#
#     fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
#     ax.bar(labels, vals)
#     ax.set_ylim(0, 1.0)
#     ax.set_ylabel("freq.", fontsize=8)
#     ax.set_title("GT yaw distribution", fontsize=9)
#     ax.tick_params(axis="x", labelrotation=18, labelsize=7)
#     ax.tick_params(axis="y", labelsize=7)
#     ax.grid(axis="y", alpha=0.25, linewidth=0.5)
#     fig.tight_layout(pad=0.6)
#     fig.canvas.draw()
#     arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
#     plt.close(fig)
#     return Image.fromarray(arr).resize(size)


def fmt_float(x: Any, nd: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def short_pose_label(c: Dict[str, Any]) -> str:
    yaw = c.get("yaw_deg")
    pitch = c.get("pitch_deg")
    area = c.get("area_frac")
    yb = c.get("y_bin")

    parts = []
    if yaw is not None:
        parts.append(f"yaw {float(yaw):.0f}")
    elif c.get("yaw_bin") is not None:
        parts.append(f"yb {int(c.get('yaw_bin'))}")

    if pitch is not None:
        parts.append(f"p {float(pitch):.0f}")
    elif c.get("pitch_bin") is not None:
        parts.append(f"pb {int(c.get('pitch_bin'))}")

    if yb is not None:
        parts.append(f"y{int(yb)}")
    if area is not None:
        parts.append(f"a {float(area):.3g}")

    return " | ".join(parts)


def score_value(c: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if c.get(key) is not None:
            try:
                return float(c.get(key))
            except Exception:
                pass
    return None


def score_label_for_method(c: Dict[str, Any], method: str) -> List[str]:
    """
    Return compact, method-correct score labels.

    Important semantic distinction:
      - Random does not use a score.
      - CLIP/DINO are embedding baselines, so show a single similarity/ranking score.
      - S_prior only is a prior/context baseline, so show prior/context only.
      - DINO + S_prior is a hybrid baseline, so show total + prior/context when available.
      - SEALECT is the trained retriever and is the only row where N/C/O decomposition
        should be interpreted as SEALECT's neural/context/observed terms.
    """
    m = method.lower().replace(" ", "_")

    if "random" in m:
        return []

    if "clip" in m and "prior" not in m:
        sim = score_value(c, [
            "clip_score",
            "similarity",
            "sim_score",
            "neural_score_used",
            "base_score",
            "score",
            "final_score",
        ])
        return [f"sim={fmt_float(sim, 2)}"] if sim is not None else []

    if "dino" in m and "prior" not in m:
        sim = score_value(c, [
            "dino_score",
            "similarity",
            "sim_score",
            "neural_score_used",
            "base_score",
            "score",
            "final_score",
        ])
        return [f"sim={fmt_float(sim, 2)}"] if sim is not None else []

    if "s_prior_only" in m or "prior_only" in m:
        prior = score_value(c, [
            "context_score_used",
            "prior_score_used",
            "prior_score",
            "base_score",
            "score",
            "final_score",
        ])
        return [f"prior={fmt_float(prior, 2)}"] if prior is not None else []

    if "dino_+_s_prior" in m or "dino+s_prior" in m or ("dino" in m and "prior" in m):
        total = score_value(c, ["final_score", "score", "base_score"])
        sim = score_value(c, ["dino_score", "similarity", "sim_score", "neural_score_used"])
        prior = score_value(c, ["context_score_used", "prior_score_used", "prior_score"])
        line = []
        if total is not None:
            line.append(f"score={fmt_float(total, 2)}")
        elif sim is not None:
            line.append(f"sim={fmt_float(sim, 2)}")
        if prior is not None:
            line.append(f"prior={fmt_float(prior, 2)}")
        return [" | ".join(line)] if line else []

    if "sealect" in m:
        total = score_value(c, ["final_score", "score"])
        neural = score_value(c, ["neural_score_used"])
        context = score_value(c, ["context_score_used", "prior_score_used"])
        obs = score_value(c, ["observed_score_used"])

        lines = []
        first = []
        if total is not None:
            first.append(f"S={fmt_float(total, 2)}")
        if neural is not None:
            first.append(f"N={fmt_float(neural, 2)}")
        if first:
            lines.append(" | ".join(first))

        second = []
        if context is not None:
            second.append(f"C={fmt_float(context, 2)}")
        if obs is not None:
            second.append(f"O={fmt_float(obs, 2)}")
        if second:
            lines.append(" | ".join(second))
        return lines

    # Fallback for unexpected method names: show only a generic score.
    val = score_value(c, ["final_score", "score", "base_score"])
    return [f"score={fmt_float(val, 2)}"] if val is not None else []


def single_numeric_label_for_method(c: Dict[str, Any], method: str) -> Optional[str]:
    """Return exactly one numeric caption, or None.

    This keeps qualitative panels clean: no pose tokens, decompositions, ids, or prose.
    Random gets no label. Baselines get their natural scalar where available; SEALECT
    shows only its final score.
    """
    m = method.lower().replace(" ", "_")
    if "random" in m:
        return None

    if "clip" in m and "prior" not in m:
        val = score_value(c, ["clip_score", "similarity", "sim_score", "neural_score_used", "base_score", "score", "final_score"])
    elif "dino" in m and "prior" not in m:
        val = score_value(c, ["dino_score", "similarity", "sim_score", "neural_score_used", "base_score", "score", "final_score"])
    elif "s_prior_only" in m or "prior_only" in m:
        val = score_value(c, ["context_score_used", "prior_score_used", "prior_score", "base_score", "score", "final_score"])
    else:
        # Hybrid and SEALECT rows: one scalar only, preferably the final ranking score.
        val = score_value(c, ["final_score", "score", "base_score", "neural_score_used", "context_score_used", "prior_score_used"])

    if val is None:
        return None
    return f"{val:.2f}"


# 
# DISPLAY-ONLY MMR
# 

def crop_feature(cand: Dict[str, Any]) -> np.ndarray:
    path = cand.get("crop_path")
    p = resolve_path(path)
    try:
        if p is not None:
            img = Image.open(p).convert("RGB").resize((48, 48), Image.Resampling.BICUBIC)
            arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
            arr = arr - arr.mean()
            return arr / (np.linalg.norm(arr) + 1e-8)
    except Exception:
        pass

    vals = np.asarray(
        [
            float(cand.get("pitch_bin", 0)) / 17.0,
            float(cand.get("yaw_bin", 0)) / 17.0,
            float(cand.get("roll_bin", 0)) / 17.0,
            float(cand.get("y_bin", 0)) / 7.0,
            float(cand.get("area_frac", 0.0)),
            float(cand.get("aspect_ratio", 0.0)),
        ],
        dtype=np.float32,
    )
    vals = vals - vals.mean()
    return vals / (np.linalg.norm(vals) + 1e-8)


def candidate_base_score(cand: Dict[str, Any]) -> float:
    for key in [
        "final_score",
        "base_score",
        "score",
        "neural_score_used",
        "context_score_used",
        "prior_score_used",
    ]:
        if cand.get(key) is not None:
            try:
                return float(cand[key])
            except Exception:
                pass
    return -float(int(cand.get("rank", 999)))


def normalize_scores(scores: List[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def mmr_rerank(topk: List[Dict[str, Any]], display_k: int, pool_size: int, lambda_div: float) -> List[Dict[str, Any]]:
    pool = [c for c in topk if is_allowed_record(c)][:pool_size]

    if len(pool) <= display_k:
        out = []
        for rank, cand in enumerate(pool, start=1):
            cc = dict(cand)
            cc["rank"] = rank
            cc["mmr_display_reranked"] = USE_MMR_DISPLAY_RERANK
            cc["mmr_lambda_div"] = float(lambda_div)
            return_rank = cand.get("rank")
            if return_rank is not None:
                cc["original_rank"] = return_rank
            out.append(cc)
        return out

    base = normalize_scores([candidate_base_score(c) for c in pool])
    feats = np.stack([crop_feature(c) for c in pool], axis=0)

    selected = []
    selected_local = set()

    for _ in range(min(display_k, len(pool))):
        best_i = None
        best_score = -1e18

        for i, cand in enumerate(pool):
            if i in selected_local:
                continue

            if not selected_local:
                max_sim = 0.0
            else:
                selected_idx = np.asarray(list(selected_local), dtype=int)
                max_sim = float((feats[selected_idx] @ feats[i]).max())

            score = float(base[i]) - lambda_div * max_sim

            if score > best_score:
                best_score = score
                best_i = i

        selected_local.add(best_i)
        selected.append(pool[best_i])

    out = []
    for rank, cand in enumerate(selected, start=1):
        cc = dict(cand)
        cc["original_rank"] = cand.get("rank", None)
        cc["rank"] = rank
        cc["mmr_display_reranked"] = USE_MMR_DISPLAY_RERANK
        cc["mmr_lambda_div"] = float(lambda_div)
        out.append(cc)
    return out


def display_candidates(raw_topk: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if USE_MMR_DISPLAY_RERANK:
        return mmr_rerank(
            raw_topk,
            display_k=DISPLAY_TOP_K,
            pool_size=min(top_k, len(raw_topk)),
            lambda_div=MMR_LAMBDA_DIV,
        )

    out = []
    for rank, cand in enumerate([c for c in raw_topk if is_allowed_record(c)][:DISPLAY_TOP_K], start=1):
        cc = dict(cand)
        cc["rank"] = rank
        cc["original_rank"] = cand.get("rank", rank)
        cc["mmr_display_reranked"] = False
        out.append(cc)
    return out


# 
# QUERY SELECTION
# 

def scene_label(q: Dict[str, Any]) -> str:
    return " ".join([
        str(q.get("scene_type", "")),
        str(q.get("context_key", "")),
        str(q.get("viewpoint_type", "")),
        str(q.get("scene_layout", "")),
    ]).lower()


def select_queries(method_data: Dict[str, Any], num_queries: int) -> List[Tuple[str, str]]:
    """
    Select qualitative queries present in all methods.

    Priority:
      1. honor MANUAL_QUERY_KEYS if provided;
      2. balance across source/dataset so one benchmark, e.g. SMD, cannot dominate;
      3. within each source, prefer diverse scene/context/viewpoint signatures;
      4. fill any remaining slots without violating RESTRICTED_B/private filtering.
    """
    method_names = list(method_data.keys())
    if not method_names:
        return []

    common = None
    for md in method_data.values():
        keys = set(md["by_query"].keys())
        common = keys if common is None else (common & keys)

    common = common or set()
    if not common:
        return []

    if MANUAL_QUERY_KEYS:
        manual = [(str(a), str(b)) for a, b in MANUAL_QUERY_KEYS]
        return [k for k in manual if k in common][:num_queries]

    reference_name = "SEALECT" if "SEALECT" in method_data else method_names[0]
    reference_rows = [
        item for item in method_data[reference_name]["rows"]
        if query_key(item) in common and is_allowed_record(item.get("query", {}))
    ]

    rng = random.Random(SEED)
    rng.shuffle(reference_rows)

    def qsig(q: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(q.get("viewpoint_type", "missing")),
            str(q.get("scene_layout", "missing")),
            str(q.get("context_key", "missing")),
        )

    def matched_keyword(q: Dict[str, Any]) -> Optional[str]:
        label = scene_label(q)
        for kw in TARGET_SCENE_KEYWORDS:
            if kw.lower() in label:
                return kw
        return None

    by_source = defaultdict(list)
    for item in reference_rows:
        q = item.get("query", {})
        by_source[get_source_name(q)].append(item)

    # Stable, balanced source order: sources with fewer available examples first,
    # then name, to avoid always starting with SMD when it is abundant.
    sources = sorted(by_source.keys(), key=lambda src: (len(by_source[src]), src))

    chosen: List[Tuple[str, str]] = []
    chosen_set = set()
    per_source = defaultdict(int)
    per_keyword = defaultdict(int)
    used_signatures_by_source = defaultdict(set)

    def can_take(item: Dict[str, Any], enforce_source_cap: bool = True, prefer_keyword: bool = False) -> bool:
        q = item.get("query", {})
        k = query_key(item)
        src = get_source_name(q)
        if k in chosen_set:
            return False
        if enforce_source_cap and BALANCE_BY_SOURCE and per_source[src] >= MAX_EXAMPLES_PER_SOURCE:
            return False
        if prefer_keyword and matched_keyword(q) is None:
            return False
        return True

    def take(item: Dict[str, Any]):
        q = item.get("query", {})
        k = query_key(item)
        src = get_source_name(q)
        chosen.append(k)
        chosen_set.add(k)
        per_source[src] += 1
        kw = matched_keyword(q)
        if kw is not None:
            per_keyword[kw] += 1
        used_signatures_by_source[src].add(qsig(q))

    # Pass 1: one diverse targeted scene per source.
    for src in sources:
        if len(chosen) >= num_queries:
            break
        for item in by_source[src]:
            q = item.get("query", {})
            if not can_take(item, enforce_source_cap=True, prefer_keyword=True):
                continue
            sig = qsig(q)
            if sig in used_signatures_by_source[src]:
                continue
            take(item)
            break

    # Pass 2: round-robin across sources, still source-capped and context-diverse.
    made_progress = True
    while len(chosen) < num_queries and made_progress:
        made_progress = False
        for src in sources:
            if len(chosen) >= num_queries:
                break
            for item in by_source[src]:
                q = item.get("query", {})
                if not can_take(item, enforce_source_cap=True, prefer_keyword=False):
                    continue
                sig = qsig(q)
                if sig in used_signatures_by_source[src]:
                    continue
                take(item)
                made_progress = True
                break

    # Pass 3: round-robin across sources, source-capped, allow repeated contexts.
    made_progress = True
    while len(chosen) < num_queries and made_progress:
        made_progress = False
        for src in sources:
            if len(chosen) >= num_queries:
                break
            for item in by_source[src]:
                if not can_take(item, enforce_source_cap=True, prefer_keyword=False):
                    continue
                take(item)
                made_progress = True
                break

    # Pass 4: if there are too few sources to fill num_queries under the cap,
    # relax source cap but continue round-robin to avoid SMD dominance.
    made_progress = True
    while len(chosen) < num_queries and made_progress:
        made_progress = False
        for src in sources:
            if len(chosen) >= num_queries:
                break
            for item in by_source[src]:
                if not can_take(item, enforce_source_cap=False, prefer_keyword=False):
                    continue
                take(item)
                made_progress = True
                break

    return chosen[:num_queries]




def sanitize_method_results(method_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Remove excluded/private records before query selection and visualization.

    - Drops any query whose query record is excluded.
    - Removes excluded candidates from each top-k list.
    - Returns counts for sanity checking.
    """
    counts = {
        "queries_dropped": 0,
        "candidates_dropped": 0,
        "queries_kept": 0,
    }

    sanitized = {}
    for display_name, md in method_data.items():
        rows = []
        for item in md["rows"]:
            q = item.get("query", {})
            if not is_allowed_record(q):
                counts["queries_dropped"] += 1
                continue

            new_item = dict(item)
            raw_topk = item.get("topk", [])
            clean_topk = [c for c in raw_topk if is_allowed_record(c)]
            counts["candidates_dropped"] += len(raw_topk) - len(clean_topk)
            new_item["topk"] = clean_topk
            rows.append(new_item)

        sanitized[display_name] = {
            **md,
            "rows": rows,
            "by_query": {query_key(item): item for item in rows},
        }
        counts["queries_kept"] += len(rows)

    return sanitized, counts


def assert_no_excluded_records_in_selection(
    qkeys: List[Tuple[str, str]],
    method_data: Dict[str, Any],
):
    """
    Crash loudly if any selected query or displayed candidate still appears excluded.
    This is intentionally conservative because RESTRICTED_B must not appear in generated panels.
    """
    failures = []

    for qkey in qkeys:
        for method, md in method_data.items():
            item = md["by_query"].get(qkey)
            if item is None:
                continue

            q = item.get("query", {})
            if record_has_excluded_source(q):
                failures.append({
                    "kind": "query",
                    "method": method,
                    "qkey": qkey,
                    "dataset": q.get("dataset"),
                    "source": q.get("source"),
                    "image_name": q.get("image_name"),
                    "image_path": q.get("image_path"),
                })

            for cand in item.get("topk", []):
                if record_has_excluded_source(cand):
                    failures.append({
                        "kind": "candidate",
                        "method": method,
                        "qkey": qkey,
                        "dataset": cand.get("dataset"),
                        "source": cand.get("source"),
                        "image_name": cand.get("image_name"),
                        "crop_path": cand.get("crop_path"),
                    })

    if failures:
        raise RuntimeError(
            "Excluded/private records detected in Experiment 4 visualization selection. "
            f"First failures: {json.dumps(failures[:10], indent=2)}"
        )

# 
# PANEL RENDERING
# 

def make_panel(
    qkey: Tuple[str, str],
    method_data: Dict[str, Any],
    test_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    gt_yaw_by_context: Dict[str, Counter],
    top_k: int,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Render a clean qualitative panel.

    Row 0 is now the query context row: the full query image spans the first
    two thumbnail columns, followed by vessel/object crops from that same image.
    Retrieval rows contain only images plus, optionally, one scalar caption.
    """
    method_names = list(method_data.keys())
    ref_item = next(iter(method_data.values()))["by_query"][qkey]
    query = ref_item.get("query", {})

    context = str(query.get("context_key", "missing"))
    scene_type = str(query.get("scene_type", query.get("scene_layout", "missing")))
    image_name = str(query.get("image_name", query.get("file_name", f"query_{qkey[0]}")))

    query_full_w = THUMB_W
    full_query_path = full_query_image_path(query, test_lookup)
    query_img = query_image(query, test_lookup, size=(THUMB_W, THUMB_H))
    query_full_img = None
    if full_query_path is not None:
        try:
            query_full_img = Image.open(full_query_path).convert("RGB")
        except Exception as e:
            IMAGE_LOAD_FAILURES[f"query_full_open_error::{str(full_query_path)}::{type(e).__name__}"] += 1
    if query_full_img is None:
        query_full_img = open_full_image_for_record(query, test_lookup=test_lookup)
    query_vessels = query_vessels_for_image(query, test_lookup, max_items=QUERY_VESSEL_SLOTS, method_data=method_data, qkey=qkey)

    row_h = THUMB_H + LABEL_H + ROW_GAP
    rows = 1 + len(method_names)
    grid_w = METHOD_LABEL_W + DISPLAY_TOP_K * THUMB_W + (DISPLAY_TOP_K - 1) * COL_GAP
    total_w = grid_w
    total_h = TOP_PAD + rows * row_h + 18

    canvas = Image.new("RGB", (total_w, total_h), PANEL_BG)
    draw = ImageDraw.Draw(canvas)

    # Column headers: keep them simple and useful.
    x_grid = METHOD_LABEL_W
    draw.text((x_grid + 6, 10), "QUERY", fill=TEXT, font=HEADER_FONT)
    for k in range(QUERY_VESSEL_SLOTS):
        x = x_grid + (QUERY_FULL_COLSPAN + k) * (THUMB_W + COL_GAP)
        draw.text((x + 6, 10), f"VESSEL {k + 1}", fill=TEXT, font=HEADER_FONT)

    # Row 0: query image + crops of vessels in that same image.
    y = TOP_PAD
    draw.text((METHOD_LABEL_W - 8, y + THUMB_H // 2 - 8), "QUERY", fill=TEXT, anchor="ra", font=LABEL_FONT)
    canvas.paste(query_img, (x_grid, y))
    draw.rectangle(
        [x_grid, y, x_grid + query_img.width - 1, y + query_img.height - 1],
        outline=(200, 200, 200),
        width=1,
    )

    for k in range(QUERY_VESSEL_SLOTS):
        x = x_grid + (QUERY_FULL_COLSPAN + k) * (THUMB_W + COL_GAP)
        if k < len(query_vessels):
            vessel = query_vessels[k]
            img = None
            if query_full_img is not None:
                img = crop_query_vessel_from_full_image(vessel, query_full_img, size=(THUMB_W, THUMB_H))
            if img is None:
                img = query_vessel_image_from_bbox(vessel, query_full_img, size=(THUMB_W, THUMB_H))
            is_query_ann = str(vessel.get("annotation_id")) == str(query.get("annotation_id"))
        else:
            img = load_image_or_blank(None, (THUMB_W, THUMB_H), "no vessel")
            is_query_ann = False

        canvas.paste(img, (x + 4, y))
        draw.rectangle(
            [x + 4, y, x + 4 + img.width - 1, y + img.height - 1],
            outline=(70, 70, 70) if is_query_ann else (220, 220, 220),
            width=2 if is_query_ann else 1,
        )
    # Retrieval method rows.
    for r, method in enumerate(method_names, start=1):
        y = TOP_PAD + r * row_h
        md = method_data[method]
        item = md["by_query"][qkey]

        draw.text((METHOD_LABEL_W - 8, y + THUMB_H // 2 - 8), method.upper(), fill=TEXT, anchor="ra", font=LABEL_FONT)

        shown = display_candidates(item.get("topk", []), top_k=top_k)
        for k in range(DISPLAY_TOP_K):
            x = x_grid + k * (THUMB_W + COL_GAP)
            if k >= len(shown):
                img = load_image_or_blank(None, (THUMB_W, THUMB_H), "no candidate")
                cand = {}
            else:
                cand = shown[k]
                img = candidate_image(cand)

            canvas.paste(img, (x, y))
            draw.rectangle(
                [x, y, x + img.width - 1, y + img.height - 1],
                outline=(220, 220, 220),
                width=1,
            )

    title = (
        f"context={context} | scene={scene_type} | image={image_name} | "
        f"query_id={query.get('image_id')} ann={query.get('annotation_id')} | "
        f"dataset={get_dataset_name(query)} | top_k_pool={top_k} | "
        f"display_top_k={DISPLAY_TOP_K} | "
        f"mmr_display={USE_MMR_DISPLAY_RERANK} lambda_div={MMR_LAMBDA_DIV}"
    )

    meta = {
        "image_id": qkey[0],
        "annotation_id": qkey[1],
        "context_key": context,
        "scene_type": scene_type,
        "image_name": image_name,
        "dataset": get_dataset_name(query),
        "source": get_source_name(query),
        "top_k_pool": top_k,
        "display_top_k": DISPLAY_TOP_K,
        "query_vessels_shown": len(query_vessels),
        "query_full_image_path": str(full_query_path) if full_query_path is not None else None,
        "query_vessel_crop_source": "bbox_from_full_image",
        "mmr_display_rerank": USE_MMR_DISPLAY_RERANK,
        "mmr_lambda_div": MMR_LAMBDA_DIV,
        "query": query,
    }

    return canvas, title, meta


# 
# MAIN
# 

def run_for_topk(top_k: int) -> Dict[str, Any]:
    print(f"\n=== Experiment 4 qualitative panels | K={top_k} ===")

    test_lookup = build_test_query_lookup()
    gt_yaw_by_context = build_gt_yaw_by_context(test_lookup)

    method_data, missing = load_method_results(top_k)
    method_data, filter_counts = sanitize_method_results(method_data)

    out_dir = OUT_ROOT / f"k{top_k}"
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    if missing:
        save_json(missing, out_dir / "missing_methods.json")
        print("Missing methods:")
        for m in missing:
            print(f"  {m['display_name']}: tried {m['tried']}")

    if not method_data:
        raise RuntimeError(f"No method retrieval_results.json files found for K={top_k}.")

    print("Loaded methods after privacy filtering:")
    for name, md in method_data.items():
        print(f"  {name:16s} -> {md['method_dir']} ({len(md['rows'])} queries)")
    print(f"Privacy filter counts: {filter_counts}")

    qkeys = select_queries(method_data, NUM_QUERIES)
    if not qkeys:
        raise RuntimeError(f"No common queries found across loaded methods for K={top_k}.")

    assert_no_excluded_records_in_selection(qkeys, method_data)

    source_counts = Counter()
    for qkey in qkeys:
        q = next(iter(method_data.values()))["by_query"][qkey].get("query", {})
        source_counts[get_source_name(q)] += 1
    print(f"Selected source counts: {dict(source_counts)}")

    selected_meta = []
    for i, qkey in enumerate(qkeys):
        panel, title, meta = make_panel(
            qkey=qkey,
            method_data=method_data,
            test_lookup=test_lookup,
            gt_yaw_by_context=gt_yaw_by_context,
            top_k=top_k,
        )

        scene = safe_stem(meta.get("scene_type", "scene"))
        img = safe_stem(meta.get("image_name", f"query_{i}"))
        out_path = panel_dir / f"qual_panel_{i:03d}_k{top_k}_{scene}_{img}.png"
        txt_path = panel_dir / f"qual_panel_{i:03d}_k{top_k}_{scene}_{img}.txt"

        save_info = save_panel_image(panel, out_path)
        txt_path.write_text(title)

        meta["index"] = i
        meta["output"] = str(out_path)
        meta["output_jpg"] = str(out_path.with_suffix(".jpg"))
        meta["save_info"] = save_info
        meta["title_file"] = str(txt_path)
        selected_meta.append(meta)

        print(
            f"  Saved {out_path} "
            f"({save_info['png_format']} {save_info['png_size']} "
            f"{save_info['png_size_bytes']} bytes) and {out_path.with_suffix('.jpg')}"
        )

    save_json(selected_meta, out_dir / "selected_queries.json")
    save_json(
        [
            {
                "index": m["index"],
                "source": m.get("source"),
                "dataset": m.get("dataset"),
                "output": m.get("output"),
                "output_jpg": m.get("output_jpg"),
                "save_info": m.get("save_info"),
            }
            for m in selected_meta
        ],
        out_dir / "saved_panel_manifest.json",
    )

    summary = {
        "top_k": top_k,
        "display_top_k": DISPLAY_TOP_K,
        "num_queries_requested": NUM_QUERIES,
        "num_queries_rendered": len(selected_meta),
        "use_mmr_display_rerank": USE_MMR_DISPLAY_RERANK,
        "mmr_lambda_div": MMR_LAMBDA_DIV,
        "excluded_datasets": sorted(EXCLUDE_DATASETS),
        "balance_by_source": BALANCE_BY_SOURCE,
        "max_examples_per_source": MAX_EXAMPLES_PER_SOURCE,
        "selected_source_counts": dict(source_counts),
        "privacy_filter_counts": filter_counts,
        "annotation_source_index_path": str(MEGA_ANN_JSON),
        "annotation_source_index_size": len(ANN_SOURCE_INDEX),
        "methods_loaded": {
            display_name: {
                "method_dir": md["method_dir"],
                "path": md["path"],
                "num_queries": len(md["rows"]),
            }
            for display_name, md in method_data.items()
        },
        "missing_methods": missing,
        "output_dir": str(out_dir),
        "panel_dir": str(panel_dir),
        "image_load_failures_top50": dict(IMAGE_LOAD_FAILURES.most_common(50)),
    }

    save_json(summary, out_dir / "summary.json")
    save_json(dict(IMAGE_LOAD_FAILURES.most_common(200)), out_dir / "image_load_failures.json")
    return summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Experiment 4: qualitative retrieval visualization")
    print(f"ROOT: {ROOT}")
    print(f"EXP1_ROOT: {EXP1_ROOT}")
    print(f"MEGA_IMAGE_ROOT: {MEGA_IMAGE_ROOT} ({len(IMAGE_INDEX)} indexed names/stems)")
    print(f"MEGA_ANN_JSON: {MEGA_ANN_JSON} ({len(ANN_SOURCE_INDEX)} source entries, {len(COCO_ANN_BY_ID)} annotations, {len(COCO_IMAGE_BY_ID)} images)")
    print(f"EXCLUDE_DATASETS: {sorted(EXCLUDE_DATASETS)}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"TOP_KS: {TOP_KS}")
    print(f"DISPLAY_TOP_K: {DISPLAY_TOP_K}")
    print(f"BALANCE_BY_SOURCE: {BALANCE_BY_SOURCE}, MAX_EXAMPLES_PER_SOURCE: {MAX_EXAMPLES_PER_SOURCE}")
    print(f"USE_MMR_DISPLAY_RERANK: {USE_MMR_DISPLAY_RERANK}")
    print(f"ALLOW_FULL_IMAGE_FALLBACK_FOR_CANDIDATES: {ALLOW_FULL_IMAGE_FALLBACK_FOR_CANDIDATES}")

    if not EXP1_ROOT.exists():
        raise FileNotFoundError(f"Missing Experiment 1 results root: {EXP1_ROOT}")

    all_summaries = []
    for top_k in TOP_KS:
        all_summaries.append(run_for_topk(top_k))

    save_json(
        {
            "top_ks": TOP_KS,
            "summaries": all_summaries,
            "out_root": str(OUT_ROOT),
        },
        OUT_ROOT / "qualitative_retrieval_grounded_sam2_summary.json",
    )

    print("\nDone.")
    print(f"Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
