#!/usr/bin/env python3
"""
02_make_bootplace_eval_requests_clean_masks.py

Build clean BootPlace eval request folders from Experiment 1 retrieval outputs.

Fixes included:
  - Uses annotations_fixed.json as source of truth.
  - Resolves global COCO ann IDs and local per-image ann indices.
  - Rejects candidate boats that are too small (<64x64 px by default).
  - Rejects likely truncated boats touching image borders.
  - Rejects extreme aspect-ratio crops.
  - Uses COCO polygon segmentation when available to create clean RGBA alpha masks.
  - Falls back to rectangular alpha only if no segmentation exists.
  - Optionally requires segmentation masks via --require-segmentation.
  - Stores existing query-scene vessel boxes for overlap filtering after BootPlace inference.
  - Writes visualization/debug metadata per request.
  - Writes BootPlace-compatible folders:
        test/backgrounds/
        test/backgrounds_single/
        test/objects/
        test/objects_single/
        test/location/
        test/location_single/
        test/annotations.json
        test/metadata/

Recommended:
  Run this first, then run 03_run_bootplace_inference_clean.sh.

Example:
  /path/to/project/venv/bin/python3 \
    02_make_bootplace_eval_requests_clean_masks.py \
    --overwrite \
    --num-queries 300 \
    --candidate-rank 1 \
    --require-segmentation

This SAM2 GPU version uses an existing local SAM2 repo/checkpoint and generates
bbox-prompted masks on CUDA. It does not clone/download anything.
"""

import argparse
import json
import math
import random
import shutil
import hashlib
import importlib
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


DEFAULT_IMAGE_ROOT = Path(
    "/path/to/datasets/"
    "datasets/data/merged/images"
)
DEFAULT_COCO_JSON = DEFAULT_IMAGE_ROOT.parent / "annotations_fixed.json"
DEFAULT_EXP1_ROOT = Path(
    "/path/to/sealect_root/experiments/"
    "experiment1/exp1_retrieval_quality"
)
DEFAULT_OUT_ROOT = Path(
    "/path/to/sealect_root/experiments/experiment5_1/bootplace_eval_requests_clean"
)

BOOTPLACE_W = 960
BOOTPLACE_H = 540

# sealect is loaded ONLY for balanced query selection; we generate requests for GALA/DiscoFOS.
METHODS = [
    ("sealect", ["sealect_dinofeat_k{K}", "sealect_dinofeat", "sweep_ln1_lc1_lo0p10"]),
    ("gala", ["gala_render_k{K}"]),
    ("discofos", ["discofos_render_k{K}"]),
]
GENERATE_ONLY = {"gala", "discofos"}   # do NOT regenerate existing methods


SAM2_PREDICTOR = None
SAM2_IMPORT_ERROR = None

SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
SAM2_1_LARGE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_1_LARGE_CFG = "configs/sam2/sam2_hiera_l.yaml"


def run_cmd(cmd, cwd=None):
    print("[SAM2 setup] " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[SAM2 setup] Downloading {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        done = 0
        block = 1024 * 1024
        while True:
            chunk = r.read(block)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r[SAM2 setup] {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
        if total:
            print()
    tmp.rename(dst)


def ensure_sam2_assets(args):
    """Validate local SAM2 repo/checkpoint and make the repo importable.

    No clone, no pip install, no checkpoint download. This uses the paths the
    experiment machine already has.
    """
    sam2_root = Path(args.sam2_root).expanduser().resolve()
    ckpt = Path(args.sam2_checkpoint).expanduser().resolve()

    if not sam2_root.exists():
        raise RuntimeError(f"SAM2 repo not found: {sam2_root}")
    if not (sam2_root / "sam2").exists():
        raise RuntimeError(f"SAM2 root does not look like a repo; missing {sam2_root / 'sam2'}")
    if not ckpt.exists():
        raise RuntimeError(f"SAM2 checkpoint not found: {ckpt}")

    args.sam2_root = sam2_root
    args.sam2_checkpoint = ckpt

    # Choose the correct default config for your checkpoint name.
    # /sam2_checkpoints/sam2_hiera_large.pt is SAM2, not SAM2.1.
    if not getattr(args, "sam2_model_cfg", None):
        args.sam2_model_cfg = SAM2_1_LARGE_CFG

    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    try:
        import torch
        import sam2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"Could not import SAM2 from {sam2_root}. Activate the venv and install it with `pip install -e {sam2_root}` if needed. Original error: {e}"
        )

    if str(args.sam2_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--sam2-device cuda requested, but torch.cuda.is_available() is false")

    cfg_candidate = sam2_root / args.sam2_model_cfg
    if cfg_candidate.exists():
        args.sam2_model_cfg = str(cfg_candidate)

    print(f"[SAM2] repo={args.sam2_root}")
    print(f"[SAM2] checkpoint={args.sam2_checkpoint}")
    print(f"[SAM2] cfg={args.sam2_model_cfg}")
    print(f"[SAM2] device={args.sam2_device}")

def init_sam2_predictor(sam2_root: Path, checkpoint: Path, model_cfg: str, device: str):
    """Lazy-load SAM2 ImagePredictor once, on GPU by default."""
    global SAM2_PREDICTOR, SAM2_IMPORT_ERROR
    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR

    if not checkpoint or str(checkpoint).strip() == "" or not Path(checkpoint).exists():
        SAM2_IMPORT_ERROR = f"missing_sam2_checkpoint::{checkpoint}"
        return None

    try:
        if sam2_root and str(sam2_root).strip() and Path(sam2_root).exists():
            sys_path = str(Path(sam2_root).resolve())
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            SAM2_IMPORT_ERROR = "cuda_requested_but_unavailable"
            return None

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"[SAM2] Loading predictor once on {device}...")
        model = build_sam2(model_cfg, str(checkpoint), device=device)
        model.eval()
        SAM2_PREDICTOR = SAM2ImagePredictor(model)
        print("[SAM2] Predictor ready")
        return SAM2_PREDICTOR
    except Exception as e:
        SAM2_IMPORT_ERROR = f"sam2_init_failed::{type(e).__name__}::{e}"
        return None

def sam2_mask_from_box(full_rgb: Image.Image, box_xyxy, args, cache_key: str = None) -> Tuple[Optional[Image.Image], str]:
    """Generate an L-mode mask using SAM2 with a bbox prompt. Returns mask and status."""
    cache_dir = getattr(args, "sam2_cache_dir", None)
    if cache_dir and cache_key:
        cache_path = Path(cache_dir) / f"{cache_key}.png"
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("L"), "sam2_cache"
            except Exception:
                pass

    predictor = init_sam2_predictor(
        getattr(args, "sam2_root", Path("")),
        getattr(args, "sam2_checkpoint", Path("")),
        getattr(args, "sam2_model_cfg", ""),
        getattr(args, "sam2_device", "cuda"),
    )
    if predictor is None:
        return None, SAM2_IMPORT_ERROR or "sam2_unavailable"

    try:
        import torch
        arr = np.array(full_rgb.convert("RGB"))
        box_np = np.array([float(v) for v in box_xyxy], dtype=np.float32)

        use_cuda = str(getattr(args, "sam2_device", "cuda")).startswith("cuda") and torch.cuda.is_available()
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.cpu.amp.autocast(enabled=False)

        with torch.inference_mode():
            with autocast_ctx:
                predictor.set_image(arr)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np,
                    multimask_output=bool(getattr(args, "sam2_multimask", True)),
                )

        if masks is None or len(masks) == 0:
            return None, "sam2_no_masks"

        scores_np = np.array(scores).reshape(-1) if scores is not None else np.zeros((len(masks),), dtype=np.float32)
        best_idx = int(np.argmax(scores_np)) if len(scores_np) else 0
        best_score = float(scores_np[best_idx]) if len(scores_np) else 0.0
        min_score = float(getattr(args, "sam2_min_score", 0.0))
        if best_score < min_score:
            return None, f"sam2_low_score::{best_score:.4f}"

        mask = (masks[best_idx].astype(np.uint8) * 255)
        out = Image.fromarray(mask, mode="L")

        if cache_dir and cache_key:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            out.save(cache_path)

        return out, f"sam2::{best_score:.4f}"
    except Exception as e:
        return None, f"sam2_failed::{type(e).__name__}::{e}"

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_stem(x: Any) -> str:
    s = str(x)
    p = Path(s)
    stem = p.stem if p.suffix else s
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)


def source_of_record(r: Dict[str, Any]) -> str:
    for k in ["source", "dataset"]:
        if r.get(k) is not None and str(r[k]).strip():
            return str(r[k])
    return "unknown"


def build_image_index(image_root: Path) -> Dict[str, Path]:
    idx = {}
    if not image_root.exists():
        return idx
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            idx[p.name] = p
            idx[p.stem] = p
    return idx


def resolve_path(path_like: Any, image_root: Path, image_index: Dict[str, Path]) -> Optional[Path]:
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

    for key in [p.name, p.stem, s]:
        if key in image_index:
            return image_index[key]

    for cand in [image_root / p, image_root / p.name]:
        if cand.exists():
            return cand

    return None


def build_coco_index(coco_json: Path) -> Dict[str, Any]:
    coco = load_json(coco_json)
    images_by_id = {str(img["id"]): img for img in coco.get("images", [])}

    anns_by_pair = {}
    anns_by_image = defaultdict(list)

    for ann in coco.get("annotations", []):
        image_id = str(ann.get("image_id"))
        anns_by_image[image_id].append(ann)

        keys = [ann.get("id")]
        for alt in ["annotation_id", "original_annotation_id", "ann_id"]:
            if ann.get(alt) is not None:
                keys.append(ann.get(alt))

        for key in keys:
            if key is not None:
                anns_by_pair[(image_id, str(key))] = ann

    for image_id in anns_by_image:
        anns_by_image[image_id] = sorted(anns_by_image[image_id], key=lambda a: int(a.get("id", 0)))

    return {
        "coco": coco,
        "images_by_id": images_by_id,
        "anns_by_pair": anns_by_pair,
        "anns_by_image": anns_by_image,
    }


def lookup_coco_ann(record: Dict[str, Any], coco_index: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Robustly resolve retrieval record to COCO image + annotation.

    Resolution:
      1. exact/global COCO ann id
      2. local 1-based ann index within image
      3. local 0-based ann index within image
      4. single-ann fallback
    """
    image_id = record.get("image_id")
    ann_id = record.get("annotation_id")

    if image_id is None:
        return None, None, "missing_image_id"

    image_id_s = str(image_id)
    img = coco_index["images_by_id"].get(image_id_s)
    anns = list(coco_index["anns_by_image"].get(image_id_s, []))

    if ann_id is not None:
        ann = coco_index["anns_by_pair"].get((image_id_s, str(ann_id)))
        if ann is not None:
            return img, ann, "global_ann_id"

    if ann_id is not None and anns:
        try:
            idx = int(ann_id)
            if 1 <= idx <= len(anns):
                return img, anns[idx - 1], "local_1_based"
            if 0 <= idx < len(anns):
                return img, anns[idx], "local_0_based"
        except Exception:
            pass

    if len(anns) == 1:
        return img, anns[0], "single_ann_fallback"

    return img, None, "unresolved"


def resolve_coco_image_path(img: Optional[Dict[str, Any]], record: Dict[str, Any], image_root: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    candidates = []
    if img is not None:
        candidates.extend([
            img.get("file_name"),
            img.get("image_name"),
            img.get("filename"),
            img.get("path"),
        ])

    candidates.extend([
        record.get("image_path"),
        record.get("image_name"),
        record.get("file_name"),
        record.get("filename"),
        record.get("crop_path"),
    ])

    for c in candidates:
        p = resolve_path(c, image_root, image_index)
        if p is not None:
            return p

    return None


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def clamp_xyxy(box, w: int, h: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def scale_box_xyxy(box, sx: float, sy: float) -> List[float]:
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def box_area(box) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, box_area(a) + box_area(b) - inter)


def write_box_txt(path: Path, xyxy: List[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in xyxy:
            f.write(f"{float(v):.6f}\n")


def result_path_for(exp1_root: Path, method_dir: str, top_k: int) -> Path:
    return exp1_root / method_dir / f"topk_{top_k}" / "retrieval_results.json"


def resolve_method_dir(exp1_root: Path, candidates: List[str], top_k: int) -> Optional[str]:
    for template in candidates:
        name = template.replace("{K}", str(top_k))
        if result_path_for(exp1_root, name, top_k).exists():
            return name
    return None


def load_method_results(exp1_root: Path, top_k: int):
    out = {}
    missing = []
    for display, candidates in METHODS:
        method_dir = resolve_method_dir(exp1_root, candidates, top_k)
        if method_dir is None:
            missing.append((display, candidates))
            continue
        path = result_path_for(exp1_root, method_dir, top_k)
        out[display] = {
            "method_dir": method_dir,
            "path": path,
            "rows": load_json(path),
        }
    return out, missing


def rasterize_coco_segmentation(segmentation, img_w: int, img_h: int) -> Optional[Image.Image]:
    """
    Rasterize polygon COCO segmentation to L mask.
    RLE is not supported here unless pycocotools is installed.
    """
    if segmentation is None:
        return None

    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    if isinstance(segmentation, list):
        ok = False
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=255, fill=255)
            ok = True
        return mask if ok else None

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            rle = segmentation
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return Image.fromarray((decoded > 0).astype(np.uint8) * 255, mode="L")
        except Exception:
            return None

    return None


def alpha_bbox(alpha_crop: Image.Image, threshold: int = 10) -> Optional[List[int]]:
    arr = np.array(alpha_crop)
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def make_rgba_from_coco_ann(
    record: Dict[str, Any],
    coco_index: Dict[str, Any],
    image_root: Path,
    image_index: Dict[str, Path],
    min_w: int,
    min_h: int,
    edge_margin: int,
    min_mask_frac: float,
    require_segmentation: bool,
    feather_radius: float,
    crop_to_mask: bool,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], str]:
    img, ann, ann_mode = lookup_coco_ann(record, coco_index)
    if img is None or ann is None or ann.get("bbox") is None:
        return None, None, f"missing_coco_ann_or_bbox::{ann_mode}"

    img_path = resolve_coco_image_path(img, record, image_root, image_index)
    if img_path is None:
        return None, None, "missing_image_path"

    try:
        full = Image.open(img_path).convert("RGB")
    except Exception:
        return None, None, "bad_image"

    img_w, img_h = full.size
    box = clamp_xyxy(xywh_to_xyxy(ann["bbox"]), img_w, img_h)
    if box is None:
        return None, None, "bad_bbox"

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = x2 - x1
    bh = y2 - y1

    min_area = int(min_w) * int(min_h)
    if bw * bh < min_area:
        return None, None, f"too_small_area::{bw}x{bh}"

    if x1 <= edge_margin or y1 <= edge_margin or x2 >= img_w - edge_margin or y2 >= img_h - edge_margin:
        return None, None, "touches_image_edge"

    aspect = bw / max(1, bh)
    if aspect < 0.2 or aspect > 8.0:
        return None, None, f"bad_aspect::{aspect:.3f}"

    seg_mask = rasterize_coco_segmentation(ann.get("segmentation"), img_w, img_h)

    if seg_mask is not None:
        alpha_full = seg_mask
        mask_source = "coco_segmentation"
    else:
        alpha_full = None
        mask_source = "missing_segmentation"

        mask_mode = getattr(args, "mask_source", "auto") if args is not None else "auto"
        if mask_mode in {"auto", "sam2"}:
            cache_key_raw = f"{img_path}|{ann.get('id')}|{x1},{y1},{x2},{y2}|{mask_mode}"
            cache_key = hashlib.md5(cache_key_raw.encode("utf-8")).hexdigest()
            sam_mask, sam_status = sam2_mask_from_box(full, [x1, y1, x2, y2], args, cache_key=cache_key) if args is not None else (None, "sam2_no_args")
            if sam_mask is not None:
                alpha_full = sam_mask
                mask_source = sam_status
            elif mask_mode == "sam2" or require_segmentation:
                return None, None, sam_status

        if alpha_full is None:
            if require_segmentation:
                return None, None, mask_source
            alpha_full = Image.new("L", (img_w, img_h), 0)
            draw = ImageDraw.Draw(alpha_full)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            mask_source = "rect_fallback"

    alpha_crop = alpha_full.crop((x1, y1, x2, y2))
    alpha_arr = np.array(alpha_crop)
    mask_area = int((alpha_arr > 10).sum())
    bbox_area = max(1, bw * bh)
    mask_frac = mask_area / bbox_area

    if mask_frac < min_mask_frac:
        return None, None, f"mask_too_small::{mask_frac:.4f}"

    if crop_to_mask and (mask_source == "coco_segmentation" or mask_source.startswith("sam2")):
        mb = alpha_bbox(alpha_crop)
        if mb is not None:
            mx1, my1, mx2, my2 = mb
            # Keep a tiny padding around the mask.
            pad = 2
            mx1 = max(0, mx1 - pad)
            my1 = max(0, my1 - pad)
            mx2 = min(alpha_crop.width, mx2 + pad)
            my2 = min(alpha_crop.height, my2 + pad)

            x1_new = x1 + mx1
            y1_new = y1 + my1
            x2_new = x1 + mx2
            y2_new = y1 + my2

            x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new
            bw = x2 - x1
            bh = y2 - y1

            min_area = int(min_w) * int(min_h)
            if bw * bh < min_area:
                return None, None, f"too_small_area_after_mask_crop::{bw}x{bh}"

            alpha_crop = alpha_full.crop((x1, y1, x2, y2))

    if feather_radius > 0:
        alpha_crop = alpha_crop.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    crop = full.crop((x1, y1, x2, y2)).convert("RGBA")
    crop.putalpha(alpha_crop)

    info = {
        "candidate_ann_mode": ann_mode,
        "candidate_source_image": str(img_path),
        "candidate_box_xyxy_original": [x1, y1, x2, y2],
        "candidate_box_size": [bw, bh],
        "candidate_mask_source": mask_source,
        "candidate_mask_frac": mask_frac,
        "candidate_coco_image_id": img.get("id"),
        "candidate_coco_ann_id": ann.get("id"),
    }

    return crop, info, "ok"


def target_box_from_query(
    q: Dict[str, Any],
    coco_index: Dict[str, Any],
    bg_w: int,
    bg_h: int,
) -> Tuple[Optional[List[float]], str, Optional[Dict[str, Any]]]:
    q_img, q_ann, q_mode = lookup_coco_ann(q, coco_index)

    if q_ann is not None and q_ann.get("bbox") is not None:
        box_orig = xywh_to_xyxy(q_ann["bbox"])
        sx = BOOTPLACE_W / float(bg_w)
        sy = BOOTPLACE_H / float(bg_h)
        box = scale_box_xyxy(box_orig, sx, sy)
        return clamp_xyxy(box, BOOTPLACE_W, BOOTPLACE_H), f"coco_bbox::{q_mode}", q_ann

    if q.get("bbox") is not None:
        box_orig = xywh_to_xyxy(q["bbox"])
        sx = BOOTPLACE_W / float(bg_w)
        sy = BOOTPLACE_H / float(bg_h)
        box = scale_box_xyxy(box_orig, sx, sy)
        return clamp_xyxy(box, BOOTPLACE_W, BOOTPLACE_H), "record_bbox", None

    return None, "missing_target_bbox", None


def existing_scene_boxes(
    q_image_id: Any,
    q_ann_exclude: Optional[Dict[str, Any]],
    coco_index: Dict[str, Any],
    bg_w: int,
    bg_h: int,
    min_existing_size: int = 2,
) -> List[List[float]]:
    anns = coco_index["anns_by_image"].get(str(q_image_id), [])
    exclude_id = None if q_ann_exclude is None else str(q_ann_exclude.get("id"))

    sx = BOOTPLACE_W / float(bg_w)
    sy = BOOTPLACE_H / float(bg_h)

    out = []
    for ann in anns:
        if exclude_id is not None and str(ann.get("id")) == exclude_id:
            continue
        if ann.get("bbox") is None:
            continue
        box = scale_box_xyxy(xywh_to_xyxy(ann["bbox"]), sx, sy)
        box = clamp_xyxy(box, BOOTPLACE_W, BOOTPLACE_H)
        if box is None:
            continue
        if box[2] - box[0] < min_existing_size or box[3] - box[1] < min_existing_size:
            continue
        out.append(box)
    return out


def select_balanced_queries(reference_rows: List[Dict[str, Any]], n: int, seed: int):
    rng = random.Random(seed)
    rows = list(reference_rows)
    rng.shuffle(rows)

    by_source = defaultdict(list)
    for item in rows:
        by_source[source_of_record(item.get("query", {}))].append(item)

    sources = sorted(by_source.keys(), key=lambda s: (len(by_source[s]), s))
    selected = []
    selected_keys = set()

    while len(selected) < n:
        progressed = False
        for src in sources:
            if len(selected) >= n:
                break
            while by_source[src]:
                item = by_source[src].pop()
                q = item.get("query", {})
                key = (str(q.get("image_id")), str(q.get("annotation_id")))
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                progressed = True
                break
        if not progressed:
            break

    return selected


def make_request_id(method: str, idx: int, q: Dict[str, Any], cand: Dict[str, Any]):
    return safe_stem(
        f"{method}_{idx:05d}_q{q.get('image_id')}_{q.get('annotation_id')}_"
        f"c{cand.get('image_id')}_{cand.get('annotation_id')}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--coco-json", type=Path, default=DEFAULT_COCO_JSON)
    ap.add_argument("--exp1-root", type=Path, default=DEFAULT_EXP1_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--top-ks", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--candidate-rank", type=int, default=1)
    ap.add_argument("--max-rank-search", type=int, default=10, help="If candidate-rank crop fails, search top candidates up to this rank.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-crop-w", type=int, default=64)
    ap.add_argument("--min-crop-h", type=int, default=64)
    ap.add_argument("--edge-margin", type=int, default=4)
    ap.add_argument("--min-mask-frac", type=float, default=0.05)
    ap.add_argument("--require-segmentation", action="store_true")
    ap.add_argument("--feather-radius", type=float, default=1.5)
    ap.add_argument("--crop-to-mask", action="store_true", default=True)
    ap.add_argument("--no-crop-to-mask", dest="crop_to_mask", action="store_false")

    # Mask generation. COCO JSONs have boxes only; this uses SAM2 from the bbox prompt.
    ap.add_argument("--mask-source", choices=["auto", "coco", "sam2", "rect"], default="sam2",
                    help="sam2: generate masks from COCO bbox prompts. auto: COCO seg if present else SAM2. rect: no SAM2.")
    ap.add_argument("--sam2-root", type=Path, default=Path("/path/to/sealect_root/sam2"),
                    help="Existing local SAM2 repo root.")
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path("/path/to/sealect_root/sam2_checkpoints/sam2_hiera_large.pt"),
                    help="Existing local SAM2 checkpoint path.")
    ap.add_argument("--sam2-model-cfg", type=str, default=SAM2_1_LARGE_CFG,
                    help="SAM2 config path. Default matches sam2_hiera_large.pt.")
    ap.add_argument("--sam2-device", type=str, default="cuda")
    ap.add_argument("--sam2-cache-dir", type=Path, default=None,
                    help="Cache generated SAM2 masks. Defaults to <out-root>/sam2_mask_cache.")
    ap.add_argument("--sam2-min-score", type=float, default=0.0)
    ap.add_argument("--sam2-multimask", action="store_true", default=True)
    ap.add_argument("--sam2-singlemask", dest="sam2_multimask", action="store_false")

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.sam2_cache_dir is None:
        args.sam2_cache_dir = args.out_root / "sam2_mask_cache"

    if args.mask_source in {"auto", "sam2"} or args.require_segmentation:
        ensure_sam2_assets(args)

    image_index = build_image_index(args.image_root)
    coco_index = build_coco_index(args.coco_json)

    if args.out_root.exists() and args.overwrite:
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_summary = {
        "script": "02_make_bootplace_eval_requests_clean_masks.py",
        "coco_json": str(args.coco_json),
        "image_root": str(args.image_root),
        "out_root": str(args.out_root),
        "min_crop_w": args.min_crop_w,
        "min_crop_h": args.min_crop_h,
        "edge_margin": args.edge_margin,
        "require_segmentation": args.require_segmentation,
        "mask_source": args.mask_source,
        "sam2_root": str(args.sam2_root),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_model_cfg": args.sam2_model_cfg,
        "note": "Clean candidate crops, generated SAM2 masks when COCO segmentations are missing, existing boxes metadata for overlap filtering.",
    }

    for top_k in args.top_ks:
        method_results, missing = load_method_results(args.exp1_root, top_k)
        if "sealect" not in method_results:
            raise RuntimeError(f"SEALECT results required for balanced query selection at K={top_k}")

        selected = select_balanced_queries(method_results["sealect"]["rows"], args.num_queries, args.seed)
        summary_k = {"missing": missing, "selected_queries": len(selected), "methods": {}}

        for method, md in method_results.items():
            if method not in GENERATE_ONLY:      # sealect only used for query selection above
                continue
            out_base = args.out_root / f"k{top_k}" / method
            test_root = out_base / "test"

            for name in [
                "backgrounds", "backgrounds_single",
                "objects", "objects_single",
                "location", "location_single",
                "metadata", "viz_requests"
            ]:
                (test_root / name).mkdir(parents=True, exist_ok=True)

            rows_by_key = {
                (str(item.get("query", {}).get("image_id")), str(item.get("query", {}).get("annotation_id"))): item
                for item in md["rows"]
            }

            coco_images = []
            coco_anns = []
            failures = Counter()
            crop_status = Counter()
            target_status = Counter()
            used_rank_counts = Counter()
            written = 0

            for selected_item in tqdm(selected, desc=f"k{top_k} {method}"):
                q = selected_item.get("query", {})
                qkey = (str(q.get("image_id")), str(q.get("annotation_id")))
                item = rows_by_key.get(qkey)

                if item is None:
                    failures["missing_query_in_method"] += 1
                    continue

                top_list = item.get("topk", [])
                if not top_list:
                    failures["empty_topk"] += 1
                    continue

                q_img, _, _ = lookup_coco_ann(q, coco_index)
                bg_path = resolve_coco_image_path(q_img, q, args.image_root, image_index)
                if bg_path is None:
                    failures["missing_background"] += 1
                    continue

                try:
                    bg = Image.open(bg_path).convert("RGB")
                except Exception:
                    failures["bad_background"] += 1
                    continue

                # Try candidate-rank first, then subsequent candidates until a clean crop is found.
                crop = None
                crop_info = None
                chosen_cand = None
                final_status = None
                used_rank = None
                start_rank = max(1, int(args.candidate_rank))
                max_rank = min(len(top_list), max(start_rank, int(args.max_rank_search)))

                for rank in range(start_rank, max_rank + 1):
                    cand = top_list[rank - 1]
                    crop_try, info_try, status = make_rgba_from_coco_ann(
                        cand,
                        coco_index,
                        args.image_root,
                        image_index,
                        min_w=args.min_crop_w,
                        min_h=args.min_crop_h,
                        edge_margin=args.edge_margin,
                        min_mask_frac=args.min_mask_frac,
                        require_segmentation=args.require_segmentation,
                        feather_radius=args.feather_radius,
                        crop_to_mask=args.crop_to_mask,
                        args=args,
                    )
                    crop_status[status] += 1

                    if crop_try is not None:
                        crop = crop_try
                        crop_info = info_try
                        chosen_cand = cand
                        final_status = status
                        used_rank = rank
                        used_rank_counts[str(rank)] += 1
                        break

                if crop is None:
                    failures["no_clean_candidate_in_top_ranks"] += 1
                    continue

                target_box, tstatus, q_ann = target_box_from_query(q, coco_index, bg.width, bg.height)
                target_status[tstatus] += 1
                if target_box is None:
                    failures["bad_target_box"] += 1
                    continue

                existing_boxes = existing_scene_boxes(q.get("image_id"), q_ann, coco_index, bg.width, bg.height)

                request_id = make_request_id(method, written, q, chosen_cand)

                bg_resized = bg.resize((BOOTPLACE_W, BOOTPLACE_H), Image.BICUBIC)
                bg_resized.save(test_root / "backgrounds" / f"{request_id}.png")
                bg_resized.save(test_root / "backgrounds_single" / f"{request_id}.png")

                crop.save(test_root / "objects_single" / f"{request_id}.png")
                obj_subdir = test_root / "objects" / request_id
                obj_subdir.mkdir(parents=True, exist_ok=True)
                crop.save(obj_subdir / "car_00000.png")

                write_box_txt(test_root / "location_single" / f"{request_id}.txt", target_box)
                loc_subdir = test_root / "location" / request_id
                loc_subdir.mkdir(parents=True, exist_ok=True)
                write_box_txt(loc_subdir / "car_00000.txt", target_box)

                # Request debug visualization: existing boxes green, target/insertion prior blue.
                viz = bg_resized.copy()
                draw = ImageDraw.Draw(viz)
                for ex in existing_boxes:
                    draw.rectangle(ex, outline="green", width=3)
                draw.rectangle(target_box, outline="blue", width=4)
                viz.save(test_root / "viz_requests" / f"{request_id}.png")

                x1, y1, x2, y2 = target_box
                coco_images.append({
                    "id": written + 1,
                    "file_name": f"{request_id}.png",
                    "width": BOOTPLACE_W,
                    "height": BOOTPLACE_H,
                    "source": source_of_record(q),
                    "query_image_id": q.get("image_id"),
                    "query_annotation_id": q.get("annotation_id"),
                    "candidate_image_id": chosen_cand.get("image_id"),
                    "candidate_annotation_id": chosen_cand.get("annotation_id"),
                    "method": method,
                    "top_k": top_k,
                    "candidate_rank_requested": args.candidate_rank,
                    "candidate_rank_used": int(used_rank) if used_rank is not None else None,
                    "target_box_source": tstatus,
                    "candidate_crop_status": final_status,
                    **(crop_info or {}),
                })

                coco_anns.append({
                    "id": written + 1,
                    "image_id": written + 1,
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })

                save_json({
                    "request_id": request_id,
                    "method": method,
                    "top_k": top_k,
                    "query": q,
                    "candidate": chosen_cand,
                    "target_box_xyxy": target_box,
                    "existing_boxes_xyxy": existing_boxes,
                    "background_path": str(bg_path),
                    "candidate_crop_info": crop_info,
                    "candidate_crop_status": final_status,
                    "target_status": tstatus,
                }, test_root / "metadata" / f"{request_id}.json")

                written += 1

            save_json({
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [{"id": 0, "name": "car"}],
            }, test_root / "annotations.json")

            summary_k["methods"][method] = {
                "method_dir": md["method_dir"],
                "input_path": str(md["path"]),
                "out_path": str(out_base),
                "written": written,
                "failures": dict(failures),
                "crop_status": dict(crop_status),
                "target_status": dict(target_status),
                "used_rank_counts": dict(used_rank_counts),
            }

            print(f"\nK={top_k} method={method} written={written}")
            print("failures:", dict(failures))
            print("crop_status:", dict(crop_status))
            print("target_status:", dict(target_status))
            print("used_rank_counts:", dict(used_rank_counts))

        all_summary[f"k{top_k}"] = summary_k

    save_json(all_summary, args.out_root / "build_eval_requests_summary.json")
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
