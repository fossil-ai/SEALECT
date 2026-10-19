#!/usr/bin/env python3
"""
04_prepare_diffharmony_inputs.py

Convert PoseNet/BootPlace outputs into DiffHarmony iHarmony-style inference inputs.

Expected input per split:
  bootplace_outputs_clean/k20/sealect/composites/<request_id>.png
  bootplace_outputs_clean/k20/sealect/placement_metadata.json
  bootplace_eval_requests_clean/k20/sealect/test/objects_single/<request_id>.png
  bootplace_eval_requests_clean/k20/sealect/test/backgrounds_single/<request_id>.png

Writes per split:
  diffharmony_inputs/k20/sealect/HPose/composite_images/<request_id>_1_1.png
  diffharmony_inputs/k20/sealect/HPose/masks/<request_id>_1.png
  diffharmony_inputs/k20/sealect/HPose/real_images/<request_id>.jpg
  diffharmony_inputs/k20/sealect/test.jsonl

DiffHarmony dataset convention inferred from repo:
  composite image name: <id>_1_1.png
  mask name:            <id>_1.png
  real image name:      <id>.jpg
  test.jsonl row:       {"file_name": "HPose/composite_images/<id>_1_1.png", "text": ""}

The real_images entry is only needed because the dataset parser expects it. For inference,
we use the original background as the closest non-composited image.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageFilter

METHODS_DEFAULT = ["random", "clip", "dino", "prior_only", "dino_prior", "sealect"]
TOPKS_DEFAULT = [20, 100]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def stem_key(x: Any) -> str:
    return Path(str(x)).stem


def find_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    if folder.exists():
        for p in folder.iterdir():
            if p.is_file() and p.stem == stem:
                return p
    return None


def clamp_box_xyxy(box: List[float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def make_inserted_object_mask(
    obj_path: Path,
    paste_box_xyxy: List[float],
    image_size: Tuple[int, int],
    feather_radius: float = 0.0,
    threshold: int = 8,
) -> Image.Image:
    """Create full-image binary mask for inserted object using object alpha + paste box."""
    img_w, img_h = image_size
    box = clamp_box_xyxy(paste_box_xyxy, img_w, img_h)
    mask_full = Image.new("L", (img_w, img_h), 0)
    if box is None:
        return mask_full

    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 1 or box_h <= 1:
        return mask_full

    obj = Image.open(obj_path).convert("RGBA")
    alpha = obj.getchannel("A")

    # Match BootPlace/test_v2_safe_compose composition: object is resized into paste_bbox.
    alpha = alpha.resize((box_w, box_h), Image.BICUBIC)
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    alpha = alpha.point(lambda px: 255 if px > threshold else 0)
    mask_full.paste(alpha, (x1, y1))
    return mask_full


def copy_as_rgb(src: Path, dst: Path, size: Optional[Tuple[int, int]] = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    # DiffHarmony parser expects .jpg real_images, but composite/mask stay png.
    if dst.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(dst, quality=95)
    else:
        img.save(dst)


def prepare_split(
    k: int,
    method: str,
    req_root: Path,
    bootplace_out_root: Path,
    diffharmony_out_root: Path,
    include_rejected: bool,
    feather_radius: float,
    overwrite: bool,
) -> Dict[str, Any]:
    req_split = req_root / f"k{k}" / method / "test"
    out_split = bootplace_out_root / f"k{k}" / method
    dh_split = diffharmony_out_root / f"k{k}" / method

    composite_dir = out_split / "composites"
    placement_meta_path = out_split / "placement_metadata.json"
    postprocess_report_path = out_split / "postprocess_overlap_viz_report.json"
    obj_dir = req_split / "objects_single"
    bg_dir = req_split / "backgrounds_single"

    hroot = dh_split / "HPose"
    comp_out = hroot / "composite_images"
    mask_out = hroot / "masks"
    real_out = hroot / "real_images"
    test_jsonl = dh_split / "test.jsonl"

    summary = {
        "k": k,
        "method": method,
        "req_split": str(req_split),
        "bootplace_split": str(out_split),
        "diffharmony_split": str(dh_split),
        "written": 0,
        "skipped_rejected": 0,
        "missing_composite": 0,
        "missing_object": 0,
        "missing_background": 0,
        "missing_box": 0,
        "errors": [],
    }

    if not placement_meta_path.exists():
        summary["errors"].append(f"Missing {placement_meta_path}")
        return summary
    if not composite_dir.exists():
        summary["errors"].append(f"Missing {composite_dir}")
        return summary

    if overwrite and dh_split.exists():
        shutil.rmtree(dh_split)
    for d in [comp_out, mask_out, real_out]:
        d.mkdir(parents=True, exist_ok=True)

    rows = load_json(placement_meta_path)
    if not isinstance(rows, list):
        summary["errors"].append(f"Expected list in {placement_meta_path}")
        return summary

    rejected = set()
    if postprocess_report_path.exists():
        try:
            rep = load_json(postprocess_report_path)
            for item in rep.get("items", []):
                if item.get("rejected_overlap"):
                    rejected.add(stem_key(item.get("request_id") or item.get("file_name")))
        except Exception as e:
            summary["errors"].append(f"Could not read postprocess report: {e}")

    jsonl_rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("file_name") or row.get("request_id") or row.get("image_id")
        if not name:
            continue
        request_id = stem_key(name)

        if (not include_rejected) and request_id in rejected:
            summary["skipped_rejected"] += 1
            continue

        box = row.get("paste_bbox_xyxy") or row.get("bbox_xyxy") or row.get("box_xyxy")
        if not box or len(box) < 4:
            summary["missing_box"] += 1
            continue

        comp_path = find_image_by_stem(composite_dir, request_id)
        obj_path = find_image_by_stem(obj_dir, request_id)
        bg_path = find_image_by_stem(bg_dir, request_id)

        if comp_path is None:
            summary["missing_composite"] += 1
            continue
        if obj_path is None:
            summary["missing_object"] += 1
            continue
        if bg_path is None:
            summary["missing_background"] += 1
            continue

        try:
            comp_img = Image.open(comp_path).convert("RGB")
            img_w, img_h = comp_img.size

            # DiffHarmony naming convention.
            # dh_comp_name = f"{request_id}_1_1.png"
            # dh_mask_name = f"{request_id}_1.png"
            # dh_real_name = f"{request_id}.jpg"

            # DiffHarmony naming convention.
            base_id = request_id.split("_q")[0]

            dh_comp_name = f"{request_id}_1_1.png"
            dh_mask_name = f"{base_id}.png"
            dh_real_name = f"{base_id}.jpg"

            comp_dst = comp_out / dh_comp_name
            mask_dst = mask_out / dh_mask_name
            real_dst = real_out / dh_real_name

            comp_img.save(comp_dst)
            mask = make_inserted_object_mask(
                obj_path=obj_path,
                paste_box_xyxy=[float(v) for v in box[:4]],
                image_size=(img_w, img_h),
                feather_radius=feather_radius,
            )
            mask.save(mask_dst)
            copy_as_rgb(bg_path, real_dst, size=(img_w, img_h))

            jsonl_rows.append({
                "file_name": f"HPose/composite_images/{dh_comp_name}",
                "text": "",
            })
            summary["written"] += 1
        except Exception as e:
            summary["errors"].append(f"{request_id}: {e}")

    save_jsonl(jsonl_rows, test_jsonl)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-root", type=Path, default=Path("/path/to/sealect_root/experiments/experiment5_1/bootplace_eval_requests_clean"))
    ap.add_argument("--bootplace-out-root", type=Path, default=Path("/path/to/sealect_root/experiments/experiment5_1/bootplace_outputs_clean"))
    ap.add_argument("--out-root", type=Path, default=Path("/path/to/sealect_root/experiments/experiment5_1/diffharmony_inputs"))
    ap.add_argument("--top-ks", type=int, nargs="+", default=TOPKS_DEFAULT)
    ap.add_argument("--methods", nargs="+", default=METHODS_DEFAULT)
    ap.add_argument("--include-rejected", action="store_true", help="Include overlap-rejected composites too. Default: skip rejected overlap items.")
    ap.add_argument("--feather-radius", type=float, default=0.0, help="Optional blur radius for harmonization mask before thresholding.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for k in args.top_ks:
        for method in args.methods:
            print(f"Preparing DiffHarmony input for k{k}/{method}")
            s = prepare_split(
                k=k,
                method=method,
                req_root=args.req_root,
                bootplace_out_root=args.bootplace_out_root,
                diffharmony_out_root=args.out_root,
                include_rejected=args.include_rejected,
                feather_radius=args.feather_radius,
                overwrite=args.overwrite,
            )
            summaries.append(s)
            print(json.dumps({
                "k": s["k"],
                "method": s["method"],
                "written": s["written"],
                "skipped_rejected": s["skipped_rejected"],
                "missing_composite": s["missing_composite"],
                "missing_object": s["missing_object"],
                "missing_background": s["missing_background"],
                "missing_box": s["missing_box"],
                "errors": len(s["errors"]),
            }, indent=2))

    summary_path = args.out_root / "prepare_diffharmony_inputs_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path("/path/to/sealect_root/experiments/experiment5_1")

    REQ_ROOT = ROOT / "bootplace_eval_requests_clean"
    BOOTPLACE_OUT_ROOT = ROOT / "bootplace_outputs_clean"
    DIFFHARMONY_OUT_ROOT = ROOT / "diffharmony_inputs"

    OVERWRITE = True
    INCLUDE_REJECTED = False
    FEATHER_RADIUS = 2.0

    TOPKS = [20, 100]
    METHODS = [
        "random",
        "clip",
        "dino",
        "prior_only",
        "dino_prior",
        "sealect",
    ]

    for k in TOPKS:
        for method in METHODS:
            print("=" * 80)
            print(f"Preparing DiffHarmony inputs: k{k}/{method}")

            try:
                summary = prepare_split(
                    k=k,
                    method=method,
                    req_root=REQ_ROOT,
                    bootplace_out_root=BOOTPLACE_OUT_ROOT,
                    diffharmony_out_root=DIFFHARMONY_OUT_ROOT,
                    include_rejected=INCLUDE_REJECTED,
                    feather_radius=FEATHER_RADIUS,
                    overwrite=OVERWRITE,
                )
                print(summary)

            except Exception as e:
                print(f"[ERROR] k{k}/{method}: {e}")

    print("\nDone.")