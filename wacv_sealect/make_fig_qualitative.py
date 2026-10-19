#!/usr/bin/env python3
"""
Regenerate the qualitative retrieval figure (paper Fig. 4), cleanly.

New layout (per query scene, one figure): the QUERY SCENE on top, then one ROW PER METHOD showing
that method's top-5 *retrieved candidate crops*. No confusing multi-query top row; current baseline
names. We render several scenes to `sec/images/qual_new/` and pick the best for the paper.

Public-only: restricted (non-public) candidates are skipped (top-5 among shareable crops), and only public query
scenes are used. Crops come from the scene-pose library; scenes from the merged image root.

Usage:
  python3 make_fig_qualitative.py                 # auto-pick N diverse scenes
  python3 make_fig_qualitative.py --n 12
  python3 make_fig_qualitative.py --scenes 36714_116615 3868_5021   # specific backgrounds
"""
import argparse, json, os, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

E1 = "/path/to/sealect_root/experiments/experiment1/exp1_retrieval_quality"
CROP_ROOT = "/path/to/sealect_root/experiments/dataset_decide"
CROP_DIR = "merged_scene_pose_library_out_grounded_sam2/crops"
IMG_ROOT = "/path/to/datasets/data/merged/images"
COCO = os.path.join(os.path.dirname(IMG_ROOT), "annotations_fixed.json")
RESTRICTED = {"RESTRICTED_A", "RESTRICTED_B"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "fig4_candidates")     # dedicated folder for the candidate figures
CTX_MARGIN = 0.30                                    # context padding around the vessel bbox

# (display name, retrieval-results dir). Edit this list to change which methods/rows appear.
METHODS = [
    ("Random",   "random_k20"),
    ("DINO",     "dino_k20"),
    ("GALA",     "gala_render_k20"),
    ("DiscoFOS", "discofos_render_k20"),
    ("SEALECT",  "sealect_dinofeat_k20"),
]
K = 5
CELL_W, CELL_H = 220, 165          # every shown crop is resized to exactly this (fill, no padding)
DEDUP_COS = 0.86                    # skip a candidate whose crop is >this cosine-similar to one already shown
DRAW_QUERY_BOX = True               # draw the held-out-vessel box on the query scene (disable via --no-box)


def load():
    """Return per-method {(image_id,annotation_id): record}, and image_id->image_name."""
    by_method, name_of = {}, {}
    for disp, d in METHODS:
        recs = json.load(open(f"{E1}/{d}/topk_20/retrieval_results.json"))
        idx = {}
        for r in recs:
            q = r["query"]
            idx[(q["image_id"], q["annotation_id"])] = r
            for e in [q] + r.get("topk", []):
                if e.get("image_id") is not None and e.get("image_name"):
                    name_of.setdefault(e["image_id"], e["image_name"])
        by_method[disp] = idx
    return by_method, name_of


def crop_path(cand, name_of):
    """Resolve a candidate's cropped-vessel image path, or None."""
    p = cand.get("crop_path")
    if p:
        fp = os.path.join(CROP_ROOT, p)
        if os.path.exists(fp):
            return fp
    src = cand.get("source")
    name = cand.get("image_name") or name_of.get(cand.get("image_id"))
    if src and name:
        fp = os.path.join(CROP_ROOT, CROP_DIR,
                          f"{src}__{os.path.splitext(name)[0]}__ann{cand['annotation_id']}.jpg")
        if os.path.exists(fp):
            return fp
    return None


def topk_public_crops(record, name_of, coco_bbox, coco_imgs, k=K):
    """Top-k shareable, visually DISTINCT candidate RECORDS for one method on one query. Skips
    restricted (non-public), the same source frame, near-duplicate crops (correlated video frames), and any candidate
    without a resolvable bbox+source (needed for the context window). Dedup uses the fast tight crop;
    the figure itself renders a source-image context window via window_crop()."""
    out, sigs, used_img = [], [], set()
    for c in record.get("topk", []):
        if (c.get("source") in RESTRICTED) or (str(c.get("candidate_id", "")).split("_")[0] in RESTRICTED):
            continue
        if c.get("image_id") in used_img:
            continue
        if (c.get("image_id"), c.get("annotation_id")) not in coco_bbox:
            continue
        ii = coco_imgs.get(c.get("image_id"))
        if not ii or not os.path.exists(os.path.join(IMG_ROOT, ii[2])):
            continue
        iw, ih, _ = ii
        bx, by, bw, bh = coco_bbox[(c.get("image_id"), c.get("annotation_id"))]
        # Skip corrupt entries that render as blank upscaled smears: malformed ABOships
        # thumbnails (tiny "image" or a bbox larger than the image) and vessels too small
        # to show as anything but a blur.
        # ABOships is excluded from the qualitative figure: its COCO metadata is systematically
        # corrupt (recorded image dims disagree with the real 1280x720 files, and bboxes are in the
        # phantom coordinate space), so window_crop lands on the wrong region (usually sky). The
        # remaining clauses drop the leftover malformed/too-small crops from the other datasets.
        if (c.get("source") == "ABOships"
                or iw < 100 or ih < 100 or bw > iw + 1 or bh > ih + 1 or min(bw, bh) < 30):
            continue
        fp = crop_path(c, name_of)
        if not fp:
            continue
        s = sig(fp)
        if any(float(s @ t) > DEDUP_COS for t in sigs):
            continue
        out.append(c); sigs.append(s); used_img.add(c.get("image_id"))
        if len(out) >= k:
            break
    return out


def scene_path(rec):
    q = rec["query"]
    fp = os.path.join(IMG_ROOT, q.get("image_name", ""))
    return fp if os.path.exists(fp) else None


def load_coco():
    """(image_id,ann_id) -> bbox[x,y,w,h]; image_id -> (w,h,file_name)."""
    d = json.load(open(COCO))
    bbox = {(a["image_id"], a["id"]): a["bbox"] for a in d["annotations"]}
    imgs = {i["id"]: (i.get("width"), i.get("height"), i["file_name"]) for i in d["images"]}
    return bbox, imgs


def window_crop(cand, coco_bbox, coco_imgs, ar=CELL_W / CELL_H, margin=CTX_MARGIN):
    """Crop a fixed-aspect (ar) window from the SOURCE image around the vessel: vessel + surrounding
    water, so it fills the cell with context (no white padding) without zooming into the vessel."""
    bb = coco_bbox.get((cand["image_id"], cand["annotation_id"]))
    ii = coco_imgs.get(cand["image_id"])
    if bb is None or ii is None:
        return None
    iw, ih, fn = ii
    src = os.path.join(IMG_ROOT, fn)
    if not os.path.exists(src):
        return None
    x, y, w, h = bb
    cx, cy = x + w / 2, y + h / 2
    mw, mh = w * (1 + 2 * margin), h * (1 + 2 * margin)
    ww, wh = (mh * ar, mh) if mw / mh < ar else (mw, mw / ar)   # extend shorter side to hit ar
    if ww > iw:
        ww, wh = iw, iw / ar
    if wh > ih:
        wh, ww = ih, ih * ar
    x0 = min(max(0.0, cx - ww / 2), iw - ww)
    y0 = min(max(0.0, cy - wh / 2), ih - wh)
    im = Image.open(src).convert("RGB").crop((int(x0), int(y0), int(x0 + ww), int(y0 + wh)))
    return im.resize((CELL_W, CELL_H), Image.LANCZOS)


def sig(path):
    """Small normalized descriptor for near-duplicate detection (correlated video frames)."""
    a = np.asarray(Image.open(path).convert("L").resize((12, 12)), dtype=float).flatten()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def render(scene_id, by_method, name_of, coco_bbox, coco_imgs, out_png):
    qi, qa = map(int, scene_id.split("_"))
    ref = by_method["SEALECT"].get((qi, qa))
    if ref is None:
        return False
    sp = scene_path(ref)
    if sp is None:
        return False
    rows = {}
    for disp, _ in METHODS:
        r = by_method[disp].get((qi, qa))
        rows[disp] = topk_public_crops(r, name_of, coco_bbox, coco_imgs) if r else []
    if any(len(v) < K for v in rows.values()):        # only fully-populated scenes
        return False

    nm = len(METHODS)
    fig = plt.figure(figsize=(K * 1.7, 1.7 + nm * 1.12))
    gs = fig.add_gridspec(nm + 1, K, height_ratios=[1.7] + [1] * nm,
                          hspace=0.10, wspace=0.05,
                          left=0.13, right=0.995, top=0.95, bottom=0.005)
    # query scene: a modest centered thumbnail (columns 1..3 of 5), not a full-width banner
    axq = fig.add_subplot(gs[0, 1:K - 1])
    q = ref["query"]
    im = Image.open(sp).convert("RGB"); W, H = im.size
    axq.imshow(im); axq.set_xticks([]); axq.set_yticks([])
    for s in axq.spines.values(): s.set_edgecolor("#333"); s.set_linewidth(1.2)
    axq.set_title("Query scene", fontsize=11, fontweight="bold", pad=4)
    if DRAW_QUERY_BOX and q.get("center_x_norm") is not None and q.get("area_frac"):
        side = (q["area_frac"] ** 0.5)
        bw, bh = side * W, side * H
        axq.add_patch(patches.Rectangle((q["center_x_norm"] * W - bw / 2, q["center_y_norm"] * H - bh / 2),
                                        bw, bh, fill=False, edgecolor="#e11d48", linewidth=2))
    # method rows
    for ri, (disp, _) in enumerate(METHODS, start=1):
        for ci in range(K):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(window_crop(rows[disp][ci], coco_bbox, coco_imgs))
            ax.set_xticks([]); ax.set_yticks([])
            hi = (disp == "SEALECT")
            for s in ax.spines.values():
                s.set_edgecolor("#2563eb" if hi else "#bbb"); s.set_linewidth(2.0 if hi else 0.6)
            if ci == 0:
                ax.set_ylabel(disp, fontsize=11, rotation=0, ha="right", va="center",
                              labelpad=8, fontweight="bold" if hi else "normal",
                              color="#2563eb" if hi else "black")
            if ri == 1:
                ax.set_title(f"rank {ci+1}", fontsize=8, color="#666", pad=2)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def pick_scenes(by_method, name_of, coco_bbox, coco_imgs, n, source=None):
    """Diverse public scenes with fully-populated rows: at most one per (source, viewpoint) bucket,
    scanned in shuffled order so datasets/contexts are spread rather than front-loaded. If `source`
    is given, restrict to that dataset (e.g. 'Kolomverse') and relax the per-bucket cap accordingly."""
    import random
    keys = [k for k, r in by_method["SEALECT"].items()
            if r["query"].get("source") not in RESTRICTED
            and r["query"].get("source") != "ABOships"   # broken scene metadata (see topk_public_crops)
            and (source is None or r["query"].get("source") == source)]
    random.Random(7).shuffle(keys)
    seen, chosen = collections.defaultdict(int), []
    for cap in range(1, 40):                # progressively relax per-bucket cap until n scenes found
        for (qi, qa) in keys:
            if len(chosen) >= n:
                break
            sid = f"{qi}_{qa}"
            if sid in chosen:
                continue
            rec = by_method["SEALECT"][(qi, qa)]
            q = rec["query"]
            bucket = (q.get("source"), q.get("viewpoint_type"))
            if seen[bucket] >= cap:
                continue
            if scene_path(rec) and all(
                    len(topk_public_crops(by_method[d].get((qi, qa)) or {}, name_of, coco_bbox, coco_imgs)) >= K
                    for d, _ in METHODS):
                chosen.append(sid); seen[bucket] += 1
                print(f"    picked {sid}  [{q.get('source')} / {q.get('viewpoint_type')} / {q.get('scene_type')}]")
        if len(chosen) >= n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--source", default=None,
                    help="Restrict auto-picked scenes to this dataset source, e.g. 'Kolomverse'.")
    ap.add_argument("--out", default=OUT_DIR,
                    help="Output directory for the candidate figures (default: fig4_candidates/).")
    ap.add_argument("--no-box", action="store_true", help="Do not draw the held-out-vessel box on the query scene.")
    args = ap.parse_args()
    global DRAW_QUERY_BOX
    if args.no_box:
        DRAW_QUERY_BOX = False
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    print("loading retrieval results + COCO...")
    by_method, name_of = load()
    coco_bbox, coco_imgs = load_coco()
    scenes = args.scenes or pick_scenes(by_method, name_of, coco_bbox, coco_imgs, args.n, source=args.source)
    print(f"rendering {len(scenes)} scenes -> {out_dir}")
    made = 0
    for sid in scenes:
        out = os.path.join(out_dir, f"qual_{sid}.png")
        if render(sid, by_method, name_of, coco_bbox, coco_imgs, out):
            made += 1; print(f"  wrote qual_{sid}.png")
        else:
            print(f"  [skip] {sid} (incomplete rows / missing scene)")
    print(f"done: {made} figures. Pick the best and set it as Fig. 4.")


if __name__ == "__main__":
    main()
