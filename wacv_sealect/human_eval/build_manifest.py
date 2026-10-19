#!/usr/bin/env python3
"""
build_manifest.py  --  Build the blind human-eval study site for SEALECT vs DINO.

Produces a fully self-contained, deployable static site under ./site/ :
    site/index.html        the rating app (blind; public scene list embedded, NO method labels)
    site/images/*.jpg       the two composites per scene, named neutrally {scene_id}_a/_b.jpg

and a PRIVATE key kept OUTSIDE the site (never uploaded):
    manifest_private.json   scene_id -> {a_method, b_method, provenance, black_frac...}

score.py joins rater CSVs against manifest_private.json to compute the SEALECT preference.

Design guarantees enforced here:
  * HARD restricted constraint: two-layer filter (drop if the query BACKGROUND source OR the
    inserted CANDIDATE source is restricted) + a final per-file re-audit -> assert 0 restricted.
  * Blindness: the shipped site has no method labels in any file name, path, or embedded datum.
    The slot(a/b)->method mapping exists only in manifest_private.json.
  * No differential rendering confound: we measure the black-pixel fraction of each composite
    and record the per-pair delta (all near zero -- dark regions live in the shared background).

Run:  python3 build_manifest.py            (defaults reproduce the 171-pair public study)
      python3 build_manifest.py --max 150  (stratified subsample)
No GPU needed. Uses the project venv (has PIL).
"""
import argparse, glob, json, os, re, random, collections, hashlib, sys
import numpy as np
from PIL import Image, ImageDraw
try:
    from scipy import ndimage
except Exception:
    ndimage = None
try:
    import cv2
except Exception:
    cv2 = None

# --------------------------------------------------------------------------------------
# Fixed data locations (verified to exist on this machine)
# --------------------------------------------------------------------------------------
ROOT = "/path/to/sealect_root"
E1   = f"{ROOT}/experiments/experiment1/exp1_retrieval_quality"
COMP = f"{ROOT}/experiments/experiment5_1/diffharmony_outputs/k20"
# DiffHarmony inputs hold a per-composite mask of the INSERTED vessel (960x540), keyed by
# {method}_{idx}; we use it to outline the boat under judgement in each composite.
MASKS = f"{ROOT}/experiments/experiment5_1/diffharmony_inputs/k20"

# Retrieval JSONs give us image_id -> {source, dataset}. Loading sealect+dino covers every
# image_id that can appear in either method's composites; random is added for completeness.
RETRIEVAL_JSONS = {
    "sealect": f"{E1}/sealect_dinofeat_k20/topk_20/retrieval_results.json",
    "dino":    f"{E1}/dino_k20/topk_20/retrieval_results.json",
    "random":  f"{E1}/random_k20/topk_20/retrieval_results.json",
}
METHOD_A = "sealect"                     # the method under test (always arm A)
# METHOD_B (the opponent: "dino" or "random") and the output paths are set in main() from --vs.

RESTRICTED = {"RESTRICTED_B", "RESTRICTED_A"}              # access-restricted in-house UAV imagery -- never show
PUBLIC_DATASETS = {"ABOships", "Kolomverse", "LARS", "SMD", "SeaDronesSee", "SeaShips"}

# {method}_{idx}_q{QIMG}_{QANN}_c{CIMG}_{CANN}_1_1.png
FNAME_RE = re.compile(r"^(?P<m>[a-z_]+)_(?P<idx>\d+)_q(?P<qi>\d+)_(?P<qa>\d+)_c(?P<ci>\d+)_(?P<ca>\d+)_1_1\.png$")

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "index_template.html")
SIZE = 512


def load_horizons(method_b):
    """scene_id -> scene geometry (horizon + semantic fractions) for the placement filter."""
    geo = {}
    for key in (METHOD_A, method_b):
        for r in json.load(open(RETRIEVAL_JSONS[key])):
            q = r["query"]
            geo[f'{q["image_id"]}_{q["annotation_id"]}'] = dict(
                hy=q["horizon_y_norm"], hc=q["horizon_confidence"],
                vp=q["viewpoint_type"], sky=q["sky_frac"])
    return geo


def above_waterline(box, g, margin=0.02):
    """True if the inserted vessel's waterline (box bottom) sits above the scene horizon -- i.e.
    the boat is placed in the sky or on the far bank/land (a BootPlace placement failure). Only
    evaluated where the horizon is reliable and the view is not top-down/nadir."""
    if g is None or g["hc"] < 0.5 or g["vp"] == "nadir_view" or g["sky"] <= 0.02:
        return False
    return (box[3] / SIZE) < g["hy"] - margin


def load_source_maps():
    """image_id -> source, image_id -> dataset, asserting no source collisions."""
    img_source, img_dataset = {}, {}
    for name, p in RETRIEVAL_JSONS.items():
        if not os.path.exists(p):
            print(f"  [warn] missing retrieval json: {p}", file=sys.stderr); continue
        for rec in json.load(open(p)):
            for e in [rec["query"]] + rec.get("topk", []):
                iid, src, ds = e["image_id"], e["source"], e["dataset"]
                if iid in img_source and img_source[iid] != src:
                    raise SystemExit(f"FATAL image_id {iid} maps to two sources "
                                     f"({img_source[iid]} vs {src}); keying by image_id is unsafe.")
                img_source[iid] = src
                img_dataset[iid] = ds
    return img_source, img_dataset


def parse_composites(method, img_source, img_dataset):
    """Return scene_id -> record, after the two-layer restricted filter."""
    out, dropped = {}, collections.Counter()
    for f in sorted(glob.glob(os.path.join(COMP, method, "*.png"))):
        m = FNAME_RE.match(os.path.basename(f))
        if not m:
            dropped["unparsed"] += 1; continue
        qi, ci = int(m["qi"]), int(m["ci"])
        bg_src, cand_src = img_source.get(qi), img_source.get(ci)
        if bg_src is None or cand_src is None:
            dropped["unknown_source"] += 1; continue
        if bg_src in RESTRICTED or cand_src in RESTRICTED:
            dropped["amr_apl"] += 1; continue
        scene_id = f"{m['qi']}_{m['qa']}"          # query background key -> pairs across methods
        out[scene_id] = dict(png=f, scene_id=scene_id, qi=qi, ci=ci,
                             bg_src=bg_src, cand_src=cand_src,
                             dataset=img_dataset.get(qi, "UNKNOWN"))
    return out, dropped


def black_frac(path):
    """Fraction of near-black pixels -- proxy for a harmonization blob / dark scene."""
    im = np.asarray(Image.open(path).convert("RGB"))
    return float(((im[:, :, 0] < 28) & (im[:, :, 1] < 28) & (im[:, :, 2] < 28)).mean())


def has_black_rectangle(path, min_bbox_frac=0.004, min_fill=0.9):
    """True if the composite contains a large SOLID axis-aligned near-black block -- a masked
    region / harmonization artifact (fill ratio ~1.0), as opposed to irregular dark content
    like a person's hair (fill ratio << 1). Aerial (SeaDronesSee) scenes exhibit these."""
    if ndimage is None:
        return False
    im = np.asarray(Image.open(path).convert("RGB"))
    blk = im.max(axis=2) < 12
    if not blk.any():
        return False
    lab, n = ndimage.label(blk)
    npix = blk.size
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        if bbox / npix < min_bbox_frac:
            continue
        if (lab[sl] == i).sum() / bbox >= min_fill:
            return True
    return False


def insert_box(method, comp_basename, W, H, pad_frac):
    """Padded pixel bbox (x0,y0,x1,y1) of the inserted vessel in the WxH composite, derived
    from the DiffHarmony mask (anamorphic-scaled from mask space to composite space)."""
    m = FNAME_RE.match(comp_basename)
    if not m:
        return None
    mp = os.path.join(MASKS, method, "HPose", "masks", f"{method}_{m['idx']}.png")
    if not os.path.exists(mp):
        return None
    a = np.asarray(Image.open(mp).convert("L"))
    ys, xs = np.where(a > 127)
    if len(xs) == 0:
        return None
    sx, sy = W / a.shape[1], H / a.shape[0]
    x0, x1 = xs.min() * sx, xs.max() * sx
    y0, y1 = ys.min() * sy, ys.max() * sy
    pad = pad_frac * max(x1 - x0, y1 - y0)
    return (max(0, x0 - pad), max(0, y0 - pad), min(W - 1, x1 + pad), min(H - 1, y1 + pad))


def draw_box(img, box, width, color=(255, 60, 60)):
    d = ImageDraw.Draw(img)
    for w in range(width):
        d.rectangle([box[0] - w, box[1] - w, box[2] + w, box[3] + w], outline=color)


def fill_black(img):
    """Inpaint small pure-black artifact pixels (e.g. an uncovered aerial-scene corner). Large
    black blocks are dropped upstream, so any black left here is small and blends cleanly."""
    if cv2 is None:
        return img
    a = np.asarray(img.convert("RGB"))
    mask = (a.max(axis=2) < 12).astype(np.uint8)
    if mask.sum() == 0:
        return img
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))


def stratified_sample(scene_ids, dataset_of, k, seed):
    """Down-sample to k scenes keeping per-dataset proportions (largest-remainder)."""
    if k is None or k >= len(scene_ids):
        return list(scene_ids)
    rng = random.Random(seed)
    by_ds = collections.defaultdict(list)
    for s in scene_ids:
        by_ds[s.dataset if hasattr(s, "dataset") else dataset_of[s]].append(s)
    total = len(scene_ids)
    quota = {ds: len(v) * k / total for ds, v in by_ds.items()}
    chosen = []
    for ds, v in by_ds.items():
        rng.shuffle(v)
        chosen += v[:int(quota[ds])]
    # fill remainder by largest fractional part
    rem = k - len(chosen)
    leftovers = []
    for ds, v in by_ds.items():
        leftovers += [(quota[ds] - int(quota[ds]), s) for s in v[int(quota[ds]):]]
    leftovers.sort(key=lambda t: -t[0])
    chosen += [s for _, s in leftovers[:max(0, rem)]]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", choices=["dino", "random"], default="dino",
                    help="opponent arm: 'dino' (primary study) or 'random' (calibration/floor).")
    ap.add_argument("--max", type=int, default=None,
                    help="cap number of scenes (stratified). Default: use all common public pairs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=92, help="output JPEG quality")
    ap.add_argument("--max-black-delta", type=float, default=0.05,
                    help="drop a pair if |black_frac(sealect)-black_frac(dino)| exceeds this "
                         "(guards against a one-sided rendering failure; ~none expected).")
    ap.add_argument("--keep-black-rects", action="store_true",
                    help="keep composites containing a solid black rectangle (masked-region "
                         "artifact). Default: drop them (they look broken to raters).")
    ap.add_argument("--no-inpaint", action="store_true",
                    help="do not inpaint small residual pure-black corner pixels (default: fill them).")
    ap.add_argument("--no-box", action="store_true",
                    help="do NOT outline the inserted vessel (default: outline it in red).")
    ap.add_argument("--pad-frac", type=float, default=0.08, help="box padding as frac of box size")
    ap.add_argument("--box-width", type=int, default=3, help="outline thickness in px")
    ap.add_argument("--keep-sky-land", action="store_true",
                    help="keep pairs where a vessel is placed above the waterline (sky/far bank). "
                         "Default: drop them -- a BootPlace placement failure, not a SEALECT one.")
    args = ap.parse_args()

    method_b = args.vs
    tag = "" if method_b == "dino" else method_b
    site_dir = os.path.join(HERE, "site" if method_b == "dino" else f"site_{method_b}")
    img_out = os.path.join(site_dir, "images")
    private_out = os.path.join(HERE, "manifest_private.json" if method_b == "dino"
                               else f"manifest_private_{method_b}.json")

    print(f"== SEALECT human-eval manifest builder (sealect vs {method_b}) ==")
    img_source, img_dataset = load_source_maps()
    print(f"source map: {len(img_source)} image_ids | sources={sorted(set(img_source.values()))}")
    geo = load_horizons(method_b)

    sea, dsea = parse_composites(METHOD_A, img_source, img_dataset)
    mb, dmb = parse_composites(method_b, img_source, img_dataset)
    print(f"{METHOD_A}: kept {len(sea)} (dropped {dict(dsea)})")
    print(f"{method_b}: kept {len(mb)} (dropped {dict(dmb)})")

    common = sorted(set(sea) & set(mb))
    print(f"common public backgrounds {METHOD_A}∩{method_b} = {len(common)}")

    # per-pair rendering-confound guard + inserted-vessel box (from the DiffHarmony mask)
    pairs, box_of = [], {}
    n_skyland = 0
    for sid in common:
        bfa, bfb = black_frac(sea[sid]["png"]), black_frac(mb[sid]["png"])
        if abs(bfa - bfb) > args.max_black_delta:
            print(f"  [drop] {sid}: |Δblack|={abs(bfa-bfb):.3f} (one-sided render failure)")
            continue
        if not args.keep_black_rects and (has_black_rectangle(sea[sid]["png"])
                                          or has_black_rectangle(mb[sid]["png"])):
            print(f"  [drop] {sid}: solid black-rectangle artifact")
            continue
        # inserted-vessel boxes (needed for the outline AND the placement filter)
        W, H = Image.open(sea[sid]["png"]).size
        bs = insert_box(METHOD_A, os.path.basename(sea[sid]["png"]), W, H, args.pad_frac)
        bd = insert_box(method_b, os.path.basename(mb[sid]["png"]), W, H, args.pad_frac)
        if bs is None or bd is None:          # never box one side but not the other
            print(f"  [drop] {sid}: missing inserted-vessel mask "
                  f"({METHOD_A}={bs is not None}, {method_b}={bd is not None})")
            continue
        if not args.keep_sky_land and (above_waterline(bs, geo.get(sid))
                                       or above_waterline(bd, geo.get(sid))):
            n_skyland += 1
            continue
        box_of[sid] = {METHOD_A: bs, method_b: bd}
        pairs.append((sid, bfa, bfb))
    kept_ids = [p[0] for p in pairs]

    # optional stratified subsample
    kept_ids = stratified_sample(kept_ids, {s: sea[s]["dataset"] for s in kept_ids},
                                 args.max, args.seed)
    kept_ids = sorted(kept_ids)
    bf_of = {sid: (bfa, bfb) for sid, bfa, bfb in pairs}

    if not args.keep_sky_land:
        print(f"placement filter: dropped {n_skyland} pairs with a sky/far-bank (above-waterline) "
              f"insertion (BootPlace failure, not SEALECT)")
    strat = collections.Counter(sea[s]["dataset"] for s in kept_ids)
    print(f"final study set: {len(kept_ids)} pairs | per-dataset {dict(sorted(strat.items()))}")

    # ---- write site ----
    if os.path.exists(img_out):
        for f in glob.glob(os.path.join(img_out, "*")):
            os.remove(f)
    os.makedirs(img_out, exist_ok=True)

    rng = random.Random(args.seed)
    public_scenes, private = [], []
    total_bytes = 0
    for sid in kept_ids:
        # coin-flip which method lands in slot 'a' vs 'b' (blinds the file names)
        methods = [METHOD_A, method_b]
        if rng.random() < 0.5:
            methods = methods[::-1]
        slot_png = {"a": (sea if methods[0] == METHOD_A else mb)[sid]["png"],
                    "b": (sea if methods[1] == METHOD_A else mb)[sid]["png"]}
        slot_method = {"a": methods[0], "b": methods[1]}
        for slot in ("a", "b"):
            out = os.path.join(img_out, f"{sid}_{slot}.jpg")
            img = Image.open(slot_png[slot]).convert("RGB")
            if not args.no_inpaint:
                img = fill_black(img)
            if not args.no_box:
                draw_box(img, box_of[sid][slot_method[slot]], args.box_width)
            img.save(out, "JPEG", quality=args.quality, optimize=True)
            total_bytes += os.path.getsize(out)
        public_scenes.append(dict(scene_id=sid, dataset=sea[sid]["dataset"],
                                  img_a=f"images/{sid}_a.jpg", img_b=f"images/{sid}_b.jpg"))
        bfa, bfb = bf_of[sid]
        private.append({"scene_id": sid, "dataset": sea[sid]["dataset"],
                        "a_method": slot_method["a"], "b_method": slot_method["b"],
                        f"{METHOD_A}_png": os.path.basename(sea[sid]["png"]),
                        f"{method_b}_png": os.path.basename(mb[sid]["png"]),
                        f"black_frac_{METHOD_A}": round(bfa, 4),
                        f"black_frac_{method_b}": round(bfb, 4),
                        "black_frac_delta": round(abs(bfa - bfb), 4),
                        f"box_{METHOD_A}": [round(v, 1) for v in box_of[sid][METHOD_A]],
                        f"box_{method_b}": [round(v, 1) for v in box_of[sid][method_b]]})

    # FINAL per-file restricted re-audit (belt and suspenders)
    for p in private:
        for src in (sea[p["scene_id"]]["bg_src"], sea[p["scene_id"]]["cand_src"],
                    mb[p["scene_id"]]["cand_src"]):
            assert src not in RESTRICTED, f"restricted LEAK at {p['scene_id']}"
        assert p["dataset"] in PUBLIC_DATASETS, f"non-public dataset {p['dataset']}"
    print(f"AUDIT PASSED: 0 restricted across {len(private)} pairs "
          f"({2*len(private)} images, {total_bytes/1e6:.1f} MB)")

    # render index.html from template with the PUBLIC scene list + study tag embedded
    with open(TEMPLATE) as fh:
        html = fh.read()
    html = html.replace("/*__SCENES__*/[]", json.dumps(public_scenes, separators=(",", ":")))
    html = html.replace("/*__TAG__*/\"\"", json.dumps(tag))
    with open(os.path.join(site_dir, "index.html"), "w") as fh:
        fh.write(html)

    json.dump(private, open(private_out, "w"), indent=2)
    # also drop a public copy of the scene list (handy, still label-free)
    json.dump(public_scenes, open(os.path.join(site_dir, "scenes.json"), "w"), indent=2)

    print(f"\nwrote: {os.path.join(site_dir,'index.html')}")
    print(f"       {img_out}/  ({2*len(private)} jpgs)")
    print(f"       {private_out}   (KEEP PRIVATE -- do not upload)")
    print("\nnext: preview locally with")
    print(f"   python3 -m http.server -d {site_dir} 8000   ->  http://localhost:8000")
    print(f"then deploy {os.path.basename(site_dir)}/ to any static host (see README.md).")


if __name__ == "__main__":
    main()
