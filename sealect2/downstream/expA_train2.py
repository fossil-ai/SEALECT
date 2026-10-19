#!/usr/bin/env python3
"""
Experiment A (strengthened): SAM2-masked cutouts + 3-seed error bars.

Same controlled design as expA_train.py -- three conditions share base image,
pasted crop, paste (x,y), per-image RNG; differ ONLY in paste geometry:
  noaug / plausible (scale~depth, upright) / implausible (random scale, flip/rot).
Improvements over the pilot:
  1. SAM2 vessel silhouette alpha (removes feathered-rect water-halo confound).
  2. seeds {42,123,7} -> mean+-std, error bars on FP-rate (the robust axis).
Lead metric = false positives on vessel-empty LARS images (precision/robustness).

Run from a NON-sam2 cwd (scratchpad) so the sam2 package isn't shadowed.
  pool stage:  build+cache SAM2 crop pool (short), writes sanity viz + coverage.
  run  stage:  seeds x modes train+eval, incremental results_multiseed.json.
Public data only (SeaShips/SMD/LARS). project venv.
"""
import os, sys, json, math, random, argparse, tempfile, pickle
from pathlib import Path
import numpy as np
import cv2
import torch

DATA = Path("/path/to/datasets/data")
OUT  = Path("/path/to/sealect_root/sealect2/downstream/expA_out2")
OUT.mkdir(parents=True, exist_ok=True)
POOL_CACHE = OUT / "crop_pool_sam2.pkl"
SAM2_CKPT = "/path/to/sealect_root/sam2_checkpoints/sam2_hiera_large.pt"
SAM2_CFG  = "sam2_hiera_l.yaml"
SEEDS = [42, 123, 7]
POOL_SEED = 42  # crop pool identical across all training seeds

SMD_VESSEL  = {"Vessel/ship", "Boat", "Speed boat"}
LARS_VESSEL = {"Boat/ship", "Row boats"}


def _index_images(root: Path):
    idx = {}
    for p in root.rglob("*.jpg"): idx.setdefault(p.name, p)
    for p in root.rglob("*.png"): idx.setdefault(p.name, p)
    return idx


def load_coco_vessels(json_path, img_index, keep_names):
    d = json.load(open(json_path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    keep_ids = set(cats) if keep_names is None else {i for i, n in cats.items() if n in keep_names}
    byimg = {im["id"]: {"file_name": im["file_name"], "boxes": []} for im in d["images"]}
    for a in d["annotations"]:
        if a["category_id"] not in keep_ids: continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1: continue
        byimg[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
    out = []
    for iid, rec in byimg.items():
        name = os.path.basename(rec["file_name"])
        p = img_index.get(name) or img_index.get(rec["file_name"])
        if p is None: continue
        out.append({"image_id": iid, "path": str(p), "boxes": rec["boxes"]})
    return out


def build_base():
    ss_idx = _index_images(DATA / "SeaShips" / "JPEGImages")
    smd_idx = _index_images(DATA / "SMD" / "images")
    ss = load_coco_vessels(DATA / "SeaShips/instances_train.json", ss_idx, None)
    smd = load_coco_vessels(DATA / "SMD/smd_coco.json", smd_idx, SMD_VESSEL)
    return [r for r in (ss + smd) if r["boxes"]]


# ---------------- SAM2 crop pool ----------------
def _feather_alpha(H, W):
    m = max(2, int(0.08 * min(H, W)))
    if H > 2*m and W > 2*m:
        a = cv2.copyMakeBorder(np.ones((H-2*m, W-2*m), np.float32), m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    else:
        a = np.ones((H, W), np.float32)
    return cv2.GaussianBlur(a, (0, 0), sigmaX=max(1, m/2))


def build_crop_pool_sam2(base, max_crops=800, max_side=256):
    if POOL_CACHE.exists():
        pool = pickle.load(open(POOL_CACHE, "rb"))
        print(f"[pool] loaded cached SAM2 pool: {len(pool)} crops "
              f"({sum(p['sam2'] for p in pool)} SAM2, {sum(1-p['sam2'] for p in pool)} fallback)", flush=True)
        return pool
    rng = random.Random(POOL_SEED)
    recs = []
    for r in base:
        for b in r["boxes"]:
            w, h = b[2]-b[0], b[3]-b[1]
            if w < 48 or h < 32: continue
            if not (0.4 <= w/h <= 12): continue
            recs.append((r["path"], b))
    rng.shuffle(recs)

    sys.path.insert(0, "/tmp")  # ensure not shadowed
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(SAM2_CFG, SAM2_CKPT, device=dev)
    predictor = SAM2ImagePredictor(sam)

    pool, n_sam2, n_fb, tried = [], 0, 0, 0
    for path, b in recs:
        if len(pool) >= max_crops: break
        tried += 1
        img = cv2.imread(path)
        if img is None: continue
        Himg, Wimg = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in b]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(Wimg, x2), min(Himg, y2)
        if x2-x1 < 20 or y2-y1 < 12: continue
        # padded ROI for SAM2 context
        pw, ph = x2-x1, y2-y1
        px, py = int(0.2*pw), int(0.2*ph)
        rx1, ry1 = max(0, x1-px), max(0, y1-py); rx2, ry2 = min(Wimg, x2+px), min(Himg, y2+py)
        roi = img[ry1:ry2, rx1:rx2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        alpha_tight = None; is_sam2 = 0
        try:
            predictor.set_image(rgb)
            box_local = np.array([x1-rx1, y1-ry1, x2-rx1, y2-ry1], dtype=np.float32)
            with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(box=box_local, multimask_output=False)
            mask = masks[0].astype(np.float32)                       # ROI-sized
            mask_tight = mask[y1-ry1:y2-ry1, x1-rx1:x2-rx1]
            cov = float(mask_tight.mean())
            if 0.05 <= cov <= 0.98:
                alpha_tight = mask_tight; is_sam2 = 1
        except Exception as e:
            if tried <= 3: print(f"[pool] SAM2 err: {e}", flush=True)
        crop = img[y1:y2, x1:x2].copy()
        if alpha_tight is None:
            alpha_tight = _feather_alpha(crop.shape[0], crop.shape[1]); n_fb += 1
        else:
            n_sam2 += 1
        # resize down
        H, W = crop.shape[:2]
        s = min(1.0, max_side/max(H, W))
        if s < 1.0:
            crop = cv2.resize(crop, (int(W*s), int(H*s)))
            alpha_tight = cv2.resize(alpha_tight, (crop.shape[1], crop.shape[0]))
        alpha_tight = cv2.GaussianBlur(alpha_tight.astype(np.float32), (0, 0), sigmaX=1.2)
        alpha_tight = np.clip(alpha_tight, 0, 1)
        pool.append({"img": crop, "alpha": alpha_tight, "sam2": is_sam2})
        if len(pool) % 100 == 0:
            print(f"[pool] {len(pool)}/{max_crops}  (sam2={n_sam2} fb={n_fb}, tried={tried})", flush=True)
    del sam, predictor; torch.cuda.empty_cache()
    pickle.dump(pool, open(POOL_CACHE, "wb"))
    print(f"[pool] built {len(pool)} crops: {n_sam2} SAM2, {n_fb} fallback -> cached", flush=True)
    return pool


def paste(base_img, boxes, crop_rec, mode, rng):
    H, W = base_img.shape[:2]
    crop, alpha = crop_rec["img"], crop_rec["alpha"]
    ch, cw = crop.shape[:2]; ar = cw/ch
    cx = rng.uniform(0.1, 0.9) * W
    yc = rng.uniform(0.50, 0.85) * H
    t = (yc/H - 0.50) / 0.35
    if mode == "plausible":
        vh = (0.05 + 0.16*t) * H; flip_v, rot = False, 0.0
    else:
        vh = rng.uniform(0.05, 0.45) * H; flip_v = rng.random() < 0.5; rot = rng.uniform(-35, 35)
    vw = vh * ar
    if vw < 12 or vw > 0.9*W or vh > 0.9*H: return None
    obj = cv2.resize(crop, (int(vw), int(vh))); a = cv2.resize(alpha, (int(vw), int(vh)))
    if flip_v: obj = cv2.flip(obj, 0); a = cv2.flip(a, 0)
    if abs(rot) > 1:
        M = cv2.getRotationMatrix2D((vw/2, vh/2), rot, 1.0)
        obj = cv2.warpAffine(obj, M, (int(vw), int(vh)), borderMode=cv2.BORDER_REFLECT)
        a = cv2.warpAffine(a, M, (int(vw), int(vh)))
    ph, pw = obj.shape[:2]
    x1 = max(0, min(W-pw, int(cx-pw/2))); y1 = max(0, min(H-ph, int(yc-ph/2)))
    x2, y2 = x1+pw, y1+ph
    for b in boxes:
        ix1, iy1 = max(x1, b[0]), max(y1, b[1]); ix2, iy2 = min(x2, b[2]), min(y2, b[3])
        inter = max(0, ix2-ix1)*max(0, iy2-iy1); ua = pw*ph + (b[2]-b[0])*(b[3]-b[1]) - inter
        if ua > 0 and inter/ua > 0.25: return None
    roi = base_img[y1:y2, x1:x2].astype(np.float32); a3 = np.clip(a, 0, 1)[:, :, None]
    base_img[y1:y2, x1:x2] = (a3*obj.astype(np.float32) + (1-a3)*roi).astype(np.uint8)
    return [x1, y1, x2, y2]


class VesselDet(torch.utils.data.Dataset):
    def __init__(self, base, pool, mode, seed):
        self.base, self.pool, self.mode, self.seed = base, pool, mode, seed
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        r = self.base[i]
        img = cv2.imread(r["path"])
        if img is None: img = np.zeros((720, 1280, 3), np.uint8)
        boxes = [list(map(float, b)) for b in r["boxes"]]
        if self.mode != "noaug" and self.pool:
            rng = random.Random(self.seed*100003 + i)   # shared crop+placement across modes within a seed
            crop = self.pool[rng.randrange(len(self.pool))]
            nb = paste(img, boxes, crop, self.mode, rng)
            if nb: boxes.append([float(v) for v in nb])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float()/255.0
        if not boxes: boxes = [[0, 0, 1, 1]]
        bx = torch.tensor(boxes, dtype=torch.float32)
        return t, {"boxes": bx, "labels": torch.ones((len(boxes),), dtype=torch.int64), "image_id": torch.tensor([i])}


def collate(b): return tuple(zip(*b))


def get_model():
    import torchvision
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


def train_one(mode, base, pool, epochs, device, seed, lr=0.005):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ds = VesselDet(base, pool, mode, seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    model = get_model().to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs-3), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(); model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for imgs, tgts in dl:
            imgs = [im.to(device) for im in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = sum(model(imgs, tgts).values())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss); n += 1
        sched.step()
        print(f"  [{mode} s{seed}] epoch {ep+1}/{epochs} loss={tot/max(1,n):.4f}", flush=True)
    ck = OUT / f"model_{mode}_s{seed}.pt"
    torch.save(model.state_dict(), ck)
    return ck


def build_lars_gt():
    lars_idx = _index_images(DATA / "LARS" / "images")
    d = json.load(open(DATA / "LARS/lars_coco.json"))
    keep_ids = {c["id"] for c in d["categories"] if c["name"] in LARS_VESSEL}
    images = [im for im in d["images"] if os.path.basename(im["file_name"]) in lars_idx]
    valid = {im["id"] for im in images}
    anns, aid = [], 1
    for a in d["annotations"]:
        if a["category_id"] in keep_ids and a["image_id"] in valid and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            anns.append({"id": aid, "image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"],
                         "area": a["bbox"][2]*a["bbox"][3], "iscrowd": 0}); aid += 1
    gt = {"images": [{"id": im["id"], "file_name": im["file_name"]} for im in images],
          "annotations": anns, "categories": [{"id": 1, "name": "vessel"}]}
    pathmap = {im["id"]: str(lars_idx[os.path.basename(im["file_name"])]) for im in images}
    return gt, pathmap


@torch.no_grad()
def evaluate(ckpt, gt, pathmap, device, conf=0.5):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model = get_model().to(device); model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(gt, tmp); tmp.close()
    coco_gt = COCO(tmp.name)
    results = []; empty_ids = {im["id"] for im in gt["images"]} - {a["image_id"] for a in gt["annotations"]}
    fp_on_empty, n_empty = 0, 0
    ids = [im["id"] for im in gt["images"]]
    for k, iid in enumerate(ids):
        img = cv2.imread(pathmap[iid])
        if img is None: continue
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().to(device)/255.0
        out = model([t])[0]
        b = out["boxes"].cpu().numpy(); s = out["scores"].cpu().numpy()
        for bb, sc in zip(b, s):
            x1,y1,x2,y2 = bb
            results.append({"image_id": int(iid), "category_id": 1, "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)], "score": float(sc)})
        if iid in empty_ids:
            n_empty += 1; fp_on_empty += int((s >= conf).sum())
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AR100": float(ev.stats[8]),
            "fp_per_empty_img": fp_on_empty/max(1, n_empty), "n_empty": n_empty, "n_dets_total": len(results)}


def agg(vals):
    a = np.array(vals, float); return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "vals": [float(v) for v in a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "run"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--modes", default="noaug,plausible,implausible")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = build_base()
    print(f"base train images: {len(base)} (vessel-bearing)", flush=True)
    modes = args.modes.split(",")

    if args.stage == "pool":
        pool = build_crop_pool_sam2(base)
        # sanity viz
        r = base[123]; scratch = Path("experiments/expA_scratch"); scratch.mkdir(parents=True, exist_ok=True)
        for mode in ["plausible", "implausible"]:
            img = cv2.imread(r["path"]); boxes = [list(map(float, b)) for b in r["boxes"]]
            rng = random.Random(42*100003 + 123); crop = pool[rng.randrange(len(pool))]
            nb = paste(img, boxes, crop, mode, rng)
            if nb: cv2.rectangle(img, (nb[0],nb[1]), (nb[2],nb[3]), (0,0,255), 3)
            cv2.imwrite(str(scratch/f"sam2_sanity_{mode}.jpg"), img)
        cov = np.mean([p["alpha"].mean() for p in pool])
        print(f"[pool] mean alpha coverage={cov:.3f}  sanity viz written", flush=True)
        return

    # run: seeds x modes
    pool = build_crop_pool_sam2(base)
    gt, pathmap = build_lars_gt()
    print(f"LARS OOD test: {len(gt['images'])} imgs, {len(gt['annotations'])} vessel boxes", flush=True)
    res_path = OUT / "results_multiseed.json"
    per = json.load(open(res_path)) if res_path.exists() else {}
    for seed in SEEDS:
        for mode in modes:
            key = f"{mode}_s{seed}"
            if key in per:
                print(f"=== SKIP {key} (cached) ===", flush=True); continue
            print(f"=== TRAIN {mode} seed={seed} ===", flush=True)
            ck = train_one(mode, base, pool, args.epochs, device, seed)
            print(f"=== EVAL {mode} seed={seed} ===", flush=True)
            per[key] = evaluate(ck, gt, pathmap, device)
            per[key]["seed"] = seed; per[key]["mode"] = mode
            json.dump(per, open(res_path, "w"), indent=2)   # incremental
            print(f"  {key}: {per[key]}", flush=True)

    # aggregate
    summary = {}
    for mode in modes:
        vals = [per[f"{mode}_s{s}"] for s in SEEDS if f"{mode}_s{s}" in per]
        if not vals: continue
        summary[mode] = {m: agg([v[m] for v in vals]) for m in ["AP", "AP50", "AR100", "fp_per_empty_img"]}
    json.dump({"per_run": per, "summary": summary}, open(OUT/"results_multiseed_summary.json", "w"), indent=2)
    print("\n==== SUMMARY (LARS OOD, mean+-std over 3 seeds) ====", flush=True)
    print(f"{'mode':14s} {'AP':>13} {'AP50':>13} {'AR100':>13} {'FP/empty':>15}", flush=True)
    for mode in modes:
        if mode not in summary: continue
        s = summary[mode]
        def f(m): return f"{s[m]['mean']:.3f}+-{s[m]['std']:.3f}"
        print(f"{mode:14s} {f('AP'):>13} {f('AP50'):>13} {f('AR100'):>13} {f('fp_per_empty_img'):>15}", flush=True)
    print("[DONE multiseed]", flush=True)


if __name__ == "__main__":
    main()
