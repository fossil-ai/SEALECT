#!/usr/bin/env python3
import os, json, math, glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# -----------------------
# CONFIG: edit these
# -----------------------
VAL_ANN_DIR  = "/path/to/sealect_root/unified/val_annotations"
VAL_IMG_DIR  = "/path/to/sealect_root/unified/val_images"   # folder containing all 200 images
CKPT_PATH    = "runs_posenet_v12sym_stable/checkpoints/best_geo.pt"          # or last.pt / best_yaw.pt
IMG_SIZE     = 512

# If your yaw / rotation is symmetric by 180deg (boat front/back ambiguity)
YAW_SYM_180 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Import your model
# -----------------------
# Option A: if this file lives next to your big script, import from it.
# from posenet_v5 import PoseNet, Cfg, euler_to_rotmat, deg2rad, rotmat_to_euler, rad2deg, geodesic_loss
#
# Option B: if you're running this by pasting into the same file, comment imports.
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _sorted_datasets(groups):
    # stable order: by name
    return sorted([k for k in groups.keys()])

def _bar_plot(values, labels, title, ylabel, out_path, rotate=35):
    plt.figure(figsize=(max(8, 0.6 * len(labels)), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _grouped_bar_plot(a, b, c, labels, legend, title, ylabel, out_path, rotate=35):
    # 3 series grouped
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(max(9, 0.6 * len(labels)), 5))
    plt.bar(x - w, a, width=w, label=legend[0])
    plt.bar(x,     b, width=w, label=legend[1])
    plt.bar(x + w, c, width=w, label=legend[2])
    plt.xticks(x, labels, rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _overlay_hist(series_dict, title, xlabel, out_path, bins=30):
    plt.figure(figsize=(9, 5))
    for k, v in series_dict.items():
        if len(v) == 0:
            continue
        plt.hist(v, bins=bins, alpha=0.35, density=True, label=f"{k} (n={len(v)})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _plot_cdf(series_dict, title, xlabel, out_path):
    plt.figure(figsize=(9, 5))
    for k, v in series_dict.items():
        if len(v) == 0:
            continue
        xs = np.sort(np.array(v, dtype=np.float32))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=f"{k} (n={len(xs)})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _boxplot(series_dict, title, ylabel, out_path, rotate=35):
    labels = list(series_dict.keys())
    data = [np.array(series_dict[k], dtype=np.float32) for k in labels]
    plt.figure(figsize=(max(9, 0.6 * len(labels)), 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xticks(rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def write_worst_cases(per_image_rows, out_path, topk=10):
    # Sort by geo error descending
    worst = sorted(per_image_rows, key=lambda r: r["geo_deg"], reverse=True)
    with open(out_path, "w") as f:
        f.write(f"Top {topk} worst images by geodesic error\n")
        f.write("=" * 60 + "\n")
        for r in worst[:topk]:
            f.write(
                f"{r['geo_deg']:.2f} deg | {r['dataset']} | "
                f"P/Y/R err: {r['pitch_err_deg']:.1f}/{r['yaw_err_deg']:.1f}/{r['roll_err_deg']:.1f} | "
                f"{r['image']}\n"
            )

def make_plots(groups, summary_rows, per_image_rows, out_dir):
    """
    groups: dict[dataset] -> list[per-image rows]
    summary_rows: list of dataset summary dicts (includes __OVERALL__)
    per_image_rows: all per-image dicts
    out_dir: where to write plots
    """
    ensure_dir(out_dir)

    # exclude overall for per-dataset charts
    ds_summ = [s for s in summary_rows if s["dataset"] != "__OVERALL__"]
    labels = [s["dataset"] for s in ds_summ]

    geo_mean = [s["geo_mean"] for s in ds_summ]
    geo_p95  = [s["geo_p95"] for s in ds_summ]

    pitch_mae = [s["pitch_mae"] for s in ds_summ]
    yaw_mae   = [s["yaw_mae"] for s in ds_summ]
    roll_mae  = [s["roll_mae"] for s in ds_summ]

    p_acc = [s["pitch_bin_acc"] for s in ds_summ]
    y_acc = [s["yaw_bin_acc"] for s in ds_summ]
    r_acc = [s["roll_bin_acc"] for s in ds_summ]

    _bar_plot(geo_mean, labels, "Geodesic Error Mean by Dataset", "Degrees", os.path.join(out_dir, "geo_mean_by_dataset.png"))
    _bar_plot(geo_p95,  labels, "Geodesic Error p95 by Dataset (Tail)", "Degrees", os.path.join(out_dir, "geo_p95_by_dataset.png"))

    _grouped_bar_plot(pitch_mae, yaw_mae, roll_mae, labels,
                      legend=["Pitch MAE", "Yaw MAE", "Roll MAE"],
                      title="Euler MAE by Dataset (wrap-around; yaw may be 180°-sym)",
                      ylabel="Degrees",
                      out_path=os.path.join(out_dir, "mae_by_dataset.png"))

    _grouped_bar_plot(p_acc, y_acc, r_acc, labels,
                      legend=["Pitch bin acc", "Yaw bin acc", "Roll bin acc"],
                      title="Bin Accuracy by Dataset",
                      ylabel="Accuracy",
                      out_path=os.path.join(out_dir, "binacc_by_dataset.png"))

    # Distributions (geo)
    geo_series = {ds: [r["geo_deg"] for r in rows] for ds, rows in groups.items()}
    # Keep consistent order in plots
    geo_series = {k: geo_series[k] for k in _sorted_datasets(groups)}

    _overlay_hist(geo_series,
                  title="Geodesic Error Distribution (Overlay)",
                  xlabel="Geodesic error (deg)",
                  out_path=os.path.join(out_dir, "geo_hist_overlay.png"),
                  bins=30)

    _plot_cdf(geo_series,
              title="Geodesic Error CDF (per dataset)",
              xlabel="Geodesic error (deg)",
              out_path=os.path.join(out_dir, "geo_cdf.png"))

    _boxplot(geo_series,
             title="Geodesic Error Boxplot (per dataset)",
             ylabel="Geodesic error (deg)",
             out_path=os.path.join(out_dir, "geo_box.png"))

    # Optional: write worst cases
    write_worst_cases(per_image_rows, os.path.join(out_dir, "worst_cases.txt"), topk=25)
# ---- minimal math helpers (match your training code) ----
def deg2rad(x: float) -> float:
    return x * (math.pi / 180.0)

def rad2deg(x: torch.Tensor) -> torch.Tensor:
    return x * (180.0 / math.pi)

def euler_to_rotmat(p: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    cr, sr = torch.cos(r), torch.sin(r)
    z = torch.zeros_like(cp)
    o = torch.ones_like(cp)
    Rx = torch.stack([o,z,z, z,cp,-sp, z,sp,cp], -1).reshape(-1,3,3)
    Ry = torch.stack([cy,z,sy, z,o,z, -sy,z,cy], -1).reshape(-1,3,3)
    Rz = torch.stack([cr,-sr,z, sr,cr,z, z,z,o], -1).reshape(-1,3,3)
    return Ry @ Rx @ Rz

def rotmat_to_euler(R: torch.Tensor, eps: float = 1e-6):
    x = (-R[:,1,2]).clamp(-1 + eps, 1 - eps)
    pitch = torch.asin(x)
    yaw   = torch.atan2(R[:,0,2], R[:,2,2])
    roll  = torch.atan2(R[:,1,0], R[:,1,1])
    return pitch, yaw, roll

def geodesic_loss(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    Rd = R1.transpose(-1,-2) @ R2
    tr = Rd[:,0,0] + Rd[:,1,1] + Rd[:,2,2]
    c = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    return torch.acos(c)

def wrap_abs_deg(a_deg: torch.Tensor, b_deg: torch.Tensor) -> torch.Tensor:
    # per-sample absolute angular error in degrees with wraparound
    return ((a_deg - b_deg + 180) % 360 - 180).abs()

def yaw_sym_abs_err_deg(pred_yaw_deg: torch.Tensor, gt_yaw_deg: torch.Tensor) -> torch.Tensor:
    e = wrap_abs_deg(pred_yaw_deg, gt_yaw_deg)
    if not YAW_SYM_180:
        return e
    return torch.minimum(e, (e - 180.0).abs())

def parse_dataset_key(fname: str) -> str:
    # expects ..._{dataset}.jpg  -> dataset
    base = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in base:
        return "UNKNOWN"
    return base.split("_")[-1]

def load_gt_map(ann_dir: str):
    # map: imageFileName -> dict(pitch_deg,yaw_deg,roll_deg,R,bins if you want)
    gt = {}
    for fp in glob.glob(os.path.join(ann_dir, "*.json")):
        with open(fp, "r") as f:
            d = json.load(f)
        name = d["imageFileName"]
        pose = d["annotations"][0]["pose"]
        p_d = float(pose["pitch"])
        y_d = float(pose["yaw"])
        r_d = float(pose["roll"])
        p_r = torch.tensor(deg2rad(p_d), dtype=torch.float32)
        y_r = torch.tensor(deg2rad(y_d), dtype=torch.float32)
        r_r = torch.tensor(deg2rad(r_d), dtype=torch.float32)
        R = euler_to_rotmat(p_r.view(1), y_r.view(1), r_r.view(1))[0]
        gt[name] = {
            "p_deg": p_d, "y_deg": y_d, "r_deg": r_d,
            "R": R
        }
    return gt

def preprocess_image(path: str, img_size: int):
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(t)
    return t

def tensor_percentile(x: np.ndarray, q: float):
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def summarize_group(rows):
    # rows: list of per-image dicts
    geo = np.array([r["geo_deg"] for r in rows], dtype=np.float32)
    p   = np.array([r["pitch_err_deg"] for r in rows], dtype=np.float32)
    y   = np.array([r["yaw_err_deg"] for r in rows], dtype=np.float32)
    r_  = np.array([r["roll_err_deg"] for r in rows], dtype=np.float32)

    # bin accs optional
    bp = np.array([r.get("pitch_bin_acc", np.nan) for r in rows], dtype=np.float32)
    by = np.array([r.get("yaw_bin_acc",   np.nan) for r in rows], dtype=np.float32)
    br = np.array([r.get("roll_bin_acc",  np.nan) for r in rows], dtype=np.float32)

    return {
        "n": len(rows),
        "geo_mean": float(np.mean(geo)),
        "geo_median": float(np.median(geo)),
        "geo_p90": tensor_percentile(geo, 90),
        "geo_p95": tensor_percentile(geo, 95),
        "pitch_mae": float(np.mean(p)),
        "yaw_mae": float(np.mean(y)),
        "roll_mae": float(np.mean(r_)),
        "pitch_bin_acc": float(np.nanmean(bp)),
        "yaw_bin_acc": float(np.nanmean(by)),
        "roll_bin_acc": float(np.nanmean(br)),
    }

def main():
    # --- import your model here (must match your training definitions) ---
    # If you pasted this into the same file, you already have PoseNet + cfg.
    # from posenet_v5 import PoseNet, cfg  # <-- CHANGE MODULE NAME to your script file (without .py)
    from pose3 import PoseNet, cfg
    gt_map = load_gt_map(VAL_ANN_DIR)

    # model = PoseNet(cfg.SEGFORMER_ENCODER_CKPT, cfg.SEGFORMER_UNFREEZE_LAST_N).to(DEVICE)
    model = PoseNet(cfg.SEGFORMER_ENCODER_CKPT, cfg.SEGFORMER_UNFREEZE_LAST_N).to(DEVICE)

    # ---- PROBE to build seg.projs + seg.combine ----
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=DEVICE)
        _ = model(dummy)

    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)  # now it will match
    model.eval()
    # sd = torch.load(CKPT_PATH, map_location=DEVICE)
    # model.load_state_dict(sd, strict=True)
    # model.eval()

    # If your ResNet BN stats freezing matters at eval, you can keep eval() only.
    # (eval() already freezes BN running stats.)

    img_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg")))
    if not img_paths:
        raise RuntimeError(f"No .jpg found in {VAL_IMG_DIR}")

    per_image = []
    groups = defaultdict(list)

    with torch.no_grad():
        for ip in img_paths:
            name = os.path.basename(ip)
            if name not in gt_map:
                # skip anything not in the val annotations
                continue

            ds_key = parse_dataset_key(name)

            x = preprocess_image(ip, IMG_SIZE).unsqueeze(0).to(DEVICE)
            R_gt = gt_map[name]["R"].unsqueeze(0).to(DEVICE)

            R_pred, pl, yl, rl = model(x)
            R_pred_f = R_pred.float()
            R_gt_f   = R_gt.float()

            geo_deg = rad2deg(geodesic_loss(R_pred_f, R_gt_f))[0].item()

            # Euler errors from predicted rotation (for reporting only)
            pp, yp, rp = rotmat_to_euler(R_pred_f.cpu())
            pred_deg = torch.stack([rad2deg(pp), rad2deg(yp), rad2deg(rp)], dim=-1).squeeze(0)
            gt_deg = torch.tensor([gt_map[name]["p_deg"], gt_map[name]["y_deg"], gt_map[name]["r_deg"]], dtype=torch.float32)

            pitch_err = wrap_abs_deg(pred_deg[0], gt_deg[0]).item()
            yaw_err   = yaw_sym_abs_err_deg(pred_deg[1], gt_deg[1]).item()
            roll_err  = wrap_abs_deg(pred_deg[2], gt_deg[2]).item()

            # Bin accuracies if you want them (requires reproducing your binning).
            # Here we compute only argmax-vs-GT bins by re-binning GT yaw/pitch/roll.
            # (Uses same definition as training: angle in [-pi,pi] mapped into bins.)
            def euler_bin_id(angle_rad: torch.Tensor, num_bins: int) -> torch.Tensor:
                t = (angle_rad + math.pi) / (2 * math.pi)
                return torch.floor(t * num_bins).long().clamp(0, num_bins - 1)

            p_bin_gt = euler_bin_id(torch.deg2rad(gt_deg[0]), cfg.PITCH_BINS).item()
            y_bin_gt = euler_bin_id(torch.deg2rad(gt_deg[1]), cfg.YAW_BINS).item()
            r_bin_gt = euler_bin_id(torch.deg2rad(gt_deg[2]), cfg.ROLL_BINS).item()

            pitch_bin_acc = float(pl.argmax(1).item() == p_bin_gt)
            yaw_bin_acc   = float(yl.argmax(1).item() == y_bin_gt)
            roll_bin_acc  = float(rl.argmax(1).item() == r_bin_gt)

            row = {
                "image": name,
                "dataset": ds_key,
                "geo_deg": geo_deg,
                "pitch_err_deg": pitch_err,
                "yaw_err_deg": yaw_err,
                "roll_err_deg": roll_err,
                "pitch_bin_acc": pitch_bin_acc,
                "yaw_bin_acc": yaw_bin_acc,
                "roll_bin_acc": roll_bin_acc,
            }
            per_image.append(row)
            groups[ds_key].append(row)

    # summaries
    summary_rows = []
    for ds_key in sorted(groups.keys()):
        s = summarize_group(groups[ds_key])
        s["dataset"] = ds_key
        summary_rows.append(s)

    overall = summarize_group(per_image)
    overall["dataset"] = "__OVERALL__"
    summary_rows.append(overall)

    # print nice
    def fmt(s):
        return (f"{s['dataset']:>12} | n={s['n']:3d} | "
                f"geo mean/med/p95 = {s['geo_mean']:.1f}/{s['geo_median']:.1f}/{s['geo_p95']:.1f} deg | "
                f"P/Y/R MAE = {s['pitch_mae']:.1f}/{s['yaw_mae']:.1f}/{s['roll_mae']:.1f} deg | "
                f"bin acc P/Y/R = {s['pitch_bin_acc']:.3f}/{s['yaw_bin_acc']:.3f}/{s['roll_bin_acc']:.3f}")
    print("\nPer-dataset metrics:")
    for s in summary_rows:
        print(fmt(s))

    # save CSVs (no pandas needed)
    out_dir = os.path.dirname(CKPT_PATH) or "."
    out_summary = os.path.join(out_dir, "per_dataset_metrics.csv")
    out_image   = os.path.join(out_dir, "per_image_metrics.csv")

    def write_csv(path, rows, keys):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    write_csv(out_summary, summary_rows, [
        "dataset","n",
        "geo_mean","geo_median","geo_p90","geo_p95",
        "pitch_mae","yaw_mae","roll_mae",
        "pitch_bin_acc","yaw_bin_acc","roll_bin_acc"
    ])
    write_csv(out_image, per_image, [
        "image","dataset","geo_deg",
        "pitch_err_deg","yaw_err_deg","roll_err_deg",
        "pitch_bin_acc","yaw_bin_acc","roll_bin_acc"
    ])

    print(f"\nWrote:\n  {out_summary}\n  {out_image}\n")

    plot_dir = os.path.join(out_dir, "plots_eval")
    make_plots(groups, summary_rows, per_image, plot_dir)
    print(f"Wrote plots to: {plot_dir}")

if __name__ == "__main__":
    main()