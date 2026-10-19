#!/usr/bin/env python3
"""
Result-section figures for the SEALECT paper:
  pareto   -- Recall vs Plausibility (PS) scatter; SEALECT on the top-right frontier (from Table 1).
  signflip -- the novel analytical finding: scene context reduces geometry uncertainty, and the
              scale-vs-placement relationship SIGN-FLIPS across viewpoints (so it cancels globally,
              motivating scene conditioning). Entropy-reduction bars + per-viewpoint density.
  prior    -- query-conditioned S_geom: for contrasting scenes, the kernel-weighted predicted
              placement band (on the image), yaw rose, and scale curve -- the mechanism, adapting.

Usage: python3 make_analysis_figs.py [pareto|signflip|prior|all]
Out:   wacv_sealect/results_figs/fig_{pareto,signflip,prior}.png
"""
import sys, os, json, math, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_figs")
LIB = "/path/to/sealect_root/experiments/dataset_decide/merged_scene_pose_library_out_grounded_sam2"
IMG_ROOT = "/path/to/datasets/data/merged/images"
BLUE, GREY, ORANGE, GREEN = "#2563eb", "#9aa3af", "#e8792b", "#2e8b57"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})


# ------------------------------------------------------------------ #3 Pareto
def fig_pareto():
    # Table 1, K=20: method -> (Recall, PS, group, label-offset)
    D = {
        "Random":            (0.106, 1.364, "base"),
        "$S_{\\mathrm{obs}}$ only": (0.048, 1.651, "abl"),
        "DiscoFOS":          (0.333, 1.783, "fos"),
        "$S_{\\mathrm{geom}}$ only": (0.028, 1.812, "abl"),
        "CLIP":              (0.432, 1.903, "appear"),
        "DINO$+S_{\\mathrm{geom}}$": (0.445, 1.907, "hybrid"),
        "DINO":              (0.445, 1.947, "appear"),
        "SigLIP2":           (0.493, 2.042, "appear"),
        "$S_{\\mathrm{neural}}$ only": (0.634, 2.278, "abl"),
        "SEALECT":           (0.627, 2.418, "ours"),
    }
    off = {  # explicit label offsets (pts) + h-align, to avoid collisions
        "Random": (8, -3, "left"), "$S_{\\mathrm{obs}}$ only": (8, -3, "left"),
        "DiscoFOS": (8, -3, "left"), "$S_{\\mathrm{geom}}$ only": (8, 2, "left"),
        "CLIP": (-7, -13, "right"), "DINO$+S_{\\mathrm{geom}}$": (8, -12, "left"),
        "DINO": (8, 4, "left"), "SigLIP2": (8, 3, "left"),
        "$S_{\\mathrm{neural}}$ only": (10, -3, "left"), "SEALECT": (-8, 10, "right"),
    }
    colors = {"base": GREY, "abl": GREY, "fos": GREEN, "appear": ORANGE, "hybrid": ORANGE, "ours": BLUE}
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    for name, (rec, ps, g) in D.items():
        ours = g == "ours"
        ax.scatter(rec, ps, s=230 if ours else 80, c=colors[g], zorder=3,
                   edgecolor="black", linewidth=1.3 if ours else 0.5, marker="*" if ours else "o")
        dx, dy, ha = off[name]
        ax.annotate(name, (rec, ps), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9.5, fontweight="bold" if ours else "normal",
                    color=BLUE if ours else "black")
    ax.set_xlabel("Recall@20 $\\uparrow$  (retrieval)")
    ax.set_ylabel("Plausibility Score $\\uparrow$  (geometry)")
    ax.set_title("Retrieval quality vs. plausibility", fontweight="bold")
    ax.grid(alpha=0.25); ax.set_xlim(-0.02, 0.72); ax.set_ylim(1.30, 2.52)
    ax.annotate("best", (0.627, 2.418), textcoords="offset points", xytext=(-8, 22),
                ha="right", color=BLUE, fontsize=8.5, style="italic")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_pareto.png"); fig.savefig(p, dpi=220); plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------ #2 sign-flip finding
def _entropy(counter):
    t = sum(counter.values())
    return -sum((v / t) * math.log2(v / t) for v in counter.values() if v)


def _condH(R, keyfn, valfn):
    g = collections.defaultdict(collections.Counter)
    for r in R:
        g[keyfn(r)][valfn(r)] += 1
    tot = sum(sum(x.values()) for x in g.values())
    return sum((sum(x.values()) / tot) * _entropy(x) for x in g.values())


def fig_signflip():
    R = json.load(open(f"{LIB}/scene_object_pairs.json"))
    sbin = lambda r: int(np.clip(math.log10(max(r["area_frac"], 1e-6)) + 6, 0, 11))
    axes_spec = [("Placement", lambda r: int(r["y_bin"])), ("Scale", sbin),
                 ("Yaw", lambda r: int(r["yaw_bin"])), ("Pitch", lambda r: int(r["pitch_bin"]))]
    red = []
    for nm, vf in axes_spec:
        Hm = _entropy(collections.Counter(vf(r) for r in R))
        Hc = _condH(R, lambda r: r["context_key"], vf)
        red.append((nm, 100 * (Hm - Hc) / Hm))
    cy = np.array([r["center_y_norm"] for r in R])
    logA = np.log(np.clip([r["area_frac"] for r in R], 1e-6, None))
    vp = collections.defaultdict(list)
    for i, r in enumerate(R):
        vp[r["viewpoint_type"]].append(i)
    pooled_r = np.corrcoef(cy, logA)[0, 1]

    fig = plt.figure(figsize=(10.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.15], wspace=0.32)
    # panel A: entropy reduction
    axA = fig.add_subplot(gs[0, 0])
    names = [n for n, _ in red]; vals = [v for _, v in red]
    axA.barh(range(len(names)), vals, color=[BLUE, ORANGE, GREEN, GREY])
    axA.set_yticks(range(len(names))); axA.set_yticklabels(names)
    axA.invert_yaxis()
    for i, v in enumerate(vals):
        axA.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10, fontweight="bold")
    axA.set_xlabel("uncertainty reduction (%)")
    axA.set_title("Scene context predicts\ngeometry", fontsize=11, fontweight="bold")
    axA.set_xlim(0, max(vals) * 1.25)
    # panels B,C: sign flip
    for ax, view, col in [(fig.add_subplot(gs[0, 1]), "horizon_view", ORANGE),
                          (fig.add_subplot(gs[0, 2]), "nadir_aerial", BLUE)]:
        idx = np.array(vp[view])
        x, y = cy[idx], logA[idx]
        ax.hexbin(x, y, gridsize=28, cmap="Blues", mincnt=1, alpha=0.9)
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m * xs + b, color="#d11", lw=2.4)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{view.replace('_',' ')}\nslope {'+' if m>0 else ''}{m:.2f}, r={r:+.2f}",
                     fontsize=10.5, fontweight="bold", color=col)
        ax.set_xlabel("vertical position (top$\\to$bottom)")
        ax.set_ylabel("log apparent size")
    fig.suptitle(f"The scale–placement relation sign-flips across viewpoints "
                 f"(pooled r={pooled_r:+.3f} $\\approx$ 0) — only scene conditioning captures it",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_signflip.png"); fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", p, "| entropy reductions:", red, "| pooled r:", round(pooled_r, 4))


# ------------------------------------------------------------------ #1 query-conditioned prior
def _descriptor(rec):
    vpf = rec["vertical_profile"]
    prof = vpf["water"] + vpf["sky"] + vpf.get("land", [0.0] * len(vpf["water"]))
    return [rec["water_frac"], rec["sky_frac"], rec["land_frac"], rec["horizon_y_norm"]] + prof


def fig_prior(h=None):
    R = json.load(open(f"{LIB}/scene_object_pairs.json"))
    C = json.load(open(f"{LIB}/image_contexts.json"))
    Z = np.array([_descriptor(r) for r in R], dtype=float)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    yaw = np.array([int(r["yaw_bin"]) for r in R]); NY = yaw.max() + 1
    cy = np.array([r["center_y_norm"] for r in R])
    area = np.array([r["area_frac"] for r in R])
    # choose genuinely contrasting public scenes by robust descriptors (viewpoint_type is noisy),
    # labelled descriptively so each image matches its caption.
    def ok(c):
        return c.get("dataset") not in {"RESTRICTED_A", "RESTRICTED_B"} and os.path.exists(os.path.join(IMG_ROOT, c["image_name"]))
    criteria = [
        ("open-sea horizon", lambda c: c["water_frac"] > 0.3 and c["sky_frac"] > 0.3
         and c["horizon_confidence"] > 0.75 and "aerial" not in c.get("viewpoint_type", "")),
        ("aerial (top-down)", lambda c: c.get("dataset") == "SeaDronesSee"
         and c["water_frac"] > 0.6 and c["sky_frac"] < 0.05),
        ("harbour / shore", lambda c: c["land_frac"] > 0.30 and c["water_frac"] > 0.2),
    ]
    scenes = []
    for label, cond in criteria:
        for c in C:
            if ok(c) and cond(c):
                scenes.append((c, label)); break

    def weights_ess(zq, target=300):
        """soft-kNN: choose bandwidth so the effective neighbourhood ~ target (real conditioning)."""
        d2 = ((Zs - zq) ** 2).sum(1)
        lo, hi = 0.2, 40.0
        for _ in range(24):
            hh = math.sqrt(lo * hi)
            w = np.exp(-(d2 - d2.min()) / (2 * hh * hh)); w /= w.sum()
            ess = 1.0 / (w ** 2).sum()
            if ess > target: hi = hh
            else: lo = hh
        return w, ess, hh

    fig, axes = plt.subplots(len(scenes), 3, figsize=(11, 3.0 * len(scenes)),
                             gridspec_kw={"width_ratios": [1.7, 1, 1], "wspace": 0.3, "hspace": 0.35})
    for row, (sc, label) in enumerate(scenes):
        zq = (np.array(_descriptor(sc)) - mu) / sd
        w, ess, hh = weights_ess(zq)
        yawH = -sum(p*math.log2(p) for p in [w[yaw==b].sum() for b in range(NY)] if p>0)
        print(f"  [{label:20s}] ESS={ess:.0f} h={hh:.2f} yawH={yawH:.2f} "
              f"E[size]={float((w*area).sum()):.4f}")
        # placement over image: smooth blue overlay concentrated where p(y) is high
        ax = axes[row, 0]
        im = Image.open(os.path.join(IMG_ROOT, sc["image_name"])).convert("RGB")
        W, Himg = im.size
        ax.imshow(im, extent=[0, 1, 1, 0]); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.set_xticks([]); ax.set_yticks([])
        nb = 60
        ph, edges = np.histogram(cy, bins=nb, range=(0, 1), weights=w, density=True)
        ph = np.clip(ph / (ph.max() + 1e-9), 0, 1)
        overlay = np.zeros((nb, 1, 4)); overlay[:, 0, :3] = matplotlib.colors.to_rgb(BLUE)
        overlay[:, 0, 3] = 0.6 * ph
        ax.imshow(overlay, extent=[0, 1, 1, 0], aspect="auto", interpolation="bilinear", zorder=2)
        ax.axhline(sc["horizon_y_norm"], color="w", ls="--", lw=1.3, alpha=0.9, zorder=3)
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        if row == 0:
            ax.set_title("predicted vertical placement", fontsize=10.5)
        # yaw rose
        axr = axes[row, 1]; axr.remove()
        axr = fig.add_subplot(len(scenes), 3, row * 3 + 2, projection="polar")
        yh = np.array([w[yaw == b].sum() for b in range(NY)])
        theta = np.linspace(0, 2 * np.pi, NY, endpoint=False)
        axr.bar(theta, yh, width=2 * np.pi / NY, color=ORANGE, alpha=0.85)
        axr.set_yticks([]); axr.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        axr.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=8)
        if row == 0:
            axr.set_title("predicted yaw", fontsize=10.5, pad=12)
        # scale curve
        axs = axes[row, 2]
        la = np.log10(np.clip(area, 1e-6, None))
        sh, se = np.histogram(la, bins=40, weights=w, density=True)
        axs.fill_between(0.5 * (se[:-1] + se[1:]), sh, color=GREEN, alpha=0.5)
        axs.set_yticks([]); axs.set_xlabel("log$_{10}$ rel. size", fontsize=9)
        exp = float((w * area).sum())
        axs.axvline(math.log10(max(exp, 1e-6)), color="#d11", lw=1.6)
        if row == 0:
            axs.set_title("predicted scale", fontsize=10.5)
    fig.suptitle("Scene-conditioned prior $\\hat p(\\pi\\mid\\mathbf{c}_q)$ adapts to the query scene",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(OUT, "fig_prior.png"); fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p} | scenes={[l for _,l in scenes]}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("pareto", "all"): fig_pareto()
    if which in ("signflip", "all"): fig_signflip()
    if which in ("prior", "all"): fig_prior()
