#!/usr/bin/env python3
"""
Paper Fig. 4: per-axis geometric quality of each method's top-5 retrievals.

For every public query scene (access-restricted and corrupt-metadata ABOships queries excluded), take
each method's top-5 *shareable* candidates (access-restricted candidates skipped) and report the fraction whose geometry
matches the scene's held-out vessel on each axis:
  yaw        circular yaw-bin distance <= 1 (one 20-degree bin; same rule as the paper's relevance metric)
  scale      |log area_frac difference| <= log 2 (within a factor of two)
  placement  |vertical-bin difference| <= 1
All methods are scored on the identical query set (scenes where every method returns at least one shareable candidate).

SEALECT is the kernel-prior run reported in Tables 1-3 (sealect2/runs/B3_kernel_prior). GALA-style and
DiscoFOS use the *_render_k20 copies of gala_only_k20 / discofos_ft_k20 (same rankings, dataset tags added
by experiments/experiment5_1/enrich_baselines.py).

Usage:  python3 wacv_sealect/make_fig4_axis_decomp.py [--out fig4_axis_decomp.png]
"""
import argparse, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # repository root
E1 = os.path.join(ROOT, "experiments/experiment1/exp1_retrieval_quality")
S2 = os.path.join(ROOT, "sealect2/runs")
RESTRICTED = {"RESTRICTED_A", "RESTRICTED_B"}
EXCLUDED_QUERY_SOURCES = RESTRICTED | {"ABOships"}
K = 5
YAW_TOL, YBIN_TOL, LOG_AREA_TOL = 1, 1, math.log(2.0)

# (legend name, retrieval_results.json, color)
METHODS = [
    ("Random",     f"{E1}/random_k20/topk_20/retrieval_results.json",       "#9D9D9D"),
    ("DINO",       f"{E1}/dino_k20/topk_20/retrieval_results.json",         "#4C78A8"),
    ("GALA-style", f"{E1}/gala_render_k20/topk_20/retrieval_results.json",  "#F58518"),
    ("DiscoFOS",   f"{E1}/discofos_render_k20/topk_20/retrieval_results.json", "#54A24B"),
    ("SEALECT",    f"{S2}/B3_kernel_prior/topk_20/retrieval_results.json",  "#E45756"),
]
AXES = ["Yaw", "Scale\n(scene-conditioned)", "Placement\n(scene-conditioned)"]


def circ(a, b, n=18):
    d = abs(int(a) - int(b))
    return min(d, n - d)


def log_area(x):
    return math.log(max(float(x), 1e-8))


def source(rec):
    return str(rec.get("source") or rec.get("dataset") or "")


def load(path):
    return {(r["query"]["image_id"], r["query"]["annotation_id"]): r for r in json.load(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig4_axis_decomp.png"))
    args = ap.parse_args()

    runs = {name: load(path) for name, path, _ in METHODS}
    ref = runs["Random"]  # carries full query metadata
    queries = {k for k, r in ref.items() if source(r["query"]) and source(r["query"]) not in EXCLUDED_QUERY_SOURCES}
    queries &= set.intersection(*[set(r) for r in runs.values()])

    def shareable_top(name, key):
        return [c for c in runs[name][key]["topk"] if source(c) not in RESTRICTED][:K]

    # keep only scenes where every method returns a shareable candidate, so all bars share one query set
    queries = {k for k in queries if all(shareable_top(name, k) for name in runs)}

    scores = {}
    for name, _, _ in METHODS:
        acc = np.zeros(3)
        n = 0
        for key in queries:
            q = ref[key]["query"]
            top = shareable_top(name, key)
            n += 1
            acc += np.array([
                np.mean([circ(q["yaw_bin"], c["yaw_bin"]) <= YAW_TOL for c in top]),
                np.mean([abs(log_area(q["area_frac"]) - log_area(c["area_frac"])) <= LOG_AREA_TOL for c in top]),
                np.mean([abs(int(q["y_bin"]) - int(c["y_bin"])) <= YBIN_TOL for c in top]),
            ])
        scores[name] = acc / n
        print(f"{name:11s} n={n}  yaw={scores[name][0]:.3f}  scale={scores[name][1]:.3f}  placement={scores[name][2]:.3f}")
    n_q = len(queries)

    fig, ax = plt.subplots(figsize=(6.9, 3.8), dpi=200)
    width = 0.16
    x = np.arange(len(AXES))
    ax.axvspan(0.5, 2.5, color="#F2F2F2", zorder=0)
    for i, (name, _, color) in enumerate(METHODS):
        ax.bar(x + (i - 2) * width, scores[name], width, label=name, color=color,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-5 match to the scene's held-out vessel", fontsize=9)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8, zorder=1)
    ax.set_title(f"Where each method's retrieval matches the scene geometry\n"
                 f"(mean over {n_q:,} public query scenes, top-5)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=7.5, frameon=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}  ({n_q} queries)")


if __name__ == "__main__":
    main()
