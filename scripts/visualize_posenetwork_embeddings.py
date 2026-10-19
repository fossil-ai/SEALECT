#!/usr/bin/env python3
"""
visualize_posenetwork_embeddings_better.py

Better validation-set visualization for PoseNet embeddings.

What it shows:
1. A single overall t-SNE colored by coarse pose family
2. Nearest-neighbor contact sheet in embedding space (excluding restricted images)
3. Embedding-distance vs geodesic-distance scatter plot
4. Pose-family neighbor confusion matrix
5. Retrieval-style metrics:
   - pitch purity@k
   - yaw purity@k
   - roll purity@k
   - pose-family purity@k
   - mean geodesic neighbor distance@k
6. Optional K-means analysis on PCA-reduced embeddings
7. Saves images into folders per cluster

Edit only:
    TRAIN_MODULE_NAME
    CHECKPOINT_PATH
    OUT_DIR
"""

import os
import json
import math
import shutil
import importlib
import re
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


# 
# EDIT THESE
# 
TRAIN_MODULE_NAME = "pose3"
CHECKPOINT_PATH = "imback1/checkpoints/best_yaw.pt"
OUT_DIR = "val_pose_grouping_viz_better1"
K_NEIGHBORS = 5
NN_SHEET_QUERIES = 10
NN_SHEET_PER_ROW = 6
PAIR_SAMPLE_SIZE = 12000

EXCLUDE_DATASETS_FROM_CONTACT_SHEET = {"apl"}

RUN_KMEANS = True
KMEANS_PCA_DIM = 24
KMEANS_NUM_CLUSTERS = 8

SAVE_CLUSTER_FOLDERS = True
COPY_CLUSTER_IMAGES = True
CLUSTER_THUMB_CONTACT_SHEETS = True
CLUSTER_THUMB_SIZE = 160
CLUSTER_CONTACT_SHEET_COLS = 5
# 


def safe_slug(text: str) -> str:
    text = str(text)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def parse_dataset_tag(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[-1].lower()
    if len(parts) == 1:
        return parts[0].lower()
    return "unknown"


def make_pose_family(pitch_deg: float, yaw_deg: float, roll_deg: float) -> str:
    if pitch_deg < -60:
        return "nadir-ish"
    if pitch_deg > 40:
        return "steep-up"
    if abs(yaw_deg) < 30:
        return "front-back-ish"
    if 60 <= abs(yaw_deg) <= 120:
        return "side-ish"
    if abs(roll_deg) > 25:
        return "rolled"
    return "oblique"


def auto_perplexity(n: int) -> int:
    if n < 10:
        return max(2, n - 1)
    if n < 40:
        return max(5, n // 3)
    return min(30, max(5, n // 10))


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_tsne(coords, labels, title, outpath, point_size=24):
    plt.figure(figsize=(8, 7))
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab20", max(1, len(unique)))
    label_to_idx = {lab: i for i, lab in enumerate(unique)}
    colors = [label_to_idx[x] for x in labels]

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        cmap=cmap,
        s=point_size,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="k",
    )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)

    if len(unique) <= 20:
        handles = []
        for lab, idx in label_to_idx.items():
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=str(lab),
                    markerfacecolor=cmap(idx),
                    markeredgecolor="k",
                    markersize=7,
                )
            )
        plt.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def load_rgb_square(image_path: str, size: int = 120) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (size, size), method=Image.Resampling.BILINEAR)
    return img


def add_caption(img: Image.Image, text: str, height: int = 30) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, img.height + 7), text, fill=(0, 0, 0))
    return canvas


def save_nn_contact_sheet(records, neighbors_idx, outpath, query_indices, per_row=6, thumb_size=120):
    if len(query_indices) == 0:
        return

    rows = []
    for qidx in query_indices:
        row_imgs = []
        nbrs = neighbors_idx[qidx][:per_row]

        for rank, idx in enumerate(nbrs):
            rec = records[idx]
            img = load_rgb_square(rec["image_path"], thumb_size)
            prefix = "Q" if rank == 0 else str(rank)
            caption = (
                f"{prefix} | {rec['family']} | "
                f"P{rec['pitch_bin']} Y{rec['yaw_bin']} R{rec['roll_bin']}"
            )
            img = add_caption(img, caption)
            row_imgs.append(img)

        if not row_imgs:
            continue

        row_w = sum(im.width for im in row_imgs)
        row_h = max(im.height for im in row_imgs)
        row_canvas = Image.new("RGB", (row_w, row_h), (240, 240, 240))

        x = 0
        for im in row_imgs:
            row_canvas.paste(im, (x, 0))
            x += im.width

        rows.append(row_canvas)

    if not rows:
        return

    total_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows)
    canvas = Image.new("RGB", (total_w, total_h), (230, 230, 230))

    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height

    canvas.save(outpath)


def select_diverse_queries(families, datasets, excluded_datasets=None, max_queries=10):
    if excluded_datasets is None:
        excluded_datasets = set()

    valid = [i for i, ds in enumerate(datasets) if ds not in excluded_datasets]
    if not valid:
        return []

    fam_to_indices = defaultdict(list)
    for i in valid:
        fam_to_indices[families[i]].append(i)

    chosen = []
    for fam in sorted(fam_to_indices.keys()):
        idxs = fam_to_indices[fam]
        if idxs:
            chosen.append(idxs[len(idxs) // 2])

    if len(chosen) < max_queries:
        remaining = [i for i in valid if i not in chosen]
        if remaining:
            extra = np.linspace(
                0, len(remaining) - 1,
                num=min(max_queries - len(chosen), len(remaining)),
                dtype=int
            ).tolist()
            chosen.extend([remaining[i] for i in extra])

    return chosen[:max_queries]


def build_contact_neighbors(records, nbrs_wo_self, excluded_datasets, per_row):
    result = []
    for i, rec in enumerate(records):
        row = [i]
        for j in nbrs_wo_self[i]:
            if records[j]["dataset"] in excluded_datasets:
                continue
            row.append(int(j))
            if len(row) >= per_row:
                break
        result.append(row)
    return result


def rotation_matrix_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    Rd = R1.T @ R2
    tr = float(Rd[0, 0] + Rd[1, 1] + Rd[2, 2])
    c = np.clip((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return float(np.degrees(np.arccos(c)))


def plot_embedding_vs_geodesic(emb_dists, geo_dists, outpath, spearman_val):
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_dists, geo_dists, s=10, alpha=0.25, linewidths=0)
    plt.xlabel("Embedding distance")
    plt.ylabel("GT geodesic distance (deg)")
    plt.title(f"Embedding distance vs. GT geodesic distance\nSpearman = {spearman_val:.3f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, labels, title, outpath):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Neighbor pose family")
    plt.ylabel("Query pose family")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def majority_label(labels):
    c = Counter(labels)
    return c.most_common(1)[0][0]


def compute_cluster_purity(cluster_ids, gt_labels):
    cluster_to_labels = defaultdict(list)
    for cid, lab in zip(cluster_ids, gt_labels):
        cluster_to_labels[int(cid)].append(lab)

    correct = 0
    for labs in cluster_to_labels.values():
        correct += Counter(labs).most_common(1)[0][1]
    return float(correct / max(1, len(gt_labels)))


def plot_cluster_family_matrix(cluster_ids, families, num_clusters, family_labels, outpath):
    mat = np.zeros((num_clusters, len(family_labels)), dtype=np.float64)
    fam_to_idx = {f: i for i, f in enumerate(family_labels)}

    for cid, fam in zip(cluster_ids, families):
        mat[int(cid), fam_to_idx[fam]] += 1.0

    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, np.maximum(row_sums, 1e-12))

    plt.figure(figsize=(8, 7))
    plt.imshow(mat, interpolation="nearest", aspect="auto")
    plt.title("K-means cluster composition by pose family")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(family_labels)), family_labels, rotation=35, ha="right")
    plt.yticks(range(num_clusters), [f"cluster {i}" for i in range(num_clusters)])
    plt.xlabel("Pose family")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def safe_copy_or_link(src, dst, copy_mode=True):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def make_cluster_contact_sheet(records, indices, outpath, thumb_size=160, cols=5):
    if len(indices) == 0:
        return

    tiles = []
    for idx in indices:
        rec = records[idx]
        img = load_rgb_square(rec["image_path"], thumb_size)
        caption = (
            f"{rec['dataset']} | {rec['family']}\n"
            f"P{rec['pitch_bin']} Y{rec['yaw_bin']} R{rec['roll_bin']}"
        )
        cap_img = Image.new("RGB", (img.width, img.height + 34), (255, 255, 255))
        cap_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(cap_img)
        draw.text((4, img.height + 3), caption, fill=(0, 0, 0))
        tiles.append(cap_img)

    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    tile_w = max(t.width for t in tiles)
    tile_h = max(t.height for t in tiles)

    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (235, 235, 235))
    for n, tile in enumerate(tiles):
        r = n // cols
        c = n % cols
        canvas.paste(tile, (c * tile_w, r * tile_h))

    canvas.save(outpath)


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_mod = importlib.import_module(TRAIN_MODULE_NAME)
    cfg = train_mod.cfg
    device = train_mod.DEVICE
    train_mod.seed_all(cfg.SEED)

    val_ds = train_mod.BoatPoseDataset(
        cfg.VAL_ANNOTATION_DIR,
        cfg.VAL_IMAGE_DIR,
        cfg.IMG_SIZE,
        augment=False,
    )

    print(f"Loaded validation set with {len(val_ds)} samples from {cfg.VAL_IMAGE_DIR}")

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS if os.name != "nt" else 0,
        pin_memory=cfg.PIN_MEMORY and device == "cuda",
        drop_last=False,
    )

    model = train_mod.PoseNet(
        cfg.SEGFORMER_ENCODER_CKPT,
        cfg.SEGFORMER_UNFREEZE_LAST_N
    ).to(device)

    with torch.no_grad():
        _ = model(torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device))

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    hooked = {}

    def hook_fn(module, inputs, output):
        hooked["trunk"] = output.detach()

    hook = model.trunk.register_forward_hook(hook_fn)

    all_embeddings = []
    all_gt_bins = []
    all_gt_euler = []
    all_gt_rotmats = []
    all_names = []
    all_image_paths = []
    all_datasets = []

    for batch in val_dl:
        imgs, R_gt, euler_gt, bins_gt, names = batch
        imgs = imgs.to(device, non_blocking=True)

        _ = model(imgs)
        emb = hooked["trunk"].float().cpu().numpy()

        all_embeddings.append(emb)
        all_gt_bins.append(bins_gt.cpu().numpy())
        all_gt_euler.append(euler_gt.cpu().numpy())
        all_gt_rotmats.append(R_gt.cpu().numpy())
        all_names.extend(list(names))

        for name in names:
            all_image_paths.append(os.path.join(val_ds.img_dir, name))
            all_datasets.append(parse_dataset_tag(name))

    hook.remove()

    X = np.concatenate(all_embeddings, axis=0)
    gt_bins = np.concatenate(all_gt_bins, axis=0)
    gt_euler = np.concatenate(all_gt_euler, axis=0)
    gt_rotmats = np.concatenate(all_gt_rotmats, axis=0)

    pose_family_labels = [
        make_pose_family(float(p), float(y), float(r))
        for p, y, r in gt_euler
    ]

    unique_datasets = sorted(set(all_datasets))
    dataset_counts = {d: all_datasets.count(d) for d in unique_datasets}
    print("Dataset counts inferred from filenames:")
    for k, v in dataset_counts.items():
        print(f"  {k}: {v}")

    pca_dim = min(50, X.shape[1], max(2, X.shape[0] - 1))
    X_pca = PCA(n_components=pca_dim, random_state=cfg.SEED).fit_transform(X)

    perplexity = auto_perplexity(len(X_pca))
    X_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=cfg.SEED,
    ).fit_transform(X_pca)

    plot_tsne(
        X_tsne,
        pose_family_labels,
        "Validation embeddings t-SNE (coarse pose family)",
        os.path.join(OUT_DIR, "tsne_overall_pose_family.png"),
    )

    nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, len(X)), metric="euclidean")
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X)
    nbrs_wo_self = nbrs[:, 1:]

    pitch_purity = []
    yaw_purity = []
    roll_purity = []
    family_purity = []
    mean_geo_neighbor = []

    families_unique = sorted(set(pose_family_labels))
    fam_to_idx = {f: i for i, f in enumerate(families_unique)}
    cm_counts = np.zeros((len(families_unique), len(families_unique)), dtype=np.float64)

    records = []

    for i in range(len(X)):
        query_pitch = int(gt_bins[i, 0])
        query_yaw = int(gt_bins[i, 1])
        query_roll = int(gt_bins[i, 2])
        query_family = pose_family_labels[i]

        nbr_idxs = nbrs_wo_self[i]

        pitch_hits = 0
        yaw_hits = 0
        roll_hits = 0
        family_hits = 0
        geo_vals = []

        for j in nbr_idxs:
            if int(gt_bins[j, 0]) == query_pitch:
                pitch_hits += 1
            if int(gt_bins[j, 1]) == query_yaw:
                yaw_hits += 1
            if int(gt_bins[j, 2]) == query_roll:
                roll_hits += 1
            if pose_family_labels[j] == query_family:
                family_hits += 1

            geo_vals.append(rotation_matrix_geodesic_deg(gt_rotmats[i], gt_rotmats[j]))
            cm_counts[fam_to_idx[query_family], fam_to_idx[pose_family_labels[j]]] += 1.0

        pitch_purity.append(pitch_hits / len(nbr_idxs))
        yaw_purity.append(yaw_hits / len(nbr_idxs))
        roll_purity.append(roll_hits / len(nbr_idxs))
        family_purity.append(family_hits / len(nbr_idxs))
        mean_geo_neighbor.append(float(np.mean(geo_vals)))

        records.append({
            "name": all_names[i],
            "image_path": all_image_paths[i],
            "dataset": all_datasets[i],
            "family": query_family,
            "pitch_bin": query_pitch,
            "yaw_bin": query_yaw,
            "roll_bin": query_roll,
        })

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts, np.maximum(row_sums, 1e-12))

    plot_confusion_matrix(
        cm_norm,
        families_unique,
        f"Pose-family neighbor confusion (K={K_NEIGHBORS})",
        os.path.join(OUT_DIR, "pose_family_neighbor_confusion.png"),
    )

    rng = np.random.default_rng(cfg.SEED)
    num_points = len(X)
    max_pairs = num_points * (num_points - 1) // 2
    pair_count = min(PAIR_SAMPLE_SIZE, max_pairs)

    emb_pair_dists = []
    geo_pair_dists = []

    seen = set()
    while len(emb_pair_dists) < pair_count:
        i = int(rng.integers(0, num_points))
        j = int(rng.integers(0, num_points))
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        emb_d = float(np.linalg.norm(X[a] - X[b]))
        geo_d = rotation_matrix_geodesic_deg(gt_rotmats[a], gt_rotmats[b])

        emb_pair_dists.append(emb_d)
        geo_pair_dists.append(geo_d)

    emb_pair_dists = np.asarray(emb_pair_dists, dtype=np.float64)
    geo_pair_dists = np.asarray(geo_pair_dists, dtype=np.float64)
    spearman_val = spearman_corr(emb_pair_dists, geo_pair_dists)

    plot_embedding_vs_geodesic(
        emb_pair_dists,
        geo_pair_dists,
        os.path.join(OUT_DIR, "embedding_vs_geodesic_scatter.png"),
        spearman_val,
    )

    query_indices = select_diverse_queries(
        pose_family_labels,
        all_datasets,
        excluded_datasets=EXCLUDE_DATASETS_FROM_CONTACT_SHEET,
        max_queries=NN_SHEET_QUERIES
    )

    contact_neighbors = build_contact_neighbors(
        records=records,
        nbrs_wo_self=nbrs_wo_self,
        excluded_datasets=EXCLUDE_DATASETS_FROM_CONTACT_SHEET,
        per_row=NN_SHEET_PER_ROW
    )

    save_nn_contact_sheet(
        records=records,
        neighbors_idx=contact_neighbors,
        outpath=os.path.join(OUT_DIR, "nn_contact_sheet_no_apl.png"),
        query_indices=query_indices,
        per_row=NN_SHEET_PER_ROW,
    )

    kmeans_stats = {}
    if RUN_KMEANS:
        k_pca_dim = min(KMEANS_PCA_DIM, X_pca.shape[1])
        X_kmeans = X_pca[:, :k_pca_dim]

        kmeans_k = min(KMEANS_NUM_CLUSTERS, len(X_kmeans))
        kmeans = KMeans(
            n_clusters=kmeans_k,
            random_state=cfg.SEED,
            n_init=20,
        )
        cluster_ids = kmeans.fit_predict(X_kmeans)

        plot_tsne(
            X_tsne,
            [int(c) for c in cluster_ids],
            "Validation embeddings t-SNE (K-means clusters)",
            os.path.join(OUT_DIR, "tsne_kmeans_clusters.png"),
        )

        family_purity_cluster = compute_cluster_purity(cluster_ids, pose_family_labels)
        yaw_purity_cluster = compute_cluster_purity(cluster_ids, [int(x) for x in gt_bins[:, 1]])

        plot_cluster_family_matrix(
            cluster_ids,
            pose_family_labels,
            num_clusters=kmeans_k,
            family_labels=families_unique,
            outpath=os.path.join(OUT_DIR, "kmeans_cluster_pose_family_matrix.png"),
        )

        cluster_sizes = Counter([int(c) for c in cluster_ids])

        cluster_summary = {}
        if SAVE_CLUSTER_FOLDERS:
            cluster_root = os.path.join(OUT_DIR, "clusters")
            os.makedirs(cluster_root, exist_ok=True)

            cluster_to_indices = defaultdict(list)
            for idx, cid in enumerate(cluster_ids):
                cluster_to_indices[int(cid)].append(idx)

            for cid in range(kmeans_k):
                cluster_dir = os.path.join(cluster_root, f"cluster_{cid:02d}")
                images_dir = os.path.join(cluster_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

                idxs = cluster_to_indices[cid]

                fams = [pose_family_labels[i] for i in idxs]
                yaws = [int(gt_bins[i, 1]) for i in idxs]
                datasets = [all_datasets[i] for i in idxs]

                summary = {
                    "cluster_id": int(cid),
                    "size": int(len(idxs)),
                    "majority_pose_family": majority_label(fams) if fams else None,
                    "pose_family_counts": dict(Counter(fams)),
                    "yaw_bin_counts": dict(Counter(yaws)),
                    "dataset_counts": dict(Counter(datasets)),
                    "members": [],
                }

                for rank, idx in enumerate(idxs):
                    rec = records[idx]
                    src = rec["image_path"]
                    base = os.path.basename(src)
                    dst_name = (
                        f"{rank:03d}_"
                        f"{safe_slug(rec['dataset'])}_"
                        f"{safe_slug(rec['family'])}_"
                        f"P{rec['pitch_bin']}_Y{rec['yaw_bin']}_R{rec['roll_bin']}_"
                        f"{safe_slug(base)}"
                    )
                    dst = os.path.join(images_dir, dst_name)

                    safe_copy_or_link(src, dst, copy_mode=COPY_CLUSTER_IMAGES)

                    summary["members"].append({
                        "rank_in_cluster": int(rank),
                        "source_image_path": src,
                        "saved_image_path": dst,
                        "dataset": rec["dataset"],
                        "pose_family": rec["family"],
                        "pitch_bin": int(rec["pitch_bin"]),
                        "yaw_bin": int(rec["yaw_bin"]),
                        "roll_bin": int(rec["roll_bin"]),
                        "name": rec["name"],
                    })

                with open(os.path.join(cluster_dir, "cluster_info.json"), "w") as f:
                    json.dump(summary, f, indent=2)

                if CLUSTER_THUMB_CONTACT_SHEETS:
                    make_cluster_contact_sheet(
                        records=records,
                        indices=idxs,
                        outpath=os.path.join(cluster_dir, "cluster_contact_sheet.png"),
                        thumb_size=CLUSTER_THUMB_SIZE,
                        cols=CLUSTER_CONTACT_SHEET_COLS,
                    )

                cluster_summary[f"cluster_{cid:02d}"] = {
                    "size": int(len(idxs)),
                    "majority_pose_family": summary["majority_pose_family"],
                    "pose_family_counts": summary["pose_family_counts"],
                    "dataset_counts": summary["dataset_counts"],
                }

            with open(os.path.join(cluster_root, "cluster_summary.json"), "w") as f:
                json.dump(cluster_summary, f, indent=2)

        kmeans_stats = {
            "ran_kmeans": True,
            "kmeans_num_clusters": int(kmeans_k),
            "kmeans_pca_dim": int(k_pca_dim),
            "kmeans_family_cluster_purity": float(family_purity_cluster),
            "kmeans_yaw_cluster_purity": float(yaw_purity_cluster),
            "kmeans_cluster_sizes": dict(cluster_sizes),
            "saved_cluster_folders": bool(SAVE_CLUSTER_FOLDERS),
        }
    else:
        kmeans_stats = {"ran_kmeans": False}

    stats = {
        "checkpoint": CHECKPOINT_PATH,
        "num_samples": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "pca_dim_for_tsne": int(pca_dim),
        "overall_tsne_perplexity": int(perplexity),
        "num_inferred_datasets": int(len(unique_datasets)),
        "dataset_counts": dataset_counts,
        f"pitch_purity_at_{K_NEIGHBORS}": float(np.mean(pitch_purity)),
        f"yaw_purity_at_{K_NEIGHBORS}": float(np.mean(yaw_purity)),
        f"roll_purity_at_{K_NEIGHBORS}": float(np.mean(roll_purity)),
        f"pose_family_purity_at_{K_NEIGHBORS}": float(np.mean(family_purity)),
        f"mean_geodesic_neighbor_at_{K_NEIGHBORS}_deg": float(np.mean(mean_geo_neighbor)),
        "embedding_vs_geodesic_spearman": float(spearman_val),
        "excluded_from_contact_sheet": sorted(EXCLUDE_DATASETS_FROM_CONTACT_SHEET),
        "contact_sheet_queries_used": int(len(query_indices)),
        **kmeans_stats,
    }

    with open(os.path.join(OUT_DIR, "embedding_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()