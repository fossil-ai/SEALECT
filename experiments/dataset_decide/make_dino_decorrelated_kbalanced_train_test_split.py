#!/usr/bin/env python3
"""
make_dino_decorrelated_kbalanced_train_test_split.py

Train/test split using crop-level DINO visual de-correlation.

No filename grouping.
No sequence grouping.
No validation set.

Protocol:
  1. Embed every vessel crop using DINOv2.
  2. Build crop-level near-duplicate clusters.
  3. Select a k-balanced, source-aware, visually decorrelated TEST set.
  4. Promote entire images to test if any crop from that image is selected.
  5. Expand test to include all images sharing selected duplicate clusters.
  6. Remove train images too visually close to test.
  7. Put everything else into train.
  8. Save split JSONs and reviewer-facing diagnostics.

Outputs:
  splits_dino_decorrelated_kbalanced/
    train_scene_object_pairs.json
    test_scene_object_pairs.json
    train_candidate_library.json
    test_candidate_library.json
    split_summary.json
    source_split_report.csv
    context_split_report.csv
    cluster_manifest.json
"""

import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter, deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


# ---------------- CONFIG ----------------

# SCENE_PAIRS_JSON = Path("merged_scene_pose_library_out/scene_object_pairs.json")
# CANDIDATE_LIBRARY_JSON = Path("merged_scene_pose_library_out/candidate_library.json")
SCENE_PAIRS_JSON = Path("merged_scene_pose_library_out_grounded_sam2/scene_object_pairs.json")
CANDIDATE_LIBRARY_JSON = Path("merged_scene_pose_library_out_grounded_sam2/candidate_library.json")
LIBRARY_ROOT = Path("/path/to/sealect_root/experiments/dataset_decide")
CROP_DIR = LIBRARY_ROOT / "merged_scene_pose_library_out_grounded_sam2" / "crops"

OUT_DIR = Path("splits_dino_decorrelated_kbalanced_grounded_sam2")
CACHE_DIR = OUT_DIR / "cache"

RANDOM_SEED = 42

# Target held-out test fraction at crop/object level.
# TEST_FRAC = 0.10
# MAX_TEST_PER_CONTEXT = 200
# MIN_TEST_PER_CONTEXT = 20
TEST_FRAC = 0.05
MAX_TEST_PER_CONTEXT = 120
MIN_TEST_PER_CONTEXT = 15
MAX_CLUSTER_SIZE_FOR_INITIAL_TEST = 25
# Keep test balanced enough across context bins.
# Train receives everything else.
# MAX_TEST_PER_CONTEXT = 400
# MIN_TEST_PER_CONTEXT = 30

# DINO settings.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "dinov2_vits14"
BATCH_SIZE = 64

# Duplicate clustering threshold.
# Lower = more aggressive duplicate grouping.
DUP_SIM_THRESHOLD = 0.95

# During test selection, avoid adding test images too visually close to already selected test.
TEST_TEST_SIM_THRESHOLD = 0.93

# After test is selected, remove train images too close to test.
TRAIN_TEST_SIM_THRESHOLD = 0.94

# Approximate neighbors per crop for duplicate graph.
TOP_NEIGHBORS = 80

# FAISS is strongly recommended for large datasets.
USE_FAISS_IF_AVAILABLE = True

# Torch fallback chunk size if FAISS is unavailable.
SIM_CHUNK = 1024

# ----------------------------------------
def resolve_crop_path(crop_path):
    if not crop_path:
        return None

    p = Path(crop_path)

    if p.is_absolute() and p.exists():
        return p

    if p.exists():
        return p

    candidate = LIBRARY_ROOT / p
    if candidate.exists():
        return candidate

    candidate = CROP_DIR / p.name
    if candidate.exists():
        return candidate

    return None

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def key_for_record(r):
    return (int(r["image_id"]), int(r["annotation_id"]))


def image_key(r):
    return int(r["image_id"])


def source_of(r):
    return str(r.get("source", r.get("dataset", "unknown")))


def context_of(r):
    return str(r.get("context_key", "missing"))


def dino_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@torch.no_grad()
def load_or_compute_dino_embeddings(candidates):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "candidate_dino_embeddings.pt"

    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        print(f"Loaded cached embeddings: {cache_path}")
        return obj["embeddings"].numpy().astype(np.float32), list(obj["valid_indices"])

    print("Loading DINOv2...")
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL_NAME).to(DEVICE)
    model.eval()
    tf = dino_transform()

    embs = []
    valid_indices = []

    for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="DINO crop embeddings"):
        batch = candidates[i:i + BATCH_SIZE]
        imgs = []
        idxs = []

        for j, c in enumerate(batch):
            crop_path = resolve_crop_path(c.get("crop_path"))

            if crop_path is None:
                continue
            try:
                img = Image.open(crop_path).convert("RGB")
                imgs.append(tf(img))
                idxs.append(i + j)
            except Exception:
                continue

        if not imgs:
            continue

        x = torch.stack(imgs).to(DEVICE)
        feats = model(x)

        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", None)
            if feats is None:
                raise RuntimeError("Unexpected DINO output dict.")

        feats = F.normalize(feats, dim=-1).cpu()
        embs.append(feats)
        valid_indices.extend(idxs)

    if not embs:
        raise RuntimeError("No embeddings computed. Check crop_path values.")

    embs = torch.cat(embs, dim=0).numpy().astype(np.float32)

    torch.save(
        {
            "embeddings": torch.from_numpy(embs),
            "valid_indices": valid_indices,
        },
        cache_path,
    )

    print(f"Saved cached embeddings: {cache_path}")
    return embs, valid_indices


def try_import_faiss():
    if not USE_FAISS_IF_AVAILABLE:
        return None
    try:
        import faiss
        return faiss
    except Exception:
        return None


def search_topk(emb, k):
    """
    Returns sims, indices of approximate/exact top-k cosine neighbors.
    emb must be float32 L2-normalized.
    """
    faiss = try_import_faiss()

    if faiss is not None:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb.astype(np.float32))
        sims, idxs = index.search(emb.astype(np.float32), min(k + 1, len(emb)))
        return sims[:, 1:], idxs[:, 1:]

    print("FAISS unavailable; using torch fallback. This may be slow.")
    n = emb.shape[0]
    emb_t = torch.from_numpy(emb).float()
    if DEVICE == "cuda":
        emb_t = emb_t.to(DEVICE)

    all_sims = []
    all_idxs = []

    for start in tqdm(range(0, n, SIM_CHUNK), desc="torch top-k"):
        end = min(n, start + SIM_CHUNK)
        q = emb_t[start:end]
        sims = q @ emb_t.T

        row_ids = torch.arange(start, end, device=sims.device)
        sims[torch.arange(end - start, device=sims.device), row_ids] = -1.0

        vals, idxs = torch.topk(sims, k=min(k, n - 1), dim=1)
        all_sims.append(vals.cpu().numpy())
        all_idxs.append(idxs.cpu().numpy())

    return np.concatenate(all_sims, axis=0), np.concatenate(all_idxs, axis=0)


def build_duplicate_clusters(emb):
    sims, idxs = search_topk(emb, TOP_NEIGHBORS)

    n = emb.shape[0]
    neighbors = [[] for _ in range(n)]

    for i in tqdm(range(n), desc="building duplicate graph"):
        for sim, j in zip(sims[i], idxs[i]):
            if sim >= DUP_SIM_THRESHOLD:
                neighbors[i].append(int(j))

    seen = np.zeros(n, dtype=bool)
    comps = []

    for i in range(n):
        if seen[i]:
            continue

        q = deque([i])
        seen[i] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)

        comps.append(comp)

    return comps


def cluster_centroids(emb, comps):
    cents = []
    for comp in comps:
        c = emb[comp].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        cents.append(c.astype(np.float32))
    return np.stack(cents, axis=0)


def max_sim_to_set(query_vectors, ref_vectors):
    if len(query_vectors) == 0 or len(ref_vectors) == 0:
        return np.zeros(len(query_vectors), dtype=np.float32) - 1.0

    faiss = try_import_faiss()

    if faiss is not None:
        index = faiss.IndexFlatIP(ref_vectors.shape[1])
        index.add(ref_vectors.astype(np.float32))
        sims, _ = index.search(query_vectors.astype(np.float32), 1)
        return sims[:, 0]

    q = torch.from_numpy(query_vectors).float()
    r = torch.from_numpy(ref_vectors).float()
    if DEVICE == "cuda":
        q = q.to(DEVICE)
        r = r.to(DEVICE)

    out = []
    for start in range(0, len(q), SIM_CHUNK):
        end = min(len(q), start + SIM_CHUNK)
        sims = q[start:end] @ r.T
        out.append(sims.max(dim=1).values.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_image_units(records, crop_cluster_ids, cluster_centroids_arr):
    """
    Unit = image. Entire images are assigned to train or test.
    Each image stores all crop indices and crop-cluster IDs.
    """
    by_image = defaultdict(list)
    for i, r in enumerate(records):
        by_image[image_key(r)].append(i)

    units = []

    for img_id, idxs in by_image.items():
        rows = [records[i] for i in idxs]
        clusters = sorted(set(int(crop_cluster_ids[i]) for i in idxs))

        k_counts = Counter(context_of(r) for r in rows)
        s_counts = Counter(source_of(r) for r in rows)

        # Image embedding = mean of its crop-cluster centroids.
        vec = cluster_centroids_arr[np.array(clusters)].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        units.append({
            "image_id": img_id,
            "indices": idxs,
            "clusters": clusters,
            "size": len(idxs),
            "major_context": k_counts.most_common(1)[0][0],
            "major_source": s_counts.most_common(1)[0][0],
            "context_counts": k_counts,
            "source_counts": s_counts,
            "embedding": vec.astype(np.float32),
        })

    return units


def target_test_counts(records):
    by_context = Counter(context_of(r) for r in records)
    targets = {}

    for k, n in by_context.items():
        t = int(round(TEST_FRAC * n))
        t = min(t, MAX_TEST_PER_CONTEXT)
        if n >= MIN_TEST_PER_CONTEXT:
            t = max(MIN_TEST_PER_CONTEXT, t)
        else:
            t = 0
        targets[k] = t

    return targets

def choose_test_images(units, records, seed, cluster_size_lookup):
    """
    Fast k-balanced, source-aware, visually decorrelated test selection.

    Rejects images that belong to very large duplicate clusters so they do not
    seed huge test-set expansion. Large redundant clusters are left for train.
    """
    rng = random.Random(seed)

    targets_k = target_test_counts(records)

    units_by_k = defaultdict(list)
    for u in units:
        units_by_k[u["major_context"]].append(u)

    for k in units_by_k:
        rng.shuffle(units_by_k[k])

    selected_image_ids = set()
    selected_units = []
    selected_k = Counter()
    selected_s = Counter()

    contexts = list(units_by_k.keys())
    rng.shuffle(contexts)

    skipped_large_cluster = 0
    skipped_too_similar = 0

    print("Selecting test images with fast decorrelated sampler...", flush=True)

    made_progress = True
    while made_progress:
        made_progress = False

        for k in contexts:
            target = targets_k.get(k, 0)

            if target <= 0:
                continue

            if selected_k[k] >= target:
                continue

            bucket = units_by_k[k]

            while bucket:
                u = bucket.pop()

                if u["image_id"] in selected_image_ids:
                    continue

                # Do not allow huge duplicate clusters to seed test expansion.
                largest_cluster_size = max(
                    cluster_size_lookup.get(c, 1)
                    for c in u["clusters"]
                )

                if largest_cluster_size > MAX_CLUSTER_SIZE_FOR_INITIAL_TEST:
                    skipped_large_cluster += 1
                    continue

                # Visual decorrelation against already-selected test images.
                if selected_units:
                    selected_vecs = np.stack(
                        [x["embedding"] for x in selected_units],
                        axis=0,
                    )

                    max_sim = float((selected_vecs @ u["embedding"]).max())

                    if max_sim >= TEST_TEST_SIM_THRESHOLD:
                        skipped_too_similar += 1
                        continue

                selected_units.append(u)
                selected_image_ids.add(u["image_id"])

                for i in u["indices"]:
                    selected_k[context_of(records[i])] += 1
                    selected_s[source_of(records[i])] += 1

                made_progress = True
                break

        total_selected = sum(selected_k.values())
        total_target = sum(targets_k.values())

        print(
            f"  selected crops: {total_selected}/{total_target} | "
            f"selected images: {len(selected_image_ids)} | "
            f"skipped_large_cluster: {skipped_large_cluster} | "
            f"skipped_too_similar: {skipped_too_similar}",
            flush=True,
        )

        if total_selected >= total_target:
            break

    print("Finished test image selection.", flush=True)
    print("Selected images:", len(selected_image_ids), flush=True)
    print("Selected crops by context:", dict(selected_k), flush=True)
    print("Skipped due to large duplicate cluster:", skipped_large_cluster, flush=True)
    print("Skipped due to test-test similarity:", skipped_too_similar, flush=True)

    return selected_image_ids
#
# def choose_test_images(units, records, seed):
#     """
#     Greedy k-balanced, source-aware, visually decorrelated test selection.
#     """
#     rng = random.Random(seed)
#
#     targets_k = target_test_counts(records)
#     total_target = sum(targets_k.values())
#
#     total_by_source = Counter(source_of(r) for r in records)
#     target_by_source = {
#         s: max(1, int(round(TEST_FRAC * n)))
#         for s, n in total_by_source.items()
#     }
#
#     selected_image_ids = set()
#     selected_units = []
#     selected_n = 0
#     selected_k = Counter()
#     selected_s = Counter()
#
#     remaining = units[:]
#     rng.shuffle(remaining)
#
#     while remaining and selected_n < total_target:
#         selected_vecs = (
#             np.stack([u["embedding"] for u in selected_units], axis=0)
#             if selected_units else None
#         )
#
#         best = None
#         best_score = -1e18
#
#         for u in remaining:
#             k = u["major_context"]
#             s = u["major_source"]
#
#             if targets_k.get(k, 0) <= selected_k[k]:
#                 continue
#
#             if selected_vecs is not None:
#                 sim = float(u["embedding"] @ selected_vecs.T.max(axis=0)) if False else None
#                 max_sim = float((selected_vecs @ u["embedding"]).max())
#                 if max_sim >= TEST_TEST_SIM_THRESHOLD:
#                     continue
#
#             k_need = max(0, targets_k.get(k, 0) - selected_k[k])
#             s_need = max(0, target_by_source.get(s, 0) - selected_s[s])
#
#             # Prefer underfilled contexts, underfilled sources, and smaller images.
#             score = 3.0 * k_need + 1.0 * s_need - 0.10 * math.log1p(u["size"])
#
#             if score > best_score:
#                 best_score = score
#                 best = u
#
#         if best is None:
#             break
#
#         selected_units.append(best)
#         selected_image_ids.add(best["image_id"])
#         selected_n += best["size"]
#
#         for i in best["indices"]:
#             selected_k[context_of(records[i])] += 1
#             selected_s[source_of(records[i])] += 1
#
#         remaining = [u for u in remaining if u["image_id"] != best["image_id"]]
#
#     return selected_image_ids


def expand_test_by_duplicate_clusters(test_image_ids, units):
    """
    If any image in a duplicate cluster is test, all images sharing that
    crop cluster become test. This avoids crop-cluster leakage.
    """
    cluster_to_images = defaultdict(set)

    for u in units:
        for c in u["clusters"]:
            cluster_to_images[c].add(u["image_id"])

    changed = True
    expanded = set(test_image_ids)

    while changed:
        changed = False
        current_clusters = set()

        for u in units:
            if u["image_id"] in expanded:
                current_clusters.update(u["clusters"])

        for c in current_clusters:
            for img_id in cluster_to_images[c]:
                if img_id not in expanded:
                    expanded.add(img_id)
                    changed = True

    return expanded


def split_units(records, units, test_image_ids, cluster_centroids_arr):
    test_units = [u for u in units if u["image_id"] in test_image_ids]
    train_units_raw = [u for u in units if u["image_id"] not in test_image_ids]

    test_vecs = np.stack([u["embedding"] for u in test_units], axis=0)

    train_vecs = np.stack([u["embedding"] for u in train_units_raw], axis=0) if train_units_raw else np.zeros((0, test_vecs.shape[1]), dtype=np.float32)
    max_sims = max_sim_to_set(train_vecs.astype(np.float32), test_vecs.astype(np.float32))

    train_units = []
    removed_units = []
    for u, sim in zip(train_units_raw, max_sims):
        u = dict(u)
        u["max_sim_to_test"] = float(sim)
        if sim >= TRAIN_TEST_SIM_THRESHOLD:
            removed_units.append(u)
        else:
            train_units.append(u)

    def records_from_units(us):
        idxs = []
        for u in us:
            idxs.extend(u["indices"])
        return [records[i] for i in sorted(idxs)]

    return records_from_units(train_units), records_from_units(test_units), removed_units, max_sims


def filter_candidates_for_records(candidates, records):
    allowed = set(key_for_record(r) for r in records)
    return [c for c in candidates if key_for_record(c) in allowed]


def summarize(records, name):
    by_source_records = Counter(source_of(r) for r in records)
    by_source_images = defaultdict(set)
    by_context = Counter(context_of(r) for r in records)
    by_scene = Counter(str(r.get("scene_type", "missing")) for r in records)

    for r in records:
        by_source_images[source_of(r)].add(image_key(r))

    return {
        "name": name,
        "num_crops": len(records),
        "num_images": len(set(image_key(r) for r in records)),
        "sources_crops": dict(by_source_records.most_common()),
        "sources_images": {k: len(v) for k, v in sorted(by_source_images.items())},
        "num_context_keys": len(by_context),
        "context_keys": dict(by_context.most_common()),
        "scene_types": dict(by_scene.most_common()),
    }


def make_source_report(all_records, train_records, test_records):
    def counts(records):
        crops = Counter(source_of(r) for r in records)
        imgs = defaultdict(set)
        for r in records:
            imgs[source_of(r)].add(image_key(r))
        return crops, {k: len(v) for k, v in imgs.items()}

    all_c, all_i = counts(all_records)
    tr_c, tr_i = counts(train_records)
    te_c, te_i = counts(test_records)

    sources = sorted(set(all_c) | set(tr_c) | set(te_c))
    rows = []

    for s in sources:
        rows.append({
            "source": s,
            "all_images": all_i.get(s, 0),
            "all_crops": all_c.get(s, 0),
            "train_images": tr_i.get(s, 0),
            "train_crops": tr_c.get(s, 0),
            "test_images": te_i.get(s, 0),
            "test_crops": te_c.get(s, 0),
            "test_crop_frac": round(te_c.get(s, 0) / max(1, all_c.get(s, 0)), 4),
        })

    return rows


def make_context_report(all_records, train_records, test_records):
    def counts(records):
        return Counter(context_of(r) for r in records)

    all_c = counts(all_records)
    tr_c = counts(train_records)
    te_c = counts(test_records)

    contexts = sorted(all_c)
    rows = []

    for k in contexts:
        rows.append({
            "context_key": k,
            "all_crops": all_c.get(k, 0),
            "train_crops": tr_c.get(k, 0),
            "test_crops": te_c.get(k, 0),
            "test_crop_frac": round(te_c.get(k, 0) / max(1, all_c.get(k, 0)), 4),
        })

    return rows


def annotate_records(records, split, key_to_cluster):
    out = []
    for r in records:
        rr = dict(r)
        rr["split"] = split
        rr["dedup_cluster_id"] = int(key_to_cluster[key_for_record(r)])
        rr["source"] = rr.get("source", rr.get("dataset", "unknown"))
        out.append(rr)
    return out


def annotate_candidates(candidates, split, key_to_cluster):
    out = []
    for c in candidates:
        cc = dict(c)
        cc["split"] = split
        cc["dedup_cluster_id"] = int(key_to_cluster[key_for_record(c)])
        cc["source"] = cc.get("source", cc.get("dataset", "unknown"))
        out.append(cc)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    scene_pairs = load_json(SCENE_PAIRS_JSON)
    candidates = load_json(CANDIDATE_LIBRARY_JSON)

    print(f"Loaded scene pairs: {len(scene_pairs)}")
    print(f"Loaded candidates:   {len(candidates)}")

    candidate_by_key = {key_for_record(c): c for c in candidates}
    paired = [r for r in scene_pairs if key_for_record(r) in candidate_by_key]
    candidates_paired = [candidate_by_key[key_for_record(r)] for r in paired]

    print(f"Paired records: {len(paired)}")

    emb, valid_indices = load_or_compute_dino_embeddings(candidates_paired)

    candidates_valid = [candidates_paired[i] for i in valid_indices]
    records_valid = [paired[i] for i in valid_indices]

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    print(f"Valid embedded crops: {len(records_valid)}")
    print("Building duplicate clusters...", flush=True)
    comps = build_duplicate_clusters(emb)
    cluster_size_lookup = {ci: len(comp) for ci, comp in enumerate(comps)}
    print("Computing cluster centroids...", flush=True)
    centroids = cluster_centroids(emb, comps)

    crop_to_cluster = {}
    for ci, comp in enumerate(comps):
        for i in comp:
            crop_to_cluster[i] = ci

    crop_cluster_ids = [crop_to_cluster[i] for i in range(len(records_valid))]

    key_to_cluster = {
        key_for_record(r): int(crop_cluster_ids[i])
        for i, r in enumerate(records_valid)
    }
    print("Building image units...", flush=True)
    units = build_image_units(records_valid, crop_cluster_ids, centroids)
    print("Choosing test images...", flush=True)
    # initial_test_images = choose_test_images(
    #     units=units,
    #     records=records_valid,
    #     seed=RANDOM_SEED,
    # )
    initial_test_images = choose_test_images(
        units=units,
        records=records_valid,
        seed=RANDOM_SEED,
        cluster_size_lookup=cluster_size_lookup,
    )
    print("Expanding test by duplicate clusters...", flush=True)
    expanded_test_images = expand_test_by_duplicate_clusters(initial_test_images, units)
    print("Splitting train/test...", flush=True)
    train_records_raw, test_records_raw, removed_units, train_test_sims = split_units(
        records=records_valid,
        units=units,
        test_image_ids=expanded_test_images,
        cluster_centroids_arr=centroids,
    )

    train_candidates_raw = filter_candidates_for_records(candidates_valid, train_records_raw)
    test_candidates_raw = filter_candidates_for_records(candidates_valid, test_records_raw)

    train_records = annotate_records(train_records_raw, "train", key_to_cluster)
    test_records = annotate_records(test_records_raw, "test", key_to_cluster)
    train_candidates = annotate_candidates(train_candidates_raw, "train", key_to_cluster)
    test_candidates = annotate_candidates(test_candidates_raw, "test", key_to_cluster)

    # Cluster manifest
    cluster_manifest = []
    for ci, comp in enumerate(comps):
        rows = [records_valid[i] for i in comp]
        splits = []
        for r in rows:
            if key_for_record(r) in set(key_for_record(x) for x in test_records_raw):
                splits.append("test")
            elif key_for_record(r) in set(key_for_record(x) for x in train_records_raw):
                splits.append("train")
            else:
                splits.append("removed")

        cluster_manifest.append({
            "dedup_cluster_id": ci,
            "size": len(comp),
            "split_labels": dict(Counter(splits)),
            "sources": dict(Counter(source_of(r) for r in rows)),
            "context_keys": dict(Counter(context_of(r) for r in rows).most_common()),
            "example_candidate_ids": [
                candidates_valid[i].get("candidate_id") for i in comp[:5]
            ],
        })

    save_json(train_records, OUT_DIR / "train_scene_object_pairs.json")
    save_json(test_records, OUT_DIR / "test_scene_object_pairs.json")
    save_json(train_candidates, OUT_DIR / "train_candidate_library.json")
    save_json(test_candidates, OUT_DIR / "test_candidate_library.json")
    save_json(cluster_manifest, OUT_DIR / "cluster_manifest.json")

    save_csv(make_source_report(records_valid, train_records_raw, test_records_raw), OUT_DIR / "source_split_report.csv")
    save_csv(make_context_report(records_valid, train_records_raw, test_records_raw), OUT_DIR / "context_split_report.csv")

    cluster_sizes = [len(c) for c in comps]

    summary = {
        "config": {
            "random_seed": RANDOM_SEED,
            "test_frac": TEST_FRAC,
            "max_test_per_context": MAX_TEST_PER_CONTEXT,
            "min_test_per_context": MIN_TEST_PER_CONTEXT,
            "dino_model": DINO_MODEL_NAME,
            "dup_sim_threshold": DUP_SIM_THRESHOLD,
            "test_test_sim_threshold": TEST_TEST_SIM_THRESHOLD,
            "train_test_sim_threshold": TRAIN_TEST_SIM_THRESHOLD,
            "top_neighbors": TOP_NEIGHBORS,
            "note": "No filename or sequence grouping. Train/test only. Test selected by crop-level DINO de-correlation with image-level exclusivity.",
        },
        "all_valid": summarize(records_valid, "all_valid"),
        "train": summarize(train_records_raw, "train"),
        "test": summarize(test_records_raw, "test"),
        "removed_train_images_too_close_to_test": len(removed_units),
        "removed_train_crops_too_close_to_test": int(sum(u["size"] for u in removed_units)),
        "candidate_counts": {
            "train": len(train_candidates),
            "test": len(test_candidates),
        },
        "duplicate_clusters": {
            "num_clusters": len(comps),
            "largest_cluster": int(max(cluster_sizes)),
            "mean_cluster_size": float(np.mean(cluster_sizes)),
            "median_cluster_size": float(np.median(cluster_sizes)),
            "num_singletons": int(sum(1 for x in cluster_sizes if x == 1)),
        },
        "decorrelation_audit": {
            "num_initial_test_images": len(initial_test_images),
            "num_expanded_test_images": len(expanded_test_images),
            "train_test_max_sim_mean_before_removal": float(np.mean(train_test_sims)) if len(train_test_sims) else None,
            "train_test_max_sim_p95_before_removal": float(np.percentile(train_test_sims, 95)) if len(train_test_sims) else None,
            "train_test_max_sim_max_before_removal": float(np.max(train_test_sims)) if len(train_test_sims) else None,
            "enforced_no_shared_image_ids": True,
            "enforced_no_shared_dedup_clusters": True,
        },
    }

    save_json(summary, OUT_DIR / "split_summary.json")

    print("\n=== Decorrelated K-Balanced Train/Test Split ===")
    print(json.dumps(summary["candidate_counts"], indent=2))
    print(json.dumps(summary["duplicate_clusters"], indent=2))
    print(json.dumps(summary["decorrelation_audit"], indent=2))
    print("\nSource report saved to:", OUT_DIR / "source_split_report.csv")
    print("Context report saved to:", OUT_DIR / "context_split_report.csv")
    print("Saved splits to:", OUT_DIR)


if __name__ == "__main__":
    main()