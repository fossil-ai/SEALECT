#!/usr/bin/env python3
"""
build_scene_pose_prior.py

Build ambiguity-aware scene-conditioned priors for maritime vessel placement.

Expected input:
- scene_object_pairs.json produced by build_merged_scene_pose_library.py

Outputs:
- scene_pose_prior.json

This estimates:
1. Factorized marginals:
   p(pitch_bin | c), p(yaw_bin | c), p(roll_bin | c)
   for both scene_type and context_key
2. Joint histograms:
   p(pose_bin | c)
3. Scale priors:
   p(area | y_bin, c)
4. Vertical placement priors:
   p(y_bin | pose_bin, c)
5. Yaw ambiguity statistics:
   entropy of p(yaw_bin | c)
   normalized yaw weight for later scoring

This script is designed for the appearance-alignment pose labels used in PosePlace.
"""

import os
import json
import math
from collections import defaultdict, Counter
from typing import Dict, List

# 
# EDIT THESE
# 
# SCENE_OBJECT_PAIRS_JSON = "merged_scene_pose_library_out/scene_object_pairs.json"
# OUT_PATH = "scene_pose_prior.json"
SCENE_OBJECT_PAIRS_JSON = (
    "splits_dino_decorrelated_kbalanced_grounded_sam2/"
    "train_scene_object_pairs.json"
)
OUT_PATH = "scene_pose_prior_train_grounded_sam2.json"
Y_BINS = 8
AREA_EPS = 1e-8
NUM_YAW_BINS = 18          # should match your pose model config
MIN_CONFIDENCE = 0.0       # e.g. 0.0 for all, or 0.45 to filter noisy samples
USE_CONFIDENCE_WEIGHTING = True
# 


def pose_key(pitch_bin: int, yaw_bin: int, roll_bin: int) -> str:
    return f"{pitch_bin}_{yaw_bin}_{roll_bin}"


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def safe_std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = safe_mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return float(math.sqrt(max(var, 0.0)))


def weighted_normalize_counter(counter: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {}
    return {k: float(v / total) for k, v in counter.items()}


def entropy_from_distribution(prob_dict: Dict[str, float], eps: float = 1e-12) -> float:
    h = 0.0
    for p in prob_dict.values():
        p = max(float(p), eps)
        h -= p * math.log(p)
    return float(h)


def normalized_entropy_weight(prob_dict: Dict[str, float], num_bins: int) -> float:
    """
    Returns a weight in [0,1].
    - 1 means low entropy / strongly constrained
    - 0 means high entropy / highly ambiguous
    """
    if num_bins <= 1:
        return 1.0
    h = entropy_from_distribution(prob_dict)
    hmax = math.log(num_bins)
    if hmax <= 0:
        return 1.0
    w = 1.0 - (h / hmax)
    return float(max(0.0, min(1.0, w)))


def maybe_weight(obs: Dict) -> float:
    if not USE_CONFIDENCE_WEIGHTING:
        return 1.0
    # yaw ambiguity is mostly about yaw, but we want a stable pose observation overall.
    # Use the average of the three per-axis confidences if available, else pose_confidence.
    if "pitch_conf" in obs and "yaw_conf" in obs and "roll_conf" in obs:
        return float(
            (float(obs["pitch_conf"]) + float(obs["yaw_conf"]) + float(obs["roll_conf"])) / 3.0
        )
    return float(obs.get("pose_confidence", 1.0))


def add_weighted_count(counter: Dict[str, float], key: str, weight: float):
    counter[key] += float(weight)


def main():
    with open(SCENE_OBJECT_PAIRS_JSON, "r") as f:
        observations = json.load(f)

    # -----------------------------
    # Aggregators
    # -----------------------------
    # Global by scene_type
    scene_pose_counts = defaultdict(lambda: defaultdict(float))
    scene_pitch_counts = defaultdict(lambda: defaultdict(float))
    scene_yaw_counts = defaultdict(lambda: defaultdict(float))
    scene_roll_counts = defaultdict(lambda: defaultdict(float))

    # By context_key (more specific than scene_type)
    context_pose_counts = defaultdict(lambda: defaultdict(float))
    context_pitch_counts = defaultdict(lambda: defaultdict(float))
    context_yaw_counts = defaultdict(lambda: defaultdict(float))
    context_roll_counts = defaultdict(lambda: defaultdict(float))

    # Coarse pose-family priors
    scene_pose_family_counts = defaultdict(lambda: defaultdict(float))
    context_pose_family_counts = defaultdict(lambda: defaultdict(float))
    scene_yaw_family_counts = defaultdict(lambda: defaultdict(float))
    context_yaw_family_counts = defaultdict(lambda: defaultdict(float))

    # Scale priors
    area_by_scene_ybin = defaultdict(lambda: defaultdict(list))
    area_by_context_ybin = defaultdict(lambda: defaultdict(list))

    # Vertical placement priors
    ybin_by_scene_pose = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    ybin_by_context_pose = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    # Optional dataset-conditioned summaries
    dataset_scene_counts = defaultdict(lambda: defaultdict(float))

    # Diagnostics
    used = 0
    skipped_low_conf = 0
    missing_context_key = 0
    nadir_like_count = 0
    aerial_scene_count = 0

    for obs in observations:
        pose_conf = float(obs.get("pose_confidence", 1.0))
        if pose_conf < MIN_CONFIDENCE:
            skipped_low_conf += 1
            continue

        scene_type = obs["scene_type"]
        context_key = obs.get("context_key", None)
        if context_key is None:
            # fallback so the script still runs even if context_key hasn't been added yet
            context_key = f"{scene_type}__fallback"
            missing_context_key += 1

        dataset = obs.get("dataset", "unknown")

        pitch_bin = int(obs["pitch_bin"])
        yaw_bin = int(obs["yaw_bin"])
        roll_bin = int(obs["roll_bin"])

        pkey = pose_key(pitch_bin, yaw_bin, roll_bin)
        yb = int(obs["y_bin"])
        area_frac = float(obs["area_frac"])

        weight = maybe_weight(obs)

        # full joint pose
        add_weighted_count(scene_pose_counts[scene_type], pkey, weight)
        add_weighted_count(context_pose_counts[context_key], pkey, weight)

        # factorized marginals
        add_weighted_count(scene_pitch_counts[scene_type], str(pitch_bin), weight)
        add_weighted_count(scene_yaw_counts[scene_type], str(yaw_bin), weight)
        add_weighted_count(scene_roll_counts[scene_type], str(roll_bin), weight)

        add_weighted_count(context_pitch_counts[context_key], str(pitch_bin), weight)
        add_weighted_count(context_yaw_counts[context_key], str(yaw_bin), weight)
        add_weighted_count(context_roll_counts[context_key], str(roll_bin), weight)

        # coarse families if present
        if "pose_family" in obs:
            add_weighted_count(scene_pose_family_counts[scene_type], obs["pose_family"], weight)
            add_weighted_count(context_pose_family_counts[context_key], obs["pose_family"], weight)

        if "yaw_family" in obs:
            add_weighted_count(scene_yaw_family_counts[scene_type], obs["yaw_family"], weight)
            add_weighted_count(context_yaw_family_counts[context_key], obs["yaw_family"], weight)

        # scale priors
        area_by_scene_ybin[scene_type][str(yb)].append(math.log(max(area_frac, AREA_EPS)))
        area_by_context_ybin[context_key][str(yb)].append(math.log(max(area_frac, AREA_EPS)))

        # y-position priors
        add_weighted_count(ybin_by_scene_pose[scene_type][pkey], str(yb), weight)
        add_weighted_count(ybin_by_context_pose[context_key][pkey], str(yb), weight)

        # diagnostics
        add_weighted_count(dataset_scene_counts[dataset], scene_type, weight)

        # if bool(obs.get("is_nadir_like", False)):
        #     nadir_like_count += 1
        if bool(obs.get("is_aerial_scene", False)):
            aerial_scene_count += 1

        used += 1

    # -----------------------------
    # Build output
    # -----------------------------
    out = {
        "meta": {
            "num_observations_in_file": len(observations),
            "used_observations": used,
            "skipped_low_confidence": skipped_low_conf,
            "missing_context_key": missing_context_key,
            "min_confidence": MIN_CONFIDENCE,
            "use_confidence_weighting": USE_CONFIDENCE_WEIGHTING,
            "y_bins": Y_BINS,
            "num_yaw_bins": NUM_YAW_BINS,
            "pose_key_format": "pitchBin_yawBin_rollBin",
            "aerial_scene_count": aerial_scene_count,
        },

        # Joint distributions
        "scene_pose_hist": {},
        "context_pose_hist": {},

        # Factorized marginals
        "scene_pitch_hist": {},
        "scene_yaw_hist": {},
        "scene_roll_hist": {},

        "context_pitch_hist": {},
        "context_yaw_hist": {},
        "context_roll_hist": {},

        # Coarse families
        "scene_pose_family_hist": {},
        "context_pose_family_hist": {},
        "scene_yaw_family_hist": {},
        "context_yaw_family_hist": {},

        # Scale priors
        "scale_by_scene_ybin": {},
        "scale_by_context_ybin": {},

        # Vertical placement priors
        "ybin_given_scene_pose": {},
        "ybin_given_context_pose": {},

        # Ambiguity stats
        "scene_yaw_entropy": {},
        "context_yaw_entropy": {},
        "scene_yaw_weight": {},
        "context_yaw_weight": {},

        # Optional diagnostics
        "dataset_scene_hist": {},
    }

    # Joint distributions
    for scene_type, counter in scene_pose_counts.items():
        out["scene_pose_hist"][scene_type] = weighted_normalize_counter(counter)

    for context_key, counter in context_pose_counts.items():
        out["context_pose_hist"][context_key] = weighted_normalize_counter(counter)

    # Factorized marginals
    for scene_type, counter in scene_pitch_counts.items():
        out["scene_pitch_hist"][scene_type] = weighted_normalize_counter(counter)
    for scene_type, counter in scene_yaw_counts.items():
        out["scene_yaw_hist"][scene_type] = weighted_normalize_counter(counter)
    for scene_type, counter in scene_roll_counts.items():
        out["scene_roll_hist"][scene_type] = weighted_normalize_counter(counter)

    for context_key, counter in context_pitch_counts.items():
        out["context_pitch_hist"][context_key] = weighted_normalize_counter(counter)
    for context_key, counter in context_yaw_counts.items():
        out["context_yaw_hist"][context_key] = weighted_normalize_counter(counter)
    for context_key, counter in context_roll_counts.items():
        out["context_roll_hist"][context_key] = weighted_normalize_counter(counter)

    # Coarse family histograms
    for scene_type, counter in scene_pose_family_counts.items():
        out["scene_pose_family_hist"][scene_type] = weighted_normalize_counter(counter)
    for context_key, counter in context_pose_family_counts.items():
        out["context_pose_family_hist"][context_key] = weighted_normalize_counter(counter)

    for scene_type, counter in scene_yaw_family_counts.items():
        out["scene_yaw_family_hist"][scene_type] = weighted_normalize_counter(counter)
    for context_key, counter in context_yaw_family_counts.items():
        out["context_yaw_family_hist"][context_key] = weighted_normalize_counter(counter)

    # Scale priors
    for scene_type, ydict in area_by_scene_ybin.items():
        out["scale_by_scene_ybin"][scene_type] = {}
        for yb, vals in ydict.items():
            out["scale_by_scene_ybin"][scene_type][yb] = {
                "log_area_mean": safe_mean(vals),
                "log_area_std": safe_std(vals),
                "count": len(vals),
            }

    for context_key, ydict in area_by_context_ybin.items():
        out["scale_by_context_ybin"][context_key] = {}
        for yb, vals in ydict.items():
            out["scale_by_context_ybin"][context_key][yb] = {
                "log_area_mean": safe_mean(vals),
                "log_area_std": safe_std(vals),
                "count": len(vals),
            }

    # Vertical placement priors
    for scene_type, pose_dict in ybin_by_scene_pose.items():
        out["ybin_given_scene_pose"][scene_type] = {}
        for pkey, counter in pose_dict.items():
            out["ybin_given_scene_pose"][scene_type][pkey] = weighted_normalize_counter(counter)

    for context_key, pose_dict in ybin_by_context_pose.items():
        out["ybin_given_context_pose"][context_key] = {}
        for pkey, counter in pose_dict.items():
            out["ybin_given_context_pose"][context_key][pkey] = weighted_normalize_counter(counter)

    # Yaw entropy and ambiguity-aware yaw weight
    for scene_type, hist in out["scene_yaw_hist"].items():
        h = entropy_from_distribution(hist)
        w = normalized_entropy_weight(hist, NUM_YAW_BINS)
        out["scene_yaw_entropy"][scene_type] = h
        out["scene_yaw_weight"][scene_type] = w

    for context_key, hist in out["context_yaw_hist"].items():
        h = entropy_from_distribution(hist)
        w = normalized_entropy_weight(hist, NUM_YAW_BINS)
        out["context_yaw_entropy"][context_key] = h
        out["context_yaw_weight"][context_key] = w

    # Diagnostics
    for dataset, counter in dataset_scene_counts.items():
        out["dataset_scene_hist"][dataset] = weighted_normalize_counter(counter)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out["meta"], indent=2))
    print(f"Saved scene prior to: {OUT_PATH}")


if __name__ == "__main__":
    main()