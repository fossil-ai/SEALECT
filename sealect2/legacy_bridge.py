"""
legacy_bridge.py

Thin, read-only bridge to the FROZEN original SEALECT implementation.

We do NOT copy or modify the original scoring code. We import the paper's
lambda-sweep module and reuse its exact functions to build the three raw score
components once:

    S_neural(j) = q^T e_j              (trained DINO two-tower retriever)
    S_context(j) = log p(pi_j | gamma) (empirical scene-conditioned prior)
    S_obs(j)     = log C_obs(u_j|U)    (observed-pose KDE in the same image)

Every ablation consumes these *identical* raw components, so all experiments
share the same candidates, neural embeddings, splits, seeds and candidate pools
by construction. Normalization and fusion are applied on top (see fusion.py).

IMPORTANT FINDING (documented for the study):
The shipped SEALECT (paper Table 1, run `sweep_ln1_lc1_lo0p10`) z-normalizes
each component *per query* over the full candidate pool BEFORE the fixed-lambda
linear combination (legacy `precompute_components` -> `zscore_torch`). So the
"current baseline" is a per-query-normalized fixed linear fusion, not the raw
sum. We reproduce that exactly as A0 and expose the un-normalized variant
separately as a diagnostic (A0raw).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths (all absolute; independent of CWD)
# ------------------------------------------------------------------
SEALECT_ROOT = Path(__file__).resolve().parents[1]
EXP1_DIR = SEALECT_ROOT / "experiments" / "experiment1"
LEGACY_FILE = EXP1_DIR / "run_sealect_exp1_lambda_sweep_cuda_dino.py"
MODEL_PATH = EXP1_DIR / "exp1_retrieval_quality" / "sealect_neural_dino" / "sealect_neural_retriever_dino.pt"

SEALECT2_DIR = SEALECT_ROOT / "sealect2"
CACHE_DIR = SEALECT2_DIR / "cache"
RUNS_DIR = SEALECT2_DIR / "runs"


def load_legacy():
    """Import the frozen paper lambda-sweep module and pin absolute paths."""
    spec = importlib.util.spec_from_file_location("sealect_legacy_sweep", str(LEGACY_FILE))
    legacy = importlib.util.module_from_spec(spec)
    sys.modules["sealect_legacy_sweep"] = legacy
    spec.loader.exec_module(legacy)
    # Pin the only CWD-relative constants to absolute locations.
    legacy.MODEL_PATH = MODEL_PATH
    return legacy


def _cache_signature(legacy) -> str:
    parts = []
    for p in [MODEL_PATH, legacy.PRIOR_JSON, legacy.TRAIN_CANDIDATES,
              legacy.TEST_QUERIES, legacy.DINO_CACHE]:
        p = Path(p)
        parts.append(f"{p.name}:{int(p.stat().st_mtime) if p.exists() else 0}")
    return "|".join(parts)


class RawComponents:
    """Container for the identical inputs every ablation consumes."""

    def __init__(self, queries, candidates, neural, context, obs, valid, meta):
        self.queries: List[Dict] = queries
        self.candidates: List[Dict] = candidates
        self.neural: torch.Tensor = neural      # [Nq, Ncand] float32 (raw, un-normalized)
        self.context: torch.Tensor = context    # [Nq, Ncand] float32 (log p(pi|gamma))
        self.obs: torch.Tensor = obs             # [Nq, Ncand] float32 (log C_obs)
        self.valid: torch.Tensor = valid         # [Nq, Ncand] bool
        self.meta: Dict = meta

    @property
    def n_queries(self):
        return len(self.queries)

    @property
    def n_candidates(self):
        return len(self.candidates)


def build_raw_components(use_cache: bool = True, device: str | None = None) -> RawComponents:
    """Build (or load) the three raw component matrices via the frozen code."""
    legacy = load_legacy()
    device = device or legacy.DEVICE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "raw_components.pt"
    sig = _cache_signature(legacy)

    if use_cache and cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("signature") == sig:
            print(f"[legacy_bridge] loaded cached raw components: {cache_path}")
            return RawComponents(
                blob["queries"], blob["candidates"],
                blob["neural"], blob["context"], blob["obs"], blob["valid"],
                blob["meta"],
            )
        print("[legacy_bridge] cache signature stale; recomputing.")

    # ---- Reproduce the legacy precompute path exactly, but keep RAW rows ----
    candidates = legacy.load_json(legacy.TRAIN_CANDIDATES)
    queries = legacy.load_json(legacy.TEST_QUERIES)
    prior = legacy.load_json(legacy.PRIOR_JSON)
    dino_by_key = legacy.load_dino_cache_by_key()
    print(f"[legacy_bridge] candidates={len(candidates)} queries={len(queries)}")

    with torch.no_grad():
        cand_embs = legacy.encode_candidates(legacy.load_model()[1], candidates, dino_by_key)  # [Ncand, d]
        q_embs = legacy.encode_queries(legacy.load_model()[0], queries)                        # [Nq, d]

    valid_masks = legacy.build_valid_masks(queries, candidates)              # [Nq, Ncand] bool on device
    context_cache = legacy.build_context_cache(queries, candidates, prior)   # dict key -> np[Ncand]

    from collections import defaultdict
    obs_by_image = defaultdict(list)
    for q in queries:
        obs_by_image[int(q["image_id"])].append(q)
    cand_t = legacy.build_candidate_tensors(candidates)

    Nq, Ncand = len(queries), len(candidates)
    neural = torch.empty((Nq, Ncand), dtype=torch.float32)
    context = torch.empty((Nq, Ncand), dtype=torch.float32)
    obs = torch.empty((Nq, Ncand), dtype=torch.float32)

    with torch.no_grad():
        for qi, q in enumerate(tqdm(queries, desc="[legacy_bridge] raw components")):
            neural[qi] = (cand_embs @ q_embs[qi]).detach().cpu()
            ck = (str(q.get("context_key")), str(q.get("scene_type")))
            context[qi] = torch.from_numpy(context_cache[ck])
            obs[qi] = legacy.observed_score_cuda(q, obs_by_image, cand_t).detach().cpu()

    valid = valid_masks.detach().cpu()
    meta = {"signature": sig, "device": device, "eps": legacy.EPS,
            "model_path": str(MODEL_PATH)}

    if use_cache:
        torch.save({"signature": sig, "queries": queries, "candidates": candidates,
                    "neural": neural, "context": context, "obs": obs, "valid": valid,
                    "meta": meta}, cache_path)
        print(f"[legacy_bridge] cached raw components -> {cache_path}")

    return RawComponents(queries, candidates, neural, context, obs, valid, meta)
