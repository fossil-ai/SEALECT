"""
base.py

The Variant interface every ablation implements. One variant == one deletable
folder under `variants/`. Shared infrastructure (legacy_bridge, normalization,
eval) is NOT a variant and lives at the sealect2 root.

A variant can override any of three evidence sources and the fusion:

    neural_component(raw)  -> [Nq,Ncand]   (C-series: a different trained retriever)
    context_component(raw) -> [Nq,Ncand]   (B-series: PMI/smoothed/kernel/attention prior)
    obs_component(raw)     -> [Nq,Ncand]
    normalize(x, which)                     (identity | perquery_z | global_z)
    fuse(n, c, o, meta)    -> [Nq,Ncand]    (A-series: linear/gated/mlp/attention)

Default behavior == A0 (shipped SEALECT). So a new fusion variant overrides only
`fuse`; a new context variant overrides only `context_component`; a new neural
variant overrides only `neural_component` + `prepare`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

import normalization as norm


class Variant:
    # ---- identity / bookkeeping (shown in the tracker) ----
    name: str = "unnamed"
    kind: str = "fusion"            # "fusion" | "context" | "neural_training"
    stage: int = 1
    description: str = ""
    params_added: int = 0           # learnable params this variant introduces
    learned: bool = False
    keep: bool = True               # flip to False (or delete the folder) to drop a loser

    # ---- fixed linear defaults (== A0) ----
    normalizer: str = "perquery_z"
    lambda_n: float = 1.0
    lambda_c: float = 1.0
    lambda_o: float = 0.10
    global_stats: Dict[str, tuple] = {}   # for global_z: {"neural":(mu,sig),...}

    # ---- evidence source hooks (override in B/C variants) ----
    def neural_component(self, raw) -> torch.Tensor:
        return raw.neural

    def context_component(self, raw) -> torch.Tensor:
        return raw.context

    def obs_component(self, raw) -> torch.Tensor:
        return raw.obs

    # queries-per-batch at eval (learned per-candidate scorers set this smaller)
    eval_batch: int = 256

    # ---- optional fit on training data (A2-A5, B4, C1) ----
    def prepare(self, trainctx: Optional[object] = None) -> "Variant":
        return self

    # ---- optional: precompute anything needing the FULL eval query set ----
    # (e.g. per-query r_q for all test queries, candidate geometry on device)
    def on_eval_start(self, raw, device: str = "cpu") -> None:
        return None

    # ---- normalization ----
    def normalize(self, x: torch.Tensor, which: str) -> torch.Tensor:
        if self.normalizer == "identity":
            return norm.identity(x)
        if self.normalizer == "perquery_z":
            return norm.perquery_z(x)
        if self.normalizer == "global_z":
            if which not in self.global_stats:
                raise NotImplementedError(
                    f"{self.name}: global_z needs train stats for '{which}'. "
                    "Run prepare() with training components first."
                )
            mu, sig = self.global_stats[which]
            return norm.global_z(x, mu, sig)
        raise ValueError(f"{self.name}: unknown normalizer {self.normalizer}")

    # ---- fusion (override in A-series) ----
    def fuse(self, n: torch.Tensor, c: torch.Tensor, o: torch.Tensor,
             meta: Optional[dict] = None) -> torch.Tensor:
        return self.lambda_n * n + self.lambda_c * c + self.lambda_o * o

    # ---- manifest row ----
    def manifest_row(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "stage": self.stage,
            "normalizer": self.normalizer, "params_added": self.params_added,
            "learned": self.learned, "keep": self.keep,
            "description": self.description,
        }
