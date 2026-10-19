"""C1 - geometry-hard-negative neural training (evaluated with B3 context).

Trains a new neural retriever whose representation is shaped by geometry hard
negatives (visually attractive, geometrically implausible), then evaluates it in
the full pipeline with the Stage-2 winner B3 as the context model and v1 fusion.
The v1 encoder is untouched (baseline). To isolate whether S_neural itself
improved, also run neural-only: `--lambda_c 0 --lambda_o 0` and compare to
`sweep_neural_only` (v1 neural-only = 0.634/0.830/2.278 @K20).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from base import Variant
import legacy_bridge as lb
import context_kernel as ck
import context_models as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_c1


class C1GeoHardNeg(Variant):
    name = "C1_geo_hardneg"
    kind = "neural_training"
    stage = 3
    normalizer = "perquery_z"
    params_added = 0            # retrains the encoder (same size); no inference-time params added
    learned = True
    description = "Neural retriever trained with geometry hard-negative margin loss; evaluated with B3 context + v1 fusion."

    def prepare(self, trainctx=None):
        self._ckpt = train_c1.train(use_cache=True)
        self._legacy, self._train_pairs, self._cands, _ = cm.load_ctx_assets()
        return self

    def _encode(self, raw):
        legacy = self._legacy
        blob = torch.load(self._ckpt, map_location=legacy.DEVICE)
        trainer = train_c1._load_trainer()
        s_enc = trainer.MLP(int(blob["scene_dim"]), int(blob["emb_dim"])).to(legacy.DEVICE)
        c_enc = trainer.MLP(int(blob["cand_dim"]), int(blob["emb_dim"])).to(legacy.DEVICE)
        s_enc.load_state_dict(blob["scene_enc"]); c_enc.load_state_dict(blob["cand_enc"])
        s_enc.eval(); c_enc.eval()
        dino = legacy.load_dino_cache_by_key()
        with torch.no_grad():
            cand_embs = legacy.encode_candidates(c_enc, raw.candidates, dino)
            q_embs = legacy.encode_queries(s_enc, raw.queries)
        return cand_embs, q_embs

    def neural_component(self, raw):
        cand_embs, q_embs = self._encode(raw)
        Nq, Ncand = len(raw.queries), len(raw.candidates)
        out = torch.empty(Nq, Ncand, dtype=torch.float32)
        with torch.no_grad():
            for qi in tqdm(range(Nq), desc="[C1] neural"):
                out[qi] = (cand_embs @ q_embs[qi]).cpu()
        return out

    def context_component(self, raw):
        # carry forward the Stage-2 winner (B3 kernel prior)
        return ck.kernel_context_matrix(self._legacy, self._train_pairs,
                                        raw.queries, raw.candidates, h=1.0)


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = C1GeoHardNeg()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
