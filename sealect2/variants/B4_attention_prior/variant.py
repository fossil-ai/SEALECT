"""B4 - attention-weighted empirical prior.

Keep the geometric distribution EMPIRICAL, but learn which training scenes are
relevant: a_i(q) = softmax_i( (P c_q).(P c_i) / (tau*sqrt(d)) ), then
p(pi|c_q) = sum_i a_i(q) K_pi(pi, pi_i). Attention only *weights empirical
evidence* (anchor histograms); it never directly predicts geometry. Trained with
the listwise ranking loss on the resulting context score. Compare directly to B3
(fixed RBF) to see whether learned attention beats plain kernel smoothing.
Fusion held at v1.
"""

from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Variant
import context_models as cm
import context_kernel as ck
import legacy_bridge as lb
import fitting

EPS = 1e-8


def _rel_mod():
    f = lb.EXP1_DIR / "summarize_exp1_retrieval_results.py"
    spec = importlib.util.spec_from_file_location("rel_b4", str(f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class AttnPrior(nn.Module):
    def __init__(self, d, dk=16):
        super().__init__()
        self.P = nn.Linear(d, dk, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(1))

    def weights(self, q, K):                       # q[B,d], K[Na,d] -> [B,Na]
        qp, kp = self.P(q), self.P(K)
        scale = math.sqrt(qp.shape[-1]) * torch.exp(self.log_tau)
        return F.softmax((qp @ kp.T) / scale, dim=1)

    def context_from_W(self, W, bank, cbins, gather):
        oh_p, oh_y, oh_r, oh_b, a_la = bank["ohp"], bank["ohy"], bank["ohr"], bank["ohb"], bank["ala"]
        lp_p = torch.log(W @ oh_p + EPS); lp_y = torch.log(W @ oh_y + EPS)
        lp_r = torch.log(W @ oh_r + EPS); lp_b = torch.log(W @ oh_b + EPS)
        mu = W @ a_la; var = torch.clamp(W @ (a_la ** 2) - mu ** 2, min=0.02)
        cp, cy, cr, cb, cla = cbins
        if gather:                                  # [B,S] per-query candidate bins
            sp = torch.gather(lp_p, 1, cp); sy = torch.gather(lp_y, 1, cy)
            sr = torch.gather(lp_r, 1, cr); sb = torch.gather(lp_b, 1, cb)
            scale_term = (-0.5 * torch.log(2 * math.pi * var)[:, None]
                          - 0.5 * ((cla - mu[:, None]) / torch.sqrt(var)[:, None]) ** 2)
        else:                                       # [Ncand] shared bins
            sp = lp_p[:, cp]; sy = lp_y[:, cy]; sr = lp_r[:, cr]; sb = lp_b[:, cb]
            scale_term = (-0.5 * torch.log(2 * math.pi * var)[:, None]
                          - 0.5 * ((cla[None, :] - mu[:, None]) / torch.sqrt(var)[:, None]) ** 2)
        return 2.0 * sp + sy + sr + 0.7 * scale_term + 0.5 * sb


class B4AttentionPrior(Variant):
    name = "B4_attention_prior"
    kind = "context"
    stage = 2
    normalizer = "perquery_z"
    learned = True
    keep = False   # DROPPED: worse than B3 (free RBF kernel) on all retrieval metrics; learned attention adds nothing
    n_anchor = 16384
    n_fit_q = 2000
    S = 128
    description = "Attention-weighted empirical prior: learned scene relevance over anchor histograms. Fusion = v1."

    def prepare(self, trainctx=None):
        legacy, train_pairs, candidates, _ = cm.load_ctx_assets()
        device = legacy.DEVICE
        rng = random.Random(42)
        self._legacy = legacy

        # anchor bank (subsampled for tractability)
        idx = list(range(len(train_pairs))); rng.shuffle(idx)
        anc = [train_pairs[i] for i in idx[:self.n_anchor]]
        Cx, stats = ck.scene_bank(legacy, anc)
        self._stats = stats
        self._Cx = Cx.to(device)
        self._bank = {
            "ohp": F.one_hot(ck._ints(anc, "pitch_bin").clamp(0, 17), 18).float().to(device),
            "ohy": F.one_hot(ck._ints(anc, "yaw_bin").clamp(0, 17), 18).float().to(device),
            "ohr": F.one_hot(ck._ints(anc, "roll_bin").clamp(0, 17), 18).float().to(device),
            "ohb": F.one_hot(ck._ints(anc, "y_bin").clamp(0, 7), 8).float().to(device),
            "ala": torch.tensor([math.log(max(float(p.get("area_frac", 1e-8)), 1e-8)) for p in anc],
                                device=device),
        }
        self.model = AttnPrior(Cx.shape[1]).to(device)

        # candidate bin arrays (for eval + relevance)
        cpitch = np.array([int(c["pitch_bin"]) for c in candidates])
        cyaw = np.array([int(c["yaw_bin"]) for c in candidates])
        cybin = np.array([int(c["y_bin"]) for c in candidates])
        clog = np.array([math.log(max(float(c["area_frac"]), 1e-8)) for c in candidates])
        self._cbins = (
            torch.tensor([int(c["pitch_bin"]) for c in candidates]).clamp(0, 17).to(device),
            torch.tensor([int(c["yaw_bin"]) for c in candidates]).clamp(0, 17).to(device),
            torch.tensor([int(c["roll_bin"]) for c in candidates]).clamp(0, 17).to(device),
            torch.tensor([int(c["y_bin"]) for c in candidates]).clamp(0, 7).to(device),
            torch.tensor(clog, dtype=torch.float32).to(device),
        )

        # build a compact fit set: sampled train queries x sampled candidates + relevance
        rel = _rel_mod()
        fq_idx = idx[:self.n_fit_q]
        fit_q = [train_pairs[i] for i in fq_idx]
        Qx, _ = ck.scene_bank(legacy, fit_q, stats)
        S = self.S
        cp = np.empty((len(fit_q), S), np.int64); cy = np.empty_like(cp)
        cr = np.empty_like(cp); cb = np.empty_like(cp)
        cla = np.empty((len(fit_q), S), np.float32); rr = np.empty((len(fit_q), S), np.float32)
        croll = np.array([int(c["roll_bin"]) for c in candidates])
        for r, q in enumerate(fit_q):
            s = np.zeros(len(candidates), np.int16)
            s += (np.abs(int(q["pitch_bin"]) - cpitch) <= rel.PITCH_TOL)
            d = np.abs(int(q["yaw_bin"]) - cyaw); d = np.minimum(d, 18 - d)
            s += (d <= rel.YAW_TOL)
            s += (np.abs(int(q["y_bin"]) - cybin) <= rel.YBIN_TOL)
            qla = math.log(max(float(q["area_frac"]), 1e-8))
            s += (np.abs(qla - clog) <= rel.AREA_LOG_TOL)
            pos = np.where(s >= 3)[0]; neg = np.where(s < 3)[0]
            take_pos = list(pos[rng.sample(range(len(pos)), min(len(pos), S // 3))]) if len(pos) else []
            need = S - len(take_pos)
            take_neg = list(neg[rng.sample(range(len(neg)), need)])
            samp = np.array(take_pos + take_neg)[:S]
            cp[r] = cpitch[samp]; cy[r] = cyaw[samp]; cr[r] = croll[samp]
            cb[r] = cybin[samp]; cla[r] = clog[samp]; rr[r] = s[samp]

        dev = device
        self._fit = dict(
            Qx=Qx.to(dev),
            cp=torch.tensor(cp).to(dev), cy=torch.tensor(cy).to(dev),
            cr=torch.tensor(cr).to(dev), cb=torch.tensor(cb).to(dev),
            cla=torch.tensor(cla).to(dev), rel=torch.tensor(rr).to(dev),
        )
        self._train_attention(dev)
        self.params_added = sum(p.numel() for p in self.model.parameters())
        return self

    def _train_attention(self, device, epochs=50, lr=1e-2, batch=256):
        import time
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        f = self._fit; Q = f["Qx"].shape[0]
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(Q, device=device); losses = []
            for s in range(0, Q, batch):
                b = perm[s:s + batch]
                W = self.model.weights(f["Qx"][b], self._Cx)
                cbins = (f["cp"][b], f["cy"][b], f["cr"][b], f["cb"][b], f["cla"][b])
                sc = self.model.context_from_W(W, self._bank, cbins, gather=True)
                loss = fitting.listwise_loss(sc, f["rel"][b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(float(loss.item()))
            if ep % 20 == 0 or ep == epochs - 1:
                print(f"    [B4] epoch {ep+1}/{epochs} loss={sum(losses)/len(losses):.4f}")
        self.train_time_s = round(time.time() - t0, 2)
        self.model.eval()

    @torch.no_grad()
    def context_component(self, raw):
        legacy = self._legacy; device = legacy.DEVICE
        Qx, _ = ck.scene_bank(legacy, raw.queries, self._stats)
        Nq = len(raw.queries); Ncand = len(raw.candidates)
        out = torch.empty(Nq, Ncand, dtype=torch.float32)
        for s in range(0, Nq, 256):
            e = min(s + 256, Nq)
            W = self.model.weights(Qx[s:e].to(device), self._Cx)
            Sc = self.model.context_from_W(W, self._bank, self._cbins, gather=False)
            out[s:e] = Sc.cpu()
        return out


def build(lambda_n=1.0, lambda_c=1.0, lambda_o=0.10, **_):
    v = B4AttentionPrior()
    v.lambda_n, v.lambda_c, v.lambda_o = lambda_n, lambda_c, lambda_o
    return v
