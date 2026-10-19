"""
fitting.py

Generic listwise trainer shared by the learned fusions (A2-A5). Only the fusion
head's parameters are optimized; the three frozen component scores are inputs.

Loss: listwise softmax cross-entropy toward the graded relevance used at eval.
For each training query with S sampled candidates, target = softmax(gamma * rel),
prediction = log_softmax(fused / temp); loss = -sum(target * pred). This pushes
high-relevance candidates to high fused scores without needing the full pool.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def listwise_loss(scores: torch.Tensor, rel: torch.Tensor, gamma=1.0, temp=1.0):
    target = F.softmax(gamma * rel, dim=1)
    pred = F.log_softmax(scores / temp, dim=1)
    return -(target * pred).sum(dim=1).mean()


def fit(module, forward_fn, tc, epochs=60, lr=1e-2, weight_decay=1e-4,
        gamma=1.0, temp=1.0, batch_q=256, device="cuda", seed=42, log_every=20):
    """
    module     : nn.Module holding the learnable fusion params
    forward_fn : (module, n_z,c_z,o_z, rq, geom) -> scores [B,S]
                 n_z/c_z/o_z:[B,S], rq:[B,d_r], geom:[B,S,6]
    tc         : TrainContext
    Returns    : (module, info dict) with train time + param count.
    """
    torch.manual_seed(seed)
    module = module.to(device).train()
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

    n_z, c_z, o_z = tc.n_z.to(device), tc.c_z.to(device), tc.o_z.to(device)
    rel = tc.rel.to(device)
    geom = tc.geom.to(device)
    rq = tc.rq.to(device)
    Q = n_z.shape[0]

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(Q, device=device)
        losses = []
        for s in range(0, Q, batch_q):
            b = perm[s:s + batch_q]
            scores = forward_fn(module, n_z[b], c_z[b], o_z[b], rq[b], geom[b])
            loss = listwise_loss(scores, rel[b], gamma=gamma, temp=temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    [fit] epoch {ep+1}/{epochs} loss={sum(losses)/len(losses):.4f}")

    train_time = time.time() - t0
    n_params = sum(p.numel() for p in module.parameters())
    module.eval()
    return module, {"train_time_s": round(train_time, 2), "params_added": int(n_params)}
