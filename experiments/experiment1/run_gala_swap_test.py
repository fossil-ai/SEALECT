#!/usr/bin/env python3
"""
Fair swap test: hold the appearance retriever constant (DINO), swap only the
geometry term. Compare DINO + explicit prior (the existing dino_prior baseline)
against DINO + learned GALA-style discriminative compatibility. This isolates
exactly the reviewer's question: explicit generative prior vs learned discriminative
compatibility, as the geometry signal, with appearance fixed.
"""
import json, math, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path("/path/to/sealect_root")
SP = ROOT / "experiments/dataset_decide/splits_dino_decorrelated_kbalanced_grounded_sam2"
EXP = ROOT / "experiments/experiment1/exp1_retrieval_quality"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ---- metric (verbatim from summarize_exp1_retrieval_results) ----
EPS = 1e-8
def log_area(x): return math.log(max(float(x), EPS))
def circ(a, b, n=18): d = abs(int(a) - int(b)); return min(d, n - d)
def relevance(q, c):
    s = 0
    if abs(int(q["pitch_bin"]) - int(c["pitch_bin"])) <= 0: s += 1
    if circ(int(q["yaw_bin"]), int(c["yaw_bin"])) <= 1: s += 1
    if abs(int(q["y_bin"]) - int(c["y_bin"])) <= 0: s += 1
    if abs(log_area(q["area_frac"]) - log_area(c["area_frac"])) <= 0.30: s += 1
    return s
def dcg(xs): return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(xs))
def ndcg(xs):
    d = dcg(sorted(xs, reverse=True)); return dcg(xs) / d if d > 0 else 0.0
def summarize(results, k):
    rec, nd, ps = [], [], []
    for row in results:
        q, cands = row["query"], row["topk"][:k]
        if not cands: continue
        rels = [relevance(q, c) for c in cands]
        rec.append(1.0 if max(rels) >= 4 else 0.0); nd.append(ndcg(rels)); ps.append(sum(rels)/max(1,len(rels)))
    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"Recall": m(rec), "NDCG": m(nd), "PS": m(ps)}
def znorm(x):
    x = np.asarray(x, np.float32); return (x - x.mean()) / (x.std() + 1e-6)

SCENE_KEYS = ["water_frac","sky_frac","land_frac","other_frac","horizon_y_norm",
              "water_row_mean","water_row_std","sky_row_mean","sky_row_std","land_row_mean","land_row_std"]
def scene_vec(r): return [float(r.get(k) or 0.0) for k in SCENE_KEYS]
def geom_vec(r, Np, Ny, Nr):
    def sc(b, N): a = 2*math.pi*int(b)/max(1,N); return [math.sin(a), math.cos(a)]
    return (sc(r["pitch_bin"],Np)+sc(r["yaw_bin"],Ny)+sc(r["roll_bin"],Nr)+
            [int(r["y_bin"])/8.0, log_area(r["area_frac"]),
             math.log(max(float(r.get("aspect_ratio",1.0)),1e-6)),
             float(r.get("center_x_norm") or 0.5), float(r.get("center_y_norm") or 0.5)])

def main():
    train = json.load(open(SP / "train_scene_object_pairs.json"))
    queries = json.load(open(SP / "test_scene_object_pairs.json"))
    cands = json.load(open(SP / "train_candidate_library.json"))
    qlib = json.load(open(SP / "test_candidate_library.json"))
    print(f"train {len(train)}  queries {len(queries)}  candidates {len(cands)}", flush=True)
    Np = max(int(r["pitch_bin"]) for r in train) + 1; Ny = 18; Nr = max(int(r["roll_bin"]) for r in train) + 1

    # ---- train GALA-style geometry compatibility scorer ----
    S = np.asarray([scene_vec(r) for r in train], np.float32); G = np.asarray([geom_vec(r,Np,Ny,Nr) for r in train], np.float32)
    smu, ssd = S.mean(0), S.std(0)+1e-6; gmu, gsd = G.mean(0), G.std(0)+1e-6
    St = torch.tensor((S-smu)/ssd, device=DEVICE); Gt = torch.tensor((G-gmu)/gsd, device=DEVICE)
    model = nn.Sequential(nn.Linear(S.shape[1]+G.shape[1],128), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    N = St.shape[0]; bs = 1024
    print("training GALA scorer...", flush=True)
    for ep in range(8):
        model.train(); perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; s = St[idx]; gpos = Gt[idx]; gneg = Gt[idx[torch.randperm(len(idx), device=DEVICE)]]
            x = torch.cat([torch.cat([s,gpos],1), torch.cat([s,gneg],1)], 0)
            y = torch.cat([torch.ones(len(idx),device=DEVICE), torch.zeros(len(idx),device=DEVICE)])
            opt.zero_grad(); bce(model(x).squeeze(1), y).backward(); opt.step()
    model.eval()

    # ---- candidate geometry + DINO embeddings ----
    Cg = torch.tensor((np.asarray([geom_vec(c,Np,Ny,Nr) for c in cands],np.float32)-gmu)/gsd, device=DEVICE)
    def emb_of(r):
        e = r.get("embedding"); return np.asarray(e, np.float32) if e else None
    D = np.stack([emb_of(c) if emb_of(c) is not None else np.zeros(384,np.float32) for c in cands])
    # infer dim from first valid
    dim = next((len(c["embedding"]) for c in cands if c.get("embedding")), 384)
    D = np.stack([np.asarray(c["embedding"],np.float32) if c.get("embedding") else np.zeros(dim,np.float32) for c in cands])
    D /= (np.linalg.norm(D,axis=1,keepdims=True)+1e-8)
    qemb = {(int(r["image_id"]),int(r["annotation_id"])): np.asarray(r["embedding"],np.float32)
            for r in qlib if r.get("embedding")}
    cgroup = np.asarray([str(c.get("sequence_group","")) for c in cands])
    cimg = np.asarray([int(c.get("image_id",-2)) for c in cands]); cann = np.asarray([int(c.get("annotation_id",-2)) for c in cands])
    keep = ["candidate_id","image_id","annotation_id","pitch_bin","yaw_bin","roll_bin","y_bin","area_frac","sequence_group","context_key","viewpoint_type"]
    cslim = [{k:c.get(k) for k in keep} for c in cands]

    res_gala, res_dino, res_dinogala = [], [], []
    with torch.no_grad():
        for qi, q in enumerate(queries):
            sv = torch.tensor((np.asarray(scene_vec(q),np.float32)-smu)/ssd, device=DEVICE).unsqueeze(0).expand(Cg.shape[0],-1)
            gala = model(torch.cat([sv,Cg],1)).squeeze(1).cpu().numpy()
            qe = qemb.get((int(q.get("image_id",-1)),int(q.get("annotation_id",-1))))
            app = (D @ (qe/(np.linalg.norm(qe)+1e-8))) if qe is not None else np.zeros(Cg.shape[0],np.float32)
            invalid = (cgroup==str(q.get("sequence_group",""))) | ((cimg==int(q.get("image_id",-1)))&(cann==int(q.get("annotation_id",-1))))
            qslim = {k:q.get(k) for k in ["pitch_bin","yaw_bin","y_bin","area_frac","context_key","viewpoint_type","image_id","annotation_id"]}
            for name, base, store in (("gala",gala,res_gala),("dino",app,res_dino),
                                      ("dinogala", znorm(app)+znorm(gala), res_dinogala)):
                sc = base.copy().astype(np.float32); sc[invalid] = -1e9
                top = np.argpartition(-sc,100)[:100]; top = top[np.argsort(-sc[top])]
                store.append({"query":qslim, "topk":[cslim[j] for j in top]})
            if (qi+1)%1000==0: print(f"  {qi+1}/{len(queries)}", flush=True)

    def load_ex(name,k):
        p = EXP / f"{name}_k{k}" / f"topk_{k}" / "retrieval_results.json"
        return json.load(open(p)) if p.exists() else None

    print("\n FAIR SWAP TEST: geometry term, appearance (DINO) held constant")
    for k in (20,100):
        print(f"\n--- K={k} ---")
        print(f"{'method':22s} {'Recall':>8s} {'NDCG':>8s} {'PS':>8s}")
        for disp,name in (("SEALECT (neural+prior+obs)","sealect_dinofeat"),
                          ("DINO + prior","dino_prior"), ("DINO (appearance)","dino"),
                          ("prior only","prior_only")):
            r = load_ex(name,k)
            if r: m = summarize(r,k); print(f"{disp:22s} {m['Recall']:8.3f} {m['NDCG']:8.3f} {m['PS']:8.3f}")
        for disp,r in (("GALA only (geom)",res_gala), ("DINO + GALA",res_dinogala)):
            m = summarize(r,k); star = "   <== swap" if "GALA" in disp and "DINO" in disp else ""
            print(f"{disp:22s} {m['Recall']:8.3f} {m['NDCG']:8.3f} {m['PS']:8.3f}{star}")
    print("\ndone")

if __name__ == "__main__":
    main()
