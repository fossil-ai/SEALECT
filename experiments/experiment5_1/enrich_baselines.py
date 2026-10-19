#!/usr/bin/env python3
"""
Enrich GALA / DiscoFOS retrieval JSONs so the BootPlace request builder (step 02) can render them.

Those two baselines were exported with only `candidate_id` (e.g. "SMD_1914_4950" = dataset_imageid_
annid) on each top-k entry, whereas step 02's `lookup_coco_ann` needs `image_id` + `annotation_id`.
We parse candidate_id -> image_id/annotation_id/source and (from the already-complete methods) fill
the query's source/image_name. Output goes to <method>_render_k20/topk_20/ so step 02's resolver
finds it, without touching the originals.
"""
import json, os

E1 = "/path/to/sealect_root/experiments/experiment1/exp1_retrieval_quality"
COMPLETE = ["sealect_dinofeat_k20", "dino_k20", "clip_k20", "random_k20"]
TARGETS = {"gala": "gala_only_k20", "discofos": "discofos_ft_k20"}

# image_id -> {source, dataset, image_name} from the methods that already carry full metadata
imap = {}
for p in COMPLETE:
    for r in json.load(open(f"{E1}/{p}/topk_20/retrieval_results.json")):
        for e in [r["query"]] + r.get("topk", []):
            iid = e.get("image_id")
            if iid is not None and iid not in imap:
                imap[iid] = {"source": e.get("source"), "dataset": e.get("dataset"),
                             "image_name": e.get("image_name")}
print(f"metadata map: {len(imap)} image_ids")


def parse_cid(cid):
    ds, img, ann = cid.rsplit("_", 2)   # dataset names have no underscores
    return ds, int(img), int(ann)


for name, src in TARGETS.items():
    d = json.load(open(f"{E1}/{src}/topk_20/retrieval_results.json"))
    enriched_c = miss = 0
    for r in d:
        q = r["query"]
        m = imap.get(q.get("image_id"), {})
        q.setdefault("source", m.get("source"))
        q.setdefault("dataset", m.get("dataset"))
        q.setdefault("image_name", m.get("image_name"))
        for c in r.get("topk", []):
            ds, img, ann = parse_cid(c["candidate_id"])
            c["image_id"] = img
            c["annotation_id"] = ann
            c.setdefault("source", ds)
            c.setdefault("dataset", ds)
            mm = imap.get(img)
            if mm and mm.get("image_name"):
                c["image_name"] = mm["image_name"]
            else:
                miss += 1
            enriched_c += 1
    out = f"{E1}/{name}_render_k20/topk_20"
    os.makedirs(out, exist_ok=True)
    json.dump(d, open(f"{out}/retrieval_results.json", "w"))
    print(f"{name}: {len(d)} records, {enriched_c} candidates enriched "
          f"({miss} without cached image_name) -> {out}")
