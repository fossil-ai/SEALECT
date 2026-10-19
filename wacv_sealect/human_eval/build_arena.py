#!/usr/bin/env python3
"""
build_arena.py -- Blind cross-method plausibility ARENA over all rendered methods.

Each "comparison" shows two composites of the SAME background produced by two DIFFERENT methods
(blind, boxed, artifact- and placement-filtered). A rater picks the more plausible one (or tie).
Fitting Bradley-Terry over the outcomes (arena_score.py) yields a per-method plausibility score
with CIs -- a leaderboard for {sealect, dino, clip, gala, discofos, random}.

Design mirrors build_manifest.py (restricted two-layer filter + per-file audit, black-rectangle drop,
corner inpaint, red box on the inserted vessel, above-waterline placement drop, neutral file names,
truth only in a private manifest). Schedule is weighted toward SEALECT-vs-each edges (well-powered
for the method under test) plus baseline-vs-baseline edges for Bradley-Terry connectivity.

Reuses helpers from build_manifest.py. Output: site_arena/ (deployable) + manifest_private_arena.json.
"""
import argparse, glob, json, os, re, random, collections, itertools
from PIL import Image
import build_manifest as bm   # black_frac, has_black_rectangle, insert_box, fill_black, draw_box, above_waterline, load_horizons, parse_composites

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = bm.E1
METHODS = ["sealect", "dino", "clip", "gala", "discofos", "random"]
# retrieval JSONs that carry source/dataset for the two-layer RESTRICTED_B filter (enriched for gala/discofos)
SRC_JSONS = {
    "sealect": f"{E1}/sealect_dinofeat_k20/topk_20/retrieval_results.json",
    "dino": f"{E1}/dino_k20/topk_20/retrieval_results.json",
    "clip": f"{E1}/clip_k20/topk_20/retrieval_results.json",
    "random": f"{E1}/random_k20/topk_20/retrieval_results.json",
    "gala": f"{E1}/gala_render_k20/topk_20/retrieval_results.json",
    "discofos": f"{E1}/discofos_render_k20/topk_20/retrieval_results.json",
}


def source_map():
    img_source, img_dataset = {}, {}
    for p in SRC_JSONS.values():
        for r in json.load(open(p)):
            for e in [r["query"]] + r.get("topk", []):
                iid, s, d = e.get("image_id"), e.get("source"), e.get("dataset")
                if iid is not None and s is not None:
                    img_source.setdefault(iid, s)
                    img_dataset.setdefault(iid, d)
    return img_source, img_dataset


def horizons():
    geo = {}
    for p in [SRC_JSONS["sealect"], SRC_JSONS["dino"]]:
        for r in json.load(open(p)):
            q = r["query"]
            geo[f'{q["image_id"]}_{q["annotation_id"]}'] = dict(
                hy=q["horizon_y_norm"], hc=q["horizon_confidence"],
                vp=q["viewpoint_type"], sky=q["sky_frac"])
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-per-edge", type=int, default=25,
                    help="max comparisons per SEALECT-vs-baseline edge")
    ap.add_argument("--secondary-per-edge", type=int, default=6,
                    help="max comparisons per baseline-vs-baseline edge (BT connectivity)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--pad-frac", type=float, default=0.08)
    args = ap.parse_args()

    site = os.path.join(HERE, "site_arena")
    img_out = os.path.join(site, "images")
    private_out = os.path.join(HERE, "manifest_private_arena.json")

    img_source, img_dataset = source_map()
    geo = horizons()
    print(f"source map: {len(img_source)} ids | methods: {METHODS}")

    # per method: scene_id -> composite record, after two-layer RESTRICTED_B filter
    comp = {}
    for m in METHODS:
        d, dropped = bm.parse_composites(m, img_source, img_dataset)
        comp[m] = d
        print(f"  {m:9s}: {len(d)} public composites (dropped {dict(dropped)})")

    # clean each composite once: drop black-rectangle artifacts / missing mask / above-waterline;
    # cache the inserted-vessel box. A background is usable for a method only if it passes.
    clean_box = {m: {} for m in METHODS}
    for m in METHODS:
        for sid, rec in comp[m].items():
            if bm.has_black_rectangle(rec["png"]):
                continue
            W, H = Image.open(rec["png"]).size
            box = bm.insert_box(m, os.path.basename(rec["png"]), W, H, args.pad_frac)
            if box is None or bm.above_waterline(box, geo.get(sid)):
                continue
            clean_box[m][sid] = box
        print(f"  {m:9s}: {len(clean_box[m])} clean, placed composites")

    # build the comparison schedule
    rng = random.Random(args.seed)
    edges = []
    for x in METHODS[1:]:
        edges.append(("sealect", x, args.primary_per_edge))
    for a, b in itertools.combinations(METHODS[1:], 2):
        edges.append((a, b, args.secondary_per_edge))

    schedule = []                    # (background, m1, m2)
    thin_edges = []
    for a, b, cap in edges:
        shared = sorted(set(clean_box[a]) & set(clean_box[b]))
        rng.shuffle(shared)
        take = shared[:cap]
        for sid in take:
            schedule.append((sid, a, b))
        if len(take) < cap:                       # drops reduced this edge below target
            thin_edges.append((a, b, len(take), cap))
    rng.shuffle(schedule)
    print(f"\nscheduled {len(schedule)} comparisons across "
          f"{len({frozenset((a,b)) for _,a,b in schedule})} method-edges")
    if thin_edges:
        print("[warn] edges below target after drops (fewer shared clean composites):")
        for a, b, got, cap in thin_edges:
            flag = "   <-- WEAK: Bradley-Terry connectivity risk" if got < 8 else ""
            print(f"    {a}-{b}: {got}/{cap}{flag}")

    # render the arena site (blind, neutral file names, box + inpaint)
    if os.path.exists(img_out):
        for f in glob.glob(os.path.join(img_out, "*")):
            os.remove(f)
    os.makedirs(img_out, exist_ok=True)

    public, private = [], []
    edge_counts = collections.Counter()
    for i, (sid, m1, m2) in enumerate(schedule):
        cid = f"c{i:04d}_{sid}"
        methods = [m1, m2]
        if rng.random() < 0.5:
            methods = methods[::-1]
        slot_method = {"a": methods[0], "b": methods[1]}
        for slot in ("a", "b"):
            m = slot_method[slot]
            img = Image.open(comp[m][sid]["png"]).convert("RGB")
            img = bm.fill_black(img)
            bm.draw_box(img, clean_box[m][sid], args.box_width)
            img.save(os.path.join(img_out, f"{cid}_{slot}.jpg"), "JPEG",
                     quality=args.quality, optimize=True)
        ds = img_dataset.get(int(sid.split("_")[0]), "UNKNOWN")
        public.append(dict(scene_id=cid, dataset=ds,
                           img_a=f"images/{cid}_a.jpg", img_b=f"images/{cid}_b.jpg"))
        private.append(dict(comp_id=cid, background=sid, dataset=ds,
                            a_method=slot_method["a"], b_method=slot_method["b"]))
        edge_counts[frozenset((m1, m2))] += 1

    # audit: 0 restricted, every method public
    for p in private:
        for m in (p["a_method"], p["b_method"]):
            rec = comp[m][p["background"]]
            for s in (rec["bg_src"], rec["cand_src"]):
                assert s not in bm.RESTRICTED, f"restricted leak {p['comp_id']}"
    print(f"AUDIT PASSED: 0 restricted across {len(private)} comparisons ({2*len(private)} images)")
    print("per-edge comparison counts:")
    for e, n in sorted(edge_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {' vs '.join(sorted(e)):24s} {n}")

    with open(bm.TEMPLATE) as fh:
        html = fh.read()
    html = html.replace("/*__SCENES__*/[]", json.dumps(public, separators=(",", ":")))
    html = html.replace('/*__TAG__*/""', json.dumps("arena"))
    with open(os.path.join(site, "index.html"), "w") as fh:
        fh.write(html)
    open(os.path.join(site, ".nojekyll"), "w").close()
    json.dump(private, open(private_out, "w"), indent=2)
    json.dump(public, open(os.path.join(site, "scenes.json"), "w"), indent=2)

    print(f"\nwrote: {os.path.join(site,'index.html')} ({len(private)} comparisons)")
    print(f"       {private_out}   (KEEP PRIVATE)")
    print(f"preview: python3 -m http.server -d {site} 8000")


if __name__ == "__main__":
    main()
