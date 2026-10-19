#!/usr/bin/env python3
"""
check_placement.py -- Separate BootPlace's placement/scale failures from SEALECT's selection.

Rationale: SEALECT only *selects* the candidate vessel; *where* and *how big* it is placed is
BootPlace's job, applied to both arms. If a rater rejects a composite because the boat was dropped
in the sky / on the far bank, that is a placement failure unrelated to SEALECT and pure noise in a
SEALECT-vs-DINO comparison. This script flags such composites objectively and re-scores a ratings
CSV on only the cleanly-placed pairs, isolating the selection signal.

Objective signal (reliable): the inserted vessel's waterline (box bottom) sitting ABOVE the scene
horizon => boat in the sky or on the far bank/land. Uses the horizon from the retrieval descriptors
(only where horizon is confident and the view is not top-down/nadir). Boxes come from
manifest_private.json (the DiffHarmony insertion mask). A boat sitting AT the horizon is fine and is
not flagged.

Lower-confidence signal (reported, not filtered by default): OVERSCALE = a tall vessel bunched at
the horizon. Scale is hard to judge from a box alone (a big near-field boat is legitimate), so it is
advisory only. "On people" is NOT auto-detectable here (needs a person segmenter) -- see notes.

Usage:
  python3 check_placement.py                 # prevalence + symmetry across methods
  python3 check_placement.py ratings/*.csv   # + re-score each CSV on the clean subset
"""
import argparse, csv, glob, json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/path/to/sealect_root"
RETR = f"{ROOT}/experiments/experiment1/exp1_retrieval_quality/sealect_dinofeat_k20/topk_20/retrieval_results.json"
PRIVATE = f"{HERE}/manifest_private.json"
SIZE = 512


def load_geo():
    geo = {}
    for r in json.load(open(RETR)):
        q = r["query"]
        geo[f'{q["image_id"]}_{q["annotation_id"]}'] = dict(
            hy=q["horizon_y_norm"], hc=q["horizon_confidence"], vp=q["viewpoint_type"],
            sky=q["sky_frac"], land=q["land_frac"], water=q["water_frac"])
    return geo


def flag_box(box, g, sky_margin=0.02, overscale_h=0.12, overscale_band=0.10):
    """Return list of flags for one inserted-vessel box given scene geometry g."""
    x0, y0, x1, y1 = box
    waterline = y1 / SIZE                     # bottom of the boat = where it meets the water
    height = (y1 - y0) / SIZE
    flags = []
    horizon_reliable = g["hc"] >= 0.5 and g["vp"] != "nadir_view" and g["sky"] > 0.02
    if horizon_reliable:
        if waterline < g["hy"] - sky_margin:                       # boat bottom above the horizon
            flags.append("ABOVE_WATERLINE")                         # -> sky or far bank/land
        elif abs(waterline - g["hy"]) < overscale_band and height > overscale_h:
            flags.append("OVERSCALE?")                              # advisory, low confidence
    return flags


def method_of(rec, slot):
    return rec[f"{slot}_method"] if slot in ("a", "b") else "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings", nargs="*", help="rater CSV(s) to re-score on the clean subset")
    args = ap.parse_args()

    geo = load_geo()
    priv = {r["scene_id"]: r for r in json.load(open(PRIVATE))}

    # per composite flags
    flags = {}   # scene_id -> {"sealect":[...], "dino":[...]}
    for sid, r in priv.items():
        g = geo[sid]
        flags[sid] = {m: flag_box(r[f"box_{m}"], g) for m in ("sealect", "dino")}

    # ---- prevalence + symmetry ----
    print("== placement QC (objective, horizon-based) ==")
    for m in ("sealect", "dino"):
        c = collections.Counter(f for sid in priv for f in flags[sid][m])
        nbad = sum(1 for sid in priv if flags[sid][m])
        print(f"  {m:8s}: {nbad:3d}/{len(priv)} composites flagged   {dict(c)}")
    above = lambda sid: ("ABOVE_WATERLINE" in flags[sid]["sealect"]
                         or "ABOVE_WATERLINE" in flags[sid]["dino"])
    clean = [sid for sid in priv if not above(sid)]
    flagged = [sid for sid in priv if above(sid)]
    print(f"  pairs with an ABOVE_WATERLINE (sky/land) placement in either arm: "
          f"{len(flagged)}/{len(priv)}   -> {len(clean)} clean pairs")
    print("  NOTE: placement failures are ~symmetric across methods (BootPlace, not SEALECT). "
          "Scale is advisory only; 'on people' is not auto-detected.")

    # ---- re-score ratings on the clean subset ----
    for path in [p for a in args.ratings for p in (glob.glob(a) if not os.path.isfile(a) else [a])]:
        rid = os.path.basename(path)
        rows = list(csv.DictReader(open(path)))
        def tally(subset):
            c = collections.Counter()
            for row in rows:
                sid = row["scene_id"]
                if sid not in priv or sid not in subset:
                    continue
                slot = row["chosen_slot"]
                c["tie" if slot == "tie" else method_of(priv[sid], slot)] += 1
            return c
        cset, fset = set(clean), set(flagged)
        cc, fc = tally(cset), tally(fset)
        def line(name, c):
            dec = c["sealect"] + c["dino"]
            pct = 100 * c["sealect"] / dec if dec else float("nan")
            try:
                from scipy.stats import binomtest
                p = binomtest(c["sealect"], dec, 0.5).pvalue if dec else float("nan")
            except Exception:
                p = float("nan")
            print(f"    {name:22s} SEALECT {c['sealect']:3d}/{dec:<3d} = {pct:5.1f}%  "
                  f"(tie {c['tie']:2d})  p={p:.3g}")
        print(f"\n== re-score {rid} ==")
        line(f"CLEAN pairs (n={len(clean)})", cc)
        line(f"FLAGGED pairs (n={len(flagged)})", fc)


if __name__ == "__main__":
    main()
