#!/usr/bin/env python3
"""
arena_score.py -- Bradley-Terry leaderboard from blind cross-method arena ratings.

Joins rater CSVs (chosen_slot per comp_id) with manifest_private_arena.json (a_method/b_method),
fits a Bradley-Terry model (MM algorithm) over the pairwise outcomes, and reports a per-method
plausibility score with bootstrap CIs, a pairwise win-rate matrix, and a ranking.

Ties count as half a win to each side (reported separately). Usage:
  python3 arena_score.py ratings_arena/            # dir of sealect_ratings_arena_*.csv
  python3 arena_score.py r1.csv r2.csv --private manifest_private_arena.json
"""
import argparse, csv, glob, json, os, collections, random, math

HERE = os.path.dirname(os.path.abspath(__file__))


def load_outcomes(csv_paths, priv):
    """-> list of (winner, loser) or (m, m, 'tie'); plus tie count."""
    out, ties = [], 0
    for path in csv_paths:
        for row in csv.DictReader(open(path)):
            cid = row["scene_id"] if "scene_id" in row else row.get("comp_id")
            if cid not in priv:
                continue
            rec = priv[cid]
            slot = row["chosen_slot"]
            if slot == "tie":
                out.append((rec["a_method"], rec["b_method"], "tie")); ties += 1
            else:
                win = rec[f"{slot}_method"]
                lose = rec["b_method"] if slot == "a" else rec["a_method"]
                out.append((win, lose, "win"))
    return out, ties


def per_rater_choices(csv_paths, priv):
    """comp_id -> {rater: 'A'|'B'|'T'} where A=chose a_method, B=chose b_method (slot-independent)."""
    ch = collections.defaultdict(dict)
    for path in csv_paths:
        rid = os.path.splitext(os.path.basename(path))[0]
        for row in csv.DictReader(open(path)):
            cid = row.get("scene_id") or row.get("comp_id")
            if cid not in priv:
                continue
            slot = row["chosen_slot"]
            ch[cid][rid] = "T" if slot == "tie" else ("A" if slot == "a" else "B")
    return ch


def fleiss_kappa(items):
    """items: list of category-count dicts, each item rated by the same n raters (3 cats A/B/T)."""
    cats = ["A", "B", "T"]
    rows = [it for it in items if sum(it.get(c, 0) for c in cats) >= 2]
    if len(rows) < 2:
        return float("nan")
    n = sum(rows[0].get(c, 0) for c in cats)
    if any(sum(it.get(c, 0) for c in cats) != n for it in rows) or n < 2:
        # unequal rater counts -> restrict to the modal n
        from collections import Counter
        n = Counter(sum(it.get(c, 0) for c in cats) for it in rows).most_common(1)[0][0]
        rows = [it for it in rows if sum(it.get(c, 0) for c in cats) == n]
        if len(rows) < 2 or n < 2:
            return float("nan")
    P = [(sum(it.get(c, 0) ** 2 for c in cats) - n) / (n * (n - 1)) for it in rows]
    pj = {c: sum(it.get(c, 0) for it in rows) / (len(rows) * n) for c in cats}
    Pbar, Pe = sum(P) / len(P), sum(v * v for v in pj.values())
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")


def bradley_terry(outcomes, methods, iters=500, eps=1e-9):
    """MM algorithm. Ties = 0.5 win each. Returns strength dict normalized to sum 1."""
    W = {m: 0.0 for m in methods}                 # weighted wins
    N = collections.defaultdict(float)            # N[(i,j)] total games between i,j
    for a, b, o in outcomes:
        N[(a, b)] += 1; N[(b, a)] += 1
        if o == "tie":
            W[a] += 0.5; W[b] += 0.5
        else:
            W[a] += 1.0                            # a is winner in (win) tuples
    p = {m: 1.0 for m in methods}
    for _ in range(iters):
        newp = {}
        for i in methods:
            denom = sum(N[(i, j)] / (p[i] + p[j]) for j in methods if j != i and N[(i, j)] > 0)
            newp[i] = (W[i] / denom) if denom > eps else p[i]
        s = sum(newp.values()) or 1.0
        newp = {m: v / s for m, v in newp.items()}
        if max(abs(newp[m] - p[m]) for m in methods) < 1e-10:
            p = newp; break
        p = newp
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--private", default=os.path.join(HERE, "manifest_private_arena.json"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    priv = {r["comp_id"]: r for r in json.load(open(args.private))}
    methods = sorted({r["a_method"] for r in priv.values()} | {r["b_method"] for r in priv.values()})

    files = []
    for p in args.inputs:
        files += glob.glob(os.path.join(p, "*.csv")) if os.path.isdir(p) else [p]
    outcomes, ties = load_outcomes(sorted(set(files)), priv)
    decisive = [o for o in outcomes if o[2] == "win"]
    print(f"raters files: {len(files)} | comparisons: {len(outcomes)} "
          f"({len(decisive)} decisive, {ties} ties)")

    # empirical pairwise win-rate matrix
    wins = collections.defaultdict(lambda: [0, 0])   # (i,j) -> [i_wins, total_decisive]
    for a, b, o in outcomes:
        if o == "win":
            wins[(a, b)][0] += 1; wins[(a, b)][1] += 1; wins[(b, a)][1] += 1
    print("\npairwise win-rate (row beats col, decisive only):")
    print("            " + "".join(f"{m[:5]:>8}" for m in methods))
    for i in methods:
        cells = []
        for j in methods:
            if i == j: cells.append(f"{'--':>8}"); continue
            w, n = wins[(i, j)]
            cells.append(f"{(100*w/n):>7.0f}%" if n else f"{'-':>8}")
        print(f"  {i:9s} " + "".join(cells))

    # realism check: win rate of each method vs RANDOM (should be high for all -> composites plausible)
    if "random" in methods:
        rr = []
        for m in methods:
            if m == "random": continue
            w, n = wins[(m, "random")]
            if n: rr.append((m, 100 * w / n, n))
        if rr:
            lo, hi = min(x[1] for x in rr), max(x[1] for x in rr)
            print(f"\nrealism check — win rate vs RANDOM: {lo:.0f}%–{hi:.0f}% across methods "
                  f"({'; '.join(f'{m} {p:.0f}%' for m,p,_ in rr)})")

    # inter-rater agreement (Fleiss kappa over A/B/tie), if >=2 raters
    ch = per_rater_choices(sorted(set(files)), priv)
    nraters = len({r for d in ch.values() for r in d})
    if nraters >= 2:
        items = [dict(collections.Counter(d.values())) for d in ch.values() if len(d) >= 2]
        kappa = fleiss_kappa(items)
        print(f"inter-rater agreement: Fleiss κ = {kappa:.3f} "
              f"({nraters} raters, {len(items)} shared comparisons)")

    # Bradley-Terry fit + bootstrap CIs
    p = bradley_terry(outcomes, methods)
    rng = random.Random(0)
    boot = collections.defaultdict(list)
    for _ in range(args.boot):
        sample = [outcomes[rng.randrange(len(outcomes))] for _ in range(len(outcomes))]
        bp = bradley_terry(sample, methods, iters=300)
        for m in methods: boot[m].append(bp[m])
    def ci(m):
        v = sorted(boot[m]); n = len(v)
        return v[int(0.025 * n)], v[int(0.975 * n)]

    ref = p["random"] if "random" in p else min(p.values())
    print("\nBradley-Terry plausibility leaderboard (higher = more plausible):")
    print(f"  {'method':10s}{'score':>9}{'95% CI':>20}{'rating*':>10}")
    for m in sorted(methods, key=lambda x: -p[x]):
        lo, hi = ci(m)
        rating = 400 * math.log10(max(p[m], 1e-9) / max(ref, 1e-9))
        print(f"  {m:10s}{p[m]:>9.3f}   [{lo:.3f}, {hi:.3f}]{rating:>10.0f}")
    print("  *rating = 400*log10(score/random), Elo-style relative to RANDOM.")

    # honest sentence
    order = sorted(methods, key=lambda x: -p[x])
    sealect_rank = order.index("sealect") + 1
    sw = {j: wins[("sealect", j)] for j in methods if j != "sealect"}
    beat = [j for j, (w, n) in sw.items() if n and w / n > 0.5]
    print("\nPAPER-READY:")
    print(f"  Across {len(outcomes)} blind pairwise comparisons ({len(priv)} rendered composites, "
          f"6 methods), a Bradley-Terry model ranks SEALECT #{sealect_rank} of {len(methods)}. "
          f"SEALECT is preferred over {', '.join(beat) if beat else 'no method'} "
          f"(decisive majority); see the win-rate matrix for pairwise detail.")


if __name__ == "__main__":
    main()
