#!/usr/bin/env python3
"""
sheet_to_ratings.py -- turn the collector Google Sheet into per-rater CSV files that
arena_score.py expects.

The sheet (one row per rater) has columns: rater | answered | total | done | updated_iso | csv
where `csv` holds that rater's full results as a multi-line CSV. This script writes each
rater's `csv` cell to ratings_arena/<rater>.csv.

Usage:
  1. In the sheet: File -> Download -> Comma-separated values (.csv)  -> e.g. ratings_sheet.csv
  2. python3 sheet_to_ratings.py ratings_sheet.csv           # -> ratings_arena/<rater>.csv
  3. python3 arena_score.py ratings_arena/ --private manifest_private_arena.json

By default it skips test rows (raters starting with PROBE_, __, ZZ_) and raters with 0 answers.
"""
import argparse
import csv
import os
import re
import sys

TEST_PREFIXES = ("PROBE_", "__", "ZZ_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_csv", help="CSV exported from the collector Google Sheet")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings_arena"))
    ap.add_argument("--min-answered", type=int, default=1, help="Skip raters with fewer than this many answers")
    ap.add_argument("--include-tests", action="store_true", help="Also emit PROBE_/__/ZZ_ test rows")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # allow big multi-line cells
    csv.field_size_limit(10_000_000)

    written, skipped = 0, []
    with open(args.sheet_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "csv" not in (reader.fieldnames or []):
            sys.exit(f"'{args.sheet_csv}' has no 'csv' column; is it the collector-sheet export? "
                     f"columns={reader.fieldnames}")
        for row in reader:
            rater = (row.get("rater") or "").strip()
            blob = row.get("csv") or ""
            answered = int(row.get("answered") or 0)
            if not rater or not blob.strip():
                skipped.append((rater or "<blank>", "empty")); continue
            if not args.include_tests and rater.startswith(TEST_PREFIXES):
                skipped.append((rater, "test row")); continue
            if answered < args.min_answered:
                skipped.append((rater, f"answered={answered}")); continue
            safe = re.sub(r"[^\w.\-]+", "_", rater)
            path = os.path.join(args.out, f"{safe}.csv")
            with open(path, "w", newline="", encoding="utf-8") as out:
                out.write(blob if blob.endswith("\n") else blob + "\n")
            written += 1
            print(f"  wrote {path}  ({answered} answers)")

    print(f"\n{written} rater file(s) -> {args.out}")
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(f"{r} ({why})" for r, why in skipped))
    if written:
        print(f"\nNext: python3 arena_score.py {args.out} --private manifest_private_arena.json")


if __name__ == "__main__":
    main()
