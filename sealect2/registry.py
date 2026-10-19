"""
registry.py

Discovers variants (one deletable folder per variant under `variants/`) and
maintains the tracker: `variants_manifest.json` (machine) + `VARIANTS.md`
(human). Deleting a variant folder removes it from the study automatically;
setting `keep = False` on its Variant marks it dropped without deleting.

Each `variants/<name>/variant.py` must expose:
    def build(**cfg) -> Variant
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
VARIANTS_DIR = HERE / "variants"
MANIFEST_JSON = HERE / "variants_manifest.json"
MANIFEST_MD = HERE / "VARIANTS.md"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_module(name: str):
    path = VARIANTS_DIR / name / "variant.py"
    if not path.exists():
        raise FileNotFoundError(f"no variant '{name}' at {path}")
    spec = importlib.util.spec_from_file_location(f"variant_{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_variant_names() -> List[str]:
    if not VARIANTS_DIR.exists():
        return []
    return sorted(p.name for p in VARIANTS_DIR.iterdir()
                  if p.is_dir() and (p / "variant.py").exists())


def load(name: str, **cfg):
    mod = _load_module(name)
    if not hasattr(mod, "build"):
        raise AttributeError(f"variant '{name}' has no build(**cfg)")
    return mod.build(**cfg)


def _result_metrics(name: str) -> dict:
    """Pull Recall/NDCG/PS from a run's own summary if it exists."""
    out = {}
    for k in (20, 100):
        summ = HERE / "runs" / name / f"topk_{k}" / "metrics.json"
        if summ.exists():
            m = json.load(open(summ))
            out[f"Recall@{k}"] = m.get(f"Recall@{k}")
            out[f"NDCG@{k}"] = m.get(f"NDCG@{k}")
            out[f"PS@{k}"] = m.get(f"PS@{k}")
    return out


def build_manifest() -> List[dict]:
    rows = []
    for name in list_variant_names():
        try:
            v = load(name)
            row = v.manifest_row()
        except Exception as e:  # a broken variant folder should not kill the tracker
            row = {"name": name, "kind": "?", "error": repr(e), "keep": True}
        row["folder"] = f"variants/{name}"
        row["has_run"] = (HERE / "runs" / name / "topk_20" / "retrieval_results.json").exists()
        # prefer the post-fit param count recorded at run time
        cfg_path = HERE / "runs" / name / "topk_20" / "config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            if "params_added" in cfg:
                row["params_added"] = cfg["params_added"]
        row.update(_result_metrics(name))
        rows.append(row)
    return rows


def write_manifest():
    rows = build_manifest()
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2))

    cols = ["name", "kind", "stage", "normalizer", "params_added", "learned",
            "keep", "has_run", "Recall@20", "NDCG@20", "PS@20", "description"]
    lines = ["# Variant tracker",
             "",
             "One folder per variant under `variants/`. Delete a folder (or set "
             "`keep = False`) to drop a loser; the study picks up the change "
             "automatically. Regenerate with `python registry.py`.",
             "",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        def cell(c):
            v = r.get(c, "")
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        lines.append("| " + " | ".join(cell(c) for c in cols) + " |")
    MANIFEST_MD.write_text("\n".join(lines) + "\n")
    print(f"[registry] {len(rows)} variants -> {MANIFEST_MD.name}, {MANIFEST_JSON.name}")
    return rows


if __name__ == "__main__":
    write_manifest()
