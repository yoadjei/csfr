"""Range constraint in the ablation: how far the imputed pixels leave [LO, HI]
when C3 is removed, and that no other arm leaves it.

    python scripts/ablation_range.py [--sweep-dir results/sweep_v2_govdocs1]

For every CSFR arm (csfr_variants and ablation_variants of the sweep) at the
ablation seeds: the fraction of imputed pixels outside the range, the fraction
of patches with at least one, and the largest excursion in grey levels, read
from the saved outputs. This is the only constraint-violation figure the paper
reports; it is internal to the ablation and never compared across methods.
Output: <sweep-dir>/stats/range_violations.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_objective import HI, LO  # noqa: E402

TOL = 1e-6


def range_row(x: np.ndarray, M: np.ndarray) -> dict:
    """Out-of-range statistics of the imputed pixels of a batch (B, H, W)."""
    hole = np.asarray(M) == 0
    excess = np.maximum(LO - x, 0.0) + np.maximum(x - HI, 0.0)
    out = (excess > TOL) & hole
    return {"n_patches": int(len(x)), "n_imputed": int(hole.sum()),
            "frac_pixels_out": float(out.sum() / hole.sum()) if hole.any() else 0.0,
            "frac_patches_out": float(out.any(axis=(1, 2)).mean()),
            "max_excess": float(excess[hole].max()) if hole.any() else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="results/sweep_v2_govdocs1")
    args = ap.parse_args(argv)
    sweep = Path(args.sweep_dir)
    if not sweep.is_absolute():
        sweep = ROOT / sweep
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    arms = list(cfg.get("csfr_variants", [])) + list(cfg.get("ablation_variants", []))
    rows = []
    for fam in cfg["families"]:
        for lvl in cfg["levels"]:
            for seed in cfg.get("ablation_seeds", []):
                d = sweep / f"{fam}_{lvl}" / f"seed{seed}"
                for arm in arms:
                    f = d / f"x_hat_{arm}.npy"
                    if f.exists():
                        rows.append({"family": fam, "level": lvl, "seed": seed, "method": arm,
                                     **range_row(np.load(f), np.load(d / "mask.npy"))})
    if not rows:
        print(f"[range] no ablation outputs under {sweep}")
        return 1
    out = sweep / "stats" / "range_violations.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[range] wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
