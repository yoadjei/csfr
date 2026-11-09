"""Fallback rates of the classical baselines that have a fallback.

    python scripts/baseline_fallbacks.py

Dictionary learning learns its atoms from the fully observed tiles of the
corrupted patch; with fewer such tiles than atoms it fills the whole hole with
the observed mean (src/baselines.baseline_dictlearn). Under scattered loss few
tiles are intact, so the rate decides what the method's CF1 numbers measure.
Exemplar-based inpainting fills what it cannot reach with the observed mean
when no fully observed exemplar exists at its smallest patch size, and the
sweep records that per patch.

Per sweep, cell and method: the patch-runs and how many fell back, from the
masks and per-patch files on disk. Output: results/baseline_fallbacks/fallbacks.csv.
"""
from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import baseline_dictlearn  # noqa: E402

_DL = inspect.signature(baseline_dictlearn).parameters
TILE, N_ATOMS = _DL["tile"].default, _DL["n_atoms"].default
SCAN_STEP = 2   # the tile scan in baseline_dictlearn; tests/test_baseline_fallbacks.py holds them together
SWEEPS = ("sweep_v2_govdocs1", "sweep_v2_dfrws", "sweep_v2_kodak")
OUT = ROOT / "results" / "baseline_fallbacks" / "fallbacks.csv"


def intact_tiles(m: np.ndarray, tile: int = TILE, step: int = SCAN_STEP) -> int:
    """Fully observed tile x tile windows at the scan positions of the method."""
    win = sliding_window_view(np.asarray(m, bool), (tile, tile))[::step, ::step]
    return int(win.all(axis=(-1, -2)).sum())


def dictlearn_falls_back(m: np.ndarray) -> bool:
    return intact_tiles(m) < N_ATOMS


def cell_fallbacks(seed_dir: Path) -> dict[str, tuple[int, int]]:
    """{method: (patches, fallbacks)} for one cell and seed."""
    out = {}
    if (seed_dir / "per_patch_dictlearn.npz").exists():
        M = np.load(seed_dir / "mask.npy")
        out["dictlearn"] = (len(M), sum(dictlearn_falls_back(m) for m in M))
    f = seed_dir / "per_patch_criminisi.npz"
    if f.exists():
        with np.load(f) as z:
            out["criminisi"] = (len(z["fallback"]), int(z["fallback"].sum()))
    return out


def sweep_rows(sweep: Path) -> list[dict]:
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    rows = []
    for fam in cfg["families"]:
        for lvl in cfg["levels"]:
            tot: dict[str, list[int]] = {}
            for seed in cfg["seeds"]:
                for m, (n, k) in cell_fallbacks(sweep / f"{fam}_{lvl}" / f"seed{seed}").items():
                    t = tot.setdefault(m, [0, 0, 0])
                    t[0] += n
                    t[1] += k
                    t[2] += 1
            rows += [{"sweep": sweep.name, "family": fam, "level": lvl, "method": m, "n_seeds": s,
                      "n": n, "n_fallback": k, "frac_fallback": k / n} for m, (n, k, s) in tot.items()]
    return rows


def main() -> int:
    rows = [r for s in SWEEPS if (ROOT / "results" / s / "run_metadata.json").exists()
            for r in sweep_rows(ROOT / "results" / s)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[fallbacks] wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
