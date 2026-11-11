"""What the inner iteration cap of the dictionary-learning baseline does to its output.

    python scripts/dictlearn_inner_cap.py [--sweep-dir results/sweep_v2_govdocs1] [--patches 200]

The released configuration caps the coordinate-descent sparse coding inside the
dictionary fit at fit_cd_max_iter iterations (src/baselines.baseline_dictlearn),
and the cap is often reached. For every cell of CF2 to CF4 at mask seed 0, the
first --patches patches are reconstructed at the released cap and at a cap
--reference-factor times larger, and the two are compared by hole PSNR. CF1 is
left out: there the method almost always falls back to the mean fill, which the
cap does not touch (scripts/baseline_fallbacks.py).

The run at the released cap must reproduce the sweep's saved output for those
patches exactly; matches_sweep records that it does.

Output: results/dictlearn_inner_cap/<sweep>.csv, one row per cell.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baseline_fallbacks import dictlearn_falls_back  # noqa: E402
from baselines import baseline_dictlearn, limit_threads  # noqa: E402
from quality import psnr_hole  # noqa: E402

OUT = ROOT / "results" / "dictlearn_inner_cap"
SEED = 0
FAMILIES = ("CF2", "CF3", "CF4")
RELEASED_CAP = inspect.signature(baseline_dictlearn).parameters["fit_cd_max_iter"].default


def compare(ref: np.ndarray, released: np.ndarray, reference: np.ndarray, M: np.ndarray) -> dict:
    """Hole PSNR at the released and the reference cap, and their per-patch
    difference (released minus reference)."""
    p_rel = np.array([psnr_hole(r, x, m)[0] for r, x, m in zip(ref, released, M)])
    p_ref = np.array([psnr_hole(r, x, m)[0] for r, x, m in zip(ref, reference, M)])
    d = p_rel - p_ref
    return {"psnr_released": float(p_rel.mean()), "psnr_reference": float(p_ref.mean()),
            "diff_mean": float(d.mean()), "diff_abs_max": float(np.abs(d).max())}


def cell_row(pool, cell_dir: Path, ref: np.ndarray, n: int, factor: int, workers: int) -> dict:
    y, M = np.load(cell_dir / "y.npy")[:n], np.load(cell_dir / "mask.npy")[:n]
    saved = np.load(cell_dir / "x_hat_dictlearn.npy")[:n]
    chunks = [c for c in np.array_split(np.arange(len(y)), max(1, workers) * 4) if c.size]
    outs = {}
    for cap in (RELEASED_CAP, RELEASED_CAP * factor):
        # the worker function lives in src/baselines.py, so spawned workers can import it
        fit = partial(baseline_dictlearn, rng_seed=SEED, fit_cd_max_iter=cap)
        parts = pool.map(fit, [y[c] for c in chunks], [M[c] for c in chunks])
        outs[cap] = np.concatenate(list(parts), axis=0).astype(saved.dtype)
    fam, lvl = cell_dir.parent.name.split("_")
    return {"family": fam, "level": lvl, "seed": SEED, "n_patches": int(len(y)),
            "n_learned": int(sum(not dictlearn_falls_back(m) for m in M)),
            "cap_released": RELEASED_CAP, "cap_reference": RELEASED_CAP * factor,
            **compare(ref[:n], outs[RELEASED_CAP], outs[RELEASED_CAP * factor], M),
            "matches_sweep": bool(np.array_equal(outs[RELEASED_CAP], saved))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="results/sweep_v2_govdocs1")
    ap.add_argument("--patches", type=int, default=200)
    ap.add_argument("--reference-factor", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    sweep = Path(args.sweep_dir)
    if not sweep.is_absolute():
        sweep = ROOT / sweep
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    ref = np.load(ROOT / cfg["patches_path"]).astype(np.float32)
    cells = [sweep / f"{f}_{lv}" / f"seed{SEED}" for f in cfg["families"] if f in FAMILIES
             for lv in cfg["levels"]]
    with ProcessPoolExecutor(max(1, args.workers), mp_context=get_context("spawn"),
                             initializer=limit_threads) as pool:
        rows = []
        for c in cells:
            rows.append(cell_row(pool, c, ref, args.patches, args.reference_factor, args.workers))
            print(f"[dictlearn_inner_cap] {c.parent.name}: diff_mean {rows[-1]['diff_mean']:+.4f} dB, "
                  f"max |diff| {rows[-1]['diff_abs_max']:.3f}, matches_sweep {rows[-1]['matches_sweep']}",
                  flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sweep.name}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[dictlearn_inner_cap] wrote {path}")
    if not all(r["matches_sweep"] for r in rows):
        print("[dictlearn_inner_cap] the released cap did not reproduce the sweep's output", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
