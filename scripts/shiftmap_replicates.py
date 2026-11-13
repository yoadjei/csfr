"""Run-to-run variation of shift-map, the one baseline that is not a function of
its input (src/baselines.py).

    python scripts/shiftmap_replicates.py [--sweep-dir results/sweep_v2_dfrws] [--runs 3]

Every call of baseline_shiftmap runs in a fresh interpreter, so no output depends
on what the harness ran before it; outputs still differ between interpreters.
For every cell of the sweep at seed 0, shift-map is run --runs more times on the
sweep's own input, and each run is set beside the sweep's output: the share of
patches whose output differs in any run, the cell's mean hole PSNR in every
realisation, its standard deviation and range across realisations, and the
largest range of a single patch's hole PSNR.

Output: results/shiftmap_replicates/<sweep>.csv. Each cell is saved when it
finishes (results/shiftmap_replicates/units/<sweep>/), so an interrupted run
resumes at the cells it had not finished.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import baseline_shiftmap  # noqa: E402
from checkpoint import UnitStore, digests  # noqa: E402
from quality import psnr_hole  # noqa: E402

OUT = ROOT / "results" / "shiftmap_replicates"
SEED = 0


def spread(ref: np.ndarray, outs: list[np.ndarray], M: np.ndarray) -> dict:
    """Variation across realisations outs[0], outs[1], ... of one cell."""
    ps = np.array([[psnr_hole(r, x, m)[0] for r, x, m in zip(ref, o, M)] for o in outs])
    differ = np.zeros(len(ref), bool)
    for o in outs[1:]:
        differ |= np.array([not np.array_equal(a, b) for a, b in zip(outs[0], o)])
    means = ps.mean(axis=1)
    return {"n_patches": int(len(ref)), "n_realisations": len(outs),
            "frac_patches_differ": float(differ.mean()),
            "cell_means": " ".join(f"{v:.4f}" for v in means),
            "cell_mean_sd": float(means.std(ddof=1)), "cell_mean_range": float(np.ptp(means)),
            "patch_range_max": float(np.ptp(ps, axis=0).max())}


def _cell(args: tuple) -> dict:
    cell_dir, ref, runs = args
    y, M = np.load(cell_dir / "y.npy"), np.load(cell_dir / "mask.npy")
    outs = [np.load(cell_dir / "x_hat_shiftmap.npy")] + [baseline_shiftmap(y, M) for _ in range(runs)]
    fam, lvl = cell_dir.parent.name.split("_")
    return {"family": fam, "level": lvl, "seed": SEED, **spread(ref, outs, M)}


def _store(units: Path, cell_dir: Path, runs: int, ref_path: Path) -> UnitStore:
    """A cell's measurement is reused only for the same inputs, sweep output and code."""
    return UnitStore(units, {"seed": SEED, "runs": runs,
                             "inputs": digests([cell_dir / "y.npy", cell_dir / "mask.npy",
                                                cell_dir / "x_hat_shiftmap.npy", ref_path]),
                             "code": digests([ROOT / "src" / "baselines.py", Path(__file__)])})


def measure(jobs: list[tuple], stores: dict, workers: int) -> None:
    """Measure the cells that have not finished, saving each as it finishes. With
    one worker the cells run here, one after another (every shift-map call runs
    in its own interpreter either way)."""
    todo = [j for j in jobs if stores[j[0]].load(j[0].parent.name) is None]
    if workers <= 1:
        for j in todo:
            stores[j[0]].save(j[0].parent.name, [_cell(j)])
            print(f"[shiftmap_replicates] {j[0].parent.name} done", flush=True)
        return
    failed = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_cell, j): j[0] for j in todo}
        for f in as_completed(futures):
            cell_dir = futures[f]
            try:
                stores[cell_dir].save(cell_dir.parent.name, [f.result()])
                print(f"[shiftmap_replicates] {cell_dir.parent.name} done", flush=True)
            except Exception as e:                 # the other cells still finish and are saved
                failed.append(f"{cell_dir.parent.name}: {e!r}")
    if failed:
        raise RuntimeError("cells failed: " + "; ".join(failed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="results/sweep_v2_dfrws")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true",
                    help="measure again although the output exists (shift-map is not deterministic, "
                         "so a second measurement gives new realisations)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    sweep = Path(args.sweep_dir)
    if not sweep.is_absolute():
        sweep = ROOT / sweep
    out = Path(args.out_dir)
    path = out / f"{sweep.name}.csv"
    units = out / "units" / sweep.name
    if args.force:                                   # new realisations, from the first cell
        path.unlink(missing_ok=True)
        shutil.rmtree(units, ignore_errors=True)
    if path.exists():
        print(f"[shiftmap_replicates] {path} exists; kept (--force to measure again)")
        return 0
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    ref_path = ROOT / cfg["patches_path"]
    ref = np.load(ref_path).astype(np.float32)
    jobs = [(sweep / f"{f}_{lv}" / f"seed{SEED}", ref, args.runs) for f in cfg["families"] for lv in cfg["levels"]]
    stores = {j[0]: _store(units, j[0], args.runs, ref_path) for j in jobs}
    measure(jobs, stores, args.workers)
    rows = [stores[j[0]].load(j[0].parent.name)[0] for j in jobs]
    out.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[shiftmap_replicates] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
