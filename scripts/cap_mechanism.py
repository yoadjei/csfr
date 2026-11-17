"""Where each cell's hole-PSNR difference to the strongest classical method
comes from: the patches that method reconstructs to the 100 dB cap, and the rest
(Reviewer 1, comment 5; Reviewer 4, comment 4).

    python scripts/cap_mechanism.py [--sweeps sweep_v2_govdocs1 sweep_v2_dfrws sweep_v2_kodak]

Per cell, the strongest classical method is the one Table PAIRED_V2 names
(render_tables_v2.cell_comparison on stats/paired_tests.csv). The unit is the
patch, its hole PSNR averaged over the seeds (run_stats_v2.load), so the mean
difference d = mean_i (CSFR_i - strongest_i) is the reported one (checked to
1e-9). It splits exactly into

  capped_part  (1/n) * the sum over the patches the strongest method
               reconstructs to the cap at one seed or more
  rest_part    (1/n) * the sum over the other patches

For each capped patch-run (patch, seed): whether the true hole is constant
(flat), CSFR's largest absolute and mean signed error over the hole in grey
levels, and whether CSFR's TV-only arm (no DCT or high-frequency term) reaches the
cap on it too. Descriptive only; the pre-specified analysis is unchanged.

Output: results/cap_mechanism/<sweep>.csv, one row per cell.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from quality import CAP  # noqa: E402


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_stats = _script("run_stats_v2")
_tables = _script("render_tables_v2")
SWEEPS = ("sweep_v2_govdocs1", "sweep_v2_dfrws", "sweep_v2_kodak")
WITHIN = 1.0          # grey levels: CSFR's largest hole error below this counts as within one grey level
TV_ONLY = "csfr_tv_only"   # CSFR without the DCT and high-frequency terms; a baseline, so it runs at every seed
TOL = 1e-9


def flat_holes(truth: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per patch: whether it has a hole (mask 0) and the true content of the
    hole is a single grey value."""
    hole = masks == 0
    top = np.where(hole, truth, -np.inf).max(axis=(1, 2))
    bottom = np.where(hole, truth, np.inf).min(axis=(1, 2))
    return hole.any(axis=(1, 2)) & (top == bottom)


def capped_runs(sweep: Path, cell: str, seeds: list[int], method: str, truth: np.ndarray) -> list[dict]:
    """The patch-runs at which method reaches the cap, with the true hole's
    flatness and CSFR's error over the hole."""
    out = []
    for s in seeds:
        d = sweep / cell / f"seed{s}"
        with np.load(d / f"per_patch_{method}.npz") as z:
            at_cap = z["psnr_hole"] >= CAP
        if not at_cap.any():
            continue
        with np.load(d / "per_patch_csfr.npz") as z:
            csfr_psnr = z["psnr_hole"]
        with np.load(d / f"per_patch_{TV_ONLY}.npz") as z:
            tv_exact = z["psnr_hole"] >= CAP
        M = np.load(d / "mask.npy")
        X = np.load(d / "x_hat_csfr.npy").astype(np.float64)
        flat = flat_holes(truth, M)
        for i in np.flatnonzero(at_cap):
            hole = M[i] == 0
            e = X[i][hole] - truth[i][hole]
            out.append({"seed": s, "patch": int(i), "flat": bool(flat[i]),
                        "csfr_max_abs_err": float(np.abs(e).max()), "csfr_mean_err": float(e.mean()),
                        "csfr_psnr_hole": float(csfr_psnr[i]), "tv_only_at_cap": bool(tv_exact[i])})
    return out


def cell_row(sweep: Path, fam: str, lv: str, paired: list[dict], seeds: list[int], truth: np.ndarray) -> dict:
    cell = f"{fam}_{lv}"
    best = _tables.cell_comparison(paired, fam, lv, "psnr_hole", sweep.name)["best"]
    method = best["method"]
    a, b = _stats.load(sweep, cell, seeds, "csfr"), _stats.load(sweep, cell, seeds, method)
    if a is None or b is None:
        raise ValueError(f"{sweep.name} {cell}: CSFR or {method} did not run at every seed")
    d = a["psnr_hole"] - b["psnr_hole"]
    if abs(d.mean() - float(best["diff_mean"])) > TOL:
        raise ValueError(f"{sweep.name} {cell}: recomputed difference {d.mean()} is not the reported "
                         f"{best['diff_mean']}")
    capped = b["at_cap_any"]
    runs = capped_runs(sweep, cell, seeds, method, truth)
    flat = [r for r in runs if r["flat"]]
    n = len(d)
    return {"sweep": sweep.name, "family": fam, "level": lv, "strongest": method,
            "n_patches": n, "n_seeds": len(seeds), "diff_mean": float(d.mean()),
            "capped_part": float(d[capped].sum() / n), "rest_part": float(d[~capped].sum() / n),
            "n_patches_capped": int(capped.sum()), "n_runs_capped": len(runs),
            "n_runs_capped_flat": len(flat),
            "n_runs_capped_csfr_within_1": sum(r["csfr_max_abs_err"] < WITHIN for r in runs),
            "n_runs_capped_flat_csfr_within_1": sum(r["csfr_max_abs_err"] < WITHIN for r in flat),
            "n_runs_capped_flat_tv_only_at_cap": sum(r["tv_only_at_cap"] for r in flat),
            "csfr_psnr_capped_median": float(np.median([r["csfr_psnr_hole"] for r in runs])) if runs else "",
            "csfr_flat_mean_err_median": float(np.median([r["csfr_mean_err"] for r in flat])) if flat else "",
            "csfr_flat_max_abs_err_median": float(np.median([r["csfr_max_abs_err"] for r in flat])) if flat else ""}


def sweep_rows(sweep: Path) -> list[dict]:
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    truth = np.load(ROOT / cfg["patches_path"]).astype(np.float64)
    with (sweep / "stats" / "paired_tests.csv").open(encoding="utf-8") as f:
        paired = list(csv.DictReader(f))
    return [cell_row(sweep, fam, lv, paired, cfg["seeds"], truth)
            for fam in cfg["families"] for lv in cfg["levels"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "cap_mechanism"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in args.sweeps:
        rows = sweep_rows(Path(args.results) / s)
        with (out / f"{s}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"[cap_mechanism] wrote {out / f'{s}.csv'} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
