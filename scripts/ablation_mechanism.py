"""Where the hole-PSNR difference between CSFR and each ablation arm comes
from: the patches whose true hole is flat, and the rest (Reviewer 1, comment 5).

    python scripts/ablation_mechanism.py [--sweep results/sweep_v2_govdocs1]

The arms and seeds are the ones the stats record (stats/analysis_plan.json:
ablation_family, ablation_seeds). The unit is the patch, its hole PSNR averaged
over those seeds (run_stats_v2.load), so the mean difference
d = mean_i (CSFR_i - arm_i) is the one stats/paired_tests_ablation.csv reports
(checked to 1e-9). It splits exactly into

  flat_part  (1/n) * the sum over the patches whose true hole is a single
             grey value at every ablation seed
  rest_part  (1/n) * the sum over the other patches

with the median difference, the share of patches on which CSFR is ahead, and
how many of the flat holes the arm reconstructs to the cap. Descriptive only;
the pre-specified analysis is unchanged.

Output: results/ablation_mechanism/<sweep>.csv, one row per (arm, cell).
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
METRIC = "psnr_hole"
TOL = 1e-9


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_stats = _script("run_stats_v2")
_cap = _script("cap_mechanism")


def cell_row(sweep: Path, fam: str, lv: str, arm: str, seeds: list[int], truth: np.ndarray,
             reported: dict) -> dict:
    cell = f"{fam}_{lv}"
    a, b = _stats.load(sweep, cell, seeds, "csfr"), _stats.load(sweep, cell, seeds, arm)
    if a is None or b is None:
        raise ValueError(f"{sweep.name} {cell}: CSFR or {arm} did not run at every ablation seed")
    d = a[METRIC] - b[METRIC]
    if abs(d.mean() - float(reported["diff_mean"])) > TOL:
        raise ValueError(f"{sweep.name} {cell} {arm}: recomputed difference {d.mean()} is not the reported "
                         f"{reported['diff_mean']}")
    flat = np.all([_cap.flat_holes(truth, np.load(sweep / cell / f"seed{s}" / "mask.npy")) for s in seeds],
                  axis=0)
    n = len(d)
    return {"sweep": sweep.name, "method": arm, "family": fam, "level": lv,
            "n_patches": n, "n_seeds": len(seeds), "diff_mean": float(d.mean()),
            "diff_median": float(np.median(d)), "frac_patches_csfr_ahead": float((d > 0).mean()),
            "p_holm": float(reported["p_holm"]), "n_flat": int(flat.sum()),
            "flat_part": float(d[flat].sum() / n), "rest_part": float(d[~flat].sum() / n),
            "n_flat_arm_at_cap": int((flat & b["at_cap_any"]).sum())}


def sweep_rows(sweep: Path) -> list[dict]:
    cfg = json.loads((sweep / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    plan = json.loads((sweep / "stats" / "analysis_plan.json").read_text(encoding="utf-8"))
    truth = np.load(ROOT / cfg["patches_path"]).astype(np.float64)
    with (sweep / "stats" / "paired_tests_ablation.csv").open(encoding="utf-8") as f:
        reported = {(r["method"], r["family"], r["level"]): r for r in csv.DictReader(f) if r["metric"] == METRIC}
    return [cell_row(sweep, fam, lv, arm, plan["ablation_seeds"], truth, reported[(arm, fam, lv)])
            for arm in plan["ablation_family"] for fam in cfg["families"] for lv in cfg["levels"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=str(ROOT / "results" / "sweep_v2_govdocs1"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation_mechanism"))
    args = ap.parse_args(argv)
    sweep, out = Path(args.sweep), Path(args.out)
    rows = sweep_rows(sweep)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{sweep.name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[ablation_mechanism] wrote {out / f'{sweep.name}.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
