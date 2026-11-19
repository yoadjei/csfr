"""What each wrong mask of the mask-error experiment (E10) does to the observations.

    python scripts/maskerr_geometry.py

Reads the inputs the experiment saved (results/mask_robustness/cases/<cell>/<perturbation>/
mask.npy and y.npy, and the true mask under exact/) and the ground-truth patches, and
writes results/mask_robustness/geometry.csv, one row per cell and perturbation:

  n                      patches
  hole_share             share of the pixels in the true hole
  fp_share_of_hole       hole pixels the wrong mask marks observed, as a share of the true hole
  fn_share_of_observed   observed pixels the wrong mask marks missing, as a share of the true Omega
  n_none_observed        patches in which the wrong mask marks no pixel observed
  n_hole_all_observed    patches in which the wrong mask marks the whole true hole observed
  border_fp_share        of the pixels the wrong mask marks observed next to its own hole
                         (8-connected), the share that are false positives: 1 means the
                         remaining hole is bordered by false positives only
  fp_value_max           the largest input value at a false positive (the experiment gives
                         a false positive the value 0 of the corrupted input)
  zero_fill_psnr_hole    hole PSNR of the zero-filled input over the true hole (the floor),
                         scored as the sweep scores every method

Only reads saved inputs; no method is run.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["cell", "perturbation", "n", "hole_share", "fp_share_of_hole", "fn_share_of_observed",
          "n_none_observed", "n_hole_all_observed", "border_fp_share", "fp_value_max", "zero_fill_psnr_hole"]


def _sweep():
    spec = importlib.util.spec_from_file_location("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def geometry(M_true: np.ndarray, M_wrong: np.ndarray, y: np.ndarray) -> dict:
    """The effect of a wrong mask on the observations, pooled over patches (B, H, W).
    1 = observed in both masks."""
    obs, wrong = np.asarray(M_true) == 1, np.asarray(M_wrong) == 1
    hole = ~obs
    fp, fn = wrong & hole, ~wrong & obs
    st = np.ones((3, 3), bool)
    border = np.zeros_like(wrong)
    for i in range(len(wrong)):
        border[i] = wrong[i] & binary_dilation(~wrong[i], st)        # observed, next to the wrong hole
    n_border = int(border.sum())
    return {"n": int(len(obs)), "hole_share": float(hole.mean()),
            "fp_share_of_hole": float(fp.sum() / hole.sum()) if hole.any() else float("nan"),
            "fn_share_of_observed": float(fn.sum() / obs.sum()) if obs.any() else float("nan"),
            "n_none_observed": int((~wrong.any(axis=(1, 2))).sum()),
            "n_hole_all_observed": int((fp.sum(axis=(1, 2)) == hole.sum(axis=(1, 2))).sum()),
            "border_fp_share": float((border & fp).sum() / n_border) if n_border else float("nan"),
            "fp_value_max": float(np.abs(y[fp]).max()) if fp.any() else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/patches_govdocs1_v2.npy")
    ap.add_argument("--root", default="results/mask_robustness")
    args = ap.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    ppath = Path(args.patches) if Path(args.patches).is_absolute() else ROOT / args.patches
    X = np.load(ppath).astype(np.float32)
    score = _sweep().score
    rows = []
    for cell_dir in sorted(p for p in (root / "cases").iterdir() if p.is_dir()):
        M_true = np.load(cell_dir / "exact" / "mask.npy")
        Xc = X[: len(M_true)]
        y0 = (Xc * M_true).astype(np.float32)
        floor = float(np.mean(score(Xc, y0.astype(np.float64), y0, M_true)["psnr_hole"]))
        for pert_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            M, y = np.load(pert_dir / "mask.npy"), np.load(pert_dir / "y.npy")
            rows.append({"cell": cell_dir.name, "perturbation": pert_dir.name, **geometry(M_true, M, y),
                         "zero_fill_psnr_hole": floor})
            print(f"[geometry] {cell_dir.name} {pert_dir.name}: " +
                  ", ".join(f"{k} {rows[-1][k]:.4g}" for k in FIELDS[3:]), flush=True)
    with (root / "geometry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[geometry] wrote {root / 'geometry.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
