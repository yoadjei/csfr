"""Mask-error robustness (E10): what a wrong mask costs each method.

    python scripts/run_mask_robustness.py

On the GovDocs1 corpus at seed 0, cells CF1, CF2 and CF3 at L3 and L5, the true
mask (generated exactly as the sweep generates it) is perturbed with
masks.perturb_mask: Omega eroded by 1 and 2 pixels (observed pixels next to the
hole marked missing: false negatives), Omega dilated by 1 and 2 pixels (hole
pixels next to the boundary marked observed, carrying their corrupted value 0:
false positives), and 1 and 5 percent of the pixels flipped. The input is
y = patches * M_true * M_wrong: a pixel the wrong mask marks observed shows what
the corrupted input holds there. Every method reconstructs from (y, M_wrong) and
is scored over the TRUE hole against the ground truth.

Outputs (results/mask_robustness): robustness.csv, per method, cell and
perturbation, the mean hole PSNR, the mean degradation relative to the exact
mask (perturbed minus exact, paired by patch) with its 95 percent bootstrap
interval, and the realised false-positive and false-negative fractions. Each
(cell, perturbation) directory holds y.npy and mask.npy for the LaMa environment
(scripts/run_lama_cells.py); its outputs are scored on the next run.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_solver import VARIANTS, solve  # noqa: E402
from masks import SEVERITY, make_mask, perturb_mask  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sweep = _load("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
_paired = _load("_paired", ROOT / "scripts" / "run_paired_stats.py")

CELLS = ["CF1_L3", "CF2_L3", "CF3_L3", "CF1_L5", "CF2_L5", "CF3_L5"]
PERTURBATIONS = {"exact": None, "erode1": ("erode", 1), "erode2": ("erode", 2),
                 "dilate1": ("dilate", 1), "dilate2": ("dilate", 2),
                 "flip1pct": ("flip", 0.01), "flip5pct": ("flip", 0.05)}
METHODS = ["csfr", "bilinear", "inpaint_telea", "inpaint_ns", "biharmonic", "guleryuz", "fsr"]
EXTERNAL = ["lama"]


def wrong_mask(M_true: np.ndarray, name: str, seed: int) -> np.ndarray:
    spec = PERTURBATIONS[name]
    if spec is None:
        return M_true.copy()
    kind, amount = spec
    return perturb_mask(M_true, kind, amount, np.random.default_rng([seed, list(PERTURBATIONS).index(name)]))


def reconstruct(name: str, y: np.ndarray, M: np.ndarray, seed: int, pool, workers: int) -> np.ndarray:
    if name in VARIANTS:
        x, _ = solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(M), VARIANTS[name])
        return x.numpy()
    x, _ = _sweep.run_baseline(name, y, M, seed, pool=pool, workers=workers)
    return x.astype(np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/patches_govdocs1_v2.npy")
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="first N patches only (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="results/mask_robustness")
    args = ap.parse_args(argv)

    if "exact" not in args.perturbations:
        args.perturbations = ["exact"] + args.perturbations
    torch.set_num_threads(args.threads)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ppath = Path(args.patches) if Path(args.patches).is_absolute() else ROOT / args.patches
    X = np.load(ppath).astype(np.float32)[: args.n]
    rows = []
    pool = _sweep._make_pool(args.workers)
    try:
        for cell in args.cells:
            fam, lvl = cell.split("_")
            M_true = make_mask(X.shape, fam, SEVERITY[lvl], np.random.default_rng(args.seed))
            per: dict[tuple[str, str], np.ndarray] = {}
            for pert in args.perturbations:
                M = wrong_mask(M_true, pert, args.seed)
                y = (X * M_true * M).astype(np.float32)
                d = out / "cases" / cell / pert
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "y.npy", y)
                np.save(d / "mask.npy", M)
                fp = float(((M == 1) & (M_true == 0)).mean())
                fn = float(((M == 0) & (M_true == 1)).mean())
                for name in args.methods + EXTERNAL:
                    pfile = d / f"per_patch_{name}.npz"
                    if pfile.exists():
                        psnr = _sweep._load_npz(pfile)["psnr_hole"]
                    else:
                        if name in EXTERNAL:
                            xp = d / f"x_hat_{name}.npy"
                            if not xp.exists():
                                continue
                            x = np.load(xp).astype(np.float64)
                        else:
                            x = reconstruct(name, y, M, args.seed, pool, args.workers)
                        pp = _sweep.score(X, x, y, M_true)
                        np.savez(pfile, **pp)
                        psnr = pp["psnr_hole"]
                    per[(pert, name)] = psnr
                    rows.append({"cell": cell, "family": fam, "level": lvl, "perturbation": pert,
                                 "method": name, "n": int(len(psnr)), "psnr_hole_mean": float(psnr.mean()),
                                 "false_pos_frac": fp, "false_neg_frac": fn})
                print(f"[mask_rob] {cell} {pert}: fp {fp:.4f} fn {fn:.4f} " +
                      " ".join(f"{r['method']} {r['psnr_hole_mean']:.2f}" for r in rows
                               if r["cell"] == cell and r["perturbation"] == pert), flush=True)
            for r in rows:
                if r["cell"] != cell or (r["perturbation"], r["method"]) not in per:
                    continue
                base = per.get(("exact", r["method"]))
                if base is None:
                    continue
                diff = per[(r["perturbation"], r["method"])] - base
                lo, hi = _paired.bootstrap_ci(diff, args.bootstrap,
                                              np.random.default_rng([args.seed, CELLS.index(cell) if cell in CELLS else 99,
                                                                     list(PERTURBATIONS).index(r["perturbation"]),
                                                                     (args.methods + EXTERNAL).index(r["method"])]))
                r.update({"degradation_mean": float(diff.mean()), "degradation_ci_lo": lo,
                          "degradation_ci_hi": hi})
            with (out / "robustness.csv").open("w", newline="", encoding="utf-8") as f:
                fields = list(dict.fromkeys(k for r in rows for k in r))
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    finally:
        pool.shutdown()
    (out / "README.md").write_text("Mask-error robustness (E10); see scripts/run_mask_robustness.py.\n",
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
