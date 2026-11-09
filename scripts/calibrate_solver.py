"""Stopping-rule calibration for the convergent CSFR solver (round 2).

Iteration counts, runtime, hole PSNR and initialisation independence on the
four probe cells of the released DFRWS sweep (seed 0), for tol in
{1e-1, 1e-2, 1e-3} grey levels. The masks are the released ones, so the numbers
are directly comparable with the Adam-600 artefacts in results/csfr_sweep_2d_v2.

Writes results/solver_calibration/calibration.csv.

Usage:
    python scripts/calibrate_solver.py
    python scripts/calibrate_solver.py --cells CF1_L3 --tols 1e-2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_solver import SolverConfig, solve  # noqa: E402
from quality import psnr_hole  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["CF1_L3", "CF2_L3", "CF2_L5", "CF3_L5"])
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3])
    ap.add_argument("--init-check-tol", type=float, default=1e-2)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/solver_calibration")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    ref_all = np.load(ROOT / "data/patches.npy").astype(np.float32)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in args.cells:
        d = ROOT / "results/csfr_sweep_2d_v2" / cell / "seed0"
        y = torch.from_numpy(np.load(d / "y.npy"))
        M = torch.from_numpy(np.load(d / "mask.npy"))
        Mn = M.numpy()
        for tol in args.tols:
            t0 = time.time()
            x, cert = solve(y, M, SolverConfig(tol=tol, max_iter=args.max_iter))
            dt = time.time() - t0
            it = cert.iterations.numpy()
            xs = x.numpy()
            p = float(np.mean([psnr_hole(ref_all[i], xs[i], Mn[i])[0] for i in range(len(xs))]))
            gap = float("nan")
            if tol == args.init_check_tol:
                x_ns, _ = solve(y, M, SolverConfig(tol=tol, max_iter=args.max_iter, init="ns"))
                gap = float((x - x_ns).abs().max())
            rows.append({"cell": cell, "tol": tol, "n": len(xs), "psnr_hole": p,
                         "iters_median": int(np.median(it)), "iters_p90": int(np.percentile(it, 90)),
                         "iters_max": int(it.max()), "n_capped": int((~cert.converged.numpy()).sum()),
                         "seconds": dt, "max_abs_gap_zero_vs_ns_init": gap})
            print(rows[-1], flush=True)
            with (out / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
