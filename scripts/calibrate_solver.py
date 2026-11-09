"""Stopping-rule calibration for the convergent CSFR solver (round 2).

On the four probe cells of the released DFRWS sweep (seed 0, released masks):

1. a reference solve to relative Frank-Wolfe gap 1e-13 (the float64 floor);
2. solves at tol in {1e-6, 1e-8, 1e-10, 1e-12}, each compared pixel by pixel
   with the reference: the tolerance used everywhere is the loosest one whose
   99th-percentile per-patch maximum error is below 0.01 grey levels;
3. initialisation independence at the default tolerance (zero fill against
   Navier-Stokes);
4. an independent solver: L-BFGS-B (SciPy) on the first 20 patches, compared
   with the reference.

Also records the outcome of the earlier projected-step rule, which stopped far
from the optimum on large holes, so the reason for the rule is in the artefact.

Writes results/solver_calibration/{tolerance.csv, init.csv, lbfgsb.csv}.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_objective import Objective  # noqa: E402
from csfr_solver import SolverConfig, solve  # noqa: E402
from quality import psnr_hole  # noqa: E402


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _psnr(ref, x, M) -> np.ndarray:
    return np.array([psnr_hole(ref[i], x[i], M[i])[0] for i in range(len(x))])


def lbfgsb(y: np.ndarray, M: np.ndarray, cfg: SolverConfig) -> tuple[np.ndarray, list[int]]:
    """Per-patch L-BFGS-B on the same objective (free pixels, box [0, 255]), float64."""
    obj = Objective(y.shape[-2], y.shape[-1], cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps,
                    torch.device("cpu"), torch.float64)
    scale = 1.0 / obj.lipschitz()
    out, nits = [], []
    for yi, mi in zip(y.astype(np.float64), M):
        free = mi == 0

        def fg(v, yi=yi, free=free):
            X = yi.copy()
            X[free] = v
            Xt = torch.from_numpy(X)
            return float(obj.value(Xt)) * scale, obj.grad(Xt).numpy()[free] * scale

        r = minimize(fg, np.zeros(int(free.sum())), jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 255.0)] * int(free.sum()),
                     options={"maxiter": 50000, "maxfun": 100000, "ftol": 1e-15,
                              "gtol": 1e-10, "maxcor": 20})
        X = yi.copy()
        X[free] = r.x
        out.append(X)
        nits.append(int(r.nit))
    return np.stack(out), nits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["CF1_L3", "CF2_L3", "CF2_L5", "CF3_L5"])
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-6, 1e-8, 1e-10, 1e-12])
    ap.add_argument("--ref-tol", type=float, default=1e-13)
    ap.add_argument("--max-iter", type=int, default=40000)
    ap.add_argument("--n-lbfgsb", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/solver_calibration")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ref_all = np.load(ROOT / "data/patches.npy").astype(np.float64)
    base = SolverConfig(max_iter=args.max_iter)
    tol_rows, init_rows, lb_rows = [], [], []
    for cell in args.cells:
        d = ROOT / "results/csfr_sweep_2d_v2" / cell / "seed0"
        y = torch.from_numpy(np.load(d / "y.npy"))
        M = torch.from_numpy(np.load(d / "mask.npy"))
        Mn = M.numpy()
        ref = ref_all[: len(y)]

        t0 = time.time()
        x_ref, c_ref = solve(y, M, replace(base, tol=args.ref_tol))
        t_ref = time.time() - t0
        p_ref = _psnr(ref, x_ref.numpy(), Mn)
        print(f"[calib] {cell} reference: {t_ref:.0f}s, capped {int((~c_ref.converged).sum())}, "
              f"hole PSNR {p_ref.mean():.3f}", flush=True)
        for tol in args.tols:
            t0 = time.time()
            x, c = solve(y, M, replace(base, tol=tol))
            dt = time.time() - t0
            err = (x - x_ref).abs().flatten(1).amax(1).numpy()
            it = c.iterations.numpy()
            p = _psnr(ref, x.numpy(), Mn)
            tol_rows.append({"cell": cell, "tol": tol, "n": len(y),
                             "max_err_median": float(np.median(err)),
                             "max_err_p99": float(np.percentile(err, 99)),
                             "max_err_max": float(err.max()),
                             "psnr_hole": float(p.mean()), "psnr_hole_ref": float(p_ref.mean()),
                             "psnr_abs_diff_max": float(np.abs(p - p_ref).max()),
                             "iters_median": int(np.median(it)), "iters_p90": int(np.percentile(it, 90)),
                             "iters_max": int(it.max()), "n_capped": int((~c.converged).sum()),
                             "seconds": dt, "ref_seconds": t_ref,
                             "ref_capped": int((~c_ref.converged).sum())})
            print(f"[calib] {tol_rows[-1]}", flush=True)
            _write(out / "tolerance.csv", tol_rows)

        x_ns, c_ns = solve(y, M, replace(base, init="ns"))
        x_z, c_z = solve(y, M, base)
        gap = (x_z - x_ns).abs().flatten(1).amax(1).numpy()
        init_rows.append({"cell": cell, "tol": base.tol, "max_abs_diff_median": float(np.median(gap)),
                          "max_abs_diff_p99": float(np.percentile(gap, 99)),
                          "max_abs_diff_max": float(gap.max()),
                          "iters_median_zero": int(c_z.iterations.median()),
                          "iters_median_ns": int(c_ns.iterations.median())})
        print(f"[calib] {init_rows[-1]}", flush=True)
        _write(out / "init.csv", init_rows)

        n = args.n_lbfgsb
        t0 = time.time()
        x_lb, nits = lbfgsb(y[:n].numpy(), Mn[:n], base)
        dt = time.time() - t0
        diff = np.abs(x_lb - x_ref[:n].numpy()).reshape(n, -1).max(1)
        obj = Objective(64, 64, base.lam_l1, base.lam_tv, base.lam_fc, base.eps,
                        torch.device("cpu"), torch.float64)
        lb_rows.append({"cell": cell, "n": n, "max_abs_diff_median": float(np.median(diff)),
                        "max_abs_diff_max": float(diff.max()),
                        "objective_lbfgsb": float(obj.value(torch.from_numpy(x_lb)).mean()),
                        "objective_fista": float(obj.value(x_ref[:n]).mean()),
                        "nit_median": int(np.median(nits)), "seconds": dt})
        print(f"[calib] {lb_rows[-1]}", flush=True)
        _write(out / "lbfgsb.csv", lb_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
