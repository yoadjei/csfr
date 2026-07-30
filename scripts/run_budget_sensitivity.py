"""Iteration-budget sensitivity for the CSFR solver.

The convergence trace shows the objective still falling at the end of the fixed
600-iteration budget under burst corruption (CF2), the one family where CSFR is
beaten by every classical baseline. This script tests the obvious question: is
that deficit an artefact of stopping too early, or a property of the method?

For each selected cell it re-solves at several budgets from the same
initialisation and reports PSNR against the same ground truth, alongside the
strongest baseline's PSNR at that cell for reference.

Outputs ``results/budget_sensitivity/budget.csv``.

Usage:
    python scripts/run_budget_sensitivity.py
    python scripts/run_budget_sensitivity.py --budgets 600 1200 2400 4800
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from csfr_reconstruct_2d import pick_device, psnr, reconstruct, ssim2d  # noqa: E402

SOLVER = {"lam_l1": 0.02, "lam_tv": 0.10, "lam_fc": 0.005, "lr": 2.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="results/csfr_sweep_2d_v2")
    ap.add_argument("--patches", default="data/patches.npy")
    ap.add_argument("--out-dir", default="results/budget_sensitivity")
    ap.add_argument("--cells", nargs="+",
                    default=["CF1_L5", "CF2_L3", "CF2_L5", "CF3_L5", "CF4_L5"],
                    help="Cells to probe: the under-converged ones plus controls")
    ap.add_argument("--budgets", nargs="+", type=int, default=[600, 1200, 2400, 4800])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100, help="Patches per cell (subset, for cost)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = pick_device(args.device)
    ref_all = np.load(PROJECT_ROOT / args.patches).astype(np.float32)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cell in args.cells:
        seed_dir = PROJECT_ROOT / args.sweep_dir / cell / f"seed{args.seed}"
        y = np.load(seed_dir / "y.npy")[:args.n]
        M = np.load(seed_dir / "mask.npy")[:args.n]
        ref = ref_all[:args.n]
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        baselines = {k: v["psnr_db_mean"] for k, v in metrics.items()
                     if isinstance(v, dict) and k not in
                     ("csfr", "csfr_c1", "csfr_c1c2", "csfr_c1c3", "csfr_c1c4")}
        best_base = max(baselines, key=baselines.get)

        y_t, M_t = torch.from_numpy(y), torch.from_numpy(M)
        for budget in args.budgets:
            t0 = time.time()
            # ckpt_path=None and resume=False force a genuine re-solve from
            # scratch at each budget rather than reusing the sweep checkpoints.
            x, _ = reconstruct(y_t, M_t, max_iter=budget, device=dev,
                               ckpt_path=None, ckpt_every=10 ** 9, resume=False,
                               **SOLVER)
            xc = x.cpu()
            ps = float(np.mean([psnr(torch.from_numpy(ref[i]), xc[i])
                                for i in range(len(ref))]))
            ss = float(np.mean([ssim2d(ref[i], xc[i].numpy()) for i in range(len(ref))]))
            rows.append({"cell": cell, "budget": budget, "n": len(ref),
                         "psnr_db": ps, "ssim": ss,
                         "best_baseline": best_base,
                         "best_baseline_psnr_db": baselines[best_base],
                         "gap_to_best_baseline": ps - baselines[best_base],
                         "solve_s": time.time() - t0})
            print(f"[budget] {cell:8s} T={budget:5d}  PSNR {ps:6.2f}  "
                  f"(best baseline {best_base} {baselines[best_base]:.2f}, "
                  f"gap {ps - baselines[best_base]:+.2f})", flush=True)

    csv_path = out_dir / "budget.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[budget] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
