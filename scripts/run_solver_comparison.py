"""Solver comparison (E5): the released fixed-budget Adam against the convergent solver.

    python scripts/run_solver_comparison.py

On the released DFRWS patches with the released masks
(results/csfr_sweep_2d_v2/<cell>/seed<s>/), every arm is scored with the round-2
hole-only metrics (quality.patch_metrics):

  adam600_released   the released reconstructions (x_hat.npy, Linux T4 GPU), read
                     back verbatim; seconds are the released GPU runtime_s
  adam600_cpu        the released solver re-run on this CPU at --retime-seed only,
                     for a CPU runtime and the GPU-versus-CPU difference
  adam<T>_cpu        the released solver with budget T (--adam-budgets, default 2400)
  fista_gap<tol>     the convergent solver (src/csfr_solver.py) at each gap
                     tolerance (--tols; default 1e-4, the solver default, 1e-10)

The Adam arms minimise the released objective (a quadratic data term plus
unsmoothed l1, TV and frequency terms, observed pixels restored after the last
step); the FISTA arms minimise the round-2 objective with the observed pixels
fixed. The two objectives are not comparable in value, so the shared axis is
reconstruction quality against iterations and wall-clock.

Outputs (--out, default results/solver_comparison):
  per_cell.csv            arm, cell, seed: hole PSNR, hole SSIM, hole GMSD,
                          seconds, iterations, converged count, device
  quality_vs_compute.csv  per family and arm: means over levels and seeds
  trace_<cell>.csv        seed 0 of --trace-cell: hole PSNR of both solvers at
                          each of --trace-iters iterations (FISTA run without a
                          tolerance, so every patch takes every iteration), and
                          the Adam loss and FISTA gap at those points
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_reconstruct_2d import reconstruct  # noqa: E402
from csfr_solver import SolverConfig, solve  # noqa: E402

_spec = importlib.util.spec_from_file_location("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)
score = _sweep.score

ALL_CELLS = [f"CF{f}_L{l}" for f in range(1, 5) for l in range(1, 6)]
RELEASED_CFG = ROOT / "configs" / "csfr_sweep_2d.yaml"


def _released_solver() -> dict:
    s = yaml.safe_load(RELEASED_CFG.read_text())["solver"]
    return {k: s[k] for k in ("lam_l1", "lam_tv", "lam_fc", "lr")}


def adam(y: np.ndarray, M: np.ndarray, budget: int, ckpt: Path | None = None,
         resume: bool = False) -> tuple[np.ndarray, dict]:
    s = _released_solver()
    x, info = reconstruct(torch.from_numpy(y.astype(np.float32)), torch.from_numpy(M.astype(np.float32)),
                          lam_l1=s["lam_l1"], lam_tv=s["lam_tv"], lam_fc=s["lam_fc"], max_iter=budget,
                          lr=s["lr"], device=torch.device("cpu"), ckpt_path=ckpt,
                          ckpt_every=10 if ckpt else 10 ** 9, resume=resume, log_every=10)
    return x.numpy().astype(np.float64), info


def _row(arm, cell, seed, ref, x, y, M, seconds, device, **extra) -> dict:
    pp = score(ref, x, y, M)
    return {"arm": arm, "cell": cell, "family": cell.split("_")[0], "level": cell.split("_")[1],
            "seed": seed, "n": len(x), "psnr_hole": float(pp["psnr_hole"].mean()),
            "ssim_hole": float(pp["ssim_hole"].mean()), "gmsd_hole": float(pp["gmsd_hole"].mean()),
            "n_exact": int(pp["exact"].sum()), "seconds": seconds, "device": device, **extra}


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def trace(cell_dir: Path, ref: np.ndarray, iters: list[int], tmp: Path) -> list[dict]:
    y = np.load(cell_dir / "y.npy").astype(np.float64)
    M = np.load(cell_dir / "mask.npy")
    rows = []
    ckpt = tmp / "trace_adam.pt"
    if ckpt.exists():
        ckpt.unlink()
    for k in sorted(iters):
        x, info = adam(y, M, k, ckpt=ckpt, resume=True)
        loss = info["history"][-1]["loss"] if info["history"] else float("nan")
        rows.append({"arm": "adam", "iter": k, "psnr_hole": float(score(ref, x, y, M)["psnr_hole"].mean()),
                     "objective": loss, "objective_kind": "released loss (data + l1 + tv + fc)",
                     "gap_max": ""})
    ckpt.unlink(missing_ok=True)
    for k in sorted(iters):
        x, cert = solve(torch.from_numpy(y), torch.from_numpy(M), replace(SolverConfig(), tol=0.0, max_iter=k))
        rows.append({"arm": "fista", "iter": k,
                     "psnr_hole": float(score(ref, x.numpy(), y, M)["psnr_hole"].mean()),
                     "objective": float(cert.objective.mean()),
                     "objective_kind": "round-2 objective J", "gap_max": float(cert.gap.max())})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", default="results/csfr_sweep_2d_v2")
    ap.add_argument("--patches", default="data/patches.npy")
    ap.add_argument("--cells", nargs="+", default=ALL_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--adam-budgets", nargs="+", type=int, default=[2400])
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-4, SolverConfig().tol, 1e-10])
    ap.add_argument("--retime-seed", type=int, default=0)
    ap.add_argument("--trace-cell", default="CF2_L5")
    ap.add_argument("--trace-iters", nargs="+", type=int,
                    default=[25, 50, 100, 200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/solver_comparison")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    released = Path(args.released) if Path(args.released).is_absolute() else ROOT / args.released
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ppath = Path(args.patches) if Path(args.patches).is_absolute() else ROOT / args.patches
    ref = np.load(ppath).astype(np.float32)
    tols = list(dict.fromkeys(args.tols))
    rows: list[dict] = []
    for cell in args.cells:
        for seed in args.seeds:
            d = released / cell / f"seed{seed}"
            y = np.load(d / "y.npy")
            M = np.load(d / "mask.npy")
            if not np.array_equal(y, ref * M):
                raise RuntimeError(f"{d}: released y differs from patches * mask")
            y64 = y.astype(np.float64)
            meta = json.loads((d / "metrics.json").read_text())
            x_rel = np.load(d / "x_hat.npy").astype(np.float64)
            rows.append(_row("adam600_released", cell, seed, ref, x_rel, y64, M,
                             float(meta["runtime_s"]), meta.get("device", "unknown"), iterations=600))
            budgets = list(args.adam_budgets)
            if seed == args.retime_seed and 600 not in budgets:
                budgets = [600] + budgets
            for T in budgets:
                t0 = time.time()
                x, _ = adam(y64, M, T)
                dt = time.time() - t0
                extra = {"iterations": T}
                if T == 600:
                    extra["max_abs_diff_vs_released"] = float(np.abs(x - x_rel).max())
                rows.append(_row(f"adam{T}_cpu", cell, seed, ref, x, y64, M, dt, "cpu", **extra))
            for tol in tols:
                t0 = time.time()
                x, cert = solve(torch.from_numpy(y64), torch.from_numpy(M), replace(SolverConfig(), tol=tol))
                dt = time.time() - t0
                rows.append(_row(f"fista_gap{tol:.0e}", cell, seed, ref, x.numpy(), y64, M, dt, "cpu",
                                 iterations=float(cert.iterations.float().median()),
                                 n_converged=int(cert.converged.sum()), gap_max=float(cert.gap.max())))
            last = [r for r in rows if r["cell"] == cell and r["seed"] == seed]
            print(f"[solver_cmp] {cell} seed {seed}: " +
                  "  ".join(f"{r['arm']} {r['psnr_hole']:.2f}dB/{r['seconds']:.0f}s" for r in last), flush=True)
            _write(out / "per_cell.csv", rows)

    summary = []
    for fam in sorted({r["family"] for r in rows}):
        for arm in dict.fromkeys(r["arm"] for r in rows):
            rs = [r for r in rows if r["family"] == fam and r["arm"] == arm]
            if not rs:
                continue
            summary.append({"family": fam, "arm": arm, "n_cells": len(rs),
                            "seeds": " ".join(map(str, sorted({r["seed"] for r in rs}))),
                            "device": " ".join(sorted({r["device"] for r in rs})),
                            **{m: float(np.mean([r[m] for r in rs])) for m in ("psnr_hole", "ssim_hole", "gmsd_hole")},
                            "seconds_per_cell": float(np.mean([r["seconds"] for r in rs])),
                            "iterations_median": float(np.median([r["iterations"] for r in rs])),
                            "converged_frac": (float(np.mean([r["n_converged"] / r["n"] for r in rs]))
                                               if "n_converged" in rs[0] else "")})
    _write(out / "quality_vs_compute.csv", summary)
    if args.trace_cell in args.cells:
        t = trace(released / args.trace_cell / "seed0", ref, args.trace_iters, out)
        _write(out / f"trace_{args.trace_cell}.csv", t)
    (out / "README.md").write_text(
        "Solver comparison (E5); see the docstring of scripts/run_solver_comparison.py.\n"
        "adam600_released quality is from the released GPU artefacts and its seconds are GPU\n"
        "time; every other arm ran on this CPU (run_metadata of the round-2 sweeps records it).\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
