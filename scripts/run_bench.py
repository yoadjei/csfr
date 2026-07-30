"""Wall-clock runtime and peak-memory benchmark (pre-submission review §6, P1).

Each method reconstructs the same batch of patches under the same masks; the
batch is timed end to end and repeated, and peak allocation is measured in a
separate pass so the profiler's overhead never contaminates the timings.

Peak memory is reported two ways because they answer different questions:
  * ``peak_alloc_mb`` -- peak *Python-level* allocation attributable to the
    reconstruction call, via ``tracemalloc``. Comparable across methods.
  * ``rss_delta_mb``  -- growth in resident set size across the call, which also
    captures allocations made inside native libraries (OpenCV, BLAS, torch).

Outputs ``results/bench/runtime_memory.csv`` and ``environment.json``.

Usage:
    python scripts/run_bench.py
    python scripts/run_bench.py --repeats 5 --batch 64
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from csfr_reconstruct_2d import pick_device, reconstruct  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_sweep2d", PROJECT_ROOT / "scripts" / "run_csfr_sweep.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)

LABEL = {"zero_fill": "Zero-fill", "bilinear": "Linear interpolation",
         "inpaint_telea": "Inpainting (Telea)", "inpaint_ns": "Inpainting (Navier--Stokes)",
         "dictlearn": "Dictionary learning", "csfr": "CSFR (proposed)"}


def build_methods(solver: dict, device):
    import cv2

    def run_csfr(y, M, seed):
        y_t = torch.from_numpy(y)
        M_t = torch.from_numpy(M)
        x, _ = reconstruct(
            y_t, M_t, lam_l1=solver["lam_l1"], lam_tv=solver["lam_tv"],
            lam_fc=solver["lam_fc"], max_iter=solver["max_iter"], lr=solver["lr"],
            device=device, ckpt_path=None, ckpt_every=10 ** 9, resume=False)
        return x.cpu().numpy()

    return {
        "zero_fill": lambda y, M, s: _sweep.baseline_zero(y, M),
        "bilinear": lambda y, M, s: _sweep.baseline_bilinear(y, M),
        "inpaint_telea": lambda y, M, s: _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_TELEA),
        "inpaint_ns": lambda y, M, s: _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_NS),
        "dictlearn": lambda y, M, s: _sweep.baseline_dictlearn(y, M, s),
        "csfr": run_csfr,
    }


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2 ** 20


def environment() -> dict:
    import scipy
    import sklearn
    import skimage
    import cv2
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import psutil
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["total_ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        env["psutil"] = "not installed (RSS and CPU inventory unavailable)"
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="results/csfr_sweep_2d_v2/CF1_L3/seed0",
                    help="Cell supplying the corrupted batch and mask")
    ap.add_argument("--batch", type=int, default=32, help="Patches per timed call")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-dir", default="results/bench")
    ap.add_argument("--device", default="cpu",
                    help="Solver device; the reported figures use CPU so that "
                         "every method is measured on the same hardware path")
    args = ap.parse_args(argv)

    cell = PROJECT_ROOT / args.cell
    y_all = np.load(cell / "y.npy")
    M_all = np.load(cell / "mask.npy")
    y, M = y_all[:args.batch].copy(), M_all[:args.batch].copy()
    device = pick_device(args.device)

    cfg_solver = {"lam_l1": 0.02, "lam_tv": 0.10, "lam_fc": 0.005,
                  "max_iter": 600, "lr": 2.0}
    methods = build_methods(cfg_solver, device)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, fn in methods.items():
        fn(y[:2].copy(), M[:2].copy(), 0)  # warm up imports and JIT paths

        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn(y.copy(), M.copy(), 0)
            times.append(time.perf_counter() - t0)
        t = np.array(times)

        rss_before = rss_mb()
        tracemalloc.start()
        fn(y.copy(), M.copy(), 0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = rss_mb()

        rows.append({
            "method": name, "label": LABEL[name],
            "batch": args.batch, "repeats": args.repeats,
            "patch_hw": f"{y.shape[1]}x{y.shape[2]}",
            "mean_s": float(t.mean()), "sd_s": float(t.std(ddof=1)),
            "median_s": float(np.median(t)),
            "min_s": float(t.min()), "max_s": float(t.max()),
            "ms_per_patch": float(1000 * t.mean() / args.batch),
            "peak_alloc_mb": peak / 2 ** 20,
            "rss_delta_mb": (None if rss_before is None else rss_after - rss_before),
        })
        r = rows[-1]
        print(f"[bench] {name:14s} {r['ms_per_patch']:9.2f} ms/patch  "
              f"peak {r['peak_alloc_mb']:7.1f} MB", flush=True)

    csv_path = out_dir / "runtime_memory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "environment.json").write_text(
        json.dumps({"environment": environment(),
                    "cell": str(args.cell), "device": str(device),
                    "solver": cfg_solver}, indent=2), encoding="utf-8")
    print(f"[bench] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
