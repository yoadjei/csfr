"""Runtime and memory bench v2 (E15).

    python scripts/run_bench_v2.py          # on an otherwise idle machine

Protocol of scripts/run_bench.py (a 32-patch batch of a released cell, one
warm-up call, five timed repeats, in-process, CPU), for every registered
classical baseline and the CSFR variants on the convergent solver, on three
cells, since CSFR's cost depends on the hole geometry (CF1 L3, CF2 L5, CF3 L5).
Memory: the peak resident set size seen by a 50 ms sampling thread and the
tracemalloc peak (src/cost.py). For CSFR also the median iterations and
record_ms_per_query: the mean time of one sensitivity row (src/sensitivity.py)
over 16 imputed pixels, one per patch for the first 16 patches (the middle hole
pixel in raster order), with the CG iterations.

Outputs results/bench_v2/{runtime_memory.csv, environment.json}.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import BASELINES  # noqa: E402
from cost import measure  # noqa: E402
from csfr_solver import VARIANTS, solve  # noqa: E402
from sensitivity import RecordConfig, sensitivity_row  # noqa: E402

_spec = importlib.util.spec_from_file_location("_bench", ROOT / "scripts" / "run_bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

CELLS = ["CF1_L3", "CF2_L5", "CF3_L5"]
CSFR = ["csfr", "csfr_tv_only"]


def record_cost(x: torch.Tensor, y: np.ndarray, M: np.ndarray, n: int = 16) -> dict:
    cfg, rc = VARIANTS["csfr"], RecordConfig()
    times, cg = [], []
    for i in range(min(n, len(y))):
        hole = np.argwhere(M[i] == 0)
        if not len(hole):
            continue
        p = tuple(map(int, hole[len(hole) // 2]))
        t0 = time.perf_counter()
        _, info = sensitivity_row(x[i], torch.from_numpy(y[i].astype(np.float64)),
                                  torch.from_numpy(M[i] == 1), p, cfg, rc)
        times.append(time.perf_counter() - t0)
        cg.append(info["cg_iters"])
    return {"record_ms_per_query": 1000 * float(np.mean(times)), "record_queries": len(times),
            "record_cg_iters_median": float(np.median(cg))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", default="results/csfr_sweep_2d_v2")
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=CSFR + list(BASELINES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="results/bench_v2")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rel = Path(args.released) if Path(args.released).is_absolute() else ROOT / args.released
    rows = []
    for cell in args.cells:
        d = rel / cell / "seed0"
        y = np.load(d / "y.npy")[: args.batch].copy()
        M = np.load(d / "mask.npy")[: args.batch].copy()
        for name in args.methods:
            extra: dict = {}
            if name in VARIANTS:
                def fn(name=name):
                    return solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(M), VARIANTS[name])
                x, cert = fn()
                extra["iterations_median"] = float(cert.iterations.float().median())
                extra["n_converged"] = int(cert.converged.sum())
                if name == "csfr":
                    extra.update(record_cost(x, y, M))
            else:
                def fn(name=name):
                    return BASELINES[name](y.copy(), M.copy(), seed=0)
            r = measure(fn, repeats=args.repeats)
            rows.append({"cell": cell, "method": name, "batch": len(y),
                         "patch_hw": f"{y.shape[1]}x{y.shape[2]}", **r,
                         "ms_per_patch": 1000 * r["mean_s"] / len(y), **extra})
            print(f"[bench_v2] {cell} {name:14s} {rows[-1]['ms_per_patch']:9.2f} ms/patch  "
                  f"rss peak +{r['rss_peak_delta_mb']:.0f} MB" +
                  (f"  record {extra['record_ms_per_query']:.1f} ms/query" if "record_ms_per_query" in extra else ""),
                  flush=True)
            with (out / "runtime_memory.csv").open("w", newline="", encoding="utf-8") as f:
                fields = list(dict.fromkeys(k for row in rows for k in row))
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    env = _bench.environment()
    env["torch_threads"] = torch.get_num_threads()
    (out / "environment.json").write_text(json.dumps(
        {"environment": env, "protocol": {"batch": args.batch, "repeats": args.repeats, "warmup": 1,
                                          "cells": args.cells, "rss_sampling_interval_s": 0.05}},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
