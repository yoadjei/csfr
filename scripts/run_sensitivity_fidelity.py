"""Sensitivity-record fidelity campaign (E6) and record comparison (E7).

    python scripts/run_sensitivity_fidelity.py [--sweep-dir results/sweep_v2_govdocs1]
    python scripts/run_sensitivity_fidelity.py --tol-study
    python scripts/run_sensitivity_fidelity.py --records-compare

Inputs are the y.npy and mask.npy of each cell of a sweep (<cell>/seed<seed>/).
The first --patches patches are re-solved with the operational solver
configuration (the defaults of src/csfr_solver.py unless --tols is given); when
the sweep's x_hat_csfr.npy exists and the tolerance is the sweep's, the re-solved
patches are checked against it.

Fidelity (fidelity.csv). Per patch, three observed query pixels: the first in
raster order at Chebyshev distance 1, 3 and >= 8 from the hole (a distance that
does not occur is skipped). For each query and each delta the predicted column
d x_F / d y_k is compared with the perturb-and-re-solve difference quotient on
the free set, against a float64 reference re-solve (rel_l2_error_ref) and an
operational re-solve (rel_l2_error_op); sensitivity.fidelity_patch batches every
query and delta of a patch into one solve.

summary.csv: per (family, level, delta, tol), the median and 90th percentile of
the relative error, the Sensitivity Record Fidelity SRF = fraction of queries with
relative error <= 0.10, the mean top-k sign agreement, record and reconstruction
seconds, and the fraction of re-solves that converged; for the reference and the
operational comparison. summary_by_distance.csv: the same per (family, query
distance, delta, tol).

Manifests (manifests/<cell>_patch0.json): the full audit manifest of patch 0 at
the operational tolerance, with three imputed-pixel rows (the first hole pixel in
raster order at Chebyshev distance 1 and 3 from the observed region, and the
deepest hole pixel) and the fidelity rows of that patch attached.

Record comparison (--records-compare, records_compare.csv). For the same query
pixels: the CSFR column; the exact column of the bilinear baseline (Delaunay
barycentric weights); and finite-difference columns (delta 1 and 4) of bilinear,
Telea and Navier-Stokes. Per record: L1 and L2 norm, maximum, support size,
|column|-weighted mean Chebyshev distance from the query, and for finite
differences the relative difference between delta 1 and delta 4 (zero for a
linear method; large for the 8-bit PDE methods, whose output is piecewise
constant in the input).

Every unit of work (a cell at one tolerance; a cell, for the record comparison)
is saved when it finishes, under <out>/units/{fidelity,tol_study,records}/, and
reused by a later run with the same inputs, settings and code. The CSVs are
assembled from the saved units, so an interrupted run resumes where it stopped
and writes what an uninterrupted run would.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baseline_jacobians import bilinear_weights, finite_difference_column  # noqa: E402
from baselines import BASELINES  # noqa: E402
from checkpoint import UnitStore, digests  # noqa: E402
from csfr_solver import SolverConfig, solve  # noqa: E402
from sensitivity import (RecordConfig, fidelity_patch,  # noqa: E402
                         sensitivity_column, write_manifest)

ALL_CELLS = [f"CF{f}_L{l}" for f in range(1, 5) for l in range(1, 6)]
QUERY_DISTANCES = (1, 3, 8)       # the last means ">= 8"
SRF_THRESHOLD = 0.10
TOL_STUDY = {"cells": ["CF1_L3", "CF3_L5"], "tols": [1e-4, 1e-6, 1e-8, 1e-10]}
# the code that computes a unit: a change to any of it invalidates the saved units
FIDELITY_CODE = [ROOT / "src" / "csfr_solver.py", ROOT / "src" / "sensitivity.py", Path(__file__).resolve()]
RECORDS_CODE = FIDELITY_CODE + [ROOT / "src" / "baseline_jacobians.py", ROOT / "src" / "baselines.py"]


# ---------------------------------------------------------------- query pixels

def _first_at(dist: np.ndarray, sel: np.ndarray, d: int, at_least: bool) -> tuple[int, int] | None:
    hit = sel & ((dist >= d) if at_least else (dist == d))
    idx = np.argwhere(hit)
    return (int(idx[0][0]), int(idx[0][1])) if len(idx) else None


def observed_queries(Mb: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    """(distance, pixel) for the observed pixels at Chebyshev distance 1, 3 and
    >= 8 from the hole, first in raster order; missing distances skipped."""
    if Mb.all() or not Mb.any():
        return []
    dist = distance_transform_cdt(Mb, metric="chessboard")
    out = []
    for d in QUERY_DISTANCES:
        p = _first_at(dist, Mb, d, at_least=(d == QUERY_DISTANCES[-1]))
        if p is not None:
            out.append((d, p))
    return out


def imputed_queries(Mb: np.ndarray) -> list[tuple[int, int]]:
    """Hole pixels at Chebyshev distance 1 and 3 from the observed region, first in
    raster order, and the deepest hole pixel (first in raster order)."""
    hole = ~Mb
    if not hole.any() or not Mb.any():
        return []
    dist = distance_transform_cdt(hole, metric="chessboard")
    out = []
    for d in (1, 3):
        p = _first_at(dist, hole, d, at_least=False)
        if p is not None and p not in out:
            out.append(p)
    p = _first_at(dist, hole, int(dist[hole].max()), at_least=False)
    if p not in out:
        out.append(p)
    return out


# ------------------------------------------------------------------- helpers

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _cell_inputs(sweep: Path, cell: str, seed: int, n: int):
    d = sweep / cell / f"seed{seed}"
    y = np.load(d / "y.npy")[:n].astype(np.float64)
    M = np.load(d / "mask.npy")[:n]
    xs = d / "x_hat_csfr.npy"
    return y, M, (np.load(xs)[:n].astype(np.float64) if xs.exists() else None), d


def _store(units: Path, args, cell: str, code: list[Path]) -> UnitStore:
    """A unit is reused only for the same inputs, settings and code."""
    d = Path(args.sweep_dir) / cell / f"seed{args.seed}"
    return UnitStore(units, {"sweep": str(args.sweep_dir), "seed": args.seed, "patches": args.patches,
                             "deltas": list(args.deltas), "inputs": digests([d / "y.npy", d / "mask.npy"]),
                             "code": digests(code)})


def _sweep_tol(cell_dir: Path) -> float | None:
    m = cell_dir / "metrics.json"
    if not m.exists():
        return None
    entry = json.loads(m.read_text()).get("methods", {}).get("csfr", {})
    return entry.get("solver", {}).get("tol")


def _q(vals, p):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(vals, p)) if vals.size else float("nan")


def summarise(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        s = dict(zip(keys, g))
        s["n"] = len(rs)
        for arm in ("ref", "op"):
            e = [r[f"rel_l2_error_{arm}"] for r in rs if f"rel_l2_error_{arm}" in r]
            if not e:
                continue
            sa = "sign_agreement_topk" if arm == "ref" else "sign_agreement_topk_op"
            s[f"rel_err_median_{arm}"] = _q(e, 50)
            s[f"rel_err_p90_{arm}"] = _q(e, 90)
            s[f"srf_{arm}"] = float(np.mean([v <= SRF_THRESHOLD for v in e]))
            s[f"sign_agreement_mean_{arm}"] = float(np.nanmean([r[sa] for r in rs]))
            s[f"converged_frac_{arm}"] = float(np.mean([r[f"{arm}_converged"] for r in rs]))
        s["record_seconds_mean"] = float(np.mean([r["record_seconds"] for r in rs]))
        s["reconstruction_seconds_mean"] = float(np.mean([r["reconstruction_seconds"] for r in rs]))
        out.append(s)
    return out


# ------------------------------------------------------------------ fidelity

def fidelity_unit(args, cell: str, tol: float, inputs: tuple, manifest: Path | None,
                  rc: RecordConfig) -> list[dict]:
    """Every fidelity row of one cell at one solver tolerance; writes the cell's
    patch-0 manifest when manifest is given."""
    fam, lvl = cell.split("_")
    y, M, x_sweep, cell_dir = inputs
    rows: list[dict] = []
    cfg = replace(SolverConfig(), tol=tol)
    t0 = time.time()
    X, cert = solve(torch.from_numpy(y), torch.from_numpy(M), cfg)
    recon_s = (time.time() - t0) / len(y)
    if x_sweep is not None and _sweep_tol(cell_dir) == tol:
        dev = float(np.abs(X.numpy() - x_sweep).max())
        if dev > 1e-3:
            raise RuntimeError(f"{cell}: re-solved patches differ from the sweep by {dev}")
    for i in range(len(y)):
        Mb = torch.from_numpy(M[i] == 1)
        yi = torch.from_numpy(y[i])
        qs = observed_queries(M[i] == 1)
        if not qs:
            continue
        res = fidelity_patch(X[i], yi, Mb, [p for _, p in qs], cfg, rc,
                             deltas=tuple(args.deltas), operational=True)
        for r in res:
            r.update({"cell": cell, "family": fam, "level": lvl, "seed": args.seed,
                      "patch": i, "query_distance": qs[r["query"]][0], "tol": tol,
                      "reconstruction_seconds": recon_s,
                      "patch_converged": bool(cert.converged[i]),
                      "patch_iterations": int(cert.iterations[i])})
            r["query_pixel"] = f"{r['query_pixel'][0]},{r['query_pixel'][1]}"
        rows.extend(res)
        if manifest is not None and i == 0:
            one = replace(cert, iterations=cert.iterations[:1], converged=cert.converged[:1],
                          gap=cert.gap[:1], rel_gap=cert.rel_gap[:1],
                          final_step_inf=cert.final_step_inf[:1], objective=cert.objective[:1])
            write_manifest(manifest, x_hat=X[i], y=yi, Mb=Mb,
                           solver_cfg=cfg, certificate=one,
                           queries=imputed_queries(M[i] == 1),
                           patch_id=f"{args.sweep_dir}:{cell}:seed{args.seed}:patch0",
                           rc=rc, fidelity=res)
    return rows


def run_fidelity(args, cells: list[str], tols: list[float], out: Path, manifests: bool,
                 units: Path) -> list[dict]:
    rc = RecordConfig()
    rows: list[dict] = []
    for cell in cells:
        inputs = _cell_inputs(Path(args.sweep_dir), cell, args.seed, args.patches)
        store = _store(units, args, cell, FIDELITY_CODE)
        manifest = out / "manifests" / f"{cell}_patch0.json" if manifests else None
        for tol in tols:
            done = store.rows(f"{cell}_tol{tol:g}",
                              lambda tol=tol: fidelity_unit(args, cell, tol, inputs, manifest, rc),
                              outputs=[manifest] if manifest else [])
            rows += done
            e = [r["rel_l2_error_ref"] for r in done if r["delta"] == 1.0]
            print(f"[fidelity] {cell} tol {tol:g}: {len(done)} rows, median rel err (delta 1) "
                  f"{_q(e, 50):.3g}, SRF {np.mean([v <= SRF_THRESHOLD for v in e]):.2f}", flush=True)
    return rows


# ---------------------------------------------------------- record comparison

def _record_stats(col: np.ndarray, k: tuple[int, int], hole: np.ndarray) -> dict:
    a = np.abs(np.where(hole, col, 0.0))
    peak = float(a.max())
    support = int((a > 1e-6 * peak).sum()) if peak > 0 else 0
    rr, cc = np.nonzero(hole)
    d = np.maximum(np.abs(rr - k[0]), np.abs(cc - k[1]))
    w = a[rr, cc]
    return {"l1_norm": float(a.sum()), "l2_norm": float(np.sqrt((a ** 2).sum())), "max_abs": peak,
            "support": support,
            "mean_distance": float((w * d).sum() / w.sum()) if w.sum() > 0 else float("nan")}


def records_unit(args, cell: str) -> list[dict]:
    """Every record-comparison row of one cell."""
    rc = RecordConfig()
    cfg = SolverConfig()
    methods = {m: BASELINES[m] for m in ("bilinear", "inpaint_telea", "inpaint_ns")}
    rows: list[dict] = []
    y, M, _, _ = _cell_inputs(Path(args.sweep_dir), cell, args.seed, args.patches)
    X, _ = solve(torch.from_numpy(y), torch.from_numpy(M), cfg)
    for i in range(len(y)):
        Mb = M[i] == 1
        hole = ~Mb
        W = bilinear_weights(M[i])
        y1, M1 = y[i:i + 1].astype(np.float32), M[i:i + 1]
        for dist, k in observed_queries(Mb):
            base = {"cell": cell, "patch": i, "query_pixel": f"{k[0]},{k[1]}",
                    "query_distance": dist}
            col, _ = sensitivity_column(X[i], torch.from_numpy(y[i]), torch.from_numpy(Mb), k, cfg, rc)
            rows.append({**base, "method": "csfr", "record": "implicit_function",
                         **_record_stats(col.numpy(), k, hole)})
            exact = np.zeros(Mb.shape)
            for p, entries in W.items():
                for q, wt in entries:
                    if q == k:
                        exact[p] = wt
            rows.append({**base, "method": "bilinear", "record": "exact_barycentric",
                         **_record_stats(exact, k, hole)})
            for name, fn in methods.items():
                fd = {dl: finite_difference_column(lambda a, b, fn=fn: fn(a, b), y1, M1, k, dl)
                      for dl in (1.0, 4.0)}
                n4 = float(np.linalg.norm(fd[4.0]))
                consistency = (float(np.linalg.norm(fd[1.0] - fd[4.0])) / n4) if n4 > 0 else float("nan")
                for dl, c in fd.items():
                    rows.append({**base, "method": name, "record": f"finite_difference_delta{dl:g}",
                                 **_record_stats(c, k, hole), "fd_rel_diff_delta1_vs_4": consistency})
                if name == "bilinear":
                    rows[-1]["fd_vs_exact_rel_l2"] = (float(np.linalg.norm(fd[1.0] - exact))
                                                      / max(float(np.linalg.norm(exact)), 1e-300))
    return rows


def run_records_compare(args, cells: list[str], units: Path) -> list[dict]:
    rows: list[dict] = []
    for cell in cells:
        rows += _store(units, args, cell, RECORDS_CODE).rows(cell, lambda cell=cell: records_unit(args, cell))
        print(f"[records] {cell}: {len(rows)} rows so far", flush=True)
    return rows


# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="results/sweep_v2_govdocs1")
    ap.add_argument("--cells", nargs="+", default=ALL_CELLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patches", type=int, default=10)
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.25, 1.0, 4.0])
    ap.add_argument("--tols", nargs="+", type=float, default=[SolverConfig().tol])
    ap.add_argument("--tol-study", action="store_true",
                    help=f"tolerances {TOL_STUDY['tols']} on {TOL_STUDY['cells']}")
    ap.add_argument("--records-compare", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/sensitivity_fidelity")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.records_compare:
        rows = run_records_compare(args, args.cells, out / "units" / "records")
        _write_csv(out / "records_compare.csv", rows)
        return 0
    if args.tol_study:
        rows = run_fidelity(args, TOL_STUDY["cells"], TOL_STUDY["tols"], out, manifests=False,
                            units=out / "units" / "tol_study")
        _write_csv(out / "tol_study.csv", rows)
        _write_csv(out / "tol_study_summary.csv",
                   summarise(rows, ["family", "level", "delta", "tol"]))
        return 0
    rows = run_fidelity(args, args.cells, args.tols, out, manifests=True, units=out / "units" / "fidelity")
    _write_csv(out / "fidelity.csv", rows)
    _write_csv(out / "summary.csv", summarise(rows, ["family", "level", "delta", "tol"]))
    _write_csv(out / "summary_by_distance.csv",
               summarise(rows, ["family", "query_distance", "delta", "tol"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
