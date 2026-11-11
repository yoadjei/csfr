"""Round-2 sweep harness: convergent CSFR, every baseline, retained outputs.

    python scripts/run_sweep_v2.py --config configs/sweep_v2_govdocs1.yaml [--workers 8]

For every (family, level, seed) cell, under <output_dir>/<family>_<level>/seed<s>/:

  y.npy, mask.npy, realised_frac.npy   the corrupted input (masks seeded exactly as
                                        the released harness: default_rng(seed),
                                        once per cell)
  x_hat_<method>.npy                    every method's reconstruction
  per_patch_<method>.npz                per-patch hole PSNR, hole SSIM, GMSD, HRP,
                                        whole-patch PSNR, exact flag, hole fraction;
                                        CSFR adds iterations, converged, gap
  certificate_<variant>.json            the CSFR convergence certificate
  metrics.json                          per-method status, summary and provenance

Methods come in three kinds. CSFR variants (csfr_solver.VARIANTS) run through the
convergent solver; the configured `csfr_variants` run at every seed and the
`ablation_variants` at `ablation_seeds`. Classical baselines (baselines.BASELINES)
run in a process pool over contiguous patch chunks; each treats patches
independently, so the split does not change the output. External methods (LaMa)
are not run here: scripts/run_lama_cells.py writes x_hat_<name>.npy from its own
environment, and the next run of this harness scores it; until then the method
is marked "pending".

The run is resumable per method: a method whose per-patch file exists is not
recomputed unless --force. summary.csv is rebuilt from the per-patch files of
every configured cell, so it always reflects what is on disk.

Storage. Baseline outputs are stored losslessly (uint8 when every value is an
integer in [0, 255], else float32). CSFR solutions are float64 in memory, the
metrics are computed from them, and they are stored as float32; the largest
rounding error is recorded in metrics.json (it is about 1e-5 grey levels, below
the solver's own accuracy at the calibrated tolerance).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines import BASELINES, limit_threads, run_chunk  # noqa: E402
from csfr_objective import Objective  # noqa: E402
from csfr_solver import STOPPING_RULE, VARIANTS, solve  # noqa: E402
from masks import make_mask  # noqa: E402
from quality import CAP, patch_metrics  # noqa: E402

SCHEMA = "csfr-sweep-v2"
EXTERNAL = {"lama"}
# numerical controls a config may set; the objective weights of every variant
# come from csfr_solver.VARIANTS, so a config cannot silently re-enable a term
# that an ablation arm switches off
SOLVER_KEYS = {"tol", "max_iter", "check_every", "init", "dtype"}
METRICS = ["psnr_hole", "ssim_hole", "gmsd_hole", "gmsd", "hrp", "psnr_whole", "mse_hole"]
CSFR_FIELDS = ["iterations", "converged", "gap", "rel_gap", "objective"]
CSFR_STORE_MAX_ERR = 1e-4  # grey levels; float32 storage of the float64 solution


# --------------------------------------------------------------------------- io

def save_output(path: Path, x: np.ndarray, max_err: float = 0.0) -> dict:
    """Store a reconstruction in the smallest dtype that loses at most max_err.

    uint8 when every value is an integer in [0, 255], float32 when the float32
    copy is within max_err, otherwise float64. Returns the dtype and the error.
    """
    x = np.asarray(x)
    if np.all(np.isfinite(x)) and x.min() >= 0 and x.max() <= 255 and np.all(x == np.round(x)):
        np.save(path, x.astype(np.uint8))
        return {"stored_dtype": "uint8", "storage_max_abs_error": 0.0}
    x32 = x.astype(np.float32)
    err = float(np.abs(x32.astype(np.float64) - x.astype(np.float64)).max()) if x.size else 0.0
    if err <= max_err:
        np.save(path, x32)
        return {"stored_dtype": "float32", "storage_max_abs_error": err}
    np.save(path, x.astype(np.float64))
    return {"stored_dtype": "float64", "storage_max_abs_error": 0.0}


def _replace(tmp: Path, path: Path, attempts: int = 40, delay: float = 0.25) -> None:
    """os.replace with retries: Windows refuses to replace a file that another
    process holds open, and virus scanners and indexers briefly open fresh files.
    Gives up (re-raising) after attempts * delay seconds."""
    for i in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    _replace(tmp, path)


def _load_npz(path: Path) -> dict:
    """Every array of an .npz, with the file closed afterwards (an open NpzFile
    would block a later rewrite on Windows)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------- methods

def run_baseline(name: str, y: np.ndarray, M: np.ndarray, seed: int, workers: int = 1,
                 pool: ProcessPoolExecutor | None = None) -> tuple[np.ndarray, dict]:
    """A registered baseline on the whole batch, split into contiguous chunks over
    worker processes. Returns (output, merged info).

    The baselines always run in workers initialised by baselines.limit_threads,
    including when workers == 1: OpenBLAS rounds differently depending on its
    thread configuration at start-up (one float32 ulp in one pixel of dictlearn
    was observed between an 8-thread parent and a 1-thread worker), so running
    every baseline in identically initialised workers makes the output
    bit-identical whatever --workers is."""
    workers = max(1, workers)
    n_chunks = min(len(y), workers * 4)
    parts = [p for p in np.array_split(np.arange(len(y)), n_chunks) if p.size]
    own = pool is None
    if own:
        pool = _make_pool(workers)
    try:
        futures = [pool.submit(run_chunk, name, y[p], M[p], seed) for p in parts]
        results = [f.result() for f in futures]
    finally:
        if own:
            pool.shutdown()
    out = np.concatenate([r[0] for r in results], axis=0)
    info: dict = {}
    for _, part in results:
        for k, v in part.items():
            if isinstance(v, list):
                info.setdefault(k, []).extend(v)
            else:
                info[k] = info.get(k, 0) + v
    return out, info


def _make_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                               initializer=limit_threads)


def score(ref: np.ndarray, rec: np.ndarray, y: np.ndarray, M: np.ndarray) -> dict:
    """Per-patch metrics as arrays (quality.patch_metrics on every patch)."""
    rows = [patch_metrics(ref[i], rec[i], y[i], M[i]) for i in range(len(rec))]
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in METRICS}
    out["exact"] = np.array([r["exact"] for r in rows], dtype=bool)
    out["hole_frac"] = np.array([r["hole_frac"] for r in rows], dtype=np.float64)
    return out


def _summary(pp: dict) -> dict:
    s = {"n": int(len(pp["psnr_hole"])), "n_exact": int(pp["exact"].sum())}
    for k in METRICS:
        s[f"{k}_mean"] = float(np.mean(pp[k]))
    s["psnr_hole_median"] = float(np.median(pp["psnr_hole"]))
    if "converged" in pp:
        s["n_converged"] = int(np.sum(pp["converged"]))
        s["iters_median"] = float(np.median(pp["iterations"]))
        s["gap_max"] = float(np.max(pp["gap"]))
    return s


def solver_config(name: str, overrides: dict):
    unknown = set(overrides) - SOLVER_KEYS
    if unknown:
        raise ValueError(f"solver keys not allowed in a config: {sorted(unknown)} "
                         f"(objective weights come from csfr_solver.VARIANTS)")
    return replace(VARIANTS[name], **overrides)


# ------------------------------------------------------------------------ cell

def _cell_input(patches: np.ndarray, fam: str, frac: float, seed: int,
                cell: Path) -> tuple[np.ndarray, np.ndarray]:
    M = make_mask(patches.shape, fam, frac, np.random.default_rng(seed))
    y = patches * M
    if (cell / "y.npy").exists():
        y0, M0 = np.load(cell / "y.npy"), np.load(cell / "mask.npy")
        if not (np.array_equal(y0, y) and np.array_equal(M0, M)):
            raise RuntimeError(f"{cell}: stored y/mask differ from the regenerated ones; "
                               "the outputs in this directory belong to other masks")
    else:
        cell.mkdir(parents=True, exist_ok=True)
        np.save(cell / "y.npy", y)
        np.save(cell / "mask.npy", M)
        np.save(cell / "realised_frac.npy", 1.0 - M.reshape(len(M), -1).mean(1))
    return y, M


def cell_methods(cfg: dict, seed: int) -> list[tuple[str, str]]:
    """(method, kind) for one seed, in run order."""
    out = [(m, "csfr") for m in cfg.get("csfr_variants", ["csfr"])]
    if seed in cfg.get("ablation_seeds", []):
        out += [(m, "csfr") for m in cfg.get("ablation_variants", [])]
    out += [(m, "classical") for m in cfg.get("baselines", [])]
    out += [(m, "external") for m in cfg.get("external", [])]
    return out


def run_cell(patches, fam, lvl, frac, seed, cell: Path, cfg: dict, *, workers: int,
             pool, only: set | None, force: bool) -> dict:
    y, M = _cell_input(patches, fam, frac, seed, cell)
    mpath = cell / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    metrics.update({"schema": SCHEMA, "family": fam, "level": lvl, "frac": frac,
                    "seed": seed, "n": int(len(y)),
                    "y_sha256": _sha256(y), "mask_sha256": _sha256(M)})
    metrics.setdefault("methods", {})
    overrides = cfg.get("solver", {})
    for name, kind in cell_methods(cfg, seed):
        if only and name not in only:
            continue
        pp_path = cell / f"per_patch_{name}.npz"
        recorded = metrics["methods"].get(name, {}).get("status") == "done"
        if pp_path.exists() and recorded and not force:
            continue
        entry: dict = {"kind": kind}
        t0 = time.time()
        if kind == "csfr":
            scfg = solver_config(name, overrides)
            x_t, cert = solve(torch.from_numpy(y), torch.from_numpy(M), scfg)
            x = x_t.numpy()
            entry["runtime_s"] = time.time() - t0
            pp = score(patches, x, y, M)
            for k in CSFR_FIELDS:
                pp[k] = getattr(cert, k).numpy()
            entry.update(save_output(cell / f"x_hat_{name}.npy", x, CSFR_STORE_MAX_ERR))
            _write_json(cell / f"certificate_{name}.json", cert.to_json())
            entry["solver"] = asdict(scfg)
        elif kind == "classical":
            x, info = run_baseline(name, y, M, seed, workers=workers, pool=pool)
            entry["runtime_s"] = time.time() - t0
            entry["workers"] = workers
            pp = score(patches, x, y, M)
            if "fallback" in info:
                pp["fallback"] = np.asarray(info["fallback"], dtype=bool)
                entry["fallback_patches"] = int(info["fallback_patches"])
            entry.update(save_output(cell / f"x_hat_{name}.npy", x))
        else:
            xp = cell / f"x_hat_{name}.npy"
            if not xp.exists():
                metrics["methods"][name] = {"kind": kind, "status": "pending"}
                continue
            x = np.load(xp).astype(np.float64)
            entry["observed_max_abs_dev"] = float(np.abs(x - y)[M == 1].max()) if (M == 1).any() else 0.0
            meta = cell / f"{name}_meta.json"
            if meta.exists():
                entry["meta"] = json.loads(meta.read_text())
            pp = score(patches, x, y, M)
        np.savez(pp_path, **pp)
        entry["status"] = "done"
        entry["summary"] = _summary(pp)
        metrics["methods"][name] = entry
        _write_json(mpath, metrics)
        s = entry["summary"]
        extra = (f"  conv {s['n_converged']}/{s['n']}  it_med {s['iters_median']:.0f}"
                 if kind == "csfr" else "")
        print(f"    {name:>14s}  hole PSNR {s['psnr_hole_mean']:6.2f}  SSIM {s['ssim_hole_mean']:.3f}"
              f"  GMSD {s['gmsd_hole_mean']:.3f}  {entry.get('runtime_s', 0):6.1f}s{extra}", flush=True)
        if kind == "csfr" and s["n_converged"] < s["n"]:
            print(f"    WARNING {name}: {s['n'] - s['n_converged']} patches reached max_iter", flush=True)
    _write_json(mpath, metrics)
    return metrics


# --------------------------------------------------------------------- summary

def write_summary(cfg: dict, outdir: Path) -> Path:
    """summary.csv from the per-patch files on disk: per (family, level, method)
    the mean over seeds of the per-seed patch means, the between-seed SD
    (ddof 1, nan for one seed), the mean within-seed patch SD of hole PSNR, and
    counts summed over seeds."""
    rows = []
    for fam in cfg["families"]:
        for lvl, frac in cfg["levels"].items():
            methods: dict[str, list] = {}
            for seed in cfg["seeds"]:
                cell = outdir / f"{fam}_{lvl}" / f"seed{seed}"
                for name, kind in cell_methods(cfg, seed):
                    f = cell / f"per_patch_{name}.npz"
                    if f.exists():
                        methods.setdefault(name, []).append((kind, _load_npz(f), cell))
            for name, runs in methods.items():
                kind = runs[0][0]
                row = {"corpus": cfg.get("corpus", ""), "family": fam, "level": lvl, "frac": frac,
                       "method": name, "kind": kind, "n_seeds": len(runs),
                       "n_patches": int(sum(len(r[1]["psnr_hole"]) for r in runs))}
                for k in METRICS:
                    means = np.array([np.mean(r[1][k]) for r in runs])
                    row[f"{k}_mean"] = float(means.mean())
                    row[f"{k}_seedsd"] = float(means.std(ddof=1)) if len(means) > 1 else float("nan")
                row["psnr_hole_sd"] = float(np.mean([np.std(r[1]["psnr_hole"]) for r in runs]))
                row["psnr_hole_median"] = float(np.median(np.concatenate([r[1]["psnr_hole"] for r in runs])))
                row["n_exact"] = int(sum(r[1]["exact"].sum() for r in runs))
                # at the hole-PSNR cap: every exact patch, and patches an iterative
                # method reconstructs to within the cap without being bit-exact
                row["n_at_cap"] = int(sum((r[1]["psnr_hole"] >= CAP).sum() for r in runs))
                csfr = all("converged" in r[1] for r in runs)
                row["n_converged"] = int(sum(r[1]["converged"].sum() for r in runs)) if csfr else ""
                row["iters_median"] = (float(np.median(np.concatenate([r[1]["iterations"] for r in runs])))
                                       if csfr else "")
                rts = []
                for _, _, cell in runs:
                    m = json.loads((cell / "metrics.json").read_text())["methods"].get(name, {})
                    if "runtime_s" in m:
                        rts.append(m["runtime_s"])
                row["runtime_s"] = float(np.mean(rts)) if rts else ""
                rows.append(row)
    path = outdir / "summary.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return path


def _git() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except OSError:
        return {"git_commit": "unknown", "git_dirty": None}


def _invocations(path: Path, this: dict) -> list[dict]:
    """Every invocation that has written into this results directory, oldest
    first, with this one replacing its own earlier entry (each run writes the
    metadata at its start and its end). A file written before the history was
    kept contributes one entry with what it recorded."""
    if not path.exists():
        return [this]
    old = json.loads(path.read_text(encoding="utf-8"))
    previous = old.get("invocations")
    if previous is None:
        previous = [{"git_commit": old.get("git_commit"), "git_dirty": old.get("git_dirty"),
                     "written": old.get("written"), "recorded_before_invocation_history": True}]
    return [e for e in previous if e.get("id") != this["id"]] + [this]


def write_metadata(cfg: dict, outdir: Path, patches: np.ndarray, clusters, args,
                   invocation: dict) -> None:
    """Environment, corpus hashes and objectives of the sweep. `invocation`
    identifies this run (id, start time, arguments, and finish time once done);
    the top-level git fields are those of the latest invocation."""
    from run_csfr_sweep import capture_environment
    env = capture_environment()
    import cv2
    env["opencv_contrib_xphoto"] = hasattr(cv2, "xphoto")
    h, w = patches.shape[-2:]
    git = _git()
    path = outdir / "run_metadata.json"
    meta = {"schema": SCHEMA, "config": cfg, "environment": env, **git,
            "invocations": _invocations(path, {**invocation, **git}),
            "patches_sha256": _sha256(patches),
            "cluster_sha256": _sha256(clusters) if clusters is not None else None,
            "n_patches": int(len(patches)),
            "objective": {name: Objective(h, w, v.lam_l1, v.lam_tv, v.lam_fc, v.eps,
                                          torch.device("cpu"), torch.float64).as_dict()
                          for name, v in VARIANTS.items()},
            "solver_overrides": cfg.get("solver", {}),
            "stopping_rule": STOPPING_RULE,
            "workers": args.workers, "torch_threads": args.threads,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_json(path, meta)


# ------------------------------------------------------------------------ main

def _validate(cfg: dict) -> None:
    for m in cfg.get("csfr_variants", []) + cfg.get("ablation_variants", []):
        if m not in VARIANTS:
            raise ValueError(f"unknown CSFR variant {m}")
    for m in cfg.get("baselines", []):
        if m not in BASELINES:
            raise ValueError(f"unknown baseline {m}")
    for m in cfg.get("external", []):
        if m not in EXTERNAL:
            raise ValueError(f"unknown external method {m}")
    solver_config("csfr", cfg.get("solver", {}))
    if {"fsr", "shiftmap"} & set(cfg.get("baselines", [])):
        from baselines import xphoto_available
        if not xphoto_available():
            raise RuntimeError("fsr/shiftmap need opencv-contrib (cv2.xphoto); see requirements.txt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes for the classical baselines")
    ap.add_argument("--threads", type=int, default=8, help="torch threads for the CSFR solver")
    ap.add_argument("--only-family", nargs="+")
    ap.add_argument("--only-level", nargs="+")
    ap.add_argument("--only-seed", nargs="+", type=int)
    ap.add_argument("--only-method", nargs="+")
    ap.add_argument("--force", action="store_true", help="recompute the selected methods")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    _validate(cfg)
    torch.set_num_threads(args.threads)
    outdir = ROOT / cfg["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    patches = np.load(ROOT / cfg["patches_path"]).astype(np.float32)
    clusters = np.load(ROOT / cfg["cluster_path"]) if cfg.get("cluster_path") else None
    cells = [(f, l) for f in cfg["families"] for l in cfg["levels"]
             if (not args.only_family or f in args.only_family)
             and (not args.only_level or l in args.only_level)]
    seeds = [s for s in cfg["seeds"] if not args.only_seed or s in args.only_seed]
    only = set(args.only_method) if args.only_method else None
    print(f"[sweep_v2] {cfg['experiment']}: {len(patches)} patches, {len(cells)} cells x "
          f"{len(seeds)} seeds, workers {args.workers}, threads {args.threads}", flush=True)
    invocation = {"id": uuid.uuid4().hex, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "finished": None, "args": vars(args)}
    write_metadata(cfg, outdir, patches, clusters, args, invocation)
    pool = _make_pool(max(1, args.workers))
    try:
        for i, (fam, lvl) in enumerate(cells):
            for seed in seeds:
                print(f"[sweep_v2] {i + 1}/{len(cells)} {fam}_{lvl} seed {seed}", flush=True)
                run_cell(patches, fam, lvl, cfg["levels"][lvl], seed,
                         outdir / f"{fam}_{lvl}" / f"seed{seed}", cfg,
                         workers=args.workers, pool=pool, only=only, force=args.force)
                write_summary(cfg, outdir)
    finally:
        pool.shutdown()
    write_summary(cfg, outdir)
    invocation["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_metadata(cfg, outdir, patches, clusters, args, invocation)
    print(f"[sweep_v2] wrote {outdir / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
