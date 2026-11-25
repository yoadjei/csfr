"""Byte-level JPEG corruption with standard decoding (E9).

    python -I scripts/run_jpeg_corruption.py --config configs/jpeg_corruption.yaml

The GPU run of the CSFR solves (kaggle/jpeg_gpu_kernel.py) adds --csfr-only (no
baselines, no LaMa), --output-dir, --files (manifest indices in run order),
--shard k/n (every n-th of them from the k-th) and --device cuda:<k>.

The source files are downloaded data: run Python isolated (-I). For every file
of the manifest, every corruption kind and --repeats seeds:

  1. decode the clean file (luma) and the corrupted one with the tolerant decoder
     (src/jpeg_corruption.py);
  2. build the oracle, decoder and offset masks at full size, then centre-crop
     image and masks to at most max_side x max_side;
  3. per mask condition, y = decoded * mask: pixels the mask keeps carry what the
     decoder produced, garbage included, as a carver would see them; reconstruct
     with the CSFR variants and the classical baselines, and score against the
     clean decode over the pixels the oracle marks lost (hole PSNR, hole SSIM,
     hole GMSD), plus whole-image PSNR.

Each (case, condition) directory under <output_dir>/cases/ holds y.npy, mask.npy
(shape (1, H, W), the layout scripts/run_lama_cells.py reads), the outputs and a
metrics.json; LaMa outputs written there by the LaMa environment are scored on
the next run. Cases where the corruption left no visible damage (oracle mask
empty) are recorded and not reconstructed; a condition whose mask marks nothing
missing is recorded as nothing-to-fill (every method returns the decode).

Outputs: per_image.csv (one row per case, condition and method), masks_quality.csv
(mask error rates against the oracle per case and condition) and summary.csv (per
kind, condition and method: means with 95 percent bootstrap intervals over cases,
and the paired CSFR-minus-best-classical and CSFR-minus-LaMa differences with
Wilcoxon p, over the cases whose mask marks something missing and leaves
something observed; the others are counted, not compared). The CSFR solves
(SOLVE_THREADS torch threads each) and the baselines of every case and condition
of a file run concurrently in the worker pool, so the recorded seconds are not
runtime measurements (E11 and E15 are). A file whose pool loses a worker runs
again on a new pool (on_live_pool); its finished conditions are kept.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import run_chunk  # noqa: E402
from csfr_solver import solve_task  # noqa: E402
from jpeg_corruption import (corrupt, decode_tolerant, decoder_mask, locate_scan,  # noqa: E402
                             mask_errors, mcu_height, offset_mask, oracle_mask)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sweep = _load("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
_paired = _load("_paired", ROOT / "scripts" / "run_paired_stats.py")
KIND_INDEX = {"truncate": 0, "overwrite": 1, "flip": 2}
METRICS = ["psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole"]
POOL_RETRIES = 3     # new pools for one file before the run gives up
POOL_WAIT_S = 120    # before each new pool, for the memory pressure that killed a worker to pass
# torch threads per CSFR solve: the count the stage ran its first 48 files with, so later solves
# are the same bits (torch splits its CPU sums by thread count); 4 such solves at once in the
# pool gave ~10% more solves per hour than one, the solve being bound by memory bandwidth
SOLVE_THREADS = 8


def parse_files(spec: str) -> list[int]:
    """Manifest indices in run order: '48-104,0-47' -> [48, ..., 104, 0, ..., 47]."""
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def on_live_pool(fn, pool, make_pool, retries: int, wait: float, sleep=time.sleep):
    """fn(pool), run again on a new pool from make_pool() while a dead worker breaks it.

    A worker killed from outside the run (a process allocating when the system's
    commit charge is at its limit is one way) breaks a ProcessPoolExecutor for
    good, and every later submit fails. Other errors propagate at once: a bug
    does not go away on a new pool. Returns (fn's result, the pool now in use).
    """
    for attempt in range(1, retries + 2):
        try:
            return fn(pool), pool
        except BrokenProcessPool as e:
            if attempt > retries:
                raise
            print(f"[jpeg] a pool worker died ({e}); new pool in {wait:.0f} s, retry {attempt}/{retries}",
                  flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            sleep(wait)
            pool = make_pool()


def centre_crop(shape: tuple[int, int], max_side: int) -> tuple[slice, slice]:
    h, w = shape
    ch, cw = min(h, max_side), min(w, max_side)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return slice(r0, r0 + ch), slice(c0, c0 + cw)


def build_case(raw: bytes, kind: str, rng: np.random.Generator, max_side: int,
               margin_rows: int) -> dict:
    """Clean and corrupted decodes, the three masks, cropped; plus case metadata."""
    scan = locate_scan(raw)
    clean, _ = decode_tolerant(raw)
    bad, off = corrupt(raw, kind, rng, scan)
    dec, used = decode_tolerant(bad, clean.shape)
    if dec.shape != clean.shape:
        raise RuntimeError(f"decoded shape {dec.shape} differs from clean {clean.shape}")
    masks = {"oracle": oracle_mask(clean, dec), "decoder": decoder_mask(dec),
             "offset": offset_mask(clean.shape, scan[0], scan[1], off, mcu_height(raw), margin_rows)}
    rs, cs = centre_crop(clean.shape, max_side)
    return {"clean": clean[rs, cs], "decoded": dec[rs, cs], "masks": {k: m[rs, cs] for k, m in masks.items()},
            "meta": {"kind": kind, "corrupt_offset": int(off), "scan_start": int(scan[0]),
                     "scan_end": int(scan[1]), "scan_frac": (off - scan[0]) / max(scan[1] - scan[0], 1),
                     "decoder_used": used, "orig_shape": list(clean.shape),
                     "crop": [rs.start, rs.stop, cs.start, cs.stop], "mcu_h": mcu_height(raw)}}


def _record(job: dict, name: str, kind: str, x: np.ndarray, seconds: float, **extra) -> None:
    pp = _sweep.score(job["ref"], x, job["y"], job["Mo"])
    job["metrics"]["methods"][name] = {"kind": kind, "status": "done", "seconds": seconds,
                                       **{k: float(pp[k][0]) for k in METRICS}, "exact": bool(pp["exact"][0]),
                                       **extra}


def start_condition(d: Path, clean: np.ndarray, dec: np.ndarray, M: np.ndarray, M_oracle: np.ndarray,
                    cfg: dict, pool, seed: int, force: bool, solve_threads: int = SOLVE_THREADS,
                    device: str = "cpu") -> dict:
    """Write a (case, condition)'s inputs and submit its unfinished CSFR solves and
    baselines to the pool; finish_condition collects them. Returns the job."""
    d.mkdir(parents=True, exist_ok=True)
    y = (dec * M)[None].astype(np.float32)
    Mb = M[None].astype(np.float32)
    np.save(d / "y.npy", y)
    np.save(d / "mask.npy", Mb)
    mpath = d / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() and not force else {}
    metrics.setdefault("methods", {})
    inputs = {"y": hashlib.sha256(y.tobytes()).hexdigest(), "mask": hashlib.sha256(Mb.tobytes()).hexdigest()}
    if metrics.get("inputs_sha256", inputs) != inputs:
        raise RuntimeError(f"{d}: its records were solved on other inputs (inputs_sha256)")
    metrics["inputs_sha256"] = inputs
    nothing = bool((M == 1).all())
    metrics["nothing_to_fill"] = nothing
    job = {"d": d, "mpath": mpath, "metrics": metrics, "y": y, "ref": clean[None].astype(np.float32),
           "Mo": M_oracle[None].astype(np.float32), "cfg": cfg, "force": force, "csfr": {}, "classical": {}}
    if nothing:
        for name in cfg["csfr_variants"] + cfg["baselines"]:
            if name not in metrics["methods"]:
                _record(job, name, "csfr" if name.startswith("csfr") else "classical", y.astype(np.float64), 0.0,
                        note="nothing to fill: the mask keeps every pixel")
        return job
    for name in cfg["csfr_variants"]:
        if force or name not in metrics["methods"]:
            scfg = _sweep.solver_config(name, cfg.get("solver", {}))
            job["csfr"][name] = (scfg.tol, pool.submit(solve_task, y, Mb, scfg, solve_threads, device))
    for b in cfg["baselines"]:
        if force or b not in metrics["methods"]:
            job["classical"][b] = (time.time(), pool.submit(run_chunk, b, y, Mb, seed))
    return job


def finish_condition(job: dict) -> dict:
    """Collect a started condition's results (CSFR first, then the baselines, the
    order of metrics.json before the pool ran the solves), score the external
    outputs present and write metrics.json."""
    d, metrics = job["d"], job["metrics"]
    for name, (tol, fut) in job["csfr"].items():
        x, info = fut.result()
        _sweep.save_output(d / f"x_hat_{name}.npy", x, _sweep.CSFR_STORE_MAX_ERR)
        _record(job, name, "csfr", x, info["seconds"], iterations=info["iterations"],
                converged=info["converged"], gap=info["gap"], tol=tol)
    for b, (t0, fut) in job["classical"].items():
        x, info = fut.result()
        _sweep.save_output(d / f"x_hat_{b}.npy", x)
        _record(job, b, "classical", x.astype(np.float64), time.time() - t0)
    for name in job["cfg"].get("external", []):
        xp = d / f"x_hat_{name}.npy"
        if name in metrics["methods"] and metrics["methods"][name].get("status") == "done" and not job["force"]:
            continue
        if xp.exists():
            _record(job, name, "external", np.load(xp).astype(np.float64), float("nan"))
        else:
            metrics["methods"][name] = {"kind": "external", "status": "pending"}
    _sweep._write_json(job["mpath"], metrics)
    return metrics


def run_condition(d: Path, clean: np.ndarray, dec: np.ndarray, M: np.ndarray, M_oracle: np.ndarray,
                  cfg: dict, pool, seed: int, force: bool, solve_threads: int = SOLVE_THREADS,
                  device: str = "cpu") -> dict:
    """One (case, condition) start to finish."""
    return finish_condition(start_condition(d, clean, dec, M, M_oracle, cfg, pool, seed, force,
                                            solve_threads, device))


def run_file(fi: int, n_files: int, rec: dict, raw: bytes, cfg: dict, out: Path, pool, force: bool,
             solve_threads: int = SOLVE_THREADS, device: str = "cpu") -> tuple[list[dict], list[dict]]:
    """Every case and condition of one manifest file. All of them are started
    before any is collected, so the pool's workers share the file's CSFR solves
    and baselines. Returns the file's per_image and masks_quality rows."""
    started, mq = [], []
    for kind in cfg["kinds"]:
        for r in range(cfg["repeats"]):
            case = f"{Path(rec['file']).stem}_{kind}_r{r}"
            rng = np.random.default_rng([cfg["seed"], fi, KIND_INDEX[kind], r])
            c = build_case(raw, kind, rng, cfg["max_side"], cfg["offset_margin_rows"])
            cdir = out / "cases" / case
            cdir.mkdir(parents=True, exist_ok=True)
            np.save(cdir / "clean.npy", c["clean"].astype(np.uint8))
            meta = {"file": rec["file"], "corpus": rec["corpus"], "mode": rec["mode"], **c["meta"]}
            (cdir / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            Mo = c["masks"]["oracle"]
            visible = bool((Mo == 0).any())
            jobs = {}
            for cond in cfg["conditions"]:
                M = c["masks"][cond]
                mq.append({"case": case, "file": rec["file"], "corpus": rec["corpus"], "kind": kind,
                           "repeat": r, "condition": cond, "visible_damage": visible,
                           "scan_frac": meta["scan_frac"], "decoder_used": meta["decoder_used"],
                           **mask_errors(M, Mo)})
                if visible:
                    jobs[cond] = (start_condition(cdir / cond, c["clean"], c["decoded"], M, Mo, cfg, pool,
                                                  cfg["seed"], force, solve_threads, device),
                                  bool((M == 0).all()))
            started.append((case, kind, r, c["clean"].shape, visible, jobs))
    rows = []
    for case, kind, r, shape, visible, jobs in started:
        for cond, (job, nothing_observed) in jobs.items():
            m = finish_condition(job)
            rows += method_rows({"case": case, "file": rec["file"], "corpus": rec["corpus"],
                                 "kind": kind, "repeat": r, "condition": cond,
                                 "nothing_observed": nothing_observed, "H": shape[0], "W": shape[1]}, m)
        done = [x for x in rows if x["case"] == case and x["method"] == "csfr"]
        print(f"[jpeg] {fi + 1}/{n_files} {case}: visible {visible}; csfr hole PSNR " +
              " ".join(f"{x['condition']} {x['psnr_hole']:.2f}" for x in done if "psnr_hole" in x), flush=True)
    return rows, mq


def method_rows(case: dict, metrics: dict) -> list[dict]:
    """One per_image.csv row per method of a (case, condition): the case fields,
    then the method's scalar fields. A method's own "kind" (csfr, classical,
    external) is written as method_kind, so it cannot replace the corruption
    kind that summarise groups by."""
    return [{**case, "method": name, "nothing_to_fill": metrics["nothing_to_fill"],
             **{("method_kind" if k == "kind" else k): v for k, v in e.items()
                if not isinstance(v, (dict, list))}}
            for name, e in metrics["methods"].items()]


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict], classical: list[str], n_boot: int, seed: int) -> list[dict]:
    """Per kind, condition and method: means with bootstrap intervals over cases,
    and CSFR's paired differences to the best classical method and to LaMa.
    Cases whose mask marks nothing missing (every method returns the decode) or
    everything missing (no method has data; the scores are the methods' fallback
    constants) are counted in n_nothing_to_fill and n_nothing_observed and left
    out of the means and the tests."""
    out = []
    groups = sorted({(r["kind"], r["condition"]) for r in rows})
    for kind, cond in groups:
        g_all = [r for r in rows if r["kind"] == kind and r["condition"] == cond and r["status"] == "done"]
        n_fill = len({r["case"] for r in g_all if r["nothing_to_fill"]})
        n_obs = len({r["case"] for r in g_all if r.get("nothing_observed")})
        g = [r for r in g_all if not r["nothing_to_fill"] and not r.get("nothing_observed")]
        by_m: dict[str, dict] = {}
        for r in g:
            by_m.setdefault(r["method"], {})[r["case"]] = r
        means = {m: np.mean([v["psnr_hole"] for v in cases.values()]) for m, cases in by_m.items()}
        best = max((m for m in by_m if m in classical), key=lambda m: means[m], default=None)
        for m, cases in by_m.items():
            row = {"kind": kind, "condition": cond, "method": m, "n_cases": len(cases),
                   "n_nothing_to_fill": n_fill, "n_nothing_observed": n_obs}
            for k in METRICS:
                v = np.array([c[k] for c in cases.values()], dtype=np.float64)
                lo, hi = _paired.bootstrap_ci(v, n_boot, np.random.default_rng([seed, len(out), METRICS.index(k)]))
                row.update({f"{k}_mean": float(v.mean()), f"{k}_ci_lo": lo, f"{k}_ci_hi": hi})
            for other, label in ((best, "best_classical"), ("lama", "lama")):
                if m != "csfr" or other is None or other not in by_m:
                    continue
                common = sorted(set(cases) & set(by_m[other]))
                diff = np.array([cases[c]["psnr_hole"] - by_m[other][c]["psnr_hole"] for c in common])
                p = 1.0 if diff.size == 0 or np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
                row.update({f"vs_{label}": other, f"diff_psnr_hole_vs_{label}": float(diff.mean()) if diff.size else "",
                            f"p_wilcoxon_vs_{label}": p, f"n_paired_vs_{label}": int(diff.size)})
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N files only (smoke test)")
    ap.add_argument("--files", default=None, help="manifest indices in run order, e.g. 48-104,0-47")
    ap.add_argument("--shard", default=None, help="k/n: every n-th of the files from the k-th")
    ap.add_argument("--csfr-only", action="store_true", help="the CSFR variants only (the GPU run)")
    ap.add_argument("--output-dir", default=None, help="instead of the config's output_dir")
    ap.add_argument("--workers", type=int, default=4, help="pool processes for the CSFR solves and baselines")
    ap.add_argument("--solve-threads", type=int, default=SOLVE_THREADS,
                    help="torch threads per CSFR solve (another count can change the last bits)")
    ap.add_argument("--device", default="cpu", help="torch device of the CSFR solves")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.files is not None and args.limit is not None:
        ap.error("--files and --limit are exclusive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.csfr_only:
        cfg = {**cfg, "baselines": [], "external": []}
    out = ROOT / (args.output_dir or cfg["output_dir"])
    (out / "cases").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / cfg["manifest"]).read_text())["files"][: args.limit]
    order = parse_files(args.files) if args.files else list(range(len(manifest)))
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        order = order[k::n]
    src = ROOT / cfg["src_dir"]
    rows, mq = [], []
    pool = _sweep._make_pool(args.workers)
    try:
        for fi in order:
            rec = manifest[fi]
            raw = (src / rec["file"]).read_bytes()
            (f_rows, f_mq), pool = on_live_pool(
                lambda p: run_file(fi, len(manifest), rec, raw, cfg, out, p, args.force, args.solve_threads,
                                   args.device),
                pool, lambda: _sweep._make_pool(args.workers), POOL_RETRIES, POOL_WAIT_S)
            rows += f_rows
            mq += f_mq
            _write(out / "per_image.csv", rows)
            _write(out / "masks_quality.csv", mq)
    finally:
        pool.shutdown()
    _write(out / "per_image.csv", rows)
    _write(out / "masks_quality.csv", mq)
    _write(out / "summary.csv", summarise(rows, cfg["baselines"], args.bootstrap, cfg["seed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
