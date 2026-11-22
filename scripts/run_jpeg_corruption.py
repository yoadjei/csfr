"""Byte-level JPEG corruption with standard decoding (E9).

    python -I scripts/run_jpeg_corruption.py --config configs/jpeg_corruption.yaml

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
Wilcoxon p). Baselines run concurrently in the worker pool and CSFR after them,
so the recorded seconds are not runtime measurements (E11 and E15 are).
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
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import run_chunk  # noqa: E402
from csfr_solver import solve  # noqa: E402
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


def run_condition(d: Path, clean: np.ndarray, dec: np.ndarray, M: np.ndarray, M_oracle: np.ndarray,
                  cfg: dict, pool, seed: int, force: bool) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    y = (dec * M)[None].astype(np.float32)
    Mb = M[None].astype(np.float32)
    np.save(d / "y.npy", y)
    np.save(d / "mask.npy", Mb)
    mpath = d / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() and not force else {}
    metrics.setdefault("methods", {})
    nothing = bool((M == 1).all())
    metrics["nothing_to_fill"] = nothing
    ref = clean[None].astype(np.float32)
    Mo = M_oracle[None].astype(np.float32)

    def record(name, kind, x, seconds, **extra):
        pp = _sweep.score(ref, x, y, Mo)
        metrics["methods"][name] = {"kind": kind, "status": "done", "seconds": seconds,
                                    **{k: float(pp[k][0]) for k in METRICS}, "exact": bool(pp["exact"][0]),
                                    **extra}

    todo_b = [b for b in cfg["baselines"] if force or b not in metrics["methods"]]
    if nothing:
        for name in cfg["csfr_variants"] + cfg["baselines"]:
            if name not in metrics["methods"]:
                record(name, "csfr" if name.startswith("csfr") else "classical", y.astype(np.float64), 0.0,
                       note="nothing to fill: the mask keeps every pixel")
    else:
        futures = {b: (time.time(), pool.submit(run_chunk, b, y, Mb, seed)) for b in todo_b}
        for name in cfg["csfr_variants"]:
            if name in metrics["methods"] and not force:
                continue
            scfg = _sweep.solver_config(name, cfg.get("solver", {}))
            t0 = time.time()
            x, cert = solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(Mb), scfg)
            dt = time.time() - t0
            _sweep.save_output(d / f"x_hat_{name}.npy", x.numpy(), _sweep.CSFR_STORE_MAX_ERR)
            record(name, "csfr", x.numpy(), dt, iterations=int(cert.iterations[0]),
                   converged=bool(cert.converged[0]), gap=float(cert.gap[0]), tol=scfg.tol)
        for b, (t0, fut) in futures.items():
            x, info = fut.result()
            _sweep.save_output(d / f"x_hat_{b}.npy", x)
            record(b, "classical", x.astype(np.float64), time.time() - t0)
    for name in cfg.get("external", []):
        xp = d / f"x_hat_{name}.npy"
        if name in metrics["methods"] and metrics["methods"][name].get("status") == "done" and not force:
            continue
        if xp.exists():
            record(name, "external", np.load(xp).astype(np.float64), float("nan"))
        else:
            metrics["methods"][name] = {"kind": "external", "status": "pending"}
    _sweep._write_json(mpath, metrics)
    return metrics


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
    out = []
    groups = sorted({(r["kind"], r["condition"]) for r in rows})
    for kind, cond in groups:
        g = [r for r in rows if r["kind"] == kind and r["condition"] == cond and r["status"] == "done"]
        by_m: dict[str, dict] = {}
        for r in g:
            by_m.setdefault(r["method"], {})[r["case"]] = r
        means = {m: np.mean([v["psnr_hole"] for v in cases.values()]) for m, cases in by_m.items()}
        best = max((m for m in by_m if m in classical), key=lambda m: means[m], default=None)
        for m, cases in by_m.items():
            row = {"kind": kind, "condition": cond, "method": m, "n_cases": len(cases)}
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(args.threads)
    out = ROOT / cfg["output_dir"]
    (out / "cases").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / cfg["manifest"]).read_text())["files"][: args.limit]
    src = ROOT / cfg["src_dir"]
    rows, mq = [], []
    pool = _sweep._make_pool(args.workers)
    try:
        for fi, rec in enumerate(manifest):
            raw = (src / rec["file"]).read_bytes()
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
                    for cond in cfg["conditions"]:
                        M = c["masks"][cond]
                        mq.append({"case": case, "file": rec["file"], "corpus": rec["corpus"], "kind": kind,
                                   "repeat": r, "condition": cond, "visible_damage": visible,
                                   "scan_frac": meta["scan_frac"], "decoder_used": meta["decoder_used"],
                                   **mask_errors(M, Mo)})
                        if not visible:
                            continue
                        m = run_condition(cdir / cond, c["clean"], c["decoded"], M, Mo, cfg, pool,
                                          cfg["seed"], args.force)
                        rows += method_rows({"case": case, "file": rec["file"], "corpus": rec["corpus"],
                                             "kind": kind, "repeat": r, "condition": cond,
                                             "H": c["clean"].shape[0], "W": c["clean"].shape[1]}, m)
                    done = [x for x in rows if x["case"] == case and x["method"] == "csfr"]
                    print(f"[jpeg] {fi + 1}/{len(manifest)} {case}: visible {visible}; csfr hole PSNR " +
                          " ".join(f"{x['condition']} {x['psnr_hole']:.2f}" for x in done if "psnr_hole" in x),
                          flush=True)
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
