"""Full-resolution colour reconstruction, cost and LaMa in its own regime (E11).

    python -I scripts/run_fullres.py
    .venv-lama/Scripts/python scripts/run_lama_cells.py --root results/fullres/cases
    python -I scripts/run_fullres.py              # scores the LaMa outputs

(-I because three of the images are downloaded GovDocs1 JPEGs.) Images: the 24
Kodak images at native size, the six Nick Mikus carved JPEGs (decoded PNGs in
data/full_mikus_*), and the first three files of the JPEG-corruption set with
both sides at least 512, centre-cropped to 512 x 768. Corruptions CF1, CF2 and
CF3 at L3 (30 percent) at full resolution with seed 0; one mask per image,
shared by the three channels. Methods run per channel (the three channels as a
batch): CSFR (FFT DCT above side 128), bilinear, Telea, Navier-Stokes,
biharmonic, Guleryuz, FSR and shift-map. Dictionary learning and Criminisi are
not run at full resolution (cost; see tasks/todo.md E11). LaMa runs on the RGB
image at native resolution, which is its training regime.

Metrics: per channel, hole PSNR, hole SSIM and GMSD, averaged over the channels.
Cost per image and method: wall-clock of one call and the peak resident set
size from a 50 ms sampling thread (src/cost.py); the tracemalloc peak is taken
on the first image of each corruption only, in a separate call, so that tracing
never inflates a timing. For CSFR also iterations, convergence, and the time
and CG iterations of one sensitivity row (the middle hole pixel of channel 0).

Outputs (results/fullres): per_image.csv, summary.csv (per corruption and
method: means over images), cost.csv.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import BASELINES  # noqa: E402
from cost import PeakRSS  # noqa: E402
from csfr_solver import VARIANTS, solve  # noqa: E402
from masks import SEVERITY, make_mask  # noqa: E402
from sensitivity import RecordConfig, sensitivity_row  # noqa: E402

_spec = importlib.util.spec_from_file_location("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)

CORRUPTIONS = ["CF1_L3", "CF2_L3", "CF3_L3"]
METHODS = ["csfr", "bilinear", "inpaint_telea", "inpaint_ns", "biharmonic", "guleryuz", "fsr", "shiftmap"]
EXTERNAL = ["lama"]
GOVDOCS_CROP = (512, 768)


def image_list() -> list[tuple[str, str, Path]]:
    """(name, corpus, path) in a fixed order."""
    out = [(p.stem, "kodak", p) for p in sorted((ROOT / "data/kodak").glob("kodim*.png"))]
    for d in sorted((ROOT / "data").glob("full_mikus_*")):
        out += [(f"{d.name}_{p.stem}", "nick_mikus_dftt", p) for p in sorted(d.glob("img*.png"))]
    manifest = json.loads((ROOT / "data/jpeg_corruption/manifest.json").read_text())["files"]
    big = [f for f in manifest if f["corpus"] == "govdocs1" and min(f["width"], f["height"]) >= 512][:3]
    out += [(Path(f["file"]).stem, "govdocs1", ROOT / "data/jpeg_corruption/src" / f["file"]) for f in big]
    return out


def load_rgb(path: Path, corpus: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32)
    if corpus == "govdocs1":
        h, w = GOVDOCS_CROP
        r0, c0 = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
        a = a[r0:r0 + h, c0:c0 + w]
    return a


def run_method(name: str, y3: np.ndarray, M3: np.ndarray, trace_memory: bool,
               max_iter: int) -> tuple[np.ndarray, dict]:
    def call():
        if name in VARIANTS:
            x, cert = solve(torch.from_numpy(y3.astype(np.float64)), torch.from_numpy(M3),
                            replace(VARIANTS[name], max_iter=max_iter))
            return x.numpy(), {"iterations_max": int(cert.iterations.max()),
                               "n_converged": int(cert.converged.sum())}
        return BASELINES[name](y3, M3, seed=0).astype(np.float64), {}

    with PeakRSS() as p:
        t0 = time.perf_counter()
        x, extra = call()
        dt = time.perf_counter() - t0
    cost = {"seconds": dt, "rss_start_mb": p.start_mb, "rss_peak_mb": p.peak_mb,
            "rss_peak_delta_mb": p.peak_mb - p.start_mb, **extra}
    if trace_memory:
        tracemalloc.start()
        call()
        cost["tracemalloc_peak_mb"] = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
    return x, cost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="+", help="image names (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=100000,
                    help="solver cap; deep full-resolution holes need more iterations than patches")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/fullres")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    (out / "cases").mkdir(parents=True, exist_ok=True)
    per, costs = [], []
    images = [i for i in image_list() if not args.only or i[0] in args.only][: args.limit]
    for corr in args.corruptions:
        fam, lvl = corr.split("_")
        for k, (name, corpus, path) in enumerate(images):
            img = load_rgb(path, corpus)
            H, W, _ = img.shape
            M = make_mask((1, H, W), fam, SEVERITY[lvl], np.random.default_rng(args.seed))[0]
            ref3 = np.moveaxis(img, -1, 0)
            M3 = np.repeat(M[None], 3, axis=0)
            y3 = (ref3 * M3).astype(np.float32)
            d = out / "cases" / corr / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "y.npy", np.moveaxis(y3, 0, -1))           # (H, W, 3) for the LaMa runner
            np.save(d / "mask.npy", M)
            mpath = d / "metrics.json"
            metrics = json.loads(mpath.read_text()) if mpath.exists() else {"methods": {}}
            metrics.update({"image": name, "corpus": corpus, "H": H, "W": W, "corruption": corr})
            for m in args.methods + EXTERNAL:
                if metrics["methods"].get(m, {}).get("status") == "done":
                    continue
                if m in EXTERNAL:
                    xp = d / f"x_hat_{m}.npy"
                    if not xp.exists():
                        metrics["methods"][m] = {"status": "pending"}
                        continue
                    x = np.moveaxis(np.load(xp).astype(np.float64), -1, 0)
                    cost = {}
                else:
                    x, cost = run_method(m, y3, M3, trace_memory=(k == 0), max_iter=args.max_iter)
                    _sweep.save_output(d / f"x_hat_{m}.npy", np.moveaxis(x, 0, -1),
                                       _sweep.CSFR_STORE_MAX_ERR if m in VARIANTS else 0.0)
                    if m == "csfr":
                        hole = np.argwhere(M == 0)
                        p = tuple(map(int, hole[len(hole) // 2]))
                        t0 = time.perf_counter()
                        _, info = sensitivity_row(torch.from_numpy(x[0]), torch.from_numpy(y3[0].astype(np.float64)),
                                                  torch.from_numpy(M == 1), p, VARIANTS["csfr"], RecordConfig())
                        cost.update({"record_seconds": time.perf_counter() - t0,
                                     "record_cg_iters": info["cg_iters"]})
                pp = _sweep.score(ref3, x, y3, M3)
                q = {kk: float(np.mean(pp[kk])) for kk in ("psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole")}
                metrics["methods"][m] = {"status": "done", **q, **cost}
                _sweep._write_json(mpath, metrics)
                print(f"[fullres] {corr} {name} {m}: hole PSNR {q['psnr_hole']:.2f}"
                      + (f"  {cost['seconds']:.1f}s" if "seconds" in cost else ""), flush=True)
            _sweep._write_json(mpath, metrics)
            for m, e in metrics["methods"].items():
                if e.get("status") != "done":
                    continue
                base = {"corruption": corr, "image": name, "corpus": corpus, "H": H, "W": W, "method": m}
                per.append({**base, **{kk: e[kk] for kk in ("psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole")}})
                if "seconds" in e:
                    costs.append({**base, **{kk: v for kk, v in e.items()
                                             if kk not in ("status", "psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole")}})
    summary = []
    for corr in args.corruptions:
        for m in dict.fromkeys(r["method"] for r in per):
            rs = [r for r in per if r["corruption"] == corr and r["method"] == m]
            if rs:
                summary.append({"corruption": corr, "method": m, "n_images": len(rs),
                                **{kk: float(np.mean([r[kk] for r in rs]))
                                   for kk in ("psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole")}})
    for fname, rows in (("per_image.csv", per), ("summary.csv", summary), ("cost.csv", costs)):
        if rows:
            with (out / fname).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(kk for r in rows for kk in r)))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
