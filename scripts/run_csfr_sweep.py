"""CF1..CF4 × L1..L5 sweep on real DFRWS image patches, batched + resumable.

Usage:
    python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml
    python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml --only-family CF1 --only-level L1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# resolve project root relative to this script

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from csfr_reconstruct_2d import (
    reconstruct, pick_device, psnr, ssim2d, hrp, cvr_metrics, SCHEMA,
)


def make_mask(shape: tuple[int, int, int], family: str, frac: float,
              rng: np.random.Generator) -> np.ndarray:
    b, h, w = shape
    M = np.ones(shape, dtype=np.float32)
    if family == "CF1":
        n_miss = int(round(frac * h * w))
        for i in range(b):
            idx = rng.choice(h * w, n_miss, replace=False)
            M[i].reshape(-1)[idx] = 0.0
    elif family == "CF2":
        target = frac * h * w
        for i in range(b):
            placed = 0
            while placed < target:
                bh = rng.integers(4, max(5, h // 2))
                bw = rng.integers(4, max(5, w // 2))
                y0 = rng.integers(0, max(1, h - bh))
                x0 = rng.integers(0, max(1, w - bw))
                M[i, y0:y0 + bh, x0:x0 + bw] = 0.0
                placed = int((1 - M[i]).sum())
    elif family == "CF3":
        rows = int(round(frac * h))
        M[:, :rows, :] = 0.0
    elif family == "CF4":
        rows = int(round(frac * h))
        M[:, h - rows:, :] = 0.0
    else:
        raise ValueError(family)
    return M


def baseline_zero(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[M == 0] = 0.0
    return out


def baseline_bilinear(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    from scipy.interpolate import griddata
    b, h, w = y.shape
    out = np.zeros_like(y)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(b):
        Mi = M[i] == 1
        if not Mi.any():
            continue
        pts = np.stack([yy[Mi], xx[Mi]], axis=1)
        vals = y[i][Mi]
        grid = griddata(pts, vals, (yy, xx), method="linear", fill_value=0.0)
        out[i] = grid
    return out


def baseline_inpaint_cv(y: np.ndarray, M: np.ndarray, flag: int) -> np.ndarray:
    """Classical PDE inpainting (Telea / Navier-Stokes) via OpenCV."""
    import cv2
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        img8 = np.clip(y[i], 0, 255).astype(np.uint8)
        hole = (M[i] == 0).astype(np.uint8)
        out[i] = cv2.inpaint(img8, hole, inpaintRadius=3, flags=flag).astype(np.float32)
    return out


def baseline_dictlearn(y: np.ndarray, M: np.ndarray, rng_seed: int,
                       tile: int = 8, stride: int = 8,
                       n_atoms: int = 64, n_nonzero: int = 8) -> np.ndarray:
    """Per-image dictionary-learning inpainting (online dictionary learning,
    Mairal et al. 2009 style): learn atoms from fully-observed tiles of the
    SAME corrupted image, then reconstruct each incomplete tile by
    orthogonal matching pursuit on the observed pixels only. Tiles with no
    observations fall back to the observed-region mean."""
    from sklearn.decomposition import MiniBatchDictionaryLearning

    def omp_numpy(D_obs: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
        """Bounded orthogonal matching pursuit (numpy only — sklearn's OMP can
        deadlock under mixed torch/OpenMP threadpools on Windows)."""
        code = np.zeros(D_obs.shape[1])
        resid = target.astype(np.float64).copy()
        support: list[int] = []
        norms = np.linalg.norm(D_obs, axis=0)
        norms[norms < 1e-10] = 1e-10
        for _ in range(k):
            corr = np.abs(D_obs.T @ resid) / norms
            corr[support] = -1.0
            j = int(np.argmax(corr))
            if corr[j] <= 1e-12:
                break
            support.append(j)
            sub = D_obs[:, support]
            sol, *_ = np.linalg.lstsq(sub, target, rcond=None)
            resid = target - sub @ sol
        if support:
            code[support] = sol
        return code

    b, h, w = y.shape
    out = y.copy()
    for i in range(b):
        img, m = y[i], M[i]
        obs_tiles = []
        for r in range(0, h - tile + 1, 2):
            for c in range(0, w - tile + 1, 2):
                if m[r:r + tile, c:c + tile].all():
                    obs_tiles.append(img[r:r + tile, c:c + tile].reshape(-1))
        obs_mean = float(img[m == 1].mean()) if (m == 1).any() else 127.5
        if len(obs_tiles) < n_atoms:
            out[i][m == 0] = obs_mean  # too little intact area to learn from
            continue
        X = np.asarray(obs_tiles, dtype=np.float64)
        if len(X) > 400:  # cap fit cost; deterministic subsample
            X = X[np.random.default_rng(rng_seed).choice(len(X), 400, replace=False)]
        mu = X.mean(axis=0)
        dl = MiniBatchDictionaryLearning(
            n_components=n_atoms, alpha=1.0, batch_size=32, max_iter=30,
            random_state=rng_seed, fit_algorithm="cd",
            transform_algorithm="lasso_cd", transform_max_iter=200)
        dl.fit(X - mu)
        Dt = dl.components_.T  # (tile*tile, n_atoms)

        acc = np.zeros((h, w), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.float64)
        for r in range(0, h - tile + 1, stride):
            for c in range(0, w - tile + 1, stride):
                tm = m[r:r + tile, c:c + tile].reshape(-1)
                ty = img[r:r + tile, c:c + tile].reshape(-1).astype(np.float64)
                if tm.all():
                    rec = ty
                elif not tm.any():
                    rec = np.full(tile * tile, obs_mean)
                else:
                    obs = tm == 1
                    D_obs = Dt[obs]
                    code = omp_numpy(D_obs, (ty - mu)[obs],
                                     min(n_nonzero, int(obs.sum())))
                    rec = Dt @ code + mu
                    rec[obs] = ty[obs]  # keep observed pixels exact
                acc[r:r + tile, c:c + tile] += rec.reshape(tile, tile)
                cnt[r:r + tile, c:c + tile] += 1.0
        filled = np.divide(acc, np.maximum(cnt, 1), out=np.full((h, w), obs_mean), where=cnt > 0)
        res = img.astype(np.float64).copy()
        res[m == 0] = filled[m == 0]
        out[i] = np.clip(res, 0, 255).astype(np.float32)
    return out


def eval_batch(rec_t, ref_t, y_t, M_t, D, hf, tv_thresh, hf_thresh):
    n = rec_t.shape[0]
    psnrs = [psnr(ref_t[i], rec_t[i]) for i in range(n)]
    ssims = [ssim2d(ref_t[i].cpu().numpy(), rec_t[i].cpu().numpy()) for i in range(n)]
    hrps = [hrp(rec_t[i], y_t[i], M_t[i]) for i in range(n)]
    cvrs = np.array([cvr_metrics(rec_t[i], y_t[i], M_t[i], D, hf, tv_thresh, hf_thresh)
                     for i in range(n)])
    ssims_ok = [s for s in ssims if s is not None]
    return {
        "psnr_db_mean": float(np.mean(psnrs)),
        "psnr_db_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims_ok)),
        "ssim_std": float(np.std(ssims_ok)),
        "hrp_mean": float(np.mean(hrps)),
        "hrp_std": float(np.std(hrps)),
        "cvr_c1": float(cvrs[:, 0].mean()),
        "cvr_c2": float(cvrs[:, 1].mean()),
        "cvr_c3": float(cvrs[:, 2].mean()),
        "cvr_c4": float(cvrs[:, 3].mean()),
    }


# CSFR ablation variants (red comment R20). The data term (C1) and the l1
# sparsity prior are the reconstruction engine and stay on in all variants.
CSFR_VARIANTS = {
    "csfr":          {"enable_tv": True,  "enable_fc": True,  "enable_bounds": True},
    "csfr_c1":       {"enable_tv": False, "enable_fc": False, "enable_bounds": False},
    "csfr_c1c2":     {"enable_tv": True,  "enable_fc": False, "enable_bounds": False},
    "csfr_c1c3":     {"enable_tv": False, "enable_fc": False, "enable_bounds": True},
    "csfr_c1c4":     {"enable_tv": False, "enable_fc": True,  "enable_bounds": False},
}


def capture_environment() -> dict:
    """Interpreter, platform, and versions of every library the sweep depends on."""
    import cv2
    import scipy
    import sklearn
    import skimage
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(),
           "torch": torch.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
           "scikit-image": skimage.__version__, "opencv": cv2.__version__}
    try:
        import psutil
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        env["total_ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    return env


def selected_variants(cfg: dict) -> dict:
    """CSFR variants to run for this config.

    Defaults to the full ablation set. A config may set ``variants: [csfr]`` to
    run only the full solver -- used by the external-validation sweeps, where the
    constraint ablation is not repeated.
    """
    names = cfg.get("variants")
    if not names:
        return CSFR_VARIANTS
    unknown = set(names) - set(CSFR_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    return {k: CSFR_VARIANTS[k] for k in names}


def run_cell(patches, fam, lvl, frac, seed, cell_dir, dev, cfg, force):
    """One (family, level, seed) cell. Resumable per METHOD: existing entries
    in metrics.json are kept unless --force."""
    import cv2

    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cell_dir / "metrics.json"
    out = (json.loads(metrics_path.read_text())
           if metrics_path.exists() and not force else {})
    out.update({"family": fam, "level": lvl, "frac": frac,
                "n": int(patches.shape[0]), "seed": seed,
                "device": str(dev), "schema": SCHEMA})

    rng = np.random.default_rng(seed)
    M = make_mask(patches.shape, fam, frac, rng)
    y = patches * M
    np.save(cell_dir / "y.npy", y)
    np.save(cell_dir / "mask.npy", M)

    y_t = torch.from_numpy(y); M_t = torch.from_numpy(M)
    ref_t = torch.from_numpy(patches)
    sol, ev = cfg["solver"], cfg["evaluation"]

    def measure(arr_t, D, hf_m):
        return eval_batch(arr_t.to(dev), ref_t.to(dev), y_t.to(dev), M_t.to(dev),
                          D, hf_m, ev["tv_thresh"], ev["hf_thresh"])

    D = hf_m = None
    for name, flags in selected_variants(cfg).items():
        if name in out:
            continue
        t0 = time.time()
        x_t, info = reconstruct(
            y_t, M_t,
            lam_l1=sol["lam_l1"], lam_tv=sol["lam_tv"], lam_fc=sol["lam_fc"],
            max_iter=sol["max_iter"], lr=sol["lr"], device=dev,
            ckpt_path=cell_dir / f"ckpt_{name}.pt", ckpt_every=sol["ckpt_every"],
            resume=True, **flags,
        )
        D, hf_m = info["D"], info["hf"]
        x_cpu = x_t.cpu()
        if name == "csfr":
            np.save(cell_dir / "x_hat.npy", x_cpu.numpy())
            out["runtime_s"] = time.time() - t0
        out[name] = measure(x_cpu, D, hf_m)
        metrics_path.write_text(json.dumps(out, indent=2))
    if D is None:  # all variants cached; rebuild transforms for baselines
        from csfr_reconstruct_2d import dct_matrix, hf_mask
        n = y.shape[-1]
        D = dct_matrix(n, dev, y_t.dtype); hf_m = hf_mask(n, dev, y_t.dtype)

    baselines = {
        "zero_fill": lambda: baseline_zero(y, M),
        "bilinear": lambda: baseline_bilinear(y, M),
        "inpaint_telea": lambda: baseline_inpaint_cv(y, M, cv2.INPAINT_TELEA),
        "inpaint_ns": lambda: baseline_inpaint_cv(y, M, cv2.INPAINT_NS),
        "dictlearn": lambda: baseline_dictlearn(y, M, seed),
    }
    for name, fn in baselines.items():
        if name in out:
            continue
        out[name] = measure(torch.from_numpy(fn()), D, hf_m)
        metrics_path.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CSFR 2D sweep: 4 families × 5 levels")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--only-level", default=None)
    ap.add_argument("--force", action="store_true", help="Re-run even if metrics exist")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = pick_device(cfg.get("device", "auto"))
    outdir = PROJECT_ROOT / cfg["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    patches_path = PROJECT_ROOT / cfg["patches_path"]
    patches = np.load(patches_path).astype(np.float32)
    print(f"[sweep2d] {patches.shape[0]} patches  "
          f"{patches.shape[1]}x{patches.shape[2]}  dev={dev}")

    levels = cfg["levels"]
    families = cfg["families"]
    cells = [(f, l) for f in families for l in levels
             if (args.only_family is None or f == args.only_family)
             and (args.only_level is None or l == args.only_level)]

    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    methods = list(selected_variants(cfg)) + ["zero_fill", "bilinear", "inpaint_telea",
                                     "inpaint_ns", "dictlearn"]
    metric_keys = ["psnr_db_mean", "psnr_db_std", "ssim_mean", "ssim_std",
                   "hrp_mean", "hrp_std", "cvr_c1", "cvr_c2", "cvr_c3", "cvr_c4"]

    per_seed = {}  # (fam, lvl, seed) -> metrics dict
    for i, (fam, lvl) in enumerate(cells):
        frac = levels[lvl]
        for seed in seeds:
            print(f"[sweep2d] {i+1}/{len(cells)}  {fam}/{lvl}  seed={seed}  miss={frac:.0%}",
                  flush=True)
            cell_dir = outdir / f"{fam}_{lvl}" / f"seed{seed}"
            r = run_cell(patches, fam, lvl, frac, seed, cell_dir, dev, cfg, args.force)
            per_seed[(fam, lvl, seed)] = r
            print(f"    csfr {r['csfr']['psnr_db_mean']:.2f}dB  "
                  f"dict {r['dictlearn']['psnr_db_mean']:.2f}  "
                  f"telea {r['inpaint_telea']['psnr_db_mean']:.2f}  "
                  f"zero {r['zero_fill']['psnr_db_mean']:.2f}", flush=True)

    # cross-seed aggregate: mean over seeds of each metric, plus between-seed std

    csv_path = outdir / "summary.csv"
    fields = ["family", "level", "frac", "method", "n_seeds"] + metric_keys + \
             [k + "_seedstd" for k in metric_keys]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (fam, lvl) in cells:
            frac = levels[lvl]
            for method in methods:
                vals = {k: [per_seed[(fam, lvl, s)][method][k] for s in seeds
                            if (fam, lvl, s) in per_seed]
                        for k in metric_keys}
                row = {"family": fam, "level": lvl, "frac": frac,
                       "method": method, "n_seeds": len(seeds)}
                for k in metric_keys:
                    row[k] = float(np.mean(vals[k]))
                    row[k + "_seedstd"] = float(np.std(vals[k]))
                w.writerow(row)
    print(f"[sweep2d] wrote {csv_path}")

    # save run metadata for traceability. The exact library versions are

    # captured here rather than only in requirements.txt, so that a released
    # result set states the environment that actually produced it.
    meta = {"config": cfg, "device": str(dev), "n_cells": len(cells),
            "environment": capture_environment(),
            "patches_sha256": hashlib.sha256(patches.tobytes()).hexdigest()}
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
