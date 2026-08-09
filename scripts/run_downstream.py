"""Downstream utility harness: corrupt → reconstruct → classify.

For each corruption family (CF1-CF4), severity (L1-L5) and reconstruction
method, draw seeded class-balanced masks over a fixed SVHN test subset,
reconstruct, then score with the frozen DigitCNN from ``src/downstream.py``.

Usage (session default, n_test=200 from config)::

    python scripts/run_downstream.py --config configs/downstream.yaml

Paper-scale (n_test=1000)::

    python scripts/run_downstream.py --config configs/downstream.yaml --n-test 1000

Fast smoke (1 family × 1 level × 2 methods × ~20 patches)::

    python scripts/run_downstream.py --config configs/downstream.yaml --smoke

Outputs
-------
``results/downstream/<FAM>_<LVL>/<method>.csv``: per-patch rows (one row per patch × method)
``results/downstream/summary.csv``: cell-level aggregates (rebuilt each run; see per-cell CSVs for durable record)
``results/downstream/environment.json``: library / platform capture
``results/downstream/clean.csv``: clean model predictions on uncorrupted subset

Note: ``summary.csv`` reflects only the cells in the current run and is rebuilt each time;
the per-cell CSVs in ``<FAM>_<LVL>/`` are the durable record. Do not rely on ``summary.csv``
to accumulate across runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from csfr_reconstruct_2d import pick_device, reconstruct  # noqa: E402
from downstream import (  # noqa: E402
    load_classifier,
    load_digit_dataset,
    predict,
)
from run_csfr_sweep import (  # noqa: E402
    baseline_bilinear,
    baseline_dictlearn,
    baseline_inpaint_cv,
    baseline_zero,
    capture_environment,
    make_mask,
)

ROW_FIELDS = [
    "patch_id",
    "seed",
    "family",
    "level",
    "frac",
    "method",
    "true_label",
    "prediction",
    "confidence",
    "correct",
    "bounded_unknown",
    "isolated_missing_frac",
    "mask_overlap_centre",
    "centre_erased_frac",
]

SUMMARY_FIELDS = [
    "family",
    "level",
    "frac",
    "method",
    "seed",
    "n",
    "accuracy",
    "mean_confidence",
    "n_correct",
    "n_bounded_unknown",
]

def select_class_balanced(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_test: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Seeded, class-balanced subset across digits 0-9.

    Returns (images, labels, original_index) with ``len == n_test`` (or fewer
    if a class is exhausted).
    """
    rng = np.random.default_rng(seed)
    labs = labels.cpu().numpy()
    per_class = n_test // 10
    remainder = n_test - per_class * 10
    chosen: list[int] = []
    for c in range(10):
        idx = np.where(labs == c)[0]
        if idx.size == 0:
            continue
        take = per_class + (1 if c < remainder else 0)
        take = min(take, int(idx.size))
        pick = rng.choice(idx, size=take, replace=False)
        chosen.extend(int(i) for i in pick)
    # if some classes were short, top up from remaining indices
    if len(chosen) < n_test:
        pool = np.setdiff1d(np.arange(len(labs)), np.asarray(chosen, dtype=np.int64))
        need = n_test - len(chosen)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(need, int(pool.size)), replace=False)
            chosen.extend(int(i) for i in extra)
    chosen_arr = np.asarray(chosen[:n_test], dtype=np.int64)
    rng.shuffle(chosen_arr)
    return images[chosen_arr], labels[chosen_arr], chosen_arr


def centre_box_mask(h: int, w: int, centre_frac: float) -> np.ndarray:
    """Boolean (H, W) mask of the centred box with side fractions ``centre_frac``."""
    ch = max(1, int(round(h * centre_frac)))
    cw = max(1, int(round(w * centre_frac)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    box = np.zeros((h, w), dtype=bool)
    box[y0 : y0 + ch, x0 : x0 + cw] = True
    return box


def mask_overlap_centre(M: np.ndarray, centre_frac: float = 0.5) -> float:
    """Fraction of erased pixels that lie inside the centre box."""
    erased = M == 0
    n_erased = int(erased.sum())
    if n_erased == 0:
        return 0.0
    box = centre_box_mask(M.shape[0], M.shape[1], centre_frac)
    return float((erased & box).sum()) / float(n_erased)


def centre_erased_frac_stat(M: np.ndarray, centre_frac: float = 0.5) -> float:
    """Fraction of centre box pixels that are erased.

    Measures how much of the discriminative signal was destroyed.
    Non-constant in erasure fraction (unlike mask_overlap_centre) and
    suitable for defining signal-covering subsets.
    """
    erased = M == 0
    box = centre_box_mask(M.shape[0], M.shape[1], centre_frac)
    n_box = int(box.sum())
    if n_box == 0:
        return 0.0
    return float((erased & box).sum()) / float(n_box)


def compute_bounded_unknown_stats(M: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
    """Per-patch Bounded-Unknown statistics.

    Computes both the boolean flag and the continuous isolated_missing_frac statistic.
    The rule is a priori and must not be tuned after seeing results.

    Paper meaning: regions with no observed neighbours / underdetermined.
    Fixed rule:
      - isolated_missing_frac = fraction of missing pixels with zero observed 4-neighbours
      - bounded_unknown = True iff isolated_missing_frac > threshold, OR observed pixel count is zero

    Returns
    -------
    (bounded_unknown_flag: bool, isolated_missing_frac: float)
    """
    obs = M == 1
    miss = M == 0
    if int(obs.sum()) == 0:
        return True, 1.0  # no observations: flag True, frac 1.0 (all missing are isolated)
    n_miss = int(miss.sum())
    if n_miss == 0:
        return False, 0.0  # no missing: flag False, frac 0.0 (nothing isolated)
    # count observed 4-neighbours at every site via axis shifts
    n_obs_nb = np.zeros(M.shape, dtype=np.int32)
    n_obs_nb[1:, :] += obs[:-1, :].astype(np.int32)
    n_obs_nb[:-1, :] += obs[1:, :].astype(np.int32)
    n_obs_nb[:, 1:] += obs[:, :-1].astype(np.int32)
    n_obs_nb[:, :-1] += obs[:, 1:].astype(np.int32)
    isolated = miss & (n_obs_nb == 0)
    frac = float(isolated.sum()) / float(n_miss)
    return frac > threshold, frac


def _to_uint_scale(images01: torch.Tensor, scale: str = "infer") -> np.ndarray:
    """(N,1,H,W) or (N,H,W) float in [0,1] → (N,H,W) float in [0,255].

    By default infers the scale: if max <= 1.5 treats as [0,1], else assumes
    already scaled. Explicit modes: scale='01' for [0,1], scale='uint8' for
    [0,255]. Asserts the expected scale; small overruns (e.g., from floating-point
    rounding in DL inpainters) are clipped to [0,255] rather than rejected.
    """
    x = images01.detach().cpu().numpy().astype(np.float32)
    if x.ndim == 4:
        x = x[:, 0]
    x_max = float(x.max())
    x_min = float(x.min())

    if scale == "infer":
        # infer: if clearly in [0, 1] range, scale to [0, 255]
        if x_max > 1.5:
            # already in [0, 255] range (or small overshoot); check not genuinely wrong
            assert x_max <= 300, \
                f"scale inference failed: input max={x_max} is far out of range (not ~255)"
            assert x_min >= -10, \
                f"scale inference failed: input min={x_min} is far out of range (not ~0)"
            # clip small overruns to valid range
            return np.clip(x, 0, 255)
        else:
            # in [0, 1] range; scale to [0, 255]
            return x * 255.0
    elif scale == "01":
        assert x_max <= 1.1 and x_min >= -0.1, \
            f"expected scale [0,1], got min={x_min}, max={x_max}"
        return x * 255.0
    elif scale == "uint8":
        assert x_max <= 256 and x_min >= -1, \
            f"expected scale [0,255], got min={x_min}, max={x_max}"
        # clip small overruns to valid range
        return np.clip(x, 0, 255)
    else:
        raise ValueError(f"unknown scale: {scale}")


def _to_classifier_tensor(rec255: np.ndarray, scale: str = "uint8") -> torch.Tensor:
    """(N,H,W) float in [0,255] → (N,1,H,W) float in [0,1] for DigitCNN.

    Scale parameter: 'uint8' expects [0,255], 'infer' guesses from input.
    Asserts the expected range; small overruns are clipped to the valid range.
    """
    x = rec255.astype(np.float32)
    x_max = float(x.max())
    x_min = float(x.min())

    if scale == "uint8":
        assert x_max <= 256 and x_min >= -1, \
            f"expected scale [0,255], got min={x_min}, max={x_max}"
        # clip small overruns to [0, 255]
        x_clipped = np.clip(x, 0, 255)
        return torch.from_numpy(x_clipped / 255.0).unsqueeze(1).contiguous()
    elif scale == "infer":
        if x_max > 1.5:
            # assume [0, 255]
            assert x_max <= 300 and x_min >= -10, \
                f"scale inference failed: max={x_max}, min={x_min} far out of range"
            x_clipped = np.clip(x, 0, 255)
            return torch.from_numpy(x_clipped / 255.0).unsqueeze(1).contiguous()
        else:
            # already in [0, 1]
            return torch.from_numpy(x).unsqueeze(1).contiguous()
    else:
        raise ValueError(f"unknown scale: {scale}")


def baseline_lama(
    y: np.ndarray,
    M: np.ndarray,
    weights: Optional[Path],
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """External-prior generative inpainting: LaMa (preferred) or Stable Diffusion.

    This is a Phase-2 baseline that tests the paper's argument about hallucination
    in generative priors. The method carries NO per-pixel provenance (TS=0 by
    construction). It is included empirically to show it outperforms CSFR on PSNR
    for severe corruption, but produces confident errors on ambiguous regions.

    Parameters
    ----------
    y : (N, H, W) float [0, 255]
        Corrupted (masked) images.
    M : (N, H, W) float {0, 1}
        Mask where 1=observed, 0=erased.
    weights : Path or None
        Unused; included for interface compatibility. Model weights are
        downloaded via diffusers' hub cache on first use.
    device : str
        Torch device string. Pass the harness's resolved device so the model
        runs where the rest of the pipeline does.

    Returns
    -------
    (N, H, W) float [0, 255] or None
        Inpainted images, or None when a required dependency is missing.

    Only a missing dependency (ImportError) returns None, which is the harness's
    signal to skip the method. Every other failure raises: a download failure or
    an out-of-memory error must not be reported as "weights unavailable", because
    a long run would then produce nothing and still look like it succeeded.
    """
    try:
        from generative_inpainter import inpaint
    except ImportError:
        return None

    # backend auto prefers LaMa and falls back to Stable Diffusion.
    # seed 0 for deterministic output; seeding is per-run, not per-cell.
    # announced in the harness log for auditability.
    result = inpaint(y, M, backend="auto", seed=0, num_inference_steps=20,
                     device=str(device))
    from generative_inpainter import _SELECTED_BACKEND
    print(f"[downstream] lama backend: {_SELECTED_BACKEND}", flush=True)
    return result


def reconstruct_method(
    name: str,
    y: np.ndarray,
    M: np.ndarray,
    *,
    seed: int,
    sol: dict,
    device: torch.device,
    lama_weights: Optional[Path],
) -> Optional[np.ndarray]:
    """Run one reconstruction method; return (N,H,W) float [0,255] or None to skip."""
    import cv2

    if name == "zero_fill":
        return baseline_zero(y, M)
    if name == "bilinear":
        return baseline_bilinear(y, M)
    if name == "telea":
        return baseline_inpaint_cv(y, M, cv2.INPAINT_TELEA)
    if name == "ns":
        return baseline_inpaint_cv(y, M, cv2.INPAINT_NS)
    if name == "dictlearn":
        return baseline_dictlearn(y, M, seed)
    if name == "csfr":
        y_t = torch.from_numpy(y)
        M_t = torch.from_numpy(M)
        x_t, _ = reconstruct(
            y_t,
            M_t,
            lam_l1=sol["lam_l1"],
            lam_tv=sol["lam_tv"],
            lam_fc=sol["lam_fc"],
            max_iter=sol["max_iter"],
            lr=sol["lr"],
            device=device,
            ckpt_path=None,
            ckpt_every=sol.get("ckpt_every", 9999),
            resume=False,
        )
        return x_t.detach().cpu().numpy().astype(np.float32)
    if name == "lama":
        return baseline_lama(y, M, lama_weights, device=str(device))
    raise ValueError(f"unknown method: {name}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def apply_smoke_overrides(cfg: dict) -> dict:
    """Shrink the grid for CI / session verification."""
    cfg = dict(cfg)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": cfg.get("levels", {}).get("L1", 0.05)}
    cfg["methods"] = ["zero_fill", "bilinear"]
    cfg["seeds"] = [cfg.get("seeds", [0])[0]]
    # keep CSFR off the smoke path so verify stays fast
    return cfg


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_pipeline(cfg: dict, *, force: bool = False) -> Path:
    """Execute the full (or smoke-shrunk) utility grid. Returns output_dir."""
    outdir = _resolve_path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.get("device", "auto"))
    clf_path = _resolve_path(cfg["classifier_path"])
    if not clf_path.exists():
        raise FileNotFoundError(f"classifier checkpoint missing: {clf_path}")

    n_test = int(cfg["n_test"])
    seeds = list(cfg.get("seeds", [0]))
    families = list(cfg["families"])
    levels: dict = dict(cfg["levels"])
    methods = list(cfg["methods"])
    sol = dict(cfg["solver"])
    bu_thresh = float(cfg.get("bounded_unknown_threshold", 0.5))
    centre_frac = float(cfg.get("centre_box_frac", 0.5))
    lama_raw = cfg.get("lama_weights")
    lama_weights = _resolve_path(lama_raw) if lama_raw else None

    prefer = str(cfg.get("corpus", "svhn"))
    images, labels, corpus = load_digit_dataset(prefer=prefer, split="test")
    # selection seed is the first experiment seed so the subset is stable
    sel_seed = int(seeds[0])
    images, labels, patch_ids = select_class_balanced(images, labels, n_test, sel_seed)
    patches = _to_uint_scale(images)  # (N,H,W) in [0,255]
    true_labels = labels.cpu().numpy().astype(np.int64)

    model = load_classifier(clf_path, device=device, freeze_weights=True)
    print(
        f"[downstream] corpus={corpus}  n={patches.shape[0]}  "
        f"{patches.shape[1]}x{patches.shape[2]}  device={device}  "
        f"methods={methods}",
        flush=True,
    )

    # record clean predictions on uncorrupted subset (Fix 1)
    clean_pred, clean_conf = predict(model, _to_classifier_tensor(patches, scale="uint8"), device=device)
    clean_pred_np = clean_pred.numpy().astype(np.int64)
    clean_conf_np = clean_conf.numpy().astype(np.float64)
    clean_correct = clean_pred_np == true_labels
    clean_accuracy = float(clean_correct.mean())

    clean_rows = []
    for i in range(patches.shape[0]):
        clean_rows.append({
            "patch_id": int(patch_ids[i]),
            "true_label": int(true_labels[i]),
            "prediction": int(clean_pred_np[i]),
            "confidence": float(clean_conf_np[i]),
            "correct": int(clean_correct[i]),
        })

    # write clean predictions immediately before the cell loop to avoid stale data during interrupted runs
    outdir.mkdir(parents=True, exist_ok=True)
    CLEAN_FIELDS = ["patch_id", "true_label", "prediction", "confidence", "correct"]
    write_csv(outdir / "clean.csv", clean_rows, CLEAN_FIELDS)
    print(f"[downstream] wrote {outdir / 'clean.csv'}", flush=True)

    summary_rows: list[dict] = []
    cells = [(f, l) for f in families for l in levels]

    for fam, lvl in cells:
        frac = float(levels[lvl])
        cell_dir = outdir / f"{fam}_{lvl}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # read existing CSVs once per (cell, method) before seed loop to avoid duplication
        # partition into "current seeds" and "other seeds" for preservation logic
        existing_per_method: dict[str, list[dict]] = {}
        other_seed_rows_per_method: dict[str, list[dict]] = {}
        for method in methods:
            csv_path = cell_dir / f"{method}.csv"
            if csv_path.exists() and not force and method != "lama":
                existing = list(csv.DictReader(csv_path.open(encoding="utf-8")))
                existing_per_method[method] = existing
                # rows belonging to seeds NOT in this run (to preserve)
                other_seed_rows = [r for r in existing if int(r["seed"]) not in seeds]
                other_seed_rows_per_method[method] = other_seed_rows
            else:
                existing_per_method[method] = []
                other_seed_rows_per_method[method] = []

        # accumulate per-method rows across seeds, then write one CSV each
        method_rows: dict[str, list[dict]] = {m: [] for m in methods}

        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            M = make_mask(patches.shape, fam, frac, rng)
            y = patches * M

            overlaps = np.array(
                [mask_overlap_centre(M[i], centre_frac) for i in range(M.shape[0])],
                dtype=np.float64,
            )
            centre_fracs = np.array(
                [centre_erased_frac_stat(M[i], centre_frac) for i in range(M.shape[0])],
                dtype=np.float64,
            )
            # compute both boolean flag and continuous statistic from one code path
            bu_stats = [compute_bounded_unknown_stats(M[i], bu_thresh) for i in range(M.shape[0])]
            bu_flags = np.array([s[0] for s in bu_stats], dtype=bool)
            isolated_fracs = np.array([s[1] for s in bu_stats], dtype=np.float64)

            for method in methods:
                # check completeness from pre-read data
                if existing_per_method.get(method):
                    seed_rows_existing = [r for r in existing_per_method[method]
                                         if int(r["seed"]) == int(seed)]
                    # complete iff we have exactly n_test rows for this seed
                    if len(seed_rows_existing) >= patches.shape[0]:
                        print(
                            f"[downstream] skip {fam}/{lvl}/{method} seed={seed}",
                            flush=True,
                        )
                        method_rows[method].extend(seed_rows_existing)
                        n_ok = sum(int(float(r["correct"])) for r in seed_rows_existing)
                        n_bu = sum(
                            1 for r in seed_rows_existing
                            if str(r.get("bounded_unknown", "")).lower() in ("1", "true")
                        )
                        summary_rows.append({
                            "family": fam, "level": lvl, "frac": frac,
                            "method": method, "seed": seed, "n": len(seed_rows_existing),
                            "accuracy": n_ok / max(len(seed_rows_existing), 1),
                            "mean_confidence": float(np.mean(
                                [float(r["confidence"]) for r in seed_rows_existing]
                            )),
                            "n_correct": n_ok,
                            "n_bounded_unknown": n_bu,
                        })
                        continue

                t0 = time.time()
                try:
                    rec = reconstruct_method(
                        method, y, M,
                        seed=int(seed), sol=sol, device=device,
                        lama_weights=lama_weights,
                    )
                except NotImplementedError as exc:
                    print(f"[downstream] skip {method}: {exc}", flush=True)
                    continue
                if rec is None:
                    print(
                        f"[downstream] skip {method}: required dependency missing",
                        flush=True,
                    )
                    continue

                pred, conf = predict(model, _to_classifier_tensor(rec), device=device)
                pred_np = pred.numpy().astype(np.int64)
                conf_np = conf.numpy().astype(np.float64)
                correct = pred_np == true_labels

                rows = []
                for i in range(patches.shape[0]):
                    # Bounded-Unknown boolean is recorded for CSFR only; others stay False.
                    # isolated_missing_frac is recorded for all methods.
                    bu_val = bool(bu_flags[i]) if method == "csfr" else False
                    rows.append({
                        "patch_id": int(patch_ids[i]),
                        "seed": int(seed),
                        "family": fam,
                        "level": lvl,
                        "frac": frac,
                        "method": method,
                        "true_label": int(true_labels[i]),
                        "prediction": int(pred_np[i]),
                        "confidence": float(conf_np[i]),
                        "correct": int(correct[i]),
                        "bounded_unknown": bu_val,
                        "isolated_missing_frac": float(isolated_fracs[i]),
                        "mask_overlap_centre": float(overlaps[i]),
                        "centre_erased_frac": float(centre_fracs[i]),
                    })
                method_rows[method].extend(rows)
                acc = float(correct.mean())
                summary_rows.append({
                    "family": fam, "level": lvl, "frac": frac,
                    "method": method, "seed": seed, "n": int(patches.shape[0]),
                    "accuracy": acc,
                    "mean_confidence": float(conf_np.mean()),
                    "n_correct": int(correct.sum()),
                    "n_bounded_unknown": int(bu_flags.sum()) if method == "csfr" else 0,
                })
                print(
                    f"[downstream] {fam}/{lvl}  {method:10s}  seed={seed}  "
                    f"acc={acc:.3f}  ({time.time() - t0:.1f}s)",
                    flush=True,
                )

        # write CSVs: add other-seed rows (preserved exactly once), then current-seed rows
        for method, rows in method_rows.items():
            if rows or other_seed_rows_per_method.get(method):
                # assemble: other seeds first, then current seed rows
                final_rows = other_seed_rows_per_method.get(method, []) + rows

                # assert uniqueness of (patch_id, seed) pairs
                pairs = set()
                for row in final_rows:
                    pair = (int(row["patch_id"]), int(row["seed"]))
                    assert pair not in pairs, \
                        f"duplicate (patch_id, seed)={pair} in {fam}/{lvl}/{method}"
                    pairs.add(pair)

                write_csv(cell_dir / f"{method}.csv", final_rows, ROW_FIELDS)

    write_csv(outdir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    # environment.json is written at the end to capture run-completion information (n_test reflects actual patches from this run)
    env_data = capture_environment()
    # add the inpainter backend selection after a run that used it
    try:
        from generative_inpainter import get_model_info
        env_data["inpainter"] = get_model_info(device=str(device))
    except ImportError:
        pass
    env_payload = {
        "config": cfg,
        "device": str(device),
        "corpus": corpus,
        "n_test": int(patches.shape[0]),
        "classifier_path": str(clf_path),
        "clean_accuracy": clean_accuracy,
        "environment": env_data,
    }
    (outdir / "environment.json").write_text(
        json.dumps(env_payload, indent=2), encoding="utf-8"
    )
    print(f"[downstream] wrote {outdir / 'summary.csv'}", flush=True)
    return outdir


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Downstream utility: corrupt → reconstruct → classify",
    )
    ap.add_argument("--config", default="configs/downstream.yaml",
                    help="path to YAML config")
    ap.add_argument("--smoke", action="store_true",
                    help="1 family × 1 level × 2 methods × ~20 patches")
    ap.add_argument("--n-test", type=int, default=None,
                    help="override config n_test (paper-scale: 1000)")
    ap.add_argument("--output-dir", default=None,
                    help="override config output_dir (for smoke tests, pass a temp directory)")
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--only-level", default=None)
    ap.add_argument("--only-method", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if method CSVs exist")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.n_test is not None:
        cfg["n_test"] = int(args.n_test)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["device"] = args.device
    if args.only_family is not None:
        cfg["families"] = [args.only_family]
    if args.only_level is not None:
        lvl = args.only_level
        if lvl not in cfg["levels"]:
            raise ValueError(f"unknown level {lvl!r}")
        cfg["levels"] = {lvl: cfg["levels"][lvl]}
    if args.only_method is not None:
        cfg["methods"] = [args.only_method]

    run_pipeline(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
