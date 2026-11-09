"""Bounded-Unknown evaluation (E14): does the mask predict reconstruction failure?

    python scripts/run_bu_evaluation.py

For every patch, cell and seed of the round-2 GovDocs1 and DFRWS sweeps, three
predictors computed from the mask alone: the isolated-missing fraction (the
statistic behind the released Bounded-Unknown flag: missing pixels with no
observed 4-neighbour, run_downstream.compute_bounded_unknown_stats), the mean
Euclidean distance of hole pixels to the nearest observed pixel, and the hole
fraction. Outcomes: CSFR hole PSNR below 20 dB and below 25 dB.

Reported: AUROC of each predictor pooled over all cells, per family, and within
each cell where the outcome varies (auroc.csv); a logistic model of the three
predictors fitted with 10-fold cross-validation, its reliability diagram in 10
bins and ECE (calibration.csv). decision.json applies the rule fixed in
tasks/todo.md before the data were seen: Bounded-Unknown stays a contribution
only if the pooled AUROC of the best predictor exceeds 0.80 AND the within-cell
AUROC of that predictor exceeds 0.65 in at least half of the cells with outcome
variance; otherwise it is reported as a diagnostic that tracks corruption
geometry.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from downstream import expected_calibration_error  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ds", ROOT / "scripts" / "run_downstream.py")
_ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ds)

PREDICTORS = ["isolated_missing_frac", "mean_hole_distance", "hole_frac"]
THRESHOLDS = [20.0, 25.0]
RULE = {"pooled_auroc_min": 0.80, "within_cell_auroc_min": 0.65, "within_cell_fraction_min": 0.5}


def predictors(M: np.ndarray) -> dict:
    hole = M == 0
    _, iso = _ds.compute_bounded_unknown_stats(M, 0.5)
    dist = distance_transform_edt(hole) if hole.any() and (~hole).any() else np.zeros(M.shape)
    return {"isolated_missing_frac": float(iso),
            "mean_hole_distance": float(dist[hole].mean()) if hole.any() else 0.0,
            "hole_frac": float(hole.mean())}


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def collect(sweep: Path) -> list[dict]:
    cfg = json.loads((sweep / "run_metadata.json").read_text())["config"]
    rows = []
    for fam in cfg["families"]:
        for lvl in cfg["levels"]:
            for seed in cfg["seeds"]:
                d = sweep / f"{fam}_{lvl}" / f"seed{seed}"
                f = d / "per_patch_csfr.npz"
                if not f.exists():
                    continue
                M = np.load(d / "mask.npy")
                with np.load(f) as z:
                    psnr = z["psnr_hole"]
                for i in range(len(M)):
                    rows.append({"sweep": cfg.get("experiment", sweep.name), "family": fam, "level": lvl,
                                 "seed": seed, "patch": i, "psnr_hole": float(psnr[i]), **predictors(M[i])})
    return rows


def evaluate(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    au, cal, decision = [], [], {"rule": RULE, "by_threshold": {}}
    X = np.array([[r[p] for p in PREDICTORS] for r in rows])
    cells = sorted({(r["sweep"], r["family"], r["level"]) for r in rows})
    for thr in THRESHOLDS:
        y = np.array([r["psnr_hole"] < thr for r in rows])
        pooled = {}
        for j, p in enumerate(PREDICTORS):
            a = _auroc(y, X[:, j])
            pooled[p] = a
            au.append({"threshold_db": thr, "scope": "pooled", "group": "all", "predictor": p,
                       "n": len(y), "n_fail": int(y.sum()), "auroc": a})
            for fam in sorted({r["family"] for r in rows}):
                sel = np.array([r["family"] == fam for r in rows])
                au.append({"threshold_db": thr, "scope": "family", "group": fam, "predictor": p,
                           "n": int(sel.sum()), "n_fail": int(y[sel].sum()), "auroc": _auroc(y[sel], X[sel, j])})
        within: dict[str, list[float]] = {p: [] for p in PREDICTORS}
        for sw, fam, lvl in cells:
            sel = np.array([(r["sweep"], r["family"], r["level"]) == (sw, fam, lvl) for r in rows])
            for j, p in enumerate(PREDICTORS):
                a = _auroc(y[sel], X[sel, j])
                if a is not None:
                    within[p].append(a)
                au.append({"threshold_db": thr, "scope": "cell", "group": f"{sw}:{fam}_{lvl}", "predictor": p,
                           "n": int(sel.sum()), "n_fail": int(y[sel].sum()), "auroc": a})
        valid = {p: a for p, a in pooled.items() if a is not None}
        best = max(valid, key=valid.get) if valid else None
        frac_ok = (float(np.mean([a > RULE["within_cell_auroc_min"] for a in within[best]]))
                   if best and within[best] else 0.0)
        keep = bool(best and valid[best] > RULE["pooled_auroc_min"] and frac_ok >= RULE["within_cell_fraction_min"])
        decision["by_threshold"][f"{thr:g}dB"] = {
            "best_predictor": best, "pooled_auroc": valid.get(best) if best else None,
            "cells_with_outcome_variance": len(within[best]) if best else 0,
            "fraction_cells_within_auroc_above_min": frac_ok, "keep_as_contribution": keep}
        if 0 < y.sum() < len(y):
            k = min(10, int(min(y.sum(), len(y) - y.sum())))
            model = make_pipeline(StandardScaler(), LogisticRegression())
            prob = cross_val_predict(model, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                                     method="predict_proba")[:, 1]
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, 9)
            for b in range(10):
                s = idx == b
                if s.any():
                    cal.append({"threshold_db": thr, "bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(s.sum()),
                                "mean_predicted": float(prob[s].mean()), "observed_rate": float(y[s].mean())})
            decision["by_threshold"][f"{thr:g}dB"].update({
                "logistic_cv_auroc": float(roc_auc_score(y, prob)),
                "logistic_cv_ece_10bins": expected_calibration_error(prob, y, n_bins=10), "cv_folds": k})
    return au, cal, decision


def _write(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", default=["results/sweep_v2_govdocs1", "results/sweep_v2_dfrws"])
    ap.add_argument("--out", default="results/bu_evaluation")
    args = ap.parse_args(argv)
    rows = []
    for s in args.sweeps:
        p = Path(s) if Path(s).is_absolute() else ROOT / s
        rows += collect(p)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    au, cal, decision = evaluate(rows)
    decision["n_patches"] = len(rows)
    decision["sweeps"] = args.sweeps
    _write(out / "auroc.csv", au)
    _write(out / "calibration.csv", cal)
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    for k, v in decision["by_threshold"].items():
        print(f"[bu] {k}: best {v['best_predictor']} pooled AUROC {v['pooled_auroc']}, "
              f"within-cell ok {v['fraction_cells_within_auroc_above_min']:.2f} -> keep {v['keep_as_contribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
