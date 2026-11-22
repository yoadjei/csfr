"""Downstream statistics for round 2 (E12).

    python scripts/run_downstream_stats_v2.py [--root results/downstream_v2]

For every classifier directory under --root (cnn, resnet), cell and method:
accuracy, and the expected calibration error of the temperature-scaled
confidence (the harness fits the temperature on clean SVHN images disjoint
from the evaluation subset) and of the raw confidence; and CSFR against every
other method by an exact McNemar test on correctness paired by (patch, seed),
Holm-adjusted across the methods of a cell. The clean images give the ceiling.

Selective prediction: the classifier answers when its (temperature-scaled)
confidence reaches a threshold. metrics.csv gives the area under the
risk-coverage curve (aurc_ts, aurc_raw; lower is better), and coverage.csv the
selective accuracy on a 5% coverage grid, per cell and per level pooled over
the families (family "all"), as the accuracy-coverage figure draws it.

Reads <root>/<classifier>/<FAM>_<LVL>/<method>.csv and <root>/<classifier>/clean.csv
(scripts/run_downstream.py). Writes <root>/stats/metrics.csv, stats.csv and coverage.csv.
The functions are the round-2 ones (src/downstream.expected_calibration_error,
run_paired_stats.holm_bonferroni): the round-1 helpers of
run_downstream_stats.py drop confidences of exactly 1 from the ECE bins and do
not enforce monotone Holm-adjusted p-values.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from downstream import expected_calibration_error  # noqa: E402

_spec = importlib.util.spec_from_file_location("_paired", ROOT / "scripts" / "run_paired_stats.py")
_paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paired)
holm_bonferroni = _paired.holm_bonferroni

REFERENCE = "csfr"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _keyed(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(r["patch_id"]), int(r["seed"])): r for r in rows}


# coverages at which the accuracy-coverage curves are tabulated
COVERAGE_GRID = tuple(round(0.05 * k, 2) for k in range(1, 21))


def risk_coverage(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    """Selective prediction: the classifier answers when its confidence is at
    least a threshold. For each distinct threshold, from the most confident
    down: the coverage (share of inputs answered) and the selective accuracy
    (accuracy on them). Tied confidences enter together, so the curve does not
    depend on how ties are ordered."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    c, k = conf[order], correct[order]
    last = np.r_[c[1:] != c[:-1], True]           # the last input of each tie group
    answered = np.arange(1, len(c) + 1)[last]
    return answered / len(c), np.cumsum(k)[last] / answered


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve: the selective risk (1 - selective
    accuracy) integrated over coverage, a step per threshold. Without ties it is
    the mean risk over the n top-k sets. Lower is better."""
    cov, acc = risk_coverage(conf, correct)
    return float(((1.0 - acc) * np.diff(np.r_[0.0, cov])).sum())


def selective_accuracy(conf, correct, coverage: float) -> tuple[float, float]:
    """(coverage, accuracy) at the highest threshold whose coverage reaches
    `coverage`: ties can lift the coverage past the one asked for."""
    cov, acc = risk_coverage(conf, correct)
    i = int(np.searchsorted(cov, coverage - 1e-9))
    return float(cov[i]), float(acc[i])


def _confidence(rows: list[dict]) -> np.ndarray:
    """The temperature-scaled confidence where the harness wrote one, else the raw one."""
    key = "confidence_ts" if rows and rows[0].get("confidence_ts", "") != "" else "confidence"
    return np.array([float(r[key]) for r in rows])


def metrics_row(rows: list[dict]) -> dict:
    correct = np.array([float(r["correct"]) for r in rows])
    raw = np.array([float(r["confidence"]) for r in rows])
    out = {"n": len(rows), "accuracy": float(correct.mean()),
           "ece_raw": expected_calibration_error(raw, correct), "aurc_raw": aurc(raw, correct)}
    if rows and rows[0].get("confidence_ts", "") != "":
        ts = _confidence(rows)
        out["ece_ts"] = expected_calibration_error(ts, correct)
        out["aurc_ts"] = aurc(ts, correct)
    return out


def coverage_rows(key: dict, rows: list[dict]) -> list[dict]:
    """The accuracy-coverage curve of `rows` on COVERAGE_GRID."""
    conf, correct = _confidence(rows), np.array([float(r["correct"]) for r in rows])
    return [{**key, "coverage_target": g, **dict(zip(("coverage", "accuracy"), selective_accuracy(conf, correct, g)))}
            for g in COVERAGE_GRID]


def mcnemar(a: dict, b: dict) -> dict:
    """Exact McNemar on the (patch, seed) pairs both methods have. b_count:
    pairs CSFR gets right and the other method wrong; statistic b - c is
    positive when CSFR is ahead."""
    keys = sorted(set(a) & set(b))
    ca = np.array([float(a[k]["correct"]) > 0.5 for k in keys])
    cb = np.array([float(b[k]["correct"]) > 0.5 for k in keys])
    n_b, n_c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if n_b + n_c == 0 else float(binomtest(n_b, n_b + n_c, 0.5).pvalue)
    return {"n_pairs": len(keys), "b_csfr_only": n_b, "c_other_only": n_c, "statistic": n_b - n_c,
            "p_mcnemar": p}


def classifier_stats(d: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Metrics, paired tests and accuracy-coverage curves of one classifier:
    curves per cell, and per level pooled over the families (family "all")."""
    clf = d.name
    metrics, stats, curves = [], [], []
    if (d / "clean.csv").exists():
        key = {"classifier": clf, "family": "clean", "level": "", "method": "clean"}
        clean = _read(d / "clean.csv")
        metrics.append({**key, **metrics_row(clean)})
        curves += coverage_rows(key, clean)
    pooled: dict[tuple[str, str], list[dict]] = {}
    for cell in sorted(p for p in d.iterdir() if p.is_dir() and "_" in p.name):
        fam, lvl = cell.name.split("_")
        per = {f.stem: _read(f) for f in sorted(cell.glob("*.csv"))}
        metrics += [{"classifier": clf, "family": fam, "level": lvl, "method": m, **metrics_row(rows)}
                    for m, rows in per.items()]
        for m, rows in per.items():
            curves += coverage_rows({"classifier": clf, "family": fam, "level": lvl, "method": m}, rows)
            pooled.setdefault((lvl, m), []).extend(rows)
        if REFERENCE not in per:
            continue
        ref = _keyed(per[REFERENCE])
        cell_rows = [{"classifier": clf, "family": fam, "level": lvl, "method": m, **mcnemar(ref, _keyed(rows))}
                     for m, rows in per.items() if m != REFERENCE]
        for r, p in zip(cell_rows, holm_bonferroni([r["p_mcnemar"] for r in cell_rows])):
            r["p_holm"] = p
        stats += cell_rows
    for (lvl, m), rows in sorted(pooled.items()):
        curves += coverage_rows({"classifier": clf, "family": "all", "level": lvl, "method": m}, rows)
    return metrics, stats, curves


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/downstream_v2")
    args = ap.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    metrics, stats, curves = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "clean.csv").exists()):
        m, s, c = classifier_stats(d)
        metrics += m
        stats += s
        curves += c
    if not metrics:
        print(f"[downstream_v2] no classifier results under {root}")
        return 1
    _write(root / "stats" / "metrics.csv", metrics)
    _write(root / "stats" / "stats.csv", stats)
    _write(root / "stats" / "coverage.csv", curves)
    print(f"[downstream_v2] wrote {root / 'stats'} ({len(metrics)} metric rows, {len(stats)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
