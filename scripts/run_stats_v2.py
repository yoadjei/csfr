"""Paired statistics for the round-2 sweeps (E1 to E4).

    python scripts/run_stats_v2.py --sweep-dir results/sweep_v2_govdocs1

Reads the per-patch files written by scripts/run_sweep_v2.py and the sweep's
run_metadata.json (configuration and cluster labels), and writes under
<sweep-dir>/stats/:

  analysis_plan.json          the plan below, written before anything is computed
  paired_tests.csv            CSFR against every method of the comparison family
  paired_tests_ablation.csv   CSFR against each ablation arm, at the ablation seeds
  strongest_baseline.csv      per cell and metric, the best method other than CSFR
  per_image.csv               per cell, method, metric and source image: the mean
                              difference and the number of patches
  summary_ci.csv              per cell, method and metric: mean with patch and
                              cluster bootstrap intervals

Analysis plan. The unit is the patch; each per-patch metric is averaged over the
seeds at which the method ran, before pairing. Differences are signed so that a
positive value favours CSFR for every metric (method minus CSFR for GMSD, where
lower is better). Per comparison: two-sided Wilcoxon signed-rank test, Holm
adjustment across every comparison of one metric within the family, the 95
percent percentile bootstrap interval resampling patches and resampling source
images (clusters), Cohen's d_z, Cliff's delta, and per source image the mean
difference, the fraction of images where CSFR leads, the image-level bootstrap
interval and the image-level Wilcoxon test. The comparison family is every
configured method except CSFR, zero-fill (the floor, not a baseline) and the
ablation arms; an external method enters once its outputs have been scored.
The functions are those of scripts/run_paired_stats.py, which produced the
released round-1 intervals.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_paired", ROOT / "scripts" / "run_paired_stats.py")
_paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paired)
cohens_dz = _paired.cohens_dz
cliffs_delta = _paired.cliffs_delta
bootstrap_ci = _paired.bootstrap_ci
cluster_bootstrap_ci = _paired.cluster_bootstrap_ci
cluster_level_test = _paired.cluster_level_test
holm_bonferroni = _paired.holm_bonferroni

METRICS = {"psnr_hole": +1, "ssim_hole": +1, "gmsd_hole": -1}   # +1: higher is better
FLOOR = {"zero_fill"}
RNG_ROOT = 2026


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng([RNG_ROOT, zlib.crc32("|".join(map(str, key)).encode())])


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load(sweep: Path, cell: str, seeds: list[int], method: str) -> dict | None:
    """Seed-averaged per-patch metrics, or None unless the method ran at every seed."""
    files = [sweep / cell / f"seed{s}" / f"per_patch_{method}.npz" for s in seeds]
    if not files or not all(f.exists() for f in files):
        return None
    runs = []
    for f in files:
        with np.load(f) as z:          # closed at once: an open NpzFile blocks rewrites on Windows
            runs.append({k: z[k] for k in z.files})
    out = {m: np.mean([r[m] for r in runs], axis=0) for m in METRICS}
    out["exact_any"] = np.any([r["exact"] for r in runs], axis=0)
    out["n_seeds"] = len(seeds)
    return out


def compare(a: dict, b: dict, metric: str, clusters: np.ndarray, n_boot: int, key: tuple) -> dict:
    """CSFR (a) against one method (b) on one metric; positive favours CSFR."""
    sign = METRICS[metric]
    diff = sign * (a[metric] - b[metric])
    p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
    labels = np.unique(clusters)
    img = np.array([diff[clusters == c].mean() for c in labels])
    p_img, _ = cluster_level_test(diff, clusters)
    lo, hi = bootstrap_ci(diff, n_boot, _rng(*key, "patch"))
    clo, chi = cluster_bootstrap_ci(diff, clusters, n_boot, _rng(*key, "cluster"))
    ilo, ihi = bootstrap_ci(img, n_boot, _rng(*key, "image"))
    return {"n": int(diff.size), "n_seeds": int(min(a["n_seeds"], b["n_seeds"])),
            "csfr_mean": float(a[metric].mean()), "method_mean": float(b[metric].mean()),
            "diff_mean": float(diff.mean()), "diff_median": float(np.median(diff)),
            "p_wilcoxon": p, "ci_patch_lo": lo, "ci_patch_hi": hi,
            "ci_cluster_lo": clo, "ci_cluster_hi": chi,
            "d_z": cohens_dz(diff),
            "cliffs_delta": sign * cliffs_delta(a[metric], b[metric]),
            "n_images": int(labels.size),
            "frac_images_csfr_leads": float(np.mean(img > 0)),
            "frac_images_tied": float(np.mean(np.abs(img) <= 1e-12)),
            "ci_image_lo": ilo, "ci_image_hi": ihi, "p_image_wilcoxon": p_img,
            "n_exact_csfr": int(a["exact_any"].sum()), "n_exact_method": int(b["exact_any"].sum())}


def _holm(rows: list[dict]) -> None:
    for metric in METRICS:
        idx = [i for i, r in enumerate(rows) if r["metric"] == metric]
        for i, p in zip(idx, holm_bonferroni([rows[i]["p_wilcoxon"] for i in idx])):
            rows[i]["p_holm"] = p


def analysis_plan(cfg: dict, family: list[str], n_boot: int) -> dict:
    return {"unit": "patch; per-patch metric averaged over the seeds at which the method ran",
            "metrics": {m: ("higher is better" if s > 0 else "lower is better") for m, s in METRICS.items()},
            "sign_convention": "difference positive when CSFR is better (method - CSFR for GMSD)",
            "test": "two-sided Wilcoxon signed-rank (scipy.stats.wilcoxon, zero_method wilcox)",
            "correction": "Holm-Bonferroni across all comparisons of one metric within a family "
                          "(comparison family; ablation family separately)",
            "comparison_family": family,
            "ablation_family": cfg.get("ablation_variants", []),
            "ablation_seeds": cfg.get("ablation_seeds", []),
            "excluded_from_comparisons": sorted(FLOOR) + ["csfr"],
            "intervals": "95 percent percentile bootstrap: patches; source images (cluster); "
                         "image-level means",
            "bootstrap_resamples": n_boot,
            "rng": f"numpy default_rng([{RNG_ROOT}, crc32(sweep|cell|method|metric|kind)])",
            "effect_sizes": ["Cohen d_z of the paired difference", "Cliff's delta (signed as above)"],
            "per_image": "mean difference per source image; fraction of images with CSFR ahead; "
                         "image-level Wilcoxon",
            "strongest_baseline": "per cell and metric, the comparison-family method with the best mean",
            "seeds": cfg["seeds"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args(argv)

    sweep = Path(args.sweep_dir)
    if not sweep.is_absolute():
        sweep = ROOT / sweep
    cfg = json.loads((sweep / "run_metadata.json").read_text())["config"]
    cpath = Path(cfg["cluster_path"])
    clusters = np.load(cpath if cpath.is_absolute() else ROOT / cpath)
    variants = [v for v in cfg.get("csfr_variants", []) if v != "csfr"]
    ablation = cfg.get("ablation_variants", [])
    family = variants + [b for b in cfg.get("baselines", []) if b not in FLOOR] + cfg.get("external", [])
    everything = ["csfr"] + variants + ablation + cfg.get("baselines", []) + cfg.get("external", [])
    outdir = sweep / "stats"
    outdir.mkdir(exist_ok=True)
    (outdir / "analysis_plan.json").write_text(
        json.dumps(analysis_plan(cfg, family, args.bootstrap), indent=2), encoding="utf-8")

    paired, abl, strongest, per_image, summary = [], [], [], [], []
    tag = cfg.get("experiment", sweep.name)
    for fam in cfg["families"]:
        for lvl in cfg["levels"]:
            cell = f"{fam}_{lvl}"
            seeds = cfg["seeds"]
            csfr = load(sweep, cell, seeds, "csfr")
            if csfr is None:
                continue
            present = {m: d for m in family if (d := load(sweep, cell, seeds, m)) is not None}
            for metric in METRICS:
                for m, d in present.items():
                    key = (tag, cell, m, metric)
                    paired.append({"sweep": tag, "family": fam, "level": lvl, "metric": metric,
                                   "method": m, **compare(csfr, d, metric, clusters, args.bootstrap, key)})
                    diff = METRICS[metric] * (csfr[metric] - d[metric])
                    for c in np.unique(clusters):
                        sel = clusters == c
                        per_image.append({"sweep": tag, "family": fam, "level": lvl, "metric": metric,
                                          "method": m, "image": int(c), "n_patches": int(sel.sum()),
                                          "diff_mean": float(diff[sel].mean())})
            a_seeds = cfg.get("ablation_seeds", [])
            if a_seeds:
                csfr_a = load(sweep, cell, a_seeds, "csfr")
                for m in ablation:
                    d = load(sweep, cell, a_seeds, m)
                    if d is None or csfr_a is None:
                        continue
                    for metric in METRICS:
                        abl.append({"sweep": tag, "family": fam, "level": lvl, "metric": metric,
                                    "method": m, **compare(csfr_a, d, metric, clusters, args.bootstrap,
                                                           (tag, cell, m, metric, "ablation"))})
            for m in everything:
                d = load(sweep, cell, seeds, m) or load(sweep, cell, cfg.get("ablation_seeds", []), m)
                if d is None:
                    continue
                for metric in METRICS:
                    x = d[metric]
                    lo, hi = bootstrap_ci(x, args.bootstrap, _rng(tag, cell, m, metric, "mean"))
                    clo, chi = cluster_bootstrap_ci(x, clusters, args.bootstrap,
                                                    _rng(tag, cell, m, metric, "mean_cluster"))
                    summary.append({"sweep": tag, "family": fam, "level": lvl, "method": m,
                                    "metric": metric, "n": int(x.size), "n_seeds": d["n_seeds"],
                                    "mean": float(x.mean()), "ci_patch_lo": lo, "ci_patch_hi": hi,
                                    "ci_cluster_lo": clo, "ci_cluster_hi": chi,
                                    "n_exact": int(d["exact_any"].sum())})
            print(f"[stats_v2] {tag} {cell}: {len(present)} methods compared", flush=True)
    _holm(paired)
    _holm(abl)
    for fam in cfg["families"]:
        for lvl in cfg["levels"]:
            for metric, sign in METRICS.items():
                rows = [r for r in paired if r["family"] == fam and r["level"] == lvl and r["metric"] == metric]
                if not rows:
                    continue
                best = max(rows, key=lambda r: sign * r["method_mean"])
                excl = best["ci_cluster_lo"] > 0 or best["ci_cluster_hi"] < 0
                verdict = ("csfr_better" if best["ci_cluster_lo"] > 0 else
                           "baseline_better" if best["ci_cluster_hi"] < 0 else "inconclusive")
                strongest.append({"sweep": tag, "family": fam, "level": lvl, "metric": metric,
                                  "method": best["method"], "method_mean": best["method_mean"],
                                  "csfr_mean": best["csfr_mean"], "diff_mean": best["diff_mean"],
                                  "ci_cluster_lo": best["ci_cluster_lo"], "ci_cluster_hi": best["ci_cluster_hi"],
                                  "p_holm": best["p_holm"],
                                  "frac_images_csfr_leads": best["frac_images_csfr_leads"],
                                  "cluster_ci_excludes_zero": excl, "verdict": verdict,
                                  "methods_compared": " ".join(r["method"] for r in rows),
                                  "lama_included": any(r["method"] == "lama" for r in rows)})
    _write(outdir / "paired_tests.csv", paired)
    _write(outdir / "paired_tests_ablation.csv", abl)
    _write(outdir / "strongest_baseline.csv", strongest)
    _write(outdir / "per_image.csv", per_image)
    _write(outdir / "summary_ci.csv", summary)
    print(f"[stats_v2] wrote {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
