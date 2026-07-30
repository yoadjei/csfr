"""Per-patch paired inference for the CSFR sweep (pre-submission review §6, P1).

Recomputes per-patch PSNR / SSIM / HRP for all six same-harness methods from the
artefacts already persisted by ``run_csfr_sweep.py`` (``mask.npy``, ``y.npy``,
``x_hat.npy``) plus the ground-truth patch set, then runs paired inference of
CSFR against each baseline.

Why recompute rather than re-run the solver: the sweep persists CSFR's
reconstruction (``x_hat.npy``) and the exact observation mask per cell and seed,
so CSFR is read back verbatim and only the deterministic baselines are re-derived
from ``(y, M)``. The numbers therefore refer to exactly the runs reported in the
manuscript tables.

Outputs (under ``results/paired_stats/``):
  per_patch/<FAM>_<LVL>_seed<S>.npz   per-patch metric vectors, all methods
  paired_tests.csv                    CSFR vs each baseline, per cell
  summary_ci.csv                      per-cell mean +/- 95% CI per method

Usage:
    python scripts/run_paired_stats.py
    python scripts/run_paired_stats.py --only-family CF1 --bootstrap 10000
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from csfr_reconstruct_2d import ssim2d  # noqa: E402

# import the sweep's baseline implementations directly so the statistical

# analysis cannot drift from the harness that produced the reported tables.
_spec = importlib.util.spec_from_file_location(
    "_sweep2d", PROJECT_ROOT / "scripts" / "run_csfr_sweep.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)

METHODS = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn", "csfr"]
BASELINES = [m for m in METHODS if m != "csfr"]
LABEL = {"zero_fill": "Zero-fill", "bilinear": "Linear", "inpaint_telea": "Telea",
         "inpaint_ns": "NS", "dictlearn": "Dict.", "csfr": "CSFR"}


# ---------------------------------------------------------------- metrics ---
def psnr_patch(ref: np.ndarray, rec: np.ndarray, idx: np.ndarray | None = None,
               max_db: float = 100.0) -> float:
    """PSNR in dB. ``idx`` restricts the MSE to a boolean pixel subset.

    ``idx=None`` reproduces the whole-patch definition used by the sweep harness
    (and therefore by every table in the manuscript). Passing the corrupted-pixel
    mask gives the stricter imputation-only variant reported alongside it.
    """
    d = (ref - rec) ** 2
    mse = float(d[idx].mean()) if idx is not None else float(d.mean())
    if mse < 1e-12:
        return max_db
    return min(max_db, 10.0 * math.log10(255.0 ** 2 / mse))


def hrp_patch(rec: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    """Hallucination Risk Proxy, matching ``csfr_reconstruct_2d.hrp`` exactly."""
    known = y[m == 1]
    if known.size == 0:
        return 0.0
    mu = float(known.mean())
    sd = max(float(known.std()), 1.0)
    imp = rec[m == 0]
    if imp.size == 0:
        return 0.0
    return float((np.abs(imp - mu) / sd).mean())


def per_patch_metrics(ref: np.ndarray, rec: np.ndarray, y: np.ndarray,
                      M: np.ndarray) -> dict[str, np.ndarray]:
    n = ref.shape[0]
    out = {k: np.empty(n, dtype=np.float64)
           for k in ("psnr", "psnr_masked", "ssim", "hrp")}
    for i in range(n):
        hole = M[i] == 0
        out["psnr"][i] = psnr_patch(ref[i], rec[i])
        out["psnr_masked"][i] = (psnr_patch(ref[i], rec[i], hole) if hole.any()
                                 else np.nan)
        out["ssim"][i] = ssim2d(ref[i], rec[i])
        out["hrp"][i] = hrp_patch(rec[i], y[i], M[i])
    return out


# ------------------------------------------------------------- inference ---
def cohens_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean difference in units of the difference SD."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1], computed from the rank-sum
    identity so the cost is O(n log n) rather than O(n^2)."""
    n_a, n_b = len(a), len(b)
    ranks = stats.rankdata(np.concatenate([a, b]))
    r_a = ranks[:n_a].sum()
    u_a = r_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u_a / (n_a * n_b) - 1.0)


def bootstrap_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling individual patches."""
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cluster_bootstrap_ci(x: np.ndarray, cluster: np.ndarray, n_boot: int,
                         rng: np.random.Generator, alpha: float = 0.05
                         ) -> tuple[float, float]:
    """Percentile bootstrap CI resampling whole *source images* with replacement.

    Patches cut from one source image share content and capture conditions, so
    they are not independent draws. Resampling clusters rather than patches
    propagates that dependence into the interval; on a corpus where a handful of
    images supply most of the patches, this interval is much wider than the
    patch-level one, and it is the honest one.
    """
    groups = [x[cluster == c] for c in np.unique(cluster)]
    k = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cluster_level_test(diff: np.ndarray, cluster: np.ndarray) -> tuple[float, int]:
    """Paired test at the source-image level: one mean difference per cluster.

    With only a handful of clusters this is a very low-powered test. That is the
    point -- it reports the evidence the design actually supports, rather than
    the evidence a false independence assumption would imply.
    """
    per_cluster = np.array([diff[cluster == c].mean() for c in np.unique(cluster)])
    if per_cluster.size < 2 or np.allclose(per_cluster, 0):
        return 1.0, int(per_cluster.size)
    return (float(stats.wilcoxon(per_cluster).pvalue), int(per_cluster.size))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, monotonicity enforced."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ------------------------------------------------------------------ main ---
def reconstruct_all(y: np.ndarray, M: np.ndarray, seed: int,
                    x_hat: np.ndarray) -> dict[str, np.ndarray]:
    import cv2
    return {
        "zero_fill": _sweep.baseline_zero(y, M),
        "bilinear": _sweep.baseline_bilinear(y, M),
        "inpaint_telea": _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_TELEA),
        "inpaint_ns": _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_NS),
        "dictlearn": _sweep.baseline_dictlearn(y, M, seed),
        "csfr": x_hat,
    }


def build_per_patch(job: tuple[str, str, int, str, str, str]) -> str:
    """Worker entry point: compute and persist one cell/seed's per-patch metrics.

    Takes plain strings so the job is cheap to pickle across processes.
    """
    fam, lvl, seed, seed_dir_s, npz_path_s, patches_s = job
    npz_path = Path(npz_path_s)
    ref = np.load(patches_s).astype(np.float32)
    seed_dir = Path(seed_dir_s)
    y = np.load(seed_dir / "y.npy")
    M = np.load(seed_dir / "mask.npy")
    x_hat = np.load(seed_dir / "x_hat.npy")
    store = {}
    for meth, rec in reconstruct_all(y, M, seed, x_hat).items():
        for key, vec in per_patch_metrics(ref, rec, y, M).items():
            store[f"{meth}/{key}"] = vec
    np.savez_compressed(npz_path, **store)
    return f"{fam}_{lvl}_seed{seed}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="results/csfr_sweep_2d_v2")
    ap.add_argument("--patches", default="data/patches.npy")
    ap.add_argument("--out-dir", default="results/paired_stats")
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel processes for the per-patch stage (1 = serial)")
    ap.add_argument("--cluster", default=None,
                    help="Per-patch source-image labels .npy (default: alongside "
                         "--patches as <stem>_cluster.npy). Enables cluster-robust "
                         "inference; omitted silently if the file is absent.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    sweep_dir = PROJECT_ROOT / args.sweep_dir
    out_dir = PROJECT_ROOT / args.out_dir
    (out_dir / "per_patch").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    patches_path = PROJECT_ROOT / args.patches
    cluster_path = (Path(args.cluster) if args.cluster
                    else patches_path.with_name(patches_path.stem + "_cluster.npy"))
    cluster = np.load(cluster_path) if cluster_path.exists() else None
    if cluster is None:
        print(f"[stats] no cluster labels at {cluster_path}; patch-level only")
    else:
        print(f"[stats] cluster labels: {len(np.unique(cluster))} source images, "
              f"largest share {np.bincount(cluster).max() / cluster.size:.1%}")

    cells = sorted(d for d in sweep_dir.iterdir()
                   if d.is_dir() and "_" in d.name
                   and (args.only_family is None or d.name.startswith(args.only_family)))

    units = []  # (fam, lvl, seed, seed_dir, npz_path)
    for cell in cells:
        fam, lvl = cell.name.split("_")
        for seed_dir in sorted(cell.glob("seed*")):
            seed = int(seed_dir.name.removeprefix("seed"))
            units.append((fam, lvl, seed, seed_dir,
                          out_dir / "per_patch" / f"{fam}_{lvl}_seed{seed}.npz"))

    # stage 1 (expensive, embarrassingly parallel): recompute baselines and

    # per-patch metrics. Cached per cell/seed, so reruns are near-free.
    todo = [(f, l, s, str(d), str(p), str(PROJECT_ROOT / args.patches))
            for f, l, s, d, p in units if not p.exists() or args.force]
    if todo:
        print(f"[stats] per-patch stage: {len(todo)} cell/seed units, "
              f"{args.workers} worker(s)", flush=True)
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for i, tag in enumerate(ex.map(build_per_patch, todo), 1):
                    print(f"[stats] {i}/{len(todo)} {tag}", flush=True)
        else:
            for i, job in enumerate(todo, 1):
                print(f"[stats] {i}/{len(todo)} {build_per_patch(job)}", flush=True)

    # stage 2 (cheap): inference over the cached per-patch vectors.

    test_rows: list[dict] = []
    ci_rows: list[dict] = []
    for fam, lvl, seed, _seed_dir, npz_path in units:
        store = dict(np.load(npz_path))
        for meth in METHODS:
            for key in ("psnr", "psnr_masked", "ssim", "hrp"):
                v = store[f"{meth}/{key}"]
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                lo, hi = bootstrap_ci(v, args.bootstrap, rng)
                ci_rows.append({
                    "family": fam, "level": lvl, "seed": seed,
                    "method": meth, "metric": key, "n": int(v.size),
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "ci_lo": lo, "ci_hi": hi})

        # paired: CSFR vs each baseline, matched patch by patch.

        for key in ("psnr", "psnr_masked", "ssim"):
            csfr_v = store[f"csfr/{key}"]
            for base in BASELINES:
                base_v = store[f"{base}/{key}"]
                ok = np.isfinite(csfr_v) & np.isfinite(base_v)
                a, b = csfr_v[ok], base_v[ok]
                if a.size < 3:
                    continue
                diff = a - b
                # Wilcoxon is the primary test (per-patch metrics are
                # skewed and bounded); the paired t is reported alongside.
                if np.allclose(diff, 0):
                    w_p, t_p = 1.0, 1.0
                else:
                    w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
                    t_p = float(stats.ttest_rel(a, b).pvalue)
                lo, hi = bootstrap_ci(diff, args.bootstrap, rng)
                row = {
                    "family": fam, "level": lvl, "seed": seed, "metric": key,
                    "baseline": base, "n": int(a.size),
                    "csfr_mean": float(a.mean()), "base_mean": float(b.mean()),
                    "mean_diff": float(diff.mean()),
                    "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "cohens_dz": cohens_dz(diff),
                    "cliffs_delta": cliffs_delta(a, b),
                    "p_wilcoxon": w_p, "p_ttest": t_p}
                if cluster is not None:
                    cl = cluster[ok]
                    c_lo, c_hi = cluster_bootstrap_ci(diff, cl, args.bootstrap, rng)
                    c_p, n_cl = cluster_level_test(diff, cl)
                    row.update({"n_clusters": n_cl,
                                "diff_ci_lo_cluster": c_lo,
                                "diff_ci_hi_cluster": c_hi,
                                "p_cluster_wilcoxon": c_p})
                test_rows.append(row)

    # stage 3: the primary analysis. Testing each seed separately reuses the

    # same 300 patches three times, which is pseudo-replication. Averaging each
    # patch's metric over the seeds first gives one paired observation per patch
    # per cell, with the seed-to-seed variation absorbed into that average.
    seedavg_rows: list[dict] = []
    by_cell: dict[tuple[str, str], list[Path]] = {}
    for fam, lvl, _seed, _sd, npz_path in units:
        by_cell.setdefault((fam, lvl), []).append(npz_path)

    for (fam, lvl), paths in sorted(by_cell.items()):
        stores = [dict(np.load(q)) for q in paths]
        for key in ("psnr", "psnr_masked", "ssim"):
            csfr_v = np.mean([s[f"csfr/{key}"] for s in stores], axis=0)
            for base in BASELINES:
                base_v = np.mean([s[f"{base}/{key}"] for s in stores], axis=0)
                ok = np.isfinite(csfr_v) & np.isfinite(base_v)
                a, b = csfr_v[ok], base_v[ok]
                if a.size < 3:
                    continue
                diff = a - b
                if np.allclose(diff, 0):
                    w_p = 1.0
                else:
                    w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
                lo, hi = bootstrap_ci(diff, args.bootstrap, rng)
                row = {"family": fam, "level": lvl, "metric": key, "baseline": base,
                       "n_seeds": len(stores), "n": int(a.size),
                       "csfr_mean": float(a.mean()), "base_mean": float(b.mean()),
                       "mean_diff": float(diff.mean()),
                       "diff_ci_lo": lo, "diff_ci_hi": hi,
                       "cohens_dz": cohens_dz(diff),
                       "cliffs_delta": cliffs_delta(a, b),
                       "p_wilcoxon": w_p}
                if cluster is not None:
                    cl = cluster[ok]
                    c_lo, c_hi = cluster_bootstrap_ci(diff, cl, args.bootstrap, rng)
                    c_p, n_cl = cluster_level_test(diff, cl)
                    row.update({"n_clusters": n_cl, "diff_ci_lo_cluster": c_lo,
                                "diff_ci_hi_cluster": c_hi, "p_cluster_wilcoxon": c_p})
                seedavg_rows.append(row)

    for key in ("psnr", "psnr_masked", "ssim"):
        idx = [i for i, r in enumerate(seedavg_rows) if r["metric"] == key]
        adj = holm_bonferroni([seedavg_rows[i]["p_wilcoxon"] for i in idx])
        for i, q in zip(idx, adj):
            seedavg_rows[i]["p_wilcoxon_holm"] = q
        if cluster is not None:
            adj_c = holm_bonferroni([seedavg_rows[i]["p_cluster_wilcoxon"] for i in idx])
            for i, q in zip(idx, adj_c):
                seedavg_rows[i]["p_cluster_holm"] = q

    # multiplicity correction within each metric family of comparisons.

    for key in ("psnr", "psnr_masked", "ssim"):
        idx = [i for i, r in enumerate(test_rows) if r["metric"] == key]
        for name, src in (("p_wilcoxon_holm", "p_wilcoxon"), ("p_ttest_holm", "p_ttest")):
            adj = holm_bonferroni([test_rows[i][src] for i in idx])
            for i, p in zip(idx, adj):
                test_rows[i][name] = p

    def dump(rows: list[dict], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[stats] wrote {path} ({len(rows)} rows)")

    dump(seedavg_rows, out_dir / "paired_tests_seedavg.csv")
    dump(test_rows, out_dir / "paired_tests_perseed.csv")
    dump(ci_rows, out_dir / "summary_ci.csv")
    (out_dir / "analysis_plan.json").write_text(json.dumps({
        "unit_of_analysis": "image patch (matched across methods by index)",
        "primary_test": "Wilcoxon signed-rank, two-sided",
        "secondary_test": "paired t-test, two-sided",
        "multiplicity": "Holm-Bonferroni within each metric across all "
                        "(family, level, seed, baseline) comparisons",
        "ci": f"percentile bootstrap, {args.bootstrap} resamples, 95%",
        "effect_sizes": ["Cohen's d_z (paired)", "Cliff's delta"],
        "clustering": (
            "Patches are nested within source images and are not independent. "
            "Where cluster labels are available the analysis additionally reports "
            "(a) a cluster bootstrap CI resampling whole source images, and "
            "(b) a Wilcoxon test on per-source-image mean differences. The "
            "cluster-level test is deliberately low-powered: it reports the "
            "evidence the sampling design supports."
            if cluster is not None else "no cluster labels available"),
        "psnr_definitions": {
            "psnr": "whole-patch MSE (definition used by the sweep harness and "
                    "all manuscript tables)",
            "psnr_masked": "MSE restricted to corrupted pixels only"},
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
