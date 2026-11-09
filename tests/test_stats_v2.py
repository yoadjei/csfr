"""Round-2 paired statistics: seed averaging, sign convention (positive favours
CSFR for every metric, GMSD included), Holm adjustment, per-image statistics,
the strongest-baseline choice, the ablation family, and the analysis plan."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["psnr_hole", "ssim_hole", "gmsd_hole", "gmsd", "hrp", "psnr_whole", "mse_hole"]


def _mod():
    spec = importlib.util.spec_from_file_location("stats_v2", ROOT / "scripts" / "run_stats_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pp(psnr, ssim, gmsd):
    n = len(psnr)
    d = {k: np.zeros(n) for k in METRICS}
    d.update(psnr_hole=np.asarray(psnr, float), ssim_hole=np.asarray(ssim, float),
             gmsd_hole=np.asarray(gmsd, float), exact=np.zeros(n, bool), hole_frac=np.full(n, 0.3))
    return d


def _sweep(tmp_path, bilinear_at_cap=0):
    """Toy sweep; bilinear_at_cap puts the first patches of bilinear at the
    hole-PSNR cap in every seed."""
    n = 40
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(8), 5)
    np.save(tmp_path / "clusters.npy", clusters)
    cfg = {"experiment": "toy", "corpus": "toy", "cluster_path": str(tmp_path / "clusters.npy"),
           "seeds": [0, 1], "families": ["CF1"], "levels": {"L3": 0.3},
           "csfr_variants": ["csfr", "csfr_tv_only"], "ablation_variants": ["csfr_no_fc"],
           "ablation_seeds": [0], "baselines": ["zero_fill", "bilinear", "inpaint_ns"],
           "external": ["lama"]}
    out = tmp_path / "sweep"
    base = 30 + rng.normal(size=n)
    for s in cfg["seeds"]:
        cell = out / "CF1_L3" / f"seed{s}"
        cell.mkdir(parents=True)
        jitter = 0.01 * s
        np.savez(cell / "per_patch_csfr.npz", **_pp(base + 2 + jitter, np.full(n, 0.8), np.full(n, 0.10)))
        np.savez(cell / "per_patch_csfr_tv_only.npz", **_pp(base + 1, np.full(n, 0.7), np.full(n, 0.12)))
        bil = base + 1.5
        bil[:bilinear_at_cap] = 100.0
        np.savez(cell / "per_patch_bilinear.npz", **_pp(bil, np.full(n, 0.75), np.full(n, 0.11)))
        np.savez(cell / "per_patch_inpaint_ns.npz", **_pp(base, np.full(n, 0.6), np.full(n, 0.20)))
        np.savez(cell / "per_patch_zero_fill.npz", **_pp(base - 20, np.full(n, 0.1), np.full(n, 0.5)))
        if s == 0:
            np.savez(cell / "per_patch_csfr_no_fc.npz", **_pp(base + 1.9, np.full(n, 0.8), np.full(n, 0.10)))
    (out / "run_metadata.json").write_text(json.dumps({"config": cfg}))
    return out


def _rows(path):
    return list(csv.DictReader(path.open()))


def test_paired_tests_sign_holm_and_family(tmp_path):
    out = _sweep(tmp_path)
    assert _mod().main(["--sweep-dir", str(out), "--bootstrap", "500"]) == 0
    rows = _rows(out / "stats" / "paired_tests.csv")
    methods = {r["method"] for r in rows}
    # zero-fill is the floor, not a baseline; ablation arms are a separate family;
    # lama has no outputs yet
    assert methods == {"csfr_tv_only", "bilinear", "inpaint_ns"}
    for r in rows:
        assert float(r["diff_mean"]) > 0, (r["metric"], r["method"])   # CSFR better on every metric
        assert float(r["p_holm"]) >= float(r["p_wilcoxon"]) - 1e-15
        assert int(r["n"]) == 40 and int(r["n_images"]) == 8
    g = next(r for r in rows if r["metric"] == "gmsd_hole" and r["method"] == "inpaint_ns")
    assert abs(float(g["diff_mean"]) - 0.10) < 1e-12        # method minus CSFR for lower-is-better
    p = next(r for r in rows if r["metric"] == "psnr_hole" and r["method"] == "bilinear")
    assert abs(float(p["diff_mean"]) - 0.505) < 1e-9        # seed-averaged: 2.005 - 1.5
    assert float(p["frac_images_csfr_leads"]) == 1.0


def test_strongest_baseline_and_ablation(tmp_path):
    out = _sweep(tmp_path)
    _mod().main(["--sweep-dir", str(out), "--bootstrap", "500"])
    sb = _rows(out / "stats" / "strongest_baseline.csv")
    psnr = next(r for r in sb if r["metric"] == "psnr_hole")
    assert psnr["method"] == "bilinear"
    assert psnr["cluster_ci_excludes_zero"] == "True" and psnr["lama_included"] == "False"
    ab = _rows(out / "stats" / "paired_tests_ablation.csv")
    # every arm that removes a part of CSFR, TV-only included, at the ablation seed only
    assert {r["method"] for r in ab} == {"csfr_tv_only", "csfr_no_fc"}
    assert all(int(r["n_seeds"]) == 1 for r in ab)
    plan = json.loads((out / "stats" / "analysis_plan.json").read_text())
    assert plan["ablation_family"] == ["csfr_tv_only", "csfr_no_fc"]


def test_cap_sensitivity_repeats_hole_psnr_on_the_cap_free_patches(tmp_path):
    out = _sweep(tmp_path, bilinear_at_cap=4)
    _mod().main(["--sweep-dir", str(out), "--bootstrap", "500"])
    # four capped patches lift bilinear above CSFR in the primary analysis ...
    sb = next(r for r in _rows(out / "stats" / "strongest_baseline.csv") if r["metric"] == "psnr_hole")
    assert sb["method"] == "bilinear" and float(sb["diff_mean"]) < 0
    # ... and on the 36 patches no method caps, CSFR leads it by the true margin
    assert sb["method_capfree"] == "bilinear" and int(sb["n_capfree"]) == 36
    assert abs(float(sb["diff_mean_capfree"]) - 0.505) < 1e-9
    assert sb["verdict_capfree"] == "csfr_better"
    p = next(r for r in _rows(out / "stats" / "paired_tests.csv")
             if r["metric"] == "psnr_hole" and r["method"] == "bilinear")
    assert int(p["n_cap_method"]) == 4 and int(p["n_cap_csfr"]) == 0
    s = next(r for r in _rows(out / "stats" / "summary_ci.csv")
             if r["metric"] == "psnr_hole" and r["method"] == "bilinear")
    assert int(s["n_at_cap"]) == 4 and int(s["n_capfree"]) == 36
    # SSIM and GMSD are not capped and carry no cap-free columns
    g = next(r for r in _rows(out / "stats" / "paired_tests.csv") if r["metric"] == "gmsd_hole")
    assert g["diff_mean_capfree"] == ""


def test_without_capped_patches_the_sensitivity_equals_the_primary(tmp_path):
    out = _sweep(tmp_path)
    _mod().main(["--sweep-dir", str(out), "--bootstrap", "500"])
    for r in _rows(out / "stats" / "paired_tests.csv"):
        if r["metric"] == "psnr_hole":
            assert int(r["n_capfree"]) == int(r["n"])
            assert float(r["diff_mean_capfree"]) == float(r["diff_mean"])


def test_analysis_plan_and_summary(tmp_path):
    out = _sweep(tmp_path)
    _mod().main(["--sweep-dir", str(out), "--bootstrap", "500"])
    plan = json.loads((out / "stats" / "analysis_plan.json").read_text())
    for k in ["unit", "test", "correction", "bootstrap_resamples", "rng", "sign_convention",
              "excluded_from_comparisons", "metrics"]:
        assert k in plan
    summ = _rows(out / "stats" / "summary_ci.csv")
    assert {r["method"] for r in summ} >= {"csfr", "zero_fill", "bilinear"}
    for r in summ:
        assert float(r["ci_patch_lo"]) <= float(r["mean"]) <= float(r["ci_patch_hi"])
