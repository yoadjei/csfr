"""Mask-error robustness (E10): the exact mask is the zero of the degradation,
dilation produces only false positives and erosion only false negatives, and
every method is scored over the true hole."""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_driver(tmp_path):
    spec = importlib.util.spec_from_file_location("mask_rob", ROOT / "scripts" / "run_mask_robustness.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main(["--patches", str(ROOT / "data" / "patches.npy"), "--n", "4", "--cells", "CF2_L3",
                   "--perturbations", "erode1", "dilate1", "--methods", "csfr", "bilinear",
                   "--workers", "2", "--bootstrap", "200", "--out", str(tmp_path)]) == 0
    rows = list(csv.DictReader((tmp_path / "robustness.csv").open()))
    assert {(r["perturbation"], r["method"]) for r in rows} == {
        (p, mth) for p in ("exact", "erode1", "dilate1") for mth in ("csfr", "bilinear")}
    for r in rows:
        if r["perturbation"] == "exact":
            assert float(r["degradation_mean"]) == 0.0
            assert float(r["false_pos_frac"]) == 0.0 == float(r["false_neg_frac"])
        if r["perturbation"] == "dilate1":
            assert float(r["false_pos_frac"]) > 0 and float(r["false_neg_frac"]) == 0
            assert float(r["degradation_mean"]) < 0      # zeros inside the true hole cost quality
        if r["perturbation"] == "erode1":
            assert float(r["false_neg_frac"]) > 0 and float(r["false_pos_frac"]) == 0
    assert (tmp_path / "cases" / "CF2_L3" / "dilate1" / "y.npy").exists()
