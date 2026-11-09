"""Bounded-Unknown evaluation (E14): predictors computed from the mask alone,
AUROC pooled, per family and within cell, a cross-validated reliability
analysis, and the pre-registered keep-or-demote rule applied mechanically."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("bu", ROOT / "scripts" / "run_bu_evaluation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_predictors_on_a_known_mask():
    m = _mod()
    M = np.ones((16, 16), np.float32)
    M[4:12, 4:12] = 0                                       # 8x8 block
    p = m.predictors(M)
    assert abs(p["hole_frac"] - 64 / 256) < 1e-12
    assert abs(p["isolated_missing_frac"] - 36 / 64) < 1e-12   # inner 6x6 has no observed 4-neighbour
    assert 1.0 < p["mean_hole_distance"] < 3.0


def _sweep(tmp, rule_holds: bool):
    rng = np.random.default_rng(0)
    cfg = {"families": ["CF1", "CF2"], "levels": {"L1": 0.05, "L3": 0.3}, "seeds": [0]}
    out = tmp / ("good" if rule_holds else "bad")
    for fam in cfg["families"]:
        for lvl, frac in cfg["levels"].items():
            d = out / f"{fam}_{lvl}" / "seed0"
            d.mkdir(parents=True)
            n = 60
            M = np.ones((n, 16, 16), np.float32)
            sizes = rng.integers(1, 12, n)
            for i, s in enumerate(sizes):
                M[i, 2:2 + s, 2:2 + s] = 0
            np.save(d / "mask.npy", M)
            psnr = 40 - 2.0 * sizes + rng.normal(0, 1, n) if rule_holds else rng.uniform(10, 40, n)
            np.savez(d / "per_patch_csfr.npz", psnr_hole=psnr)
    (out / "run_metadata.json").write_text(json.dumps({"config": {**cfg, "experiment": out.name}}))
    return out


def test_rule_keeps_an_informative_flag_and_demotes_a_useless_one(tmp_path):
    m = _mod()
    for holds in (True, False):
        sweep = _sweep(tmp_path, holds)
        out = tmp_path / f"out_{holds}"
        assert m.main(["--sweeps", str(sweep), "--out", str(out)]) == 0
        rows = list(csv.DictReader((out / "auroc.csv").open()))
        assert {r["scope"] for r in rows} >= {"pooled", "family", "cell"}
        dec = json.loads((out / "decision.json").read_text())
        assert all(v["keep_as_contribution"] is holds for v in dec["by_threshold"].values()), dec
    cal = list(csv.DictReader((out / "calibration.csv").open()))
    assert cal and all(0.0 <= float(r["mean_predicted"]) <= 1.0 for r in cal)
