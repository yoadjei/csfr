"""Range figure of the ablation: only imputed pixels count, the tolerance is
the solver's, and every CSFR arm at the ablation seeds is covered."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("ablation_range", ROOT / "scripts" / "ablation_range.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_range_row_counts_imputed_pixels_only():
    m = _mod()
    M = np.ones((2, 4, 4), np.float32)
    M[:, :2] = 0                                   # 8 imputed pixels per patch
    x = np.full((2, 4, 4), 100.0)
    x[0, 0, 0] = 255.0 + 3.0                       # imputed, out of range
    x[0, 3, 3] = -50.0                             # observed: not counted
    x[1, 1, 1] = 255.0 + 1e-7                      # within tolerance
    r = m.range_row(x, M)
    assert r["n_imputed"] == 16 and r["frac_pixels_out"] == 1 / 16
    assert r["frac_patches_out"] == 0.5 and abs(r["max_excess"] - 3.0) < 1e-12


def test_every_arm_at_the_ablation_seeds(tmp_path):
    m = _mod()
    cfg = {"families": ["CF1"], "levels": {"L3": 0.3}, "ablation_seeds": [0],
           "csfr_variants": ["csfr", "csfr_tv_only"], "ablation_variants": ["csfr_no_clip"]}
    (tmp_path / "run_metadata.json").write_text(json.dumps({"config": cfg}))
    d = tmp_path / "CF1_L3" / "seed0"
    d.mkdir(parents=True)
    M = np.zeros((1, 4, 4), np.float32)
    np.save(d / "mask.npy", M)
    for arm, v in [("csfr", 10.0), ("csfr_tv_only", 10.0), ("csfr_no_clip", 300.0)]:
        np.save(d / f"x_hat_{arm}.npy", np.full((1, 4, 4), v))
    assert m.main(["--sweep-dir", str(tmp_path)]) == 0
    rows = {r["method"]: r for r in csv.DictReader((tmp_path / "stats" / "range_violations.csv").open())}
    assert set(rows) == {"csfr", "csfr_tv_only", "csfr_no_clip"}
    assert float(rows["csfr"]["frac_pixels_out"]) == 0.0
    assert float(rows["csfr_no_clip"]["frac_pixels_out"]) == 1.0
    assert float(rows["csfr_no_clip"]["max_excess"]) == 45.0
