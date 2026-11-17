"""cap_mechanism: a cell's difference to its strongest method splits exactly into
the patches that method reconstructs to the cap and the rest, is the reported
difference, and the capped runs are described by the true hole and CSFR's error."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("cap_mechanism", ROOT / "scripts" / "cap_mechanism.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# four 4 x 4 patches, the top two rows erased; holes: 0 flat 255, 1 textured, 2 flat 100, 3 textured
TRUTH = np.stack([np.full((4, 4), 255.0), np.arange(16.0).reshape(4, 4) * 10,
                  np.full((4, 4), 100.0), np.arange(16.0).reshape(4, 4)[::-1] * 5])
MASK = np.ones((4, 4, 4), np.float32)
MASK[:, :2, :] = 0
CSFR_PSNR = {0: [60.0, 60.0, 50.0, 28.0], 1: [60.0, 25.0, 50.0, 28.0]}
STRONG_PSNR = {0: [100.0, 30.0, 40.0, 30.0], 1: [100.0, 30.0, 100.0, 30.0]}   # patch 2 capped at seed 1 only


def _npz(path, psnr):
    psnr = np.asarray(psnr)
    np.savez(path, psnr_hole=psnr, ssim_hole=np.zeros_like(psnr), gmsd_hole=np.zeros_like(psnr),
             exact=psnr >= 100)


def _sweep(tmp_path, reported_diff=None):
    sweep = tmp_path / "results" / "sweep_x"
    x_hat = TRUTH.copy()
    x_hat[0, :2, :] -= 0.2                                     # flat hole, 0.2 grey levels dark everywhere
    x_hat[2, 0, 0] += 3.0                                       # flat hole, one pixel 3 grey levels off
    for s in (0, 1):
        d = sweep / "CF3_L1" / f"seed{s}"
        d.mkdir(parents=True)
        np.save(d / "mask.npy", MASK)
        np.save(d / "x_hat_csfr.npy", x_hat.astype(np.float32))
        _npz(d / "per_patch_csfr.npz", CSFR_PSNR[s])
        _npz(d / "per_patch_criminisi.npz", STRONG_PSNR[s])
        _npz(d / "per_patch_csfr_tv_only.npz", [100.0, 20.0, 45.0, 20.0])   # exact on patch 0 only
    np.save(tmp_path / "truth.npy", TRUTH)
    (sweep / "run_metadata.json").write_text(json.dumps({"config": {
        "patches_path": str(tmp_path / "truth.npy"), "seeds": [0, 1], "families": ["CF3"], "levels": {"L1": 0.5}}}))
    a = np.mean([CSFR_PSNR[0], CSFR_PSNR[1]], axis=0)
    b = np.mean([STRONG_PSNR[0], STRONG_PSNR[1]], axis=0)
    diff = float((a - b).mean()) if reported_diff is None else reported_diff
    (sweep / "stats").mkdir()
    with (sweep / "stats" / "paired_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family", "level", "metric", "method", "diff_mean"])
        w.writeheader()
        w.writerow({"family": "CF3", "level": "L1", "metric": "psnr_hole", "method": "criminisi", "diff_mean": diff})
        w.writerow({"family": "CF3", "level": "L1", "metric": "psnr_hole", "method": "fsr", "diff_mean": 5.0})
    return sweep, a - b


def test_the_difference_splits_into_capped_patches_and_the_rest(tmp_path):
    m = _mod()
    sweep, d = _sweep(tmp_path)
    (row,) = m.sweep_rows(sweep)
    assert row["strongest"] == "criminisi"
    assert row["diff_mean"] == pytest.approx(d.mean())
    capped = np.array([True, False, True, False])               # capped at one seed or more
    assert row["capped_part"] == pytest.approx(d[capped].sum() / 4)
    assert row["rest_part"] == pytest.approx(d[~capped].sum() / 4)
    assert row["capped_part"] + row["rest_part"] == pytest.approx(row["diff_mean"])
    assert (row["n_patches_capped"], row["n_runs_capped"], row["n_runs_capped_flat"]) == (2, 3, 3)
    # patch 0 at both seeds is within 0.2 grey levels; patch 2 at seed 1 is 3 off
    assert (row["n_runs_capped_csfr_within_1"], row["n_runs_capped_flat_csfr_within_1"]) == (2, 2)
    assert row["n_runs_capped_flat_tv_only_at_cap"] == 2                 # patch 0 at both seeds
    assert row["csfr_flat_mean_err_median"] == pytest.approx(-0.2, abs=1e-5)
    assert row["csfr_flat_max_abs_err_median"] == pytest.approx(0.2, abs=1e-5)
    assert row["csfr_psnr_capped_median"] == pytest.approx(60.0)


def test_a_difference_that_is_not_the_reported_one_is_refused(tmp_path):
    m = _mod()
    sweep, _ = _sweep(tmp_path, reported_diff=-1.0)
    with pytest.raises(ValueError, match="is not the reported"):
        m.sweep_rows(sweep)


def test_main_writes_one_row_per_cell(tmp_path):
    m = _mod()
    _sweep(tmp_path)
    assert m.main(["--sweeps", "sweep_x", "--results", str(tmp_path / "results"), "--out", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader((tmp_path / "out" / "sweep_x.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["strongest"] == "criminisi"
