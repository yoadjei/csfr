"""ablation_mechanism: the difference between CSFR and an ablation arm splits
exactly into the patches whose true hole is flat and the rest, is the reported
difference, and the arms and seeds are the ones the stats record."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod(name="ablation_mechanism"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# four 4 x 4 patches, the top two rows erased; holes: 0 flat 255, 1 textured, 2 flat 100, 3 textured
TRUTH = np.stack([np.full((4, 4), 255.0), np.arange(16.0).reshape(4, 4) * 10,
                  np.full((4, 4), 100.0), np.arange(16.0).reshape(4, 4)[::-1] * 5])
MASK = np.ones((4, 4, 4), np.float32)
MASK[:, :2, :] = 0
CSFR = [60.0, 25.0, 50.0, 28.0]
ARMS = {"csfr_tv_only": [100.0, 23.0, 51.0, 27.0],    # exact on flat patch 0; d = -40, +2, -1, +1
        "csfr_no_fc": [61.0, 25.0, 50.0, 28.0]}       # d = -1, 0, 0, 0


def _npz(path, psnr):
    psnr = np.asarray(psnr)
    np.savez(path, psnr_hole=psnr, ssim_hole=np.zeros_like(psnr), gmsd_hole=np.zeros_like(psnr),
             exact=psnr >= 100)


def _sweep(tmp_path, reported=None):
    sweep = tmp_path / "results" / "sweep_x"
    d = sweep / "CF3_L1" / "seed0"
    d.mkdir(parents=True)
    np.save(d / "mask.npy", MASK)
    _npz(d / "per_patch_csfr.npz", CSFR)
    for arm, psnr in ARMS.items():
        _npz(d / f"per_patch_{arm}.npz", psnr)
    np.save(tmp_path / "truth.npy", TRUTH)
    (sweep / "run_metadata.json").write_text(json.dumps({"config": {
        "patches_path": str(tmp_path / "truth.npy"), "seeds": [0, 1, 2], "families": ["CF3"], "levels": {"L1": 0.5}}}))
    (sweep / "stats").mkdir()
    (sweep / "stats" / "analysis_plan.json").write_text(json.dumps({
        "ablation_family": list(ARMS), "ablation_seeds": [0]}))
    reported = reported or {}
    with (sweep / "stats" / "paired_tests_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family", "level", "metric", "method", "diff_mean", "p_holm"])
        w.writeheader()
        for arm, psnr in ARMS.items():
            diff = reported.get(arm, float(np.mean(np.subtract(CSFR, psnr))))
            w.writerow({"family": "CF3", "level": "L1", "metric": "psnr_hole", "method": arm,
                        "diff_mean": diff, "p_holm": 0.01})
            w.writerow({"family": "CF3", "level": "L1", "metric": "ssim_hole", "method": arm,
                        "diff_mean": 9.0, "p_holm": 0.5})
    return sweep


def test_flat_holes_need_a_hole_and_one_value():
    flat = _mod("cap_mechanism").flat_holes
    mask = MASK.copy()
    mask[3] = 1                                                   # patch 3 has no hole
    assert flat(TRUTH, mask).tolist() == [True, False, True, False]


def test_the_difference_splits_into_flat_holes_and_the_rest(tmp_path):
    m = _mod()
    rows = {r["method"]: r for r in m.sweep_rows(_sweep(tmp_path))}
    assert set(rows) == set(ARMS)                                 # the arms the stats record
    tv = rows["csfr_tv_only"]
    assert (tv["n_patches"], tv["n_seeds"], tv["n_flat"]) == (4, 1, 2)
    assert tv["diff_mean"] == pytest.approx(-38 / 4)
    assert tv["flat_part"] == pytest.approx(-41 / 4)
    assert tv["rest_part"] == pytest.approx(3 / 4)
    assert tv["flat_part"] + tv["rest_part"] == pytest.approx(tv["diff_mean"])
    assert tv["diff_median"] == pytest.approx(0.0)
    assert tv["frac_patches_csfr_ahead"] == pytest.approx(0.5)
    assert tv["n_flat_arm_at_cap"] == 1
    assert tv["p_holm"] == pytest.approx(0.01)
    assert rows["csfr_no_fc"]["frac_patches_csfr_ahead"] == 0.0   # a tie is not ahead


def test_a_difference_that_is_not_the_reported_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not the reported"):
        _mod().sweep_rows(_sweep(tmp_path, reported={"csfr_no_fc": 3.0}))


def test_main_writes_one_row_per_arm_and_cell(tmp_path):
    m = _mod()
    _sweep(tmp_path)
    assert m.main(["--sweep", str(tmp_path / "results" / "sweep_x"), "--out", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader((tmp_path / "out" / "sweep_x.csv").open(encoding="utf-8")))
    assert [(r["method"], r["family"], r["level"]) for r in rows] == [
        ("csfr_tv_only", "CF3", "L1"), ("csfr_no_fc", "CF3", "L1")]
