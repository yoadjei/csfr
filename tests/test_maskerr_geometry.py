"""The mask geometry of the mask-error experiment (E10): what a wrong mask does
to the observations, on masks small enough to count by hand."""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from masks import perturb_mask  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("maskerr_geometry", ROOT / "scripts" / "maskerr_geometry.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block_hole():
    M = np.ones((1, 8, 8), np.uint8)
    M[0, 2:6, 2:6] = 0                       # a 4 x 4 hole: 16 pixels, 48 observed
    return M


def test_dilation_borders_the_remaining_hole_with_false_positives_only():
    m = _mod()
    M = _block_hole()
    Mw = perturb_mask(M, "dilate", 1, np.random.default_rng(0))
    y = (np.full(M.shape, 200.0) * M * Mw).astype(np.float32)
    g = m.geometry(M, Mw, y)
    assert g["fp_share_of_hole"] == 12 / 16           # the ring of the hole; its 2 x 2 centre stays missing
    assert g["fn_share_of_observed"] == 0.0
    assert g["border_fp_share"] == 1.0
    assert g["fp_value_max"] == 0.0                   # a false positive shows the corrupted value
    assert (g["n_none_observed"], g["n_hole_all_observed"]) == (0, 0)


def test_erosion_counts_false_negatives_and_empty_masks():
    m = _mod()
    M = _block_hole()
    Mw = perturb_mask(M, "erode", 1, np.random.default_rng(0))
    g = m.geometry(M, Mw, (M * Mw).astype(np.float32))
    assert g["fp_share_of_hole"] == 0.0
    assert g["fn_share_of_observed"] == 20 / 48       # the 6 x 6 ring around the hole
    assert g["border_fp_share"] == 0.0                # the border of the grown hole is truly observed
    assert np.isnan(g["fp_value_max"])
    none = np.zeros_like(M)
    g = m.geometry(M, none, (M * none).astype(np.float32))
    assert g["n_none_observed"] == 1 and g["fn_share_of_observed"] == 1.0 and np.isnan(g["border_fp_share"])


def test_a_mask_that_marks_the_whole_hole_observed_is_counted():
    m = _mod()
    M = _block_hole()
    Mw = np.ones_like(M)
    g = m.geometry(M, Mw, (M * Mw).astype(np.float32))
    assert g["n_hole_all_observed"] == 1 and g["fp_share_of_hole"] == 1.0 and np.isnan(g["border_fp_share"])


def test_driver_writes_one_row_per_case_with_the_floor(tmp_path):
    m = _mod()
    X = np.full((2, 8, 8), 100.0, np.float32)
    np.save(tmp_path / "patches.npy", X)
    M = np.repeat(_block_hole(), 2, axis=0)
    for pert, Mw in (("exact", M), ("dilate1", perturb_mask(M, "dilate", 1, np.random.default_rng(0)))):
        d = tmp_path / "cases" / "CF3_L3" / pert
        d.mkdir(parents=True)
        np.save(d / "mask.npy", Mw)
        np.save(d / "y.npy", (X * M * Mw).astype(np.float32))
    assert m.main(["--patches", str(tmp_path / "patches.npy"), "--root", str(tmp_path)]) == 0
    rows = list(csv.DictReader((tmp_path / "geometry.csv").open(encoding="utf-8")))
    assert [(r["cell"], r["perturbation"]) for r in rows] == [("CF3_L3", "dilate1"), ("CF3_L3", "exact")]
    floor = 10 * np.log10(255 ** 2 / 100 ** 2)        # zero fill of a hole whose truth is 100
    assert all(abs(float(r["zero_fill_psnr_hole"]) - floor) < 1e-9 for r in rows)
    assert float(rows[0]["border_fp_share"]) == 1.0
    assert float(rows[1]["fp_share_of_hole"]) == 0.0 and float(rows[1]["border_fp_share"]) == 0.0
