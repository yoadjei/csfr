"""Corruption masks (round 2): realised fractions, geometry of the four families,
bit-identity with the released harness, and the severity table the paper prints."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from masks import make_mask, SEVERITY, describe_families, realised_fraction  # noqa: E402


def test_exact_fractions_for_cf1_cf3_cf4():
    rng = np.random.default_rng(0)
    for fam in ["CF1", "CF3", "CF4"]:
        M = make_mask((5, 64, 64), fam, 0.30, rng)
        assert np.allclose(realised_fraction(M), 0.30, atol=0.01)


def test_bursts_reach_the_target_and_never_undershoot():
    M = make_mask((5, 64, 64), "CF2", 0.30, np.random.default_rng(0))
    assert np.all(realised_fraction(M) >= 0.30)


def test_half_image_geometry():
    rng = np.random.default_rng(0)
    top = make_mask((1, 64, 64), "CF3", 0.30, rng)[0]
    bottom = make_mask((1, 64, 64), "CF4", 0.30, rng)[0]
    assert np.all(top[:19, :] == 0) and np.all(top[19:, :] == 1)
    assert np.all(bottom[45:, :] == 0) and np.all(bottom[:45, :] == 1)


def test_identical_to_the_released_harness():
    spec = importlib.util.spec_from_file_location("s", ROOT / "scripts" / "run_csfr_sweep.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    for fam in ["CF1", "CF2", "CF3", "CF4"]:
        a = make_mask((3, 64, 64), fam, 0.15, np.random.default_rng(7))
        b = s.make_mask((3, 64, 64), fam, 0.15, np.random.default_rng(7))
        assert np.array_equal(a, b)


def test_reproduces_a_released_mask():
    d = ROOT / "results" / "csfr_sweep_2d_v2" / "CF2_L3" / "seed1" / "mask.npy"
    if not d.exists():
        return
    M = make_mask((300, 64, 64), "CF2", 0.30, np.random.default_rng(1))
    assert np.array_equal(M, np.load(d))


def test_severity_table_is_complete():
    assert SEVERITY == {"L1": 0.05, "L2": 0.15, "L3": 0.30, "L4": 0.50, "L5": 0.70}
    fams = describe_families()
    assert set(fams) == {"CF1", "CF2", "CF3", "CF4"}
    assert "4" in fams["CF2"] and "31" in fams["CF2"]


def _block_mask():
    M = np.ones((1, 32, 32), np.float32)
    M[0, 10:20, 10:20] = 0
    return M


def test_erosion_marks_observed_pixels_next_to_the_hole_missing():
    from masks import perturb_mask
    M = _block_mask()
    E1 = perturb_mask(M, "erode", 1, np.random.default_rng(0))
    assert ((E1 == 1) <= (M == 1)).all()                       # only false negatives
    assert (M - E1).sum() == 12 * 12 - 10 * 10                 # a one-pixel ring
    assert E1[0, 0, 0] == 1                                     # the patch border is not a hole
    E2 = perturb_mask(M, "erode", 2, np.random.default_rng(0))
    assert (M - E2).sum() == 14 * 14 - 10 * 10


def test_dilation_marks_hole_pixels_next_to_the_boundary_observed():
    from masks import perturb_mask
    M = _block_mask()
    D1 = perturb_mask(M, "dilate", 1, np.random.default_rng(0))
    assert ((D1 == 1) >= (M == 1)).all()                       # only false positives
    assert (D1 - M).sum() == 10 * 10 - 8 * 8


def test_random_flips_change_the_stated_fraction():
    from masks import perturb_mask
    M = np.ones((3, 64, 64), np.float32)
    F = perturb_mask(M, "flip", 0.05, np.random.default_rng(0))
    changed = (F != M).reshape(3, -1).sum(1)
    assert (changed == round(0.05 * 64 * 64)).all()
    assert np.array_equal(perturb_mask(M, "flip", 0.05, np.random.default_rng(0)), F)


def test_contiguous_families_draw_nothing_at_random():
    """The paper says CF3 and CF4 seeds share one mask, and CF1 and CF2 seeds do not."""
    for fam in ["CF3", "CF4"]:
        for frac in SEVERITY.values():
            a = make_mask((4, 64, 64), fam, frac, np.random.default_rng(0))
            b = make_mask((4, 64, 64), fam, frac, np.random.default_rng(12345))
            assert np.array_equal(a, b)
    for fam in ["CF1", "CF2"]:
        a = make_mask((4, 64, 64), fam, 0.15, np.random.default_rng(0))
        b = make_mask((4, 64, 64), fam, 0.15, np.random.default_rng(1))
        assert not np.array_equal(a, b)
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    assert "CF3 and CF4 erase a fixed band of rows and draw nothing at random" in tex
