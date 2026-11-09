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
