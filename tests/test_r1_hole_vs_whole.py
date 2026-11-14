"""scripts/r1_hole_vs_whole.py: the change in a paired PSNR difference between
whole-patch and hole-only scoring splits exactly into capped patches, the rest,
and patches without a hole; with the observed pixels reproduced exactly and no
cap, the paired difference does not change."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("r1_hole_vs_whole", ROOT / "scripts" / "r1_hole_vs_whole.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _scores(m, ref, rec, M):
    """Whole-patch and hole-only PSNR per patch, as the round-1 harness scored them."""
    w = np.array([m.psnr(ref[i], rec[i]) for i in range(len(ref))])
    h = np.array([m.psnr(ref[i], rec[i], M[i] == 0) if (M[i] == 0).any() else np.nan for i in range(len(ref))])
    return w, h


def _case(seed=0, n=6, hole_rows=3):
    """float64, so the identities hold to rounding of 1e-12 (round 1 scored float32
    arrays, which leaves residuals of about 1e-7 dB)."""
    rng = np.random.default_rng(seed)
    ref = rng.integers(0, 256, (n, 16, 16)).astype(np.float64)
    M = np.ones_like(ref)
    M[:, -hole_rows:, :] = 0
    return ref, M


def test_without_the_cap_the_paired_difference_does_not_change():
    m = _mod()
    ref, M = _case()
    rng = np.random.default_rng(1)
    a = np.where(M == 1, ref, ref + rng.normal(0, 3, ref.shape))
    b = np.where(M == 1, ref, ref + rng.normal(0, 9, ref.shape))
    d = m.decompose(*_scores(m, ref, a, M), *_scores(m, ref, b, M))
    assert d["n_capped"] == 0 and d["change"] == pytest.approx(0.0, abs=1e-9)
    assert d["whole_diff"] == pytest.approx(d["hole_diff"], abs=1e-9)


def test_an_exact_hole_moves_the_difference_by_the_whole_patch_shift():
    m = _mod()
    ref, M = _case(n=4)
    rng = np.random.default_rng(2)
    a = np.where(M == 1, ref, ref + rng.normal(0, 3, ref.shape))
    b = np.where(M == 1, ref, ref + rng.normal(0, 9, ref.shape))
    b[0] = ref[0]                                            # b reconstructs patch 0 exactly
    d = m.decompose(*_scores(m, ref, a, M), *_scores(m, ref, b, M))
    shift = 10 * math.log10(M[0].size / (M[0] == 0).sum())
    assert d["n_capped"] == 1
    assert d["capped_mean_move"] == pytest.approx(-shift, abs=1e-6)
    assert d["change"] == pytest.approx(-shift / 4, abs=1e-6)
    assert d["cap_part"] == pytest.approx(d["change"], abs=1e-9) and d["rest_part"] == pytest.approx(0, abs=1e-9)


def test_the_parts_sum_to_the_change_when_patches_without_a_hole_are_dropped():
    m = _mod()
    ref, M = _case(n=5)
    M[1] = 1                                                 # patch 1 has no hole
    rng = np.random.default_rng(3)
    a = (ref + rng.normal(0, 2, ref.shape))   # observed pixels not reproduced
    b = np.where(M == 1, ref, ref + rng.normal(0, 9, ref.shape))
    b[2] = ref[2]
    d = m.decompose(*_scores(m, ref, a, M), *_scores(m, ref, b, M))
    assert (d["n_whole"], d["n_hole"]) == (5, 4)
    assert d["dropped_part"] != 0 and d["rest_part"] != 0
    assert d["change"] == pytest.approx(d["cap_part"] + d["rest_part"] + d["dropped_part"], abs=1e-12)


def test_shift_residual_is_zero_for_exact_observed_pixels_and_skips_the_cap():
    m = _mod()
    ref, M = _case(n=3)
    rec = np.where(M == 1, ref, ref + 5)
    rec[0] = ref[0]                                          # at the cap: not checked
    w, h = _scores(m, ref, rec, M)
    n_checked, worst = m.shift_residual(w, h, (M == 0).reshape(3, -1).sum(1), M[0].size)
    assert n_checked == 2 and worst < 1e-5
    rec[1, 0, 0] += 1                                        # one observed pixel off by one grey level
    w, h = _scores(m, ref, rec, M)
    assert m.shift_residual(w, h, (M == 0).reshape(3, -1).sum(1), M[0].size)[1] > 1e-3


def test_psnr_matches_the_round1_definition():
    m = _mod()
    ref = np.zeros((4, 4))
    assert m.psnr(ref, ref) == 100.0
    assert m.psnr(ref, ref + 255) == pytest.approx(0.0)
    rec = ref.copy()
    rec[0, 0] = 1e-3                                         # within the cap: 100 dB
    assert m.psnr(ref, rec) == 100.0
