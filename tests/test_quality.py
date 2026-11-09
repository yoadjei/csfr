"""Hole-only quality metrics (round 2): PSNR over the hole with the cap on exact
reconstructions, the exact whole-versus-hole offset when observed pixels are
untouched, SSIM over the hole, GMSD, HRP, and the per-patch bundle."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quality import psnr_hole, psnr_whole, ssim_hole, gmsd, hrp, patch_metrics  # noqa: E402


def _case():
    rng = np.random.default_rng(0)
    ref = (rng.random((32, 32)) * 255).astype(np.float32)
    M = (rng.random((32, 32)) > 0.4).astype(np.float32)
    rec = ref.copy()
    rec[M == 0] += 10.0
    return ref, rec, M


def test_psnr_hole_ignores_observed_pixels():
    ref, rec, M = _case()
    p, mse, exact = psnr_hole(ref, rec, M)
    assert abs(mse - 100.0) < 1e-4
    assert abs(p - 10 * np.log10(255 ** 2 / 100)) < 1e-6
    assert exact is False


def test_psnr_hole_caps_exact_reconstructions():
    ref, rec, M = _case()
    assert psnr_hole(ref, ref, M) == (100.0, 0.0, True)


def test_whole_equals_hole_plus_offset_when_observed_exact():
    ref, rec, M = _case()
    p_hole, _, _ = psnr_hole(ref, rec, M)
    offset = 10 * np.log10(M.size / (M == 0).sum())
    assert abs(psnr_whole(ref, rec) - (p_hole + offset)) < 1e-6


def test_gmsd_zero_for_identical_and_positive_otherwise():
    ref, rec, M = _case()
    assert gmsd(ref, ref) == 0.0
    assert gmsd(ref, rec) > 0.0
    assert 0.0 <= gmsd(ref, rec, M == 0) <= 1.0


def test_ssim_hole_is_one_for_identical_and_below_for_perturbed():
    ref, rec, M = _case()
    assert abs(ssim_hole(ref, ref, M) - 1.0) < 1e-9
    s = ssim_hole(ref, rec, M)
    assert -1.0 <= s < 1.0


def test_hrp_matches_released_definition():
    ref, rec, M = _case()
    y = ref * M
    known = y[M == 1]
    expected = float((np.abs(rec[M == 0] - known.mean()) / max(known.std(), 1.0)).mean())
    assert abs(hrp(rec, y, M) - expected) < 1e-9


def test_patch_metrics_bundle_keys():
    ref, rec, M = _case()
    b = patch_metrics(ref, rec, ref * M, M)
    assert set(b) >= {"psnr_hole", "mse_hole", "exact", "ssim_hole", "gmsd_hole", "gmsd",
                      "hrp", "psnr_whole", "hole_frac"}
    assert abs(b["hole_frac"] - (M == 0).mean()) < 1e-9
