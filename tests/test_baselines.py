"""Classical baselines (round 2): every method returns observed pixels exactly,
fills the hole, and is deterministic; the new methods do better than zero-fill
where they are meant to work."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baselines import (BASELINES, baseline_guleryuz, baseline_biharmonic,  # noqa: E402
                       baseline_criminisi, xphoto_available)
from masks import make_mask  # noqa: E402

NEEDS_XPHOTO = {"fsr", "shiftmap"}


def _case(fam="CF1", frac=0.3, n=32):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    ref = (50 + 100 * ((xx // 8 + yy // 8) % 2) + 0.3 * yy).astype(np.float32)[None]
    M = make_mask((1, n, n), fam, frac, rng)
    return ref, ref * M, M


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_preserves_observed_pixels_and_fills_the_hole(name):
    if name in NEEDS_XPHOTO and not xphoto_available():
        pytest.skip("opencv-contrib not installed")
    ref, y, M = _case("CF2", 0.15)
    out = BASELINES[name](y, M, seed=0)
    assert out.shape == y.shape and np.isfinite(out).all()
    assert np.array_equal(out[M == 1], y[M == 1])
    assert out.min() >= 0 and out.max() <= 255
    if name != "zero_fill":
        assert float(np.abs(out[M == 0] - y[M == 0]).mean()) > 1.0


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_is_deterministic(name):
    if name in NEEDS_XPHOTO and not xphoto_available():
        pytest.skip("opencv-contrib not installed")
    ref, y, M = _case("CF1", 0.3)
    assert np.array_equal(BASELINES[name](y, M, seed=0), BASELINES[name](y, M, seed=0))


def _hole_mse(out, ref, M):
    return float(((out - ref) ** 2)[M == 0].mean())


def test_guleryuz_beats_zero_fill_on_scattered_loss():
    ref, y, M = _case()
    assert _hole_mse(baseline_guleryuz(y, M), ref, M) < _hole_mse(y, ref, M) / 4


def test_biharmonic_beats_zero_fill_on_scattered_loss():
    ref, y, M = _case()
    assert _hole_mse(baseline_biharmonic(y, M), ref, M) < _hole_mse(y, ref, M) / 4


def test_criminisi_copies_texture_into_a_block():
    ref, y, M = _case("CF2", 0.15)
    out = baseline_criminisi(y, M)
    assert _hole_mse(out, ref, M) < _hole_mse(y, ref, M) / 4


def test_criminisi_reports_its_fallback():
    ref, y, M = _case("CF1", 0.7)
    out, info = baseline_criminisi(y, M, return_info=True)
    assert info["fallback_patches"] >= 0 and np.isfinite(out).all()
