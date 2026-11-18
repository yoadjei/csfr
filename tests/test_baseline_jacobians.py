"""Comparison records for the classical baselines (round 2): bilinear
interpolation has an exact sparse Jacobian (Delaunay barycentric weights); any
deterministic method has a finite-difference sensitivity."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_jacobians import bilinear_weights, apply_weights, finite_difference_column  # noqa: E402
from baselines import BASELINES  # noqa: E402


def _case(seed=0, n=16):
    rng = np.random.default_rng(seed)
    y = (rng.random((1, n, n)) * 255).astype(np.float32)
    M = (rng.random((1, n, n)) > 0.3).astype(np.float32)
    return y * M, M


def test_bilinear_weights_reproduce_the_released_interpolator():
    y, M = _case()
    released = BASELINES["bilinear"](y, M)[0]
    W = bilinear_weights(M[0])
    rec = apply_weights(y[0], M[0], W)
    inside = np.isfinite(rec)
    assert inside[M[0] == 0].mean() > 0.8
    assert np.allclose(rec[inside], released[inside], atol=1e-3)


def test_bilinear_weights_are_sparse_convex_combinations():
    y, M = _case(1)
    W = bilinear_weights(M[0])
    assert len(W) > 0
    for pix, v in W.items():
        assert M[0][pix] == 0
        assert 1 <= len(v) <= 3
        assert abs(sum(w for _, w in v) - 1.0) < 1e-9
        assert all(M[0][q] == 1 and w >= -1e-12 for q, w in v)


def test_bilinear_finite_difference_equals_its_exact_weights():
    y, M = _case(2)
    W = bilinear_weights(M[0])
    pix, v = next(iter(W.items()))
    q, w = max(v, key=lambda t: t[1])
    col = finite_difference_column(lambda yy, mm: BASELINES["bilinear"](yy, mm), y, M, q, delta=1.0)
    assert abs(col[pix] - w) < 1e-4


def test_finite_difference_column_is_zero_on_observed_pixels():
    y, M = _case(3)
    k = tuple(map(int, np.argwhere(M[0] == 1)[0]))
    col = finite_difference_column(lambda yy, mm: BASELINES["inpaint_ns"](yy, mm), y, M, k, delta=4.0)
    assert col.shape == (16, 16)
    assert np.all(col[M[0] == 1] == 0)


def _degenerate(observed):
    """A 16 x 16 patch observed only at the given pixels."""
    y = np.zeros((1, 16, 16), np.float32)
    M = np.zeros((1, 16, 16), np.float32)
    for k, (r, c) in enumerate(observed):
        M[0, r, c], y[0, r, c] = 1.0, 50.0 + 10 * k
    return y, M


def test_linear_interpolation_fills_a_hull_without_interior_like_outside_the_hull():
    """Fewer than three observed pixels, or all of them on one line (as a
    one-pixel erosion of a scattered-loss mask can leave), span no triangle:
    every hole pixel is outside the hull and gets the fill value, zero, and the
    exact record is empty. The observed pixels are kept."""
    for observed in ([], [(15, 3)], [(15, 3), (15, 9)], [(15, 1), (15, 6), (15, 12)],
                     [(2, 2), (5, 5), (9, 9), (12, 12)]):
        y, M = _degenerate(observed)
        out = BASELINES["bilinear"](y, M)
        assert np.array_equal(out[M == 1], y[M == 1])
        assert np.all(out[M == 0] == 0.0)
        assert bilinear_weights(M[0]) == {}


def test_linear_interpolation_is_unchanged_for_a_hull_with_interior():
    y, M = _degenerate([(15, 1), (15, 6), (12, 12)])
    out = BASELINES["bilinear"](y, M)
    assert np.any(out[M == 0] != 0.0)
    assert bilinear_weights(M[0])
