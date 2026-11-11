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


# shift-map is not a function of its input (see src/baselines.py); it is
# isolated per call instead, which test_shiftmap_runs_in_a_fresh_interpreter checks
DETERMINISTIC = sorted(set(BASELINES) - {"shiftmap"})


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_is_deterministic(name):
    if name in NEEDS_XPHOTO and not xphoto_available():
        pytest.skip("opencv-contrib not installed")
    ref, y, M = _case("CF1", 0.3)
    assert np.array_equal(BASELINES[name](y, M, seed=0), BASELINES[name](y, M, seed=0))


def test_outputs_do_not_depend_on_what_ran_before():
    """The regression that exposed shift-map: a method's output must not change
    after the other methods have run in the same process, in either order."""
    if not xphoto_available():
        pytest.skip("opencv-contrib not installed")
    ref, y, M = _case("CF2", 0.3)
    first = {n: BASELINES[n](y, M, seed=0) for n in DETERMINISTIC}
    for n in reversed(DETERMINISTIC):
        assert np.array_equal(BASELINES[n](y, M, seed=0), first[n]), n


def test_shiftmap_runs_in_a_fresh_interpreter(monkeypatch):
    import io
    import subprocess

    import baselines
    calls = []

    def fake_run(cmd, input, capture_output):
        calls.append(cmd)
        buf = io.BytesIO(input)
        y = np.load(buf)
        out = io.BytesIO()
        np.save(out, y + 1)
        return subprocess.CompletedProcess(cmd, 0, out.getvalue(), b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref, y, M = _case("CF2", 0.3)
    out = baselines.baseline_shiftmap(y.astype(np.float64), M, seed=0)
    assert len(calls) == 1 and calls[0][-2:] == [str(baselines.XPHOTO_CHILD), "INPAINT_SHIFTMAP"]
    assert out.dtype == np.float64 and np.array_equal(out, y + 1)   # the child's array, input dtype


def test_shiftmap_child_failure_is_reported(monkeypatch):
    import subprocess

    import baselines
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, input, capture_output: subprocess.CompletedProcess(cmd, 1, b"", b"boom"))
    ref, y, M = _case()
    with pytest.raises(RuntimeError, match="INPAINT_SHIFTMAP child failed: boom"):
        baselines.baseline_shiftmap(y, M)


def _hole_mse(out, ref, M):
    return float(((out - ref) ** 2)[M == 0].mean())


def test_dictlearn_does_not_flood_the_log_with_convergence_warnings():
    """The released configuration caps the coordinate-descent sparse coding of
    the dictionary fit at 200 iterations, and the cap is often reached; every
    miss printed a warning, 2.5 GB of log on one corpus. The baseline filters
    that one warning (filtering cannot change the computation);
    scripts/dictlearn_inner_cap.py measures what the cap does to the output."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    ref = np.load(Path(__file__).resolve().parent.parent / "data" / "patches.npy")[:2].astype(np.float32)
    M = make_mask(ref.shape, "CF2", 0.3, np.random.default_rng(0))     # natural content reaches the cap
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BASELINES["dictlearn"](ref * M, M)
    assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]


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
