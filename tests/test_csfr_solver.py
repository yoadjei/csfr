"""The convergent projected solver (round 2): observed pixels exact at every
step, box respected, the stated stopping rule holds at the returned point, the
solution does not depend on the initialisation, runs are deterministic, and the
variant registry matches the paper's ablation."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from csfr_solver import SolverConfig, solve, project, frank_wolfe_gap, VARIANTS  # noqa: E402
from csfr_objective import Objective  # noqa: E402


def _toy(b=2, n=16, frac=0.3, seed=0):
    """An edge plus a ramp, scattered loss, two patches."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    ref = (60 + 120 * (xx > n // 2) + 0.5 * yy).astype(np.float32)
    ref = np.repeat(ref[None], b, 0)
    M = np.ones_like(ref)
    for i in range(b):
        idx = rng.choice(n * n, int(frac * n * n), replace=False)
        M[i].reshape(-1)[idx] = 0
    return torch.from_numpy(ref), torch.from_numpy(ref * M), torch.from_numpy(M)


def test_observed_pixels_are_exact_and_in_range():
    ref, y, M = _toy()
    x, cert = solve(y, M, SolverConfig(max_iter=2000))
    assert torch.equal(x[M.bool()], y[M.bool()])
    assert float(x.min()) >= 0.0 and float(x.max()) <= 255.0


def test_certificate_bounds_the_objective_gap():
    """The returned point satisfies the stated rule: the Frank-Wolfe gap, an upper
    bound on J(x) - J*, is at most tol."""
    ref, y, M = _toy()
    cfg = SolverConfig(max_iter=20000)
    x, cert = solve(y, M, cfg)
    assert bool(cert.converged.all())
    obj = Objective(16, 16, cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps, dtype=x.dtype)
    gap = frank_wolfe_gap(obj.grad(x), x, ~M.bool(), 0.0, 255.0)
    assert bool((gap <= cfg.tol).all())
    assert torch.allclose(cert.gap, gap, rtol=1e-6, atol=1e-18)
    assert torch.allclose(cert.rel_gap, gap / obj.value(x), rtol=1e-6, atol=1e-18)
    assert "Frank-Wolfe" in cert.to_json()["stopping_rule"]


def test_large_hole_does_not_stop_early():
    """Regression for the round-2 calibration: a projected-step rule stopped a
    70 % top-of-image hole after ~140 iterations, 230 grey levels from the
    optimum. The gap rule must return the optimum to a hundredth of a grey level."""
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    ref = (40 + 3.0 * xx + 2.0 * yy + 60 * ((xx // 6) % 2)).astype(np.float32)[None]
    M = np.ones_like(ref)
    M[:, : int(0.7 * n), :] = 0
    y, Mt = torch.from_numpy(ref * M), torch.from_numpy(M)
    x, cert = solve(y, Mt, SolverConfig())
    x_ref, _ = solve(y, Mt, SolverConfig(tol=0.0, max_iter=30000))
    assert bool(cert.converged.all())
    assert float((x - x_ref).abs().max()) < 1e-2


def test_solution_is_initialisation_independent():
    ref, y, M = _toy()
    x0, _ = solve(y, M, SolverConfig(init="zero"))
    x1, _ = solve(y, M, SolverConfig(init="mean"))
    assert float((x0 - x1).abs().max()) < 1e-2


def test_recovers_a_piecewise_smooth_patch():
    ref, y, M = _toy()
    x, _ = solve(y, M, SolverConfig(max_iter=5000))
    hole = M == 0
    mse = float(((x - ref) ** 2)[hole].mean())
    assert 10 * np.log10(255 ** 2 / mse) > 30.0


def test_is_deterministic():
    ref, y, M = _toy()
    a, _ = solve(y, M, SolverConfig(max_iter=1000))
    b, _ = solve(y, M, SolverConfig(max_iter=1000))
    assert torch.equal(a, b)


def test_cap_is_reported_as_not_converged():
    ref, y, M = _toy()
    x, cert = solve(y, M, SolverConfig(tol=0.0, max_iter=50))
    assert not bool(cert.converged.any())
    assert int(cert.iterations.min()) == 50


def test_no_clip_variant_may_leave_the_box_but_keeps_observed_pixels():
    ref, y, M = _toy()
    x, _ = solve(y, M, dataclasses.replace(VARIANTS["csfr_no_clip"], max_iter=500))
    assert torch.equal(x[M.bool()], y[M.bool()])


def test_variants_registry():
    assert set(VARIANTS) >= {"csfr", "csfr_tv_only", "csfr_dct_only", "csfr_no_fc", "csfr_no_clip"}
    assert VARIANTS["csfr_tv_only"].lam_l1 == 0.0 and VARIANTS["csfr_tv_only"].lam_fc == 0.0
    assert VARIANTS["csfr_dct_only"].lam_tv == 0.0
    assert VARIANTS["csfr_no_fc"].lam_fc == 0.0
    assert VARIANTS["csfr_no_clip"].clip is False


def test_default_precision_is_float64_and_float32_agrees():
    ref, y, M = _toy()
    x64, c64 = solve(y, M, SolverConfig())
    x32, _ = solve(y, M, SolverConfig(dtype="float32", tol=1e-7))
    assert x64.dtype == torch.float64 and x32.dtype == torch.float32
    assert bool(c64.converged.all())
    assert float((x64 - x32.double()).abs().max()) < 0.05
    assert torch.equal(x64[M.bool()], y.double()[M.bool()])


def test_active_subset_matches_full_batch():
    """Freezing converged patches and computing only on the rest must not change
    the answer for the patches still running."""
    ref, y, M = _toy(b=4)
    xa, _ = solve(y, M, SolverConfig())
    xb = torch.cat([solve(y[i:i + 1], M[i:i + 1], SolverConfig())[0] for i in range(4)])
    assert float((xa - xb).abs().max()) < 1e-6


def test_solve_task_is_solve_at_the_same_thread_count():
    """The JPEG driver moved its solves from the main process into pool workers:
    at the same thread count they must be the same bits. The image is large enough
    (65,536 pixels) for torch to split its sums across threads."""
    from csfr_solver import solve_task
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    ref = (100 + 60 * np.sin(xx / 9.0) + 40 * (yy > 128) + rng.normal(0, 5, (256, 256))).clip(0, 255)
    M = (rng.uniform(size=(1, 256, 256)) > 0.3).astype(np.float32)
    y = (ref[None] * M).astype(np.float32)
    cfg = dataclasses.replace(VARIANTS["csfr"], max_iter=60)
    before = torch.get_num_threads()
    try:
        x4, i4 = solve_task(y, M, cfg, threads=4)
        assert torch.get_num_threads() == before             # the worker's own setting is restored
        torch.set_num_threads(4)
        x, cert = solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(M), cfg)
    finally:
        torch.set_num_threads(before)
    assert np.array_equal(x.numpy(), x4)
    assert (int(cert.iterations[0]), float(cert.gap[0]), bool(cert.converged[0])) == (
        i4["iterations"], i4["gap"], i4["converged"])


def test_solve_task_takes_one_image():
    import pytest
    from csfr_solver import solve_task
    ref, y, M = _toy(b=2)
    with pytest.raises(ValueError):
        solve_task(y.numpy(), M.numpy(), SolverConfig(), threads=1)
