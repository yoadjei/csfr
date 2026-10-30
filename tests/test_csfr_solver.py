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
from csfr_solver import SolverConfig, solve, project, VARIANTS  # noqa: E402
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


def test_certificate_reports_convergence_and_the_rule_holds():
    ref, y, M = _toy()
    cfg = SolverConfig(tol=1e-2, max_iter=5000)
    x, cert = solve(y, M, cfg)
    assert bool(cert.converged.all())
    obj = Objective(16, 16, cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps)
    step = 1.0 / obj.lipschitz()
    s = (project(x - step * obj.grad(x), y, M.bool()) - x).abs().flatten(1).amax(1)
    assert float(s.max()) <= cfg.tol
    assert int(cert.iterations.max()) < 5000
    assert cert.to_json()["stopping_rule"].startswith("max |P(x - grad/L) - x| <= tol")


def test_solution_is_initialisation_independent():
    ref, y, M = _toy()
    x0, _ = solve(y, M, SolverConfig(tol=1e-4, max_iter=40000, init="zero"))
    x1, _ = solve(y, M, SolverConfig(tol=1e-4, max_iter=40000, init="mean"))
    assert float((x0 - x1).abs().max()) < 0.05


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
    x, cert = solve(y, M, SolverConfig(tol=1e-9, max_iter=50))
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
