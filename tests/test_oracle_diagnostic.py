"""Oracle-support diagnostic: with the true DCT support known, least squares on
the observed pixels recovers an exactly sparse patch whenever the observed rows
of the support basis have full column rank, and the condition number reports
how far a mask is from that."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("oracle", ROOT / "scripts" / "run_oracle_diagnostic.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sparse_patch(m, n=16, k=12, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    A.flat[rng.choice(n * n, k, replace=False)] = rng.normal(scale=40, size=k)
    A[0, 0] = 128 * n
    D = m.dct_basis(n)
    return D.T @ A @ D


def test_exactly_sparse_patch_is_recovered_from_a_random_mask():
    m = _mod()
    X = _sparse_patch(m)
    M = (np.random.default_rng(1).random(X.shape) > 0.3).astype(np.float32)
    r = m.oracle_patch(X, M, kappa=13, Phi=m.pixel_dct_matrix(16))
    assert r["cond"] >= 1.0 and r["smin"] > 0
    assert r["psnr_hole"] >= 99.0          # cap, exact recovery


def test_burst_mask_makes_the_oracle_ill_conditioned():
    m = _mod()
    X = _sparse_patch(m, n=16, k=40, seed=2)
    Phi = m.pixel_dct_matrix(16)
    rand = (np.random.default_rng(3).random(X.shape) > 0.5).astype(np.float32)
    burst = np.ones_like(rand)
    burst[:, 4:12] = 0                        # half the columns gone
    c_rand = m.oracle_patch(X, rand, kappa=41, Phi=Phi)["cond"]
    c_burst = m.oracle_patch(X, burst, kappa=41, Phi=Phi)["cond"]
    assert c_burst > 10 * c_rand


def test_coherence_of_pixel_and_dct_bases():
    m = _mod()
    mu = m.mutual_coherence(64)
    assert 1 / 64 <= mu <= 2 / 64
