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


def _tiny_corpus(m, tmp_path, monkeypatch):
    """A two-patch 8 x 8 corpus in place of the real ones; no sweep masks, so
    every mask is regenerated."""
    X = np.stack([_sparse_patch(m, n=8, k=4, seed=s) for s in (0, 1)]).astype(np.float32)
    np.save(tmp_path / "patches.npy", X)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "CORPORA", {"tiny": ("patches.npy", "no_sweep")})
    return X


def test_a_rerun_reads_every_finished_cell_back(tmp_path, monkeypatch):
    m = _mod()
    _tiny_corpus(m, tmp_path, monkeypatch)
    args = ["--corpora", "tiny", "--seeds", "0", "--out", "out"]
    assert m.main(args) == 0
    first = (tmp_path / "out" / "oracle.csv").read_bytes()
    assert first.count(b"\n") == 1 + 4 * len(m.SEVERITY)              # header and one row per cell
    calls = []
    monkeypatch.setattr(m, "oracle_patch", lambda *a, **k: calls.append(1))
    assert m.main(args) == 0
    assert calls == [] and (tmp_path / "out" / "oracle.csv").read_bytes() == first


def test_sweep_masks_at_several_seeds_key_the_saved_cell(tmp_path, monkeypatch):
    """Every seed's sweep mask is named mask.npy (the campaign of 2026-11-17 died on
    that); the saved cell is keyed by each of them, and a changed mask at one seed
    recomputes that cell only."""
    m = _mod()
    X = _tiny_corpus(m, tmp_path, monkeypatch)
    monkeypatch.setattr(m, "CORPORA", {"tiny": ("patches.npy", "sweep")})
    for s in (0, 1):
        d = tmp_path / "sweep" / "CF1_L1" / f"seed{s}"
        d.mkdir(parents=True)
        np.save(d / "mask.npy", m.make_mask(X.shape, "CF1", m.SEVERITY["L1"], np.random.default_rng(s)))
    args = ["--corpora", "tiny", "--seeds", "0", "1", "--out", "out"]
    assert m.main(args) == 0
    np.save(tmp_path / "sweep" / "CF1_L1" / "seed1" / "mask.npy",
            m.make_mask(X.shape, "CF1", m.SEVERITY["L1"], np.random.default_rng(7)))
    calls = []
    real = m.oracle_patch
    monkeypatch.setattr(m, "oracle_patch", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert m.main(args) == 0
    assert len(calls) == 2 * len(X)                                    # CF1_L1 at both seeds, nothing else


def test_changed_patches_invalidate_the_saved_cells(tmp_path, monkeypatch):
    m = _mod()
    X = _tiny_corpus(m, tmp_path, monkeypatch)
    args = ["--corpora", "tiny", "--seeds", "0", "--out", "out"]
    m.main(args)
    np.save(tmp_path / "patches.npy", X[::-1].copy())
    calls = []
    real = m.oracle_patch
    monkeypatch.setattr(m, "oracle_patch", lambda *a, **k: calls.append(1) or real(*a, **k))
    m.main(args)
    assert len(calls) == 4 * len(m.SEVERITY) * len(X)                # every cell recomputed
