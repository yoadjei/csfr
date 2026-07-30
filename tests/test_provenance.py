"""Unit tests for src/provenance.py — the per-pixel attribution that backs the
Traceability Score. These lock in the properties the manuscript claims:
observed pixels are self-determined, imputed pixels attribute to observed pixels
with normalised weights, and the measured TS is 1 on a real reconstruction."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from provenance import provenance_row, measured_traceability  # noqa: E402

SOLVER = dict(lam_l1=0.02, lam_tv=0.10, lam_fc=0.005)


def _tiny_reconstruction():
    """A small deterministic problem: 16x16 patch, half the pixels observed."""
    rng = np.random.default_rng(0)
    x = (rng.random((16, 16)) * 255).astype(np.float32)
    M = (rng.random((16, 16)) > 0.5).astype(np.float32)
    y = x * M
    # A stand-in reconstruction that satisfies C1 (observed pixels exact).
    rec = np.where(M == 1, y, y[M == 1].mean()).astype(np.float32)
    return torch.from_numpy(rec), torch.from_numpy(y), torch.from_numpy(M)


def test_observed_pixel_is_self_determined():
    rec, y, M = _tiny_reconstruction()
    obs = np.argwhere(M.numpy() == 1)[0]
    w, info = provenance_row(rec, y, M, tuple(map(int, obs)), **SOLVER)
    assert info["kind"] == "observed"
    assert w[tuple(obs)] == 1.0
    assert np.isclose(np.abs(w).sum(), 1.0)


def test_imputed_pixel_attributes_to_observed_only():
    rec, y, M = _tiny_reconstruction()
    hole = np.argwhere(M.numpy() == 0)[0]
    w, info = provenance_row(rec, y, M, tuple(map(int, hole)), **SOLVER)
    assert info["kind"] == "imputed"
    # all weight mass sits on observed pixels.

    assert np.all(w[M.numpy() == 0] == 0.0)
    assert info["support"] > 0
    assert np.isclose(np.abs(w).sum(), 1.0)
    assert np.isfinite(info["residual"])


def test_measured_traceability_is_unit_on_real_patch():
    d = ROOT / "results" / "csfr_sweep_2d_v2" / "CF1_L3" / "seed0"
    if not (d / "x_hat.npy").exists():
        return  # artefacts not present in this checkout; skip
    i = 0
    rec = torch.from_numpy(np.load(d / "x_hat.npy")[i])
    y = torch.from_numpy(np.load(d / "y.npy")[i])
    M = torch.from_numpy(np.load(d / "mask.npy")[i])
    ts = measured_traceability(rec, y, M, n_probe=4, **SOLVER)
    assert ts["ts"] == 1.0
    assert ts["n_traceable"] == ts["n_probed"]
