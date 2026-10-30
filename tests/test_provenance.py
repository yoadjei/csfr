"""Unit tests for src/provenance.py — the approximate local sensitivity that
backs Audit Record Coverage. These lock in the properties the manuscript
claims: observed pixels are self-determined, imputed pixels carry normalised
sensitivity weights over observed pixels, and the measured ARC is 1 on a real
reconstruction."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from provenance import (provenance_row, measured_traceability,  # noqa: E402
                        measured_audit_record_coverage)

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


def test_measured_audit_record_coverage_is_unit_on_real_patch():
    d = ROOT / "results" / "csfr_sweep_2d_v2" / "CF1_L3" / "seed0"
    if not (d / "x_hat.npy").exists():
        return  # artefacts not present in this checkout; skip
    i = 0
    rec = torch.from_numpy(np.load(d / "x_hat.npy")[i])
    y = torch.from_numpy(np.load(d / "y.npy")[i])
    M = torch.from_numpy(np.load(d / "mask.npy")[i])
    arc = measured_audit_record_coverage(rec, y, M, n_probe=4, **SOLVER)
    assert arc["arc"] == 1.0
    assert arc["ts"] == arc["arc"], "the earlier key must keep reporting the same value"
    assert arc["n_traceable"] == arc["n_probed"]
    assert measured_audit_record_coverage is measured_traceability
