"""The implemented objective (round 2): closed-form gradient and Hessian-vector
product must agree with autograd, the FFT DCT with the matrix DCT and with
SciPy, the DC coefficient must carry no sparsity penalty, and the Lipschitz
bound must dominate the measured curvature."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn, idctn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from csfr_objective import Objective, dct2_fft, idct2_fft, dct_matrix  # noqa: E402

torch.manual_seed(0)
CPU = torch.device("cpu")


def _problem(h=16, w=12, b=3):
    x = torch.rand(b, h, w, dtype=torch.float64) * 255
    obj = Objective(h, w, lam_l1=0.02, lam_tv=0.10, lam_fc=0.005, eps=1.0,
                    device=CPU, dtype=torch.float64)
    return x, obj


def test_fft_dct_matches_matrix_and_scipy():
    x = torch.rand(2, 16, 12, dtype=torch.float64) * 255
    Dh = dct_matrix(16, CPU, torch.float64)
    Dw = dct_matrix(12, CPU, torch.float64)
    a_mat = Dh @ x @ Dw.T
    a_fft = dct2_fft(x)
    a_sci = torch.from_numpy(dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1)))
    assert torch.allclose(a_fft, a_mat, atol=1e-9)
    assert torch.allclose(a_fft, a_sci, atol=1e-9)
    x_sci = torch.from_numpy(idctn(a_fft.numpy(), type=2, norm="ortho", axes=(-2, -1)))
    assert torch.allclose(idct2_fft(a_fft), x, atol=1e-9)
    assert torch.allclose(idct2_fft(a_fft), x_sci, atol=1e-9)


def test_fft_dct_handles_odd_lengths():
    x = torch.rand(1, 7, 5, dtype=torch.float64)
    a_sci = torch.from_numpy(dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1)))
    assert torch.allclose(dct2_fft(x), a_sci, atol=1e-9)
    assert torch.allclose(idct2_fft(dct2_fft(x)), x, atol=1e-9)


def test_matrix_and_fft_paths_give_the_same_objective():
    x, obj_mat = _problem(h=16, w=16)
    obj_fft = Objective(16, 16, lam_l1=0.02, lam_tv=0.10, lam_fc=0.005, eps=1.0,
                        device=CPU, dtype=torch.float64, use_fft=True)
    assert torch.allclose(obj_mat.value(x), obj_fft.value(x), atol=1e-9)
    assert torch.allclose(obj_mat.grad(x), obj_fft.grad(x), atol=1e-12)


def test_gradient_matches_autograd():
    x, obj = _problem()
    xg = x.clone().requires_grad_(True)
    (g_auto,) = torch.autograd.grad(obj.value(xg).sum(), xg)
    g = obj.grad(x)
    assert torch.allclose(g, g_auto, atol=1e-10, rtol=1e-8)


def test_hvp_matches_autograd():
    x, obj = _problem()
    v = torch.randn_like(x)
    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(obj.value(xg).sum(), xg, create_graph=True)
    (hv_auto,) = torch.autograd.grad(g, xg, grad_outputs=v)
    hv = obj.hvp(x, v)
    assert torch.allclose(hv, hv_auto, atol=1e-10, rtol=1e-8)


def test_dc_coefficient_is_not_penalised():
    x, obj = _problem()
    # a constant offset changes only the DC coefficient: J must not move
    j0 = obj.value(x)
    j1 = obj.value(x + 3.0)
    assert torch.allclose(j0, j1, atol=1e-9)


def test_lipschitz_bound_dominates_curvature():
    x, obj = _problem(h=16, w=16)
    v = torch.randn_like(x)
    for _ in range(50):  # power iteration on the Hessian-vector product
        v = obj.hvp(x, v)
        v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1)
    lam_max = (v * obj.hvp(x, v)).flatten(1).sum(1) / (v * v).flatten(1).sum(1)
    assert float(lam_max.max()) <= obj.lipschitz() * (1 + 1e-9)


def test_objective_is_per_patch_finite_and_nonnegative():
    x, obj = _problem()
    j = obj.value(x)
    assert j.shape == (3,)
    assert torch.isfinite(j).all()
    assert (j >= 0).all()


def test_as_dict_states_every_constant():
    _, obj = _problem()
    d = obj.as_dict()
    for key in ("lam_l1", "lam_tv", "lam_fc", "eps", "hf_cutoff", "dc_excluded_from_l1",
                "tv", "normalisation", "lipschitz"):
        assert key in d
    assert d["dc_excluded_from_l1"] is True
