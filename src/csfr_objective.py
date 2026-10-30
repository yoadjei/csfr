"""Implemented CSFR objective (round 2).

Pseudo-Huber DCT sparsity (DC excluded), pseudo-Huber anisotropic total
variation and a quadratic high-frequency penalty, with closed-form gradient and
Hessian-vector products. The observed pixels are held fixed by the solver
(src/csfr_solver.py) and the box [0, 255] is enforced there too; the objective
itself has no data term, because fragment consistency is a constraint, not a
penalty.

The objective is C-infinity and strictly convex on the free pixels, so the
minimiser is unique and the solution map from the observed values to the
reconstruction is differentiable. That is what makes the sensitivity record of
src/sensitivity.py a derivative of something well defined.

Every constant here is stated in the paper and written into every manifest
through Objective.as_dict().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

LO, HI = 0.0, 255.0
HF_CUTOFF = 0.5


def dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Orthonormal 1-D DCT-II matrix (rows index frequency)."""
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    M = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    M[0] *= 1.0 / math.sqrt(2.0)
    return M


def _dct1_fft(x: Tensor) -> Tensor:
    """Orthonormal DCT-II along the last axis via one length-n FFT (Makhoul 1980)."""
    n = x.shape[-1]
    v = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * math.pi * k / (2 * n))
    out = 2.0 * (V * phase).real
    scale = torch.full((n,), math.sqrt(1.0 / (2 * n)), device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(1.0 / (4 * n))
    return out * scale


def _idct1_fft(a: Tensor) -> Tensor:
    """Inverse of _dct1_fft (orthonormal DCT-III) along the last axis."""
    n = a.shape[-1]
    k = torch.arange(n, device=a.device, dtype=a.dtype)
    scale = torch.full((n,), math.sqrt(2.0 * n), device=a.device, dtype=a.dtype)
    scale[0] = math.sqrt(4.0 * n)
    half = a * scale / 2.0                      # Re(W[k]) with W[k] = V[k] e^{-i pi k / 2n}
    half_rev = torch.cat([torch.zeros_like(half[..., :1]), half[..., 1:].flip(-1)], dim=-1)
    phase = torch.exp(1j * math.pi * k / (2 * n))
    V = (half - 1j * half_rev) * phase          # Im(W[k]) = -Re(W[n-k]), W[n] := 0
    v = torch.fft.ifft(V, dim=-1).real
    x = torch.empty_like(a)
    cut = (n + 1) // 2
    x[..., 0::2] = v[..., :cut]
    x[..., 1::2] = v[..., cut:].flip(-1)
    return x


def dct2_fft(x: Tensor) -> Tensor:
    return _dct1_fft(_dct1_fft(x).transpose(-1, -2)).transpose(-1, -2)


def idct2_fft(a: Tensor) -> Tensor:
    return _idct1_fft(_idct1_fft(a).transpose(-1, -2)).transpose(-1, -2)


def hf_mask(h: int, w: int, device: torch.device, dtype: torch.dtype,
            cutoff: float = HF_CUTOFF) -> Tensor:
    """0/1 selector of the high-frequency set H = {(k, l): k/(H-1) + l/(W-1) >= cutoff}."""
    r = torch.arange(h, device=device, dtype=dtype) / max(h - 1, 1)
    c = torch.arange(w, device=device, dtype=dtype) / max(w - 1, 1)
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    return ((rr + cc) >= cutoff).to(dtype)


def phi(z: Tensor, eps: float) -> Tensor:
    """Pseudo-Huber (Charbonnier) penalty: smooth, strictly convex, |z| - eps/2 for large |z|."""
    return torch.sqrt(z * z + eps * eps) - eps


def phi_prime(z: Tensor, eps: float) -> Tensor:
    return z / torch.sqrt(z * z + eps * eps)


def phi_second(z: Tensor, eps: float) -> Tensor:
    return (eps * eps) / torch.pow(z * z + eps * eps, 1.5)


def forward_diff(x: Tensor) -> tuple[Tensor, Tensor]:
    """Anisotropic forward differences with Neumann boundary (no difference across the border)."""
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def forward_diff_adjoint(dy: Tensor, dx: Tensor, like: Tensor) -> Tensor:
    g = torch.zeros_like(like)
    g[..., :-1, :] -= dy
    g[..., 1:, :] += dy
    g[..., :, :-1] -= dx
    g[..., :, 1:] += dx
    return g


@dataclass
class Objective:
    """J(X) for H x W patches, batched over a leading dimension or unbatched.

    J(X) = lam_l1/N * sum_{(k,l) != (0,0)} phi_eps(A_kl)
         + lam_tv * [ sum phi_eps(dy)/N_dy + sum phi_eps(dx)/N_dx ]
         + lam_fc/N * sum_{(k,l) in H} A_kl^2,        A = D_H X D_W^T.
    """
    h: int
    w: int
    lam_l1: float = 0.02
    lam_tv: float = 0.10
    lam_fc: float = 0.005
    eps: float = 1.0
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    use_fft: bool | None = None   # None: matrices up to side 128, FFT above

    def __post_init__(self) -> None:
        if self.use_fft is None:
            self.use_fft = max(self.h, self.w) > 128
        self.Dh = dct_matrix(self.h, self.device, self.dtype)
        self.Dw = dct_matrix(self.w, self.device, self.dtype)
        self.hf = hf_mask(self.h, self.w, self.device, self.dtype)
        self.l1mask = torch.ones(self.h, self.w, device=self.device, dtype=self.dtype)
        self.l1mask[0, 0] = 0.0
        self.N = self.h * self.w
        self.N_dy = (self.h - 1) * self.w
        self.N_dx = self.h * (self.w - 1)

    # transforms -------------------------------------------------------------
    def dct2(self, x: Tensor) -> Tensor:
        return dct2_fft(x) if self.use_fft else self.Dh @ x @ self.Dw.T

    def idct2(self, a: Tensor) -> Tensor:
        return idct2_fft(a) if self.use_fft else self.Dh.T @ a @ self.Dw

    # objective, gradient, curvature ----------------------------------------
    def value(self, x: Tensor) -> Tensor:
        a = self.dct2(x)
        j = self.lam_l1 * (phi(a, self.eps) * self.l1mask).sum(dim=(-2, -1)) / self.N
        dy, dx = forward_diff(x)
        j = j + self.lam_tv * (phi(dy, self.eps).sum(dim=(-2, -1)) / self.N_dy
                               + phi(dx, self.eps).sum(dim=(-2, -1)) / self.N_dx)
        j = j + self.lam_fc * ((a * self.hf) ** 2).sum(dim=(-2, -1)) / self.N
        return j

    def grad(self, x: Tensor) -> Tensor:
        a = self.dct2(x)
        g = self.idct2((self.lam_l1 / self.N) * phi_prime(a, self.eps) * self.l1mask
                       + (2.0 * self.lam_fc / self.N) * a * self.hf)
        dy, dx = forward_diff(x)
        g = g + self.lam_tv * forward_diff_adjoint(phi_prime(dy, self.eps) / self.N_dy,
                                                   phi_prime(dx, self.eps) / self.N_dx, x)
        return g

    def hvp(self, x: Tensor, v: Tensor) -> Tensor:
        a = self.dct2(x)
        av = self.dct2(v)
        hv = self.idct2((self.lam_l1 / self.N) * phi_second(a, self.eps) * self.l1mask * av
                        + (2.0 * self.lam_fc / self.N) * av * self.hf)
        dy, dx = forward_diff(x)
        vdy, vdx = forward_diff(v)
        hv = hv + self.lam_tv * forward_diff_adjoint(
            phi_second(dy, self.eps) * vdy / self.N_dy,
            phi_second(dx, self.eps) * vdx / self.N_dx, x)
        return hv

    def lipschitz(self) -> float:
        """Upper bound on the largest Hessian eigenvalue: phi'' <= 1/eps, the DCT is
        orthonormal, and each 1-D difference operator has squared norm at most 4."""
        return (self.lam_l1 / (self.N * self.eps)
                + self.lam_tv * (4.0 / (self.N_dy * self.eps) + 4.0 / (self.N_dx * self.eps))
                + 2.0 * self.lam_fc / self.N)

    def as_dict(self) -> dict:
        return {"lam_l1": self.lam_l1, "lam_tv": self.lam_tv, "lam_fc": self.lam_fc,
                "eps": self.eps, "hf_cutoff": HF_CUTOFF, "dc_excluded_from_l1": True,
                "smoothing": "pseudo-Huber sqrt(z^2 + eps^2) - eps on the DCT and TV terms",
                "tv": "anisotropic forward differences, Neumann boundary",
                "normalisation": "means over N, N_dy, N_dx",
                "transform": "fft" if self.use_fft else "matrix",
                "lipschitz": self.lipschitz()}
