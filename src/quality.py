"""Hole-only quality metrics (round 2).

Every number in the quality tables of the paper comes through patch_metrics.
PSNR and SSIM are computed over the imputed pixels only, because every evaluated
method returns the observed pixels unchanged and whole-patch scores would only
add a mask-dependent offset (10 log10(N / |hole|)) that is identical across
methods, plus a cap artefact on exactly reconstructed patches. GMSD (Xue et al.
2014) is the edge-fidelity measure for structured masks; HRP is the released
distributional proxy, kept as descriptive.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity

PEAK, CAP = 255.0, 100.0
_PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
_PREWITT_Y = _PREWITT_X.T


def psnr_hole(ref, rec, M, peak: float = PEAK, cap: float = CAP) -> tuple[float, float, bool]:
    """(PSNR over the hole, MSE over the hole, exact-reconstruction flag)."""
    hole = np.asarray(M) == 0
    d = (np.asarray(ref, np.float64) - np.asarray(rec, np.float64)) ** 2
    mse = float(d[hole].mean()) if hole.any() else float("nan")
    if mse < 1e-12:
        return cap, 0.0, True
    return min(cap, 10.0 * math.log10(peak * peak / mse)), mse, False


def psnr_whole(ref, rec, peak: float = PEAK, cap: float = CAP) -> float:
    mse = float(((np.asarray(ref, np.float64) - np.asarray(rec, np.float64)) ** 2).mean())
    return cap if mse < 1e-12 else min(cap, 10.0 * math.log10(peak * peak / mse))


def ssim_hole(ref, rec, M, win: int = 7) -> float:
    """Mean of the SSIM map over hole pixels (window 7, data range 255)."""
    _, smap = structural_similarity(np.asarray(ref, np.float64), np.asarray(rec, np.float64),
                                    data_range=PEAK, win_size=win, full=True)
    hole = np.asarray(M) == 0
    return float(smap[hole].mean()) if hole.any() else float("nan")


def gmsd(ref, rec, region=None, c: float = 170.0) -> float:
    """Gradient magnitude similarity deviation (lower is better), optionally over a region."""
    r = np.asarray(ref, np.float64)
    d = np.asarray(rec, np.float64)
    gr = np.hypot(convolve(r, _PREWITT_X, mode="nearest"), convolve(r, _PREWITT_Y, mode="nearest"))
    gd = np.hypot(convolve(d, _PREWITT_X, mode="nearest"), convolve(d, _PREWITT_Y, mode="nearest"))
    gms = (2.0 * gr * gd + c) / (gr * gr + gd * gd + c)
    if region is not None:
        gms = gms[np.asarray(region)]
    return float(gms.std())


def hrp(rec, y, M) -> float:
    """Hallucination Risk Proxy, the released definition: mean |z| of imputed pixels
    against the observed-region mean and standard deviation (floor 1)."""
    known = np.asarray(y, np.float64)[np.asarray(M) == 1]
    if known.size == 0:
        return 0.0
    mu, sd = known.mean(), max(known.std(), 1.0)
    imp = np.asarray(rec, np.float64)[np.asarray(M) == 0]
    return float((np.abs(imp - mu) / sd).mean()) if imp.size else 0.0


def patch_metrics(ref, rec, y, M) -> dict:
    p, mse, exact = psnr_hole(ref, rec, M)
    hole = np.asarray(M) == 0
    return {"psnr_hole": p, "mse_hole": mse, "exact": exact,
            "ssim_hole": ssim_hole(ref, rec, M),
            "gmsd_hole": gmsd(ref, rec, hole), "gmsd": gmsd(ref, rec),
            "hrp": hrp(rec, y, M), "psnr_whole": psnr_whole(ref, rec),
            "hole_frac": float(hole.mean())}
