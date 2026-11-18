"""Comparison records for the classical baselines (round 2).

Bilinear (Delaunay linear) interpolation has an exact, sparse Jacobian: each
imputed pixel is a convex combination of at most three observed pixels, the
vertices of the Delaunay triangle containing it, with barycentric weights. That
record is cheap and exact, and it is what the CSFR record is compared against.
Telea and Navier-Stokes inpainting are deterministic but have no closed-form
Jacobian; their sensitivity is measured by finite differences only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import Delaunay

from baselines import spans_a_triangle

Pixel = tuple[int, int]


def bilinear_weights(M: np.ndarray) -> dict[Pixel, list[tuple[Pixel, float]]]:
    """Exact interpolation weights of the released bilinear baseline for one mask.

    Returns {hole pixel: [(observed pixel, weight), ...]}. Hole pixels outside the
    convex hull of the observed pixels are omitted: the released interpolator
    fills them with zero, which has an empty record.
    """
    h, w = M.shape
    yy, xx = np.mgrid[0:h, 0:w]
    obs = M == 1
    pts = np.stack([yy[obs], xx[obs]], axis=1).astype(np.float64)
    if not spans_a_triangle(pts):          # no hull interior: every hole pixel gets the fill
        return {}
    tri = Delaunay(pts)
    hole = np.argwhere(~obs).astype(np.float64)
    simplex = tri.find_simplex(hole)
    out: dict[Pixel, list[tuple[Pixel, float]]] = {}
    for p, s in zip(hole, simplex):
        if s < 0:
            continue
        T = tri.transform[s]
        b = T[:2] @ (p - T[2])
        bary = np.array([b[0], b[1], 1.0 - b[0] - b[1]])
        verts = tri.simplices[s]
        entries = [((int(pts[v, 0]), int(pts[v, 1])), float(c))
                   for v, c in zip(verts, bary) if abs(c) > 1e-12]
        out[(int(p[0]), int(p[1]))] = entries
    return out


def apply_weights(y: np.ndarray, M: np.ndarray, W: dict) -> np.ndarray:
    """Reconstruct from the weights: observed pixels unchanged, NaN where no record."""
    rec = np.where(M == 1, y.astype(np.float64), np.nan)
    for pix, entries in W.items():
        rec[pix] = sum(wt * float(y[q]) for q, wt in entries)
    return rec


def finite_difference_column(method: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             y: np.ndarray, M: np.ndarray, k: Pixel, delta: float) -> np.ndarray:
    """(method(y + delta e_k) - method(y)) / delta for one patch, zero on observed pixels.

    y, M: (1, H, W). The perturbation direction flips where +delta would leave [0, 255].
    """
    s = -1.0 if float(y[0][k]) + delta > 255.0 else 1.0
    y2 = y.copy()
    y2[0][k] += s * delta
    base = method(y, M)[0].astype(np.float64)
    pert = method(y2, M)[0].astype(np.float64)
    col = (pert - base) / (s * delta)
    col[M[0] == 1] = 0.0
    return col
