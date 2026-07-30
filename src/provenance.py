"""Per-pixel provenance for CSFR reconstructions (Stage 6 of the framework).

What this computes
------------------
For a reconstructed pixel :math:`\\hat{x}_i`, provenance is the question "which
observed pixels determine this value, and how strongly?". Formally that is the
row :math:`\\partial \\hat{x}_i / \\partial y` of the solution's Jacobian with
respect to the observation.

Differentiating through the 600 Adam steps that produce :math:`\\hat{x}` would
require unrolling the whole optimisation, which is intractable and would in any
case attribute the answer to the *optimiser* rather than to the *problem*. We
use the implicit-function theorem instead. At a stationary point of the solver
objective :math:`J(x, y)` we have :math:`\\nabla_x J(\\hat{x}, y) = 0`, and
differentiating that identity in :math:`y` gives

.. math::
    \\frac{\\partial \\hat{x}}{\\partial y}
      = -\\bigl(\\nabla^2_{xx} J\\bigr)^{-1} \\nabla^2_{xy} J
      = \\bigl(\\nabla^2_{xx} J\\bigr)^{-1} M ,

because the only coupling to :math:`y` is the data term
:math:`\\tfrac{1}{2}\\lVert M(x-y) \\rVert^2`, whose mixed second derivative is
:math:`-M`. So the provenance row for pixel :math:`i` is obtained by solving the
linear system :math:`(\\nabla^2_{xx} J)\\, v = e_i` and reading off
:math:`v` on the observed set. We solve it by conjugate gradients using
Hessian-vector products from automatic differentiation, so nothing about the
Hessian is ever formed or approximated in closed form.

Honest limits
-------------
* The identity holds *at a stationary point*. The released solver runs a fixed
  600-iteration budget and is not certified to have reached one (see the
  convergence discussion in the paper), so these weights describe the problem
  the solver is targeting rather than a certificate about the returned iterate.
* The :math:`\\ell_1` and TV terms are non-smooth on a measure-zero set. Their
  curvature is taken at the returned point, which is well defined off that set
  and is what the CG solve uses.
* For observed pixels the terminal C1 projection makes the answer exact and
  trivial: :math:`\\hat{x}_i = y_i`, so the row is the indicator of :math:`i`.

This module is what makes the Traceability Score a *measurement* rather than a
constant: a pixel counts as traceable when this computation returns a non-empty
contributing set with finite weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from csfr_reconstruct_2d import LO, HI, SCHEMA, dct2, dct_matrix, hf_mask, tv_aniso


def _objective(x: Tensor, y: Tensor, M: Tensor, D: Tensor, hf: Tensor,
               lam_l1: float, lam_tv: float, lam_fc: float,
               enable_tv: bool, enable_fc: bool) -> Tensor:
    """The solver objective for a single patch, as in ``reconstruct``."""
    resid = (x - y) * M
    coeff = dct2(x, D)
    j = 0.5 * (resid ** 2).sum() + lam_l1 * coeff.abs().mean()
    if enable_tv:
        j = j + lam_tv * tv_aniso(x.unsqueeze(0)).squeeze(0)
    if enable_fc:
        j = j + lam_fc * ((coeff * hf) ** 2).mean()
    return j


def _hessian_vector_product(x: Tensor, grad: Tensor, v: Tensor) -> Tensor:
    """H @ v via a second backward pass; H is never materialised."""
    return torch.autograd.grad(grad, x, grad_outputs=v, retain_graph=True)[0]


def provenance_row(x_hat: Tensor, y: Tensor, M: Tensor, pixel: tuple[int, int],
                   *, lam_l1: float, lam_tv: float, lam_fc: float,
                   enable_tv: bool = True, enable_fc: bool = True,
                   cg_iters: int = 200, tol: float = 1e-8,
                   damping: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Attribution weights of one reconstructed pixel over the observed pixels.

    Returns ``(weights, info)`` where ``weights`` has the patch shape, is zero
    off the observed set, and sums to one over its support (so it reads as a
    contribution share). ``info`` records the solve diagnostics that make the
    number auditable rather than asserted.
    """
    r, c = pixel
    dev = x_hat.device
    n = x_hat.shape[-1]
    D, hf = dct_matrix(n, dev, x_hat.dtype), hf_mask(n, dev, x_hat.dtype)

    if bool(M[r, c].item()):  # observed: fixed exactly by the C1 projection
        w = np.zeros(tuple(x_hat.shape), dtype=np.float64)
        w[r, c] = 1.0
        return w, {"kind": "observed", "cg_iters": 0, "residual": 0.0,
                   "support": 1, "note": "returned unmodified by the C1 projection"}

    x = x_hat.clone().detach().requires_grad_(True)
    j = _objective(x, y, M, D, hf, lam_l1, lam_tv, lam_fc, enable_tv, enable_fc)
    grad = torch.autograd.grad(j, x, create_graph=True)[0]

    e = torch.zeros_like(x_hat)
    e[r, c] = 1.0

    # conjugate gradients on (H + damping*I) v = e. The damping keeps the solve

    # well posed where the objective is flat (an exactly flat direction means
    # the data does not determine that pixel, which is itself the answer).
    v = torch.zeros_like(e)
    resid = e.clone()
    p = resid.clone()
    rs = float((resid * resid).sum())
    used = 0
    for used in range(1, cg_iters + 1):
        hp = _hessian_vector_product(x, grad, p) + damping * p
        denom = float((p * hp).sum())
        if abs(denom) < 1e-30:
            break
        alpha = rs / denom
        v = v + alpha * p
        resid = resid - alpha * hp
        rs_new = float((resid * resid).sum())
        if rs_new ** 0.5 < tol:
            rs = rs_new
            break
        p = resid + (rs_new / rs) * p
        rs = rs_new

    # dx_i/dy = (H^-1) M restricted to observed pixels.
    w = (v * M).detach().cpu().numpy().astype(np.float64)
    total = np.abs(w).sum()
    support = int((np.abs(w) > 1e-12).sum())
    if total > 0:
        w = w / total
    return w, {"kind": "imputed", "cg_iters": used, "residual": rs ** 0.5,
               "support": support,
               "note": "implicit-function attribution over observed pixels"}


def measured_traceability(x_hat: Tensor, y: Tensor, M: Tensor, *,
                          lam_l1: float, lam_tv: float, lam_fc: float,
                          n_probe: int = 16, seed: int = 0, **kw) -> dict:
    """Traceability Score, measured rather than assumed.

    A pixel is traceable when the attribution computation returns a non-empty
    contributing set with finite weights. Observed pixels satisfy this trivially.
    Imputed pixels are sampled, since a full evaluation would mean one linear
    solve per pixel.
    """
    rng = np.random.default_rng(seed)
    holes = np.argwhere(M.cpu().numpy() == 0)
    if len(holes) == 0:
        return {"ts": 1.0, "n_probed": 0, "n_traceable": 0,
                "note": "no imputed pixels in this patch"}
    idx = rng.choice(len(holes), min(n_probe, len(holes)), replace=False)
    traceable = 0
    supports = []
    for k in idx:
        r, c = (int(v) for v in holes[k])
        _, info = provenance_row(x_hat, y, M, (r, c), lam_l1=lam_l1,
                                 lam_tv=lam_tv, lam_fc=lam_fc, **kw)
        supports.append(info["support"])
        if info["support"] > 0 and np.isfinite(info["residual"]):
            traceable += 1
    return {"ts": traceable / len(idx), "n_probed": int(len(idx)),
            "n_traceable": int(traceable),
            "mean_support": float(np.mean(supports)),
            "median_support": float(np.median(supports))}


def write_manifest(path: Path, *, x_hat: Tensor, y: Tensor, M: Tensor,
                   solver: dict, queries: list[tuple[int, int]],
                   patch_id: str, top_k: int = 12) -> dict:
    """Emit the auditable provenance manifest for one reconstruction.

    The manifest is the artefact an opposing expert would be handed: what was
    observed, what configuration processed it, and for each queried pixel the
    observed pixels that determine it with their contribution shares.
    """
    obs = M.cpu().numpy() == 1
    entries = []
    for (r, c) in queries:
        w, info = provenance_row(x_hat, y, M, (r, c), lam_l1=solver["lam_l1"],
                                 lam_tv=solver["lam_tv"], lam_fc=solver["lam_fc"])
        flat = np.abs(w).ravel()
        order = np.argsort(flat)[::-1][:top_k]
        contributors = [
            {"pixel": [int(o // w.shape[1]), int(o % w.shape[1])],
             "weight": float(w.ravel()[o]),
             "observed_value": float(y.cpu().numpy().ravel()[o])}
            for o in order if flat[o] > 1e-12]
        entries.append({
            "pixel": [int(r), int(c)],
            "reconstructed_value": float(x_hat[r, c].item()),
            "status": info["kind"],
            "n_contributing_observed_pixels": info["support"],
            "weight_mass_in_top_k": float(flat[order].sum()),
            "top_contributors": contributors,
            "solve": {"cg_iterations": info["cg_iters"],
                      "residual_norm": info["residual"]},
        })

    manifest = {
        "schema": SCHEMA,
        "artefact": "csfr-provenance-manifest",
        "patch_id": patch_id,
        "observation": {
            "n_pixels": int(M.numel()),
            "n_observed": int(obs.sum()),
            "n_imputed": int((~obs).sum()),
            "mask_sha256": hashlib.sha256(
                M.cpu().numpy().astype(np.uint8).tobytes()).hexdigest(),
            "observed_sha256": hashlib.sha256(
                (y.cpu().numpy() * obs).tobytes()).hexdigest(),
        },
        "solver_configuration": solver,
        "constraints": {
            "C1_fragment_consistency": "hard; observed pixels returned unmodified "
                                       "by the terminal projection",
            "C2_spatial_continuity": f"soft penalty, lambda_tv={solver['lam_tv']}",
            "C3_intensity_bounds": f"hard; clipped to [{LO}, {HI}] each iteration",
            "C4_frequency_coherence": f"soft penalty, lambda_fc={solver['lam_fc']}",
        },
        "attribution_method": (
            "implicit-function theorem at the returned iterate: "
            "dx/dy = (grad^2_xx J)^-1 M, solved per queried pixel by conjugate "
            "gradients with autograd Hessian-vector products"),
        "caveats": [
            "Weights describe the stationarity condition of the solver objective; "
            "the fixed iteration budget is not certified to have reached a "
            "stationary point.",
            "Weights are normalised to sum to one in absolute value over their "
            "support and are contribution shares, not probabilities.",
        ],
        "queries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
