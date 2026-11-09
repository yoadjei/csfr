"""Sensitivity record at the converged solution (round 2).

At the minimiser X* of the objective in src/csfr_objective.py with the observed
pixels fixed, the solution map Y_Omega -> X*_F on the free set F (hole pixels
not at an active box bound) is differentiable, and stationarity
grad_F J(X*) = 0 gives

    d X*_F / d Y_Omega = - H_FF^{-1} H_{F,Omega},

with H the Hessian of J at X*. Rows (one imputed pixel against every observed
pixel) and columns (every free pixel against one observed pixel) are computed
by conjugate gradients with the closed-form Hessian-vector products of the
objective, in double precision. The record is validated by perturb-and-re-solve
in fidelity_columns, two ways:

  * reference: base and perturbed problems re-solved in float64 to a tight
    tolerance, so the comparison is with the true solution map;
  * operational: re-solved with the released solver configuration, so the
    comparison is with what an examiner re-running the tool would see.

Units are grey levels of the reconstructed pixel per grey level of the observed
pixel. The record is a derivative of the solution map: it is not a
decomposition of the pixel value and not a causal attribution.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from csfr_objective import HI, LO, Objective
from csfr_solver import Certificate, SolverConfig, solve


@dataclass(frozen=True)
class RecordConfig:
    cg_rtol: float = 1e-10        # relative residual of the CG solve
    cg_max_iter: int = 5000
    damping_rel: float = 1e-6     # gamma = damping_rel * Lipschitz constant
    bound_margin: float = 1e-6    # pixels within this of 0 or 255 count as at the bound
    top_k: int = 12
    support_rel: float = 1e-6     # support = |w| above this fraction of max |w|
    ref_tol: float = 5e-12        # reference re-solve: Frank-Wolfe gap (objective units), float64
    ref_max_iter: int = 100000


def _objective_for(x: Tensor, cfg: SolverConfig) -> Objective:
    return Objective(x.shape[-2], x.shape[-1], cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps,
                     x.device, x.dtype)


def _f64(*ts: Tensor) -> tuple[Tensor, ...]:
    return tuple(t.to(torch.float64) for t in ts)


def free_set(x_hat: Tensor, Mb: Tensor, margin: float) -> Tensor:
    return (~Mb) & (x_hat > LO + margin) & (x_hat < HI - margin)


def conjugate_gradient(apply, b: Tensor, rtol: float, max_iter: int) -> tuple[Tensor, dict]:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = float((r * r).sum())
    b_norm = rs ** 0.5
    if b_norm == 0.0:
        return x, {"cg_iters": 0, "cg_residual": 0.0, "breakdown": False}
    k, rel = 0, 1.0
    for k in range(1, max_iter + 1):
        ap = apply(p)
        pap = float((p * ap).sum())
        if pap <= 0.0:
            return x, {"cg_iters": k, "cg_residual": rs ** 0.5 / b_norm, "breakdown": True}
        alpha = rs / pap
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = float((r * r).sum())
        rel = rs_new ** 0.5 / b_norm
        if rel <= rtol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, {"cg_iters": k, "cg_residual": rel, "breakdown": False}


def reduced_hessian(obj: Objective, x_hat: Tensor, free: Tensor, gamma: float):
    f = free.to(x_hat.dtype)

    def apply(v: Tensor) -> Tensor:
        v = v * f
        return obj.hvp(x_hat, v) * f + gamma * v

    return apply


def sensitivity_row(x_hat: Tensor, y: Tensor, Mb: Tensor, pixel: tuple[int, int],
                    cfg: SolverConfig, rc: RecordConfig) -> tuple[Tensor, dict]:
    """d x_hat[pixel] / d y over Omega (zero off Omega), raw units, float64."""
    x_hat, y = _f64(x_hat, y)
    r, c = pixel
    if bool(Mb[r, c]):
        w = torch.zeros_like(x_hat)
        w[r, c] = 1.0
        return w, {"kind": "observed", "cg_iters": 0, "cg_residual": 0.0, "support": 1}
    obj = _objective_for(x_hat, cfg)
    free = free_set(x_hat, Mb, rc.bound_margin)
    if not bool(free[r, c]):
        return torch.zeros_like(x_hat), {"kind": "at_bound", "cg_iters": 0,
                                         "cg_residual": 0.0, "support": 0}
    gamma = rc.damping_rel * obj.lipschitz()
    e = torch.zeros_like(x_hat)
    e[r, c] = 1.0
    w, info = conjugate_gradient(reduced_hessian(obj, x_hat, free, gamma), e,
                                 rc.cg_rtol, rc.cg_max_iter)
    row = -(obj.hvp(x_hat, w * free.to(x_hat.dtype)) * Mb.to(x_hat.dtype))
    peak = float(row.abs().max())
    thresh = rc.support_rel * peak if peak > 0 else 0.0
    info.update({"kind": "imputed", "support": int((row.abs() > thresh).sum()),
                 "damping": gamma})
    return row, info


def sensitivity_column(x_hat: Tensor, y: Tensor, Mb: Tensor, obs_pixel: tuple[int, int],
                       cfg: SolverConfig, rc: RecordConfig) -> tuple[Tensor, dict]:
    """d x_hat (free set, zero elsewhere) / d y[obs_pixel], raw units, float64."""
    x_hat, y = _f64(x_hat, y)
    r, c = obs_pixel
    if not bool(Mb[r, c]):
        raise ValueError("column queries are observed pixels")
    obj = _objective_for(x_hat, cfg)
    free = free_set(x_hat, Mb, rc.bound_margin)
    f = free.to(x_hat.dtype)
    gamma = rc.damping_rel * obj.lipschitz()
    e = torch.zeros_like(x_hat)
    e[r, c] = 1.0
    b = -(obj.hvp(x_hat, e) * f)
    u, info = conjugate_gradient(reduced_hessian(obj, x_hat, free, gamma), b,
                                 rc.cg_rtol, rc.cg_max_iter)
    info.update({"kind": "column", "damping": gamma, "n_free": int(free.sum())})
    return u * f, info


def _compare(pred: Tensor, actual: Tensor, top_k: int) -> tuple[float, float]:
    denom = float(actual.norm())
    rel = float((actual - pred).norm()) / denom if denom > 0 else float("nan")
    k = min(top_k, int(pred.numel()))
    if k == 0:
        return rel, float("nan")
    top = torch.topk(pred.abs(), k).indices
    agree = float((torch.sign(pred[top]) == torch.sign(actual[top])).to(torch.float64).mean())
    return rel, agree


def fidelity_columns(x_hat: Tensor, y: Tensor, Mb: Tensor, obs_pixel: tuple[int, int],
                     cfg: SolverConfig, rc: RecordConfig,
                     deltas: tuple[float, ...] = (0.25, 1.0, 4.0),
                     operational: bool = True) -> list[dict]:
    """fidelity_patch for a single observed pixel."""
    return fidelity_patch(x_hat, y, Mb, [obs_pixel], cfg, rc, deltas, operational)


def fidelity_patch(x_hat: Tensor, y: Tensor, Mb: Tensor, obs_pixels: list[tuple[int, int]],
                   cfg: SolverConfig, rc: RecordConfig,
                   deltas: tuple[float, ...] = (0.25, 1.0, 4.0),
                   operational: bool = True) -> list[dict]:
    """Perturb-and-re-solve check of the predicted columns of one patch.

    For every query pixel k in obs_pixels and every delta, y is perturbed at k
    and the change of the free pixels, divided by the perturbation, is compared
    with the predicted column d x_F / d y_k. The unperturbed problem and every
    perturbed one form one batch, solved once with the reference configuration
    (float64, tolerance rc.ref_tol) and, if operational, once with cfg. The
    perturbation direction flips where +delta would leave [0, 255]. One row per
    (query, delta), in that order.
    """
    free = free_set(x_hat.to(torch.float64), Mb, rc.bound_margin)
    cols = []
    for k in obs_pixels:
        t0 = time.time()
        pred, info = sensitivity_column(x_hat, y, Mb, k, cfg, rc)
        cols.append((pred[free], info, time.time() - t0))
    jobs = [(q, d, -1.0 if float(y[k[0], k[1]]) + d > HI else 1.0)
            for q, k in enumerate(obs_pixels) for d in deltas]
    Y = y.to(torch.float64).unsqueeze(0).repeat(len(jobs) + 1, 1, 1).clone()
    for j, (q, d, s) in enumerate(jobs, start=1):
        r, c = obs_pixels[q]
        Y[j, r, c] += s * d
    Ms = Mb.to(torch.float64).unsqueeze(0).repeat(len(jobs) + 1, 1, 1)

    ref_cfg = replace(cfg, dtype="float64", tol=rc.ref_tol, max_iter=rc.ref_max_iter)
    t0 = time.time()
    Xr, cr = solve(Y, Ms, ref_cfg)
    ref_s = time.time() - t0
    if operational:
        t0 = time.time()
        Xo, co = solve(Y.to(getattr(torch, cfg.dtype)), Ms, cfg)
        op_s = time.time() - t0
        Xo = Xo.to(torch.float64)

    out = []
    for j, (q, d, s) in enumerate(jobs, start=1):
        pf, info, record_s = cols[q]
        act_ref = ((Xr[j] - Xr[0]) / (s * d))[free]
        rel_ref, agree_ref = _compare(pf, act_ref, rc.top_k)
        row = {"query": q, "query_pixel": [int(obs_pixels[q][0]), int(obs_pixels[q][1])],
               "delta": d, "direction": s, "rel_l2_error_ref": rel_ref,
               "sign_agreement_topk": agree_ref, "pred_norm": float(pf.norm()),
               "actual_norm_ref": float(act_ref.norm()),
               "ref_converged": bool(cr.converged[0] and cr.converged[j]),
               "ref_iters": int(max(cr.iterations[0], cr.iterations[j])),
               "cg_iters": info["cg_iters"], "cg_residual": info["cg_residual"],
               "n_free": int(free.sum()), "record_seconds": record_s,
               "reference_seconds_batch": ref_s, "batch_size": len(jobs) + 1}
        if operational:
            act_op = ((Xo[j] - Xo[0]) / (s * d))[free]
            rel_op, agree_op = _compare(pf, act_op, rc.top_k)
            row.update({"rel_l2_error_op": rel_op, "sign_agreement_topk_op": agree_op,
                        "actual_norm_op": float(act_op.norm()),
                        "op_converged": bool(co.converged[0] and co.converged[j]),
                        "op_iters": int(max(co.iterations[0], co.iterations[j])),
                        "operational_seconds_batch": op_s})
        out.append(row)
    return out


def _certificate_summary(certificate: Certificate) -> dict:
    j = certificate.to_json()
    j.pop("trace", None)
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in j.items()}


def write_manifest(path: Path, *, x_hat: Tensor, y: Tensor, Mb: Tensor, solver_cfg: SolverConfig,
                   certificate: Certificate, queries: list[tuple[int, int]], patch_id: str,
                   rc: RecordConfig, fidelity: list[dict] | None = None) -> dict:
    """Emit the audit manifest for one reconstruction: what was observed, the
    objective and solver configuration, the convergence certificate, the record
    parameters and, for each queried pixel, its sensitivity row."""
    x64, y64 = _f64(x_hat, y)
    entries = []
    for (r, c) in queries:
        t0 = time.time()
        row, info = sensitivity_row(x64, y64, Mb, (r, c), solver_cfg, rc)
        dt = time.time() - t0
        flat = row.abs().flatten()
        order = torch.argsort(flat, descending=True)[: rc.top_k]
        total = float(flat.sum())
        contributors = [{"pixel": [int(o // row.shape[1]), int(o % row.shape[1])],
                         "weight_raw": float(row.flatten()[o]),
                         "weight_normalised": float(row.flatten()[o]) / total if total > 0 else 0.0,
                         "observed_value": float(y64.flatten()[o])}
                        for o in order.tolist() if float(flat[o]) > 0]
        entries.append({"pixel": [int(r), int(c)], "reconstructed_value": float(x64[r, c]),
                        "status": info["kind"], "support": info["support"],
                        "abs_weight_sum_raw": total, "signed_weight_sum_raw": float(row.sum()),
                        "top_contributors": contributors,
                        "solve": {"cg_iterations": info["cg_iters"],
                                  "cg_relative_residual": info["cg_residual"]},
                        "record_seconds": dt})
    obs = Mb.cpu().numpy()
    obj = _objective_for(x64, solver_cfg)
    manifest = {
        "schema": "csfr-sensitivity-manifest-2.0",
        "patch_id": patch_id,
        "observation": {
            "n_pixels": int(obs.size), "n_observed": int(obs.sum()), "n_imputed": int((~obs).sum()),
            "mask_sha256": hashlib.sha256(obs.astype(np.uint8).tobytes()).hexdigest(),
            "observed_sha256": hashlib.sha256((y64.cpu().numpy() * obs).tobytes()).hexdigest()},
        "objective": obj.as_dict(),
        "solver_configuration": asdict(solver_cfg),
        "convergence_certificate": _certificate_summary(certificate),
        "record_parameters": asdict(rc),
        "method": ("implicit-function theorem at the converged solution: d x_F / d y_Omega = "
                   "-H_FF^-1 H_F,Omega, conjugate gradients with closed-form Hessian-vector "
                   "products, float64"),
        "units": "grey level of the reconstructed pixel per grey level of the observed pixel",
        "interpretation": ("a derivative of the solution map at the released reconstruction; "
                           "not a decomposition of the pixel value and not a causal attribution"),
        "fidelity": fidelity or [],
        "queries": entries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
