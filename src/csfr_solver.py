"""Convergent projected solver for the CSFR objective (round 2).

Projected FISTA (Beck and Teboulle 2009) with the gradient-based adaptive
restart of O'Donoghue and Candes (2015) on the strictly convex objective of
src/csfr_objective.py, in double precision. The observed pixels are fixed at
every iteration (C1) and the box [0, 255] is enforced at every iteration (C3).

Stopping rule. The run stops for a patch when its Frank-Wolfe gap

    G(x) = max_{z in C} <grad J(x), x - z>,   C = {z: z_Omega = y_Omega, 0 <= z <= 255},

falls to tol (objective units). For a convex J, G(x) >= J(x) - J*, so the rule
certifies a suboptimality of at most tol; the relative value G/J is recorded
beside it. The rule is absolute because the float64 floor of G is about 2e-12
in objective units, which is a large fraction of J on flat patches, so a
relative rule cannot be met there. A projected-gradient step size is not a
valid certificate here: deep inside a large hole the gradient is near zero until
information arrives from the boundary, so steps are tiny far from the optimum
(results/solver_calibration/ records the failure). Converged patches are frozen
and the remaining iterations run only on the patches still active. max_iter caps
the run and a capped patch is reported as not converged.

Because the minimiser is unique, the initialisation changes the path, not the
answer; tests/test_csfr_solver.py checks that.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import Tensor

from csfr_objective import HI, LO, Objective

# the unclipped ablation arm has no box; its gap is evaluated over a wider box
# that contains every value the arm produces on 8-bit data
NO_CLIP_BOX = (LO - 255.0, HI + 255.0)


@dataclass(frozen=True)
class SolverConfig:
    lam_l1: float = 0.02
    lam_tv: float = 0.10
    lam_fc: float = 0.005
    eps: float = 1.0
    clip: bool = True
    # Frank-Wolfe gap in objective units, G(x) <= tol. Calibrated
    # (scripts/calibrate_solver.py, results/solver_calibration/tolerance.csv): the
    # loosest tolerance of the grid 1e-3 ... 1e-11 whose 99th-percentile per-patch
    # maximum error against a 5e-12 reference is below 0.01 grey levels on
    # CF1 L3, CF2 L3, CF2 L5 and CF3 L5 (1e-5 fails on CF2 L3 and CF3 L5).
    tol: float = 1e-6
    max_iter: int = 20000
    check_every: int = 10
    log_every: int = 100
    restart: bool = True
    init: str = "zero"         # "zero" | "mean" | "ns"
    dtype: str = "float64"


VARIANTS: dict[str, SolverConfig] = {
    "csfr": SolverConfig(),
    "csfr_tv_only": SolverConfig(lam_l1=0.0, lam_fc=0.0),
    "csfr_dct_only": SolverConfig(lam_tv=0.0),
    "csfr_no_fc": SolverConfig(lam_fc=0.0),
    "csfr_no_clip": SolverConfig(clip=False),
}

STOPPING_RULE = ("Frank-Wolfe gap: max_{z in C} <grad J(x), x - z> <= tol (objective units), "
                 "an upper bound on the suboptimality J(x) - J*; checked "
                 "every check_every iterations; converged patches are frozen; max_iter caps "
                 "the run and a capped patch is reported as not converged")


@dataclass
class Certificate:
    iterations: Tensor
    converged: Tensor
    gap: Tensor
    rel_gap: Tensor
    final_step_inf: Tensor
    objective: Tensor
    lipschitz: float
    step: float
    config: dict
    trace: list = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_json(self) -> dict:
        return {"iterations": self.iterations.tolist(),
                "converged": self.converged.tolist(),
                "gap": self.gap.tolist(),
                "rel_gap": self.rel_gap.tolist(),
                "final_step_inf": self.final_step_inf.tolist(),
                "objective": self.objective.tolist(),
                "lipschitz": self.lipschitz, "step": self.step, "config": self.config,
                "trace": self.trace, "elapsed_s": self.elapsed_s,
                "stopping_rule": STOPPING_RULE}


def project(x: Tensor, y: Tensor, Mb: Tensor, clip: bool = True) -> Tensor:
    """Exact projection onto {X: X_Omega = Y_Omega, 0 <= X <= 255}."""
    if clip:
        x = x.clamp(LO, HI)
    return torch.where(Mb, y, x)


def frank_wolfe_gap(g: Tensor, x: Tensor, free: Tensor, lo: float, hi: float) -> Tensor:
    """Per-patch max_{z in box, z = x off the free set} <g, x - z>."""
    z = torch.where(g < 0, torch.full_like(x, hi), torch.full_like(x, lo))
    return ((g * (x - z)) * free.to(x.dtype)).flatten(1).sum(1)


def initial_guess(y: Tensor, Mb: Tensor, how: str) -> Tensor:
    if how == "zero":
        return torch.where(Mb, y, torch.zeros_like(y))
    if how == "mean":
        cnt = Mb.flatten(1).sum(1).clamp(min=1).to(y.dtype)
        mean = (y * Mb).flatten(1).sum(1) / cnt
        return torch.where(Mb, y, mean.view(-1, 1, 1).expand_as(y))
    if how == "ns":
        import cv2
        out = []
        for yi, mi in zip(y.cpu().numpy(), Mb.cpu().numpy()):
            img8 = np.clip(yi, 0, 255).astype(np.uint8)
            hole = (~mi).astype(np.uint8)
            out.append(cv2.inpaint(img8, hole, 3, cv2.INPAINT_NS).astype(np.float64))
        return torch.from_numpy(np.stack(out)).to(y)
    raise ValueError(f"unknown initialisation: {how}")


def solve(y: Tensor, M: Tensor, cfg: SolverConfig = SolverConfig(),
          device: str | torch.device = "cpu") -> tuple[Tensor, Certificate]:
    """Minimise the objective over the hole pixels of every patch in the batch.

    y, M: (B, H, W); M is 1 on observed pixels. Returns (x_hat, certificate),
    x_hat in cfg.dtype with the observed pixels exactly equal to y.
    """
    dev = torch.device(device)
    y = y.to(dev, getattr(torch, cfg.dtype))
    Mb = M.to(dev).bool()
    B, H, W = y.shape
    obj = Objective(H, W, cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps, dev, y.dtype)
    lo, hi = (LO, HI) if cfg.clip else NO_CLIP_BOX
    L = obj.lipschitz()
    step = 1.0 / L
    x = project(initial_guess(y, Mb, cfg.init), y, Mb, cfg.clip)
    z = x.clone()
    t = torch.ones(B, device=dev, dtype=y.dtype)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    iters = torch.full((B,), cfg.max_iter, dtype=torch.long, device=dev)
    gap_all = torch.full((B,), float("inf"), device=dev, dtype=y.dtype)
    rel_gap = torch.full((B,), float("inf"), device=dev, dtype=y.dtype)
    step_inf = torch.full((B,), float("inf"), device=dev, dtype=y.dtype)
    trace: list[dict] = []
    t0 = time.time()
    idx = torch.arange(B, device=dev)
    xs, zs, ts, ys, Ms = x, z, t, y, Mb
    for k in range(1, cfg.max_iter + 1):
        x_new = project(zs - step * obj.grad(zs), ys, Ms, cfg.clip)
        if cfg.restart:
            crit = ((zs - x_new) * (x_new - xs)).flatten(1).sum(1) > 0
            ts = torch.where(crit, torch.ones_like(ts), ts)
        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * ts * ts))
        zs = x_new + ((ts - 1.0) / t_new).view(-1, 1, 1) * (x_new - xs)
        xs, ts = x_new, t_new
        if k % cfg.check_every == 0 or k == cfg.max_iter:
            g = obj.grad(xs)
            J = obj.value(xs)
            gap = frank_wolfe_gap(g, xs, ~Ms, lo, hi)
            rg = gap / J.clamp(min=torch.finfo(J.dtype).tiny)
            s_inf = (project(xs - step * g, ys, Ms, cfg.clip) - xs).abs().flatten(1).amax(1)
            gap_all[idx] = gap
            rel_gap[idx] = rg
            step_inf[idx] = s_inf
            done = gap <= cfg.tol
            if k % cfg.log_every == 0:
                trace.append({"iter": k, "objective_mean_active": float(J.mean()),
                              "gap_max_active": float(gap.max()), "rel_gap_max_active": float(rg.max()),
                              "n_active": int(idx.numel())})
            if bool(done.any()):
                x[idx[done]] = xs[done]
                iters[idx[done]] = k
                active[idx[done]] = False
                keep = ~done
                idx, xs, zs, ts, ys, Ms = idx[keep], xs[keep], zs[keep], ts[keep], ys[keep], Ms[keep]
                if idx.numel() == 0:
                    break
    if idx.numel() > 0:
        x[idx] = xs
    cert = Certificate(iterations=iters.cpu(), converged=(~active).cpu(), gap=gap_all.cpu(),
                       rel_gap=rel_gap.cpu(),
                       final_step_inf=step_inf.cpu(), objective=obj.value(x).cpu(),
                       lipschitz=L, step=step, config=asdict(cfg), trace=trace,
                       elapsed_s=time.time() - t0)
    return x.detach(), cert
