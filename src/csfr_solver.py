"""Convergent projected solver for the CSFR objective (round 2).

Projected FISTA (Beck and Teboulle 2009) with the gradient-based adaptive
restart of O'Donoghue and Candes (2015) on the strictly convex objective of
src/csfr_objective.py. The observed pixels are fixed at every iteration (C1),
the box [0, 255] is enforced at every iteration (C3), and the run stops when a
plain projected-gradient step would move no pixel by more than `tol` grey
levels, or at `max_iter`. Every call returns a per-patch certificate that goes
into the released metrics and into every sensitivity manifest.

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


@dataclass(frozen=True)
class SolverConfig:
    lam_l1: float = 0.02
    lam_tv: float = 0.10
    lam_fc: float = 0.005
    eps: float = 1.0
    clip: bool = True
    tol: float = 1e-2          # grey levels: max-norm of the projected-gradient step
    max_iter: int = 20000
    check_every: int = 10
    log_every: int = 100
    restart: bool = True
    init: str = "zero"         # "zero" | "mean" | "ns"


VARIANTS: dict[str, SolverConfig] = {
    "csfr": SolverConfig(),
    "csfr_tv_only": SolverConfig(lam_l1=0.0, lam_fc=0.0),
    "csfr_dct_only": SolverConfig(lam_tv=0.0),
    "csfr_no_fc": SolverConfig(lam_fc=0.0),
    "csfr_no_clip": SolverConfig(clip=False),
}


@dataclass
class Certificate:
    iterations: Tensor
    converged: Tensor
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
                "final_step_inf": self.final_step_inf.tolist(),
                "objective": self.objective.tolist(),
                "lipschitz": self.lipschitz, "step": self.step, "config": self.config,
                "trace": self.trace, "elapsed_s": self.elapsed_s,
                "stopping_rule": ("max |P(x - grad/L) - x| <= tol on the current iterate, "
                                  "checked every check_every iterations; converged patches "
                                  "are frozen; max_iter caps the run and is reported")}


def project(x: Tensor, y: Tensor, Mb: Tensor, clip: bool = True) -> Tensor:
    """Exact projection onto {X: X_Omega = Y_Omega, 0 <= X <= 255}."""
    if clip:
        x = x.clamp(LO, HI)
    return torch.where(Mb, y, x)


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
            out.append(cv2.inpaint(img8, hole, 3, cv2.INPAINT_NS).astype(np.float32))
        return torch.from_numpy(np.stack(out)).to(y)
    raise ValueError(f"unknown initialisation: {how}")


def solve(y: Tensor, M: Tensor, cfg: SolverConfig = SolverConfig(),
          device: str | torch.device = "cpu") -> tuple[Tensor, Certificate]:
    """Minimise the objective over the hole pixels of every patch in the batch.

    y, M: (B, H, W); M is 1 on observed pixels. Returns (x_hat, certificate),
    x_hat with the observed pixels exactly equal to y.
    """
    dev = torch.device(device)
    y = y.to(dev, torch.float32)
    Mb = M.to(dev).bool()
    B, H, W = y.shape
    obj = Objective(H, W, cfg.lam_l1, cfg.lam_tv, cfg.lam_fc, cfg.eps, dev, y.dtype)
    L = obj.lipschitz()
    step = 1.0 / L
    x = project(initial_guess(y, Mb, cfg.init), y, Mb, cfg.clip)
    z = x.clone()
    t = torch.ones(B, device=dev, dtype=y.dtype)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    iters = torch.zeros(B, dtype=torch.long, device=dev)
    step_inf = torch.full((B,), float("inf"), device=dev, dtype=y.dtype)
    trace: list[dict] = []
    t0 = time.time()
    k = 0
    for k in range(1, cfg.max_iter + 1):
        x_new = project(z - step * obj.grad(z), y, Mb, cfg.clip)
        if cfg.restart:
            crit = ((z - x_new) * (x_new - x)).flatten(1).sum(1) > 0
            t = torch.where(crit, torch.ones_like(t), t)
        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        beta = ((t - 1.0) / t_new).view(B, 1, 1)
        z_new = x_new + beta * (x_new - x)
        a = active.view(B, 1, 1)
        x = torch.where(a, x_new, x)
        z = torch.where(a, z_new, z)
        t = torch.where(active, t_new, t)
        if k % cfg.check_every == 0:
            s = (project(x - step * obj.grad(x), y, Mb, cfg.clip) - x).abs().flatten(1).amax(1)
            step_inf = torch.where(active, s, step_inf)
            newly = active & (s <= cfg.tol)
            iters = torch.where(newly, torch.full_like(iters, k), iters)
            active = active & ~newly
            if k % cfg.log_every == 0:
                trace.append({"iter": k, "objective_mean": float(obj.value(x).mean()),
                              "step_inf_max": float(s.max()), "n_active": int(active.sum())})
            if not bool(active.any()):
                break
    iters = torch.where(active, torch.full_like(iters, cfg.max_iter), iters)
    cert = Certificate(iterations=iters.cpu(), converged=(~active).cpu(),
                       final_step_inf=step_inf.cpu(), objective=obj.value(x).cpu(),
                       lipschitz=L, step=step, config=asdict(cfg), trace=trace,
                       elapsed_s=time.time() - t0)
    return x.detach(), cert
