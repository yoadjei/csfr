"""CSFR-2D: DCT-2D L1 + TV-2D + frequency-coherence, batched, XPU/CUDA/CPU.

No hardcoded paths. All I/O via CLI arguments or config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SCHEMA = "csfr2d-1.0"
LO, HI = 0.0, 255.0


def pick_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    M = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    M[0] *= 1.0 / math.sqrt(2.0)
    return M


def dct2(x: Tensor, D: Tensor) -> Tensor:
    return D @ x @ D.transpose(-1, -2)


def hf_mask(n: int, device: torch.device, dtype: torch.dtype,
            cutoff: float = 0.5) -> Tensor:
    r = torch.arange(n, device=device, dtype=dtype) / max(n - 1, 1)
    rr, cc = torch.meshgrid(r, r, indexing="ij")
    return ((rr + cc) >= cutoff).to(dtype)


def tv_aniso(x: Tensor) -> Tensor:
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = x[..., :, 1:] - x[..., :, :-1]
    return dy.abs().mean(dim=(-2, -1)) + dx.abs().mean(dim=(-2, -1))


def cvr_metrics(rec: Tensor, y: Tensor, M: Tensor, D: Tensor,
                hf: Tensor, tv_thresh: float, hf_thresh: float,
                c1_thresh: float = 0.5
                ) -> tuple[float, float, float, float]:
    c1 = float(((rec - y).abs()[M.bool()] > c1_thresh).float().mean()) if M.any() else 0.0
    dy = (rec[..., 1:, :] - rec[..., :-1, :]).abs()
    dx = (rec[..., :, 1:] - rec[..., :, :-1]).abs()
    c2 = float(((dy > tv_thresh).float().mean() + (dx > tv_thresh).float().mean()) / 2.0)
    c3 = float(((rec < LO - 1e-6) | (rec > HI + 1e-6)).float().mean())
    coeff = dct2(rec, D)
    c4 = float(((coeff * hf).abs() > hf_thresh).float().mean())
    return c1, c2, c3, c4


def psnr(ref: Tensor, rec: Tensor, max_db: float = 100.0) -> float:
    mse = float(((ref - rec) ** 2).mean())
    if mse < 1e-12:
        return max_db
    return min(max_db, 10.0 * math.log10(255.0 ** 2 / mse))


def ssim2d(ref: np.ndarray, rec: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as f
    except ImportError:
        return None
    return float(f(ref, rec, data_range=ref.max() - ref.min() + 1e-9))


def hrp(rec: Tensor, y: Tensor, M: Tensor) -> float:
    Mb = M.bool()
    known = y[Mb]
    if known.numel() == 0:
        return 0.0
    mu = float(known.mean()); sd = max(float(known.std()), 1.0)
    imp = rec[~Mb]
    if imp.numel() == 0:
        return 0.0
    return float(((imp - mu).abs() / sd).mean())


def reconstruct(y: Tensor, M: Tensor, *, lam_l1: float, lam_tv: float,
                lam_fc: float, max_iter: int, lr: float, device: torch.device,
                ckpt_path: Path | None, ckpt_every: int, resume: bool,
                log_every: int = 200,
                enable_tv: bool = True, enable_fc: bool = True,
                enable_bounds: bool = True) -> tuple[Tensor, dict]:
    """Run the CSFR-2D solver.  Returns (x_hat, info_dict).

    Ablation switches map to the paper's constraint classes: the data-fidelity
    term (C1, fragment consistency) and the l1 sparsity prior are always
    active (they are the reconstruction engine); `enable_tv` toggles C2
    (spatial smoothness), `enable_bounds` toggles C3 (intensity bounds,
    the clipped projection), `enable_fc` toggles C4 (frequency consistency).
    """
    y = y.to(device); M = M.to(device)
    n = y.shape[-1]
    D = dct_matrix(n, device, y.dtype)
    hf = hf_mask(n, device, y.dtype)
    x = (y * M).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    start = 0
    history: list[dict] = []
    if resume and ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        x.data.copy_(ck["x"].to(device))
        opt.load_state_dict(ck["opt"])
        start = int(ck["iter"])
        history = ck.get("history", [])
        print(f"[csfr2d] resume @ iter {start}")
    for it in range(start, max_iter):
        opt.zero_grad(set_to_none=True)
        resid = (x - y) * M
        data = 0.5 * (resid ** 2).sum(dim=(-2, -1)).mean()
        coeff = dct2(x, D)
        l1 = coeff.abs().mean()
        tv = tv_aniso(x).mean()
        fc = ((coeff * hf) ** 2).mean()
        loss = data + lam_l1 * l1
        if enable_tv:
            loss = loss + lam_tv * tv
        if enable_fc:
            loss = loss + lam_fc * fc
        loss.backward()
        opt.step()
        if enable_bounds:
            with torch.no_grad():
                x.clamp_(LO, HI)
        if (it + 1) % log_every == 0:
            history.append({"iter": it + 1, "loss": float(loss.detach())})
        if ckpt_path and (it + 1) % ckpt_every == 0:
            torch.save({"iter": it + 1, "x": x.detach().cpu(),
                        "opt": opt.state_dict(), "history": history,
                        "schema": SCHEMA}, ckpt_path)
    with torch.no_grad():
        # final C1 projection: observed pixels are evidence and are returned

        # exactly (x_i <- y_i on Omega), so the released reconstruction has
        # zero fragment-consistency violation by construction. Residuals may
        # occur transiently during optimisation; they never reach the output.
        x.data = torch.where(M.bool(), y, x.data)
        if enable_bounds:
            x.data.clamp_(LO, HI)
    info = {"D": D, "hf": hf, "iters_run": max_iter - start, "history": history}
    return x.detach(), info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CSFR-2D single-run solver")
    ap.add_argument("--y", required=True, help="Corrupted input .npy")
    ap.add_argument("--ref", required=True, help="Clean reference .npy")
    ap.add_argument("--mask", required=True, help="Binary mask .npy")
    ap.add_argument("--out_x", required=True, help="Output reconstructed .npy")
    ap.add_argument("--out_metrics", required=True, help="Output metrics .json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lam_l1", type=float, default=0.02)
    ap.add_argument("--lam_tv", type=float, default=0.10)
    ap.add_argument("--lam_fc", type=float, default=0.005)
    ap.add_argument("--max_iter", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--tv_thresh", type=float, default=64.0)
    ap.add_argument("--hf_thresh", type=float, default=32.0)
    args = ap.parse_args(argv)

    dev = pick_device(args.device)
    y = torch.from_numpy(np.load(args.y).astype(np.float32))
    ref = torch.from_numpy(np.load(args.ref).astype(np.float32))
    M = torch.from_numpy(np.load(args.mask).astype(np.float32))
    if y.ndim == 2:
        y, ref, M = y.unsqueeze(0), ref.unsqueeze(0), M.unsqueeze(0)

    t0 = time.time()
    x_hat, info = reconstruct(
        y, M, lam_l1=args.lam_l1, lam_tv=args.lam_tv, lam_fc=args.lam_fc,
        max_iter=args.max_iter, lr=args.lr, device=dev,
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        ckpt_every=args.ckpt_every, resume=args.resume,
    )
    dt = time.time() - t0

    x_hat_cpu = x_hat.cpu()
    np.save(args.out_x, x_hat_cpu.numpy())

    psnrs = [psnr(ref[i], x_hat_cpu[i]) for i in range(x_hat_cpu.shape[0])]
    ssims = [ssim2d(ref[i].numpy(), x_hat_cpu[i].numpy())
             for i in range(x_hat_cpu.shape[0])]
    hrps = [hrp(x_hat_cpu[i], y[i], M[i]) for i in range(x_hat_cpu.shape[0])]
    cvrs = [cvr_metrics(x_hat_cpu[i].to(dev), y[i].to(dev), M[i].to(dev),
                        info["D"], info["hf"], args.tv_thresh, args.hf_thresh)
            for i in range(x_hat_cpu.shape[0])]
    cvr_arr = np.array(cvrs)

    metrics = {
        "schema": SCHEMA, "device": str(dev), "runtime_s": dt,
        "n": int(x_hat_cpu.shape[0]),
        "psnr_db_mean": float(np.mean(psnrs)),
        "psnr_db_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean([s for s in ssims if s is not None])) if ssims else None,
        "hrp_mean": float(np.mean(hrps)),
        "cvr": {"C1": float(cvr_arr[:, 0].mean()), "C2": float(cvr_arr[:, 1].mean()),
                "C3": float(cvr_arr[:, 2].mean()), "C4": float(cvr_arr[:, 3].mean())},
        "iters_run": info["iters_run"],
    }
    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2))
    print(f"[csfr2d] PSNR {metrics['psnr_db_mean']:.2f}  "
          f"SSIM {metrics['ssim_mean']:.3f}  "
          f"HRP {metrics['hrp_mean']:.3f}  t={dt:.1f}s  dev={dev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
