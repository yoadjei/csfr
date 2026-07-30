"""Full-resolution colour case study (pre-submission review §4.7 / P2).

The main evaluation is grayscale $64\\times64$ patches. This is the standing
patch-only limitation, and this script addresses it with a qualitative
demonstration on real carved colour images: whole JPEGs recovered by
signature-carving from the Nick Mikus DFTT disk images, corrupted with the same
CF1/CF3 pixel-domain families, and reconstructed by CSFR run per colour channel
at the image's native resolution.

This is a case study, not a benchmark: three images cannot support inference,
and it is presented as visual evidence that the solver and its guarantees carry
to full-resolution colour input, not as a quantitative claim. C1 preservation
and the intensity bounds hold per channel by construction exactly as in the
grayscale case.

Output: paper/figures/paper2_F11_fullres_colour.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_reconstruct_2d import reconstruct, psnr  # noqa: E402

SOLVER = dict(lam_l1=0.02, lam_tv=0.10, lam_fc=0.005, max_iter=600, lr=2.0)


def corrupt(img: np.ndarray, family: str, frac: float, seed: int):
    """Pixel-domain corruption on a full colour image; mask shared across channels."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    M = np.ones((h, w), np.float32)
    if family == "CF1":
        idx = rng.choice(h * w, int(frac * h * w), replace=False)
        M.reshape(-1)[idx] = 0.0
    elif family == "CF3":
        M[: int(frac * h), :] = 0.0
    return M


def reconstruct_colour(y: np.ndarray, M: np.ndarray, device="cpu") -> np.ndarray:
    """CSFR per channel; the mask is shared, so C1/C3 hold channel-wise."""
    out = np.empty_like(y, dtype=np.float32)
    for c in range(y.shape[2]):
        yc = torch.from_numpy(y[:, :, c].astype(np.float32))[None]
        Mc = torch.from_numpy(M.astype(np.float32))[None]
        xc, _ = reconstruct(yc, Mc, device=torch.device(device),
                            ckpt_path=None, ckpt_every=10 ** 9, resume=False, **SOLVER)
        out[:, :, c] = xc[0].cpu().numpy()
    return np.clip(out, 0, 255)


def main() -> int:
    imgs = [ROOT / "data/full_mikus_11-carve-fat/img000.png",
            ROOT / "data/full_mikus_12-carve-ext2/img000.png",
            ROOT / "data/full_mikus_12-carve-ext2/img001.png"]
    imgs = [p for p in imgs if p.exists()][:3]
    regimes = [("CF1", 0.40), ("CF3", 0.50), ("CF1", 0.40)]

    fig, axes = plt.subplots(len(imgs), 3, figsize=(7.5, 2.6 * len(imgs)))
    if len(imgs) == 1:
        axes = axes[None, :]
    for ri, (p, (fam, frac)) in enumerate(zip(imgs, regimes)):
        gt = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        # the released DCT solver operates on square input; take the largest

        # centred square at native resolution (still whole-image scale, not a
        # 64x64 patch), so this exercises the solver well outside its 64x64
        # development size.
        h, w = gt.shape[:2]
        s = min(h, w)
        gt = gt[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        M = corrupt(gt, fam, frac, seed=0)
        y = gt * M[:, :, None]
        rec = reconstruct_colour(y, M)
        # PSNR over the whole image, averaged across channels.
        ps = float(np.mean([psnr(torch.from_numpy(gt[:, :, c]),
                                 torch.from_numpy(rec[:, :, c])) for c in range(3)]))
        vis_corrupt = np.where(M[:, :, None] == 1, y, np.nan)
        for ci, (title, im) in enumerate(
                [("Ground truth", gt), (f"Corrupted ({fam}, {int(frac*100)}%)", vis_corrupt),
                 (f"CSFR ({ps:.1f} dB)", rec)]):
            ax = axes[ri, ci]
            ax.imshow(np.clip(im, 0, 255).astype(np.uint8) if title.startswith(("Ground", "CSFR"))
                      else im / 255.0)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(title.split(" (")[0], fontsize=10)
            ax.set_xlabel(title, fontsize=8)
        axes[ri, 0].set_ylabel(f"{gt.shape[1]}x{gt.shape[0]}", fontsize=8)

    fig.tight_layout()
    out = ROOT / "paper/figures/paper2_F11_fullres_colour.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
