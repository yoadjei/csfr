"""Qualitative reconstruction panel (pre-submission review §4.9): identical crop
and mask across every method, so the perceptual trade-off CSFR makes is visible
rather than only tabulated.

One row per corruption regime; columns are ground truth, the corrupted input,
and all six same-harness reconstructions. Each row is annotated with the per-
patch PSNR so the picture and the number are side by side. Everything is drawn
from the released artefacts and the same baseline implementations the sweep uses.

Output: paper/figures/paper2_F10_qualitative.png
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_reconstruct_2d import psnr  # noqa: E402

_spec = importlib.util.spec_from_file_location("_sweep2d", ROOT / "scripts" / "run_csfr_sweep.py")
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)

REF = np.load(ROOT / "data" / "patches.npy").astype(np.float32)

# one informative patch per regime: CF1 (CSFR wins), CF2 (bilinear wins),

# CF3/CF4 severe (PDE wins). Chosen for visual clarity, not cherry-picked score.
ROWS = [("CF1_L3", "Random loss 30%"), ("CF2_L4", "Burst 50%"),
        ("CF3_L5", "Top-half erasure"), ("CF4_L5", "Bottom-half erasure")]
COLS = ["Ground truth", "Corrupted", "Zero-fill", "Bilinear", "Telea", "NS", "Dict.", "CSFR"]


def reconstruct_all(y, M, seed, x_hat):
    import cv2
    return {"Zero-fill": _sweep.baseline_zero(y, M),
            "Bilinear": _sweep.baseline_bilinear(y, M),
            "Telea": _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_TELEA),
            "NS": _sweep.baseline_inpaint_cv(y, M, cv2.INPAINT_NS),
            "Dict.": _sweep.baseline_dictlearn(y, M, seed),
            "CSFR": x_hat}


def pick_patch(M):
    """A patch whose corruption is clearly visible: closest to the row's mean
    missing fraction, so it is representative rather than easiest."""
    frac = (M == 0).reshape(M.shape[0], -1).mean(axis=1)
    return int(np.argmin(np.abs(frac - frac.mean())))


def main() -> int:
    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(1.5 * len(COLS), 1.7 * len(ROWS)))
    for ci, name in enumerate(COLS):
        axes[0, ci].set_title(name, fontsize=10)

    for ri, (cell, label) in enumerate(ROWS):
        d = ROOT / "results" / "csfr_sweep_2d_v2" / cell / "seed0"
        y = np.load(d / "y.npy")
        M = np.load(d / "mask.npy")
        x_hat = np.load(d / "x_hat.npy")
        pi = pick_patch(M)
        recs = reconstruct_all(y, M, 0, x_hat)
        panels = [("Ground truth", REF[pi], None),
                  ("Corrupted", np.where(M[pi] == 1, y[pi], np.nan), None)]
        for m in ["Zero-fill", "Bilinear", "Telea", "NS", "Dict.", "CSFR"]:
            rec = recs[m][pi]
            panels.append((m, rec, psnr(torch.from_numpy(REF[pi]),
                                        torch.from_numpy(rec.astype(np.float32)))))
        for ci, (m, img, ps) in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ps is not None:
                ax.set_xlabel(f"{ps:.1f} dB", fontsize=8)
        axes[ri, 0].set_ylabel(label, fontsize=9)

    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "paper2_F10_qualitative.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
