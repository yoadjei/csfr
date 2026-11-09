"""Corruption masks for the four families (CF1-CF4) at five severity levels.

Moved verbatim from scripts/run_csfr_sweep.py so every harness generates the
identical masks; tests/test_masks.py checks bit-identity with the released
harness and with a released mask. The severity table printed in the paper is
built from SEVERITY and describe_families().
"""
from __future__ import annotations

import numpy as np

SEVERITY = {"L1": 0.05, "L2": 0.15, "L3": 0.30, "L4": 0.50, "L5": 0.70}
BURST_MIN_SIDE = 4


def make_mask(shape: tuple[int, int, int], family: str, frac: float,
              rng: np.random.Generator) -> np.ndarray:
    """(B, H, W) float32 mask, 1 = observed, 0 = erased."""
    b, h, w = shape
    M = np.ones(shape, dtype=np.float32)
    if family == "CF1":
        n_miss = int(round(frac * h * w))
        for i in range(b):
            idx = rng.choice(h * w, n_miss, replace=False)
            M[i].reshape(-1)[idx] = 0.0
    elif family == "CF2":
        target = frac * h * w
        for i in range(b):
            placed = 0
            while placed < target:
                bh = rng.integers(BURST_MIN_SIDE, max(5, h // 2))
                bw = rng.integers(BURST_MIN_SIDE, max(5, w // 2))
                y0 = rng.integers(0, max(1, h - bh))
                x0 = rng.integers(0, max(1, w - bw))
                M[i, y0:y0 + bh, x0:x0 + bw] = 0.0
                placed = int((1 - M[i]).sum())
    elif family == "CF3":
        rows = int(round(frac * h))
        M[:, :rows, :] = 0.0
    elif family == "CF4":
        rows = int(round(frac * h))
        M[:, h - rows:, :] = 0.0
    else:
        raise ValueError(family)
    return M


def realised_fraction(M: np.ndarray) -> np.ndarray:
    """Erased fraction per patch."""
    return 1.0 - M.reshape(M.shape[0], -1).mean(axis=1)


def describe_families(h: int = 64, w: int = 64) -> dict[str, str]:
    """Exact mask-generation rule per family, as printed in the severity table."""
    hi_h, hi_w = max(5, h // 2) - 1, max(5, w // 2) - 1
    return {
        "CF1": "uniform random pixel loss: exactly round(f*H*W) pixels erased, drawn without replacement",
        "CF2": (f"axis-aligned bursts: height and width uniform on {{{BURST_MIN_SIDE},...,{hi_h}}} "
                f"and {{{BURST_MIN_SIDE},...,{hi_w}}}, top-left corner uniform, blocks added until the "
                "erased fraction reaches f (the last block may overshoot)"),
        "CF3": "half-image overwrite: the top round(f*H) rows erased",
        "CF4": "truncation: the bottom round(f*H) rows erased",
    }


def perturb_mask(M: np.ndarray, kind: str, amount, rng: np.random.Generator) -> np.ndarray:
    """A wrong mask, for the mask-error robustness experiment (E10).

    M: (B, H, W), 1 = observed. kind:
      "erode"  Omega eroded by `amount` pixels (8-connected): observed pixels next
               to the hole are marked missing (false negatives only);
      "dilate" Omega dilated by `amount` pixels: hole pixels next to the boundary
               are marked observed (false positives only; they keep the corrupted
               value of the input);
      "flip"   `amount` (a fraction) of the pixels of every patch flipped, chosen
               uniformly without replacement.
    The patch border is not a hole: erosion treats pixels outside the patch as
    observed.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    obs = np.asarray(M) == 1
    out = np.empty_like(obs)
    st = np.ones((3, 3), bool)
    for i in range(obs.shape[0]):
        if kind == "erode":
            out[i] = binary_erosion(obs[i], st, iterations=int(amount), border_value=1)
        elif kind == "dilate":
            out[i] = binary_dilation(obs[i], st, iterations=int(amount))
        elif kind == "flip":
            n = int(round(float(amount) * obs[i].size))
            idx = rng.choice(obs[i].size, n, replace=False)
            f = obs[i].copy().reshape(-1)
            f[idx] = ~f[idx]
            out[i] = f.reshape(obs[i].shape)
        else:
            raise ValueError(f"unknown perturbation {kind}")
    return out.astype(np.asarray(M).dtype)
