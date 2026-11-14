"""Why the round-1 paired PSNR differences changed when only the hole was scored
(Reviewer 4, comment 4).

When a method reproduces the observed pixels exactly, its whole-patch MSE is
|hole|/N times its hole MSE, so its whole-patch PSNR exceeds its hole PSNR by
10 log10(N/|hole|), the same for every method on that patch: the paired
difference of two methods is the same under both definitions. Three things can
break this, and each is measured here on the round-1 per-patch scores:

  cap_part      patches on which a score of either method reaches the 100 dB cap
                (an exact hole scores 100 dB under both definitions, so the
                other method's shift is not matched)
  rest_part     the other patches with a hole: non-zero only if observed pixels
                were not reproduced exactly (or by floating-point rounding)
  dropped_part  patches without a hole, which enter the whole-patch comparison
                but have no hole score

change = hole_diff - whole_diff = cap_part + rest_part + dropped_part exactly.
Every recomputed difference must equal the one in the round-1
results/paired_stats/paired_tests.csv, so the decomposition is of the numbers
that were reported.

    python scripts/r1_hole_vs_whole.py

Inputs: results/paired_stats/per_patch/*.npz and the masks of
results/csfr_sweep_2d_v2 (round-1 artefacts, not in the repository). Outputs
under results/r1_hole_vs_whole/: comparisons.csv (one row per cell, seed and
baseline) and shift_residual.csv (per method, the largest departure of
whole-minus-hole PSNR from 10 log10(N/|hole|) on patches below the cap).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAP = 100.0
METHOD = "csfr"
BASELINES = ("zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn")


def psnr(ref: np.ndarray, rec: np.ndarray, idx: np.ndarray | None = None, max_db: float = CAP) -> float:
    """PSNR as round 1 scored it (scripts/run_paired_stats.py at ieee-access-r1-submitted)."""
    d = (ref - rec) ** 2
    mse = float(d[idx].mean()) if idx is not None else float(d.mean())
    if mse < 1e-12:
        return max_db
    return min(max_db, 10.0 * math.log10(255.0 ** 2 / mse))


def _at_cap(x: np.ndarray, cap: float) -> np.ndarray:
    return np.isfinite(x) & (x >= cap - 1e-9)


def decompose(cw, ch, bw, bh, cap: float = CAP) -> dict:
    """Paired difference (method minus baseline) under both definitions, and the
    split of its change. cw/ch: the method's whole-patch and hole PSNR per patch
    (hole NaN for a patch without a hole); bw/bh: the baseline's."""
    cw, ch, bw, bh = (np.asarray(v, dtype=np.float64) for v in (cw, ch, bw, bh))
    okw = np.isfinite(cw) & np.isfinite(bw)
    okh = np.isfinite(ch) & np.isfinite(bh)
    if (okh & ~okw).any():
        raise ValueError("a patch has a hole score but no whole-patch score")
    dw, dh = cw - bw, ch - bh
    capped = okh & (_at_cap(cw, cap) | _at_cap(bw, cap) | _at_cap(ch, cap) | _at_cap(bh, cap))
    rest = okh & ~capped
    n_w, n_h = int(okw.sum()), int(okh.sum())
    whole, hole = float(dw[okw].mean()), float(dh[okh].mean())
    move = dh[capped] - dw[capped]
    return {"n_whole": n_w, "n_hole": n_h, "n_capped": int(capped.sum()),
            "whole_diff": whole, "hole_diff": hole, "change": hole - whole,
            "cap_part": float(move.sum() / n_h),
            "rest_part": float((dh[rest] - dw[rest]).sum() / n_h),
            "dropped_part": float(dw[okh].mean() - whole),
            "capped_mean_move": float(move.mean()) if move.size else float("nan")}


def shift_residual(w, h, n_hole, n_pix: int, cap: float = CAP) -> tuple[int, float]:
    """(patches checked, largest |w - h - 10 log10(n_pix/n_hole)|) over the
    patches with a hole on which neither score is at the cap."""
    w, h, n_hole = np.asarray(w, float), np.asarray(h, float), np.asarray(n_hole)
    ok = (n_hole > 0) & np.isfinite(h) & ~_at_cap(w, cap) & ~_at_cap(h, cap)
    if not ok.any():
        return 0, 0.0
    expect = 10 * np.log10(n_pix / n_hole[ok])
    return int(ok.sum()), float(np.abs(w[ok] - h[ok] - expect).max())


def _released(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return {(r["family"], r["level"], int(r["seed"]), r["metric"], r["baseline"]): float(r["mean_diff"])
                for r in csv.DictReader(f)}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-patch", default="results/paired_stats/per_patch")
    ap.add_argument("--sweep", default="results/csfr_sweep_2d_v2")
    ap.add_argument("--released", default="results/paired_stats/paired_tests.csv")
    ap.add_argument("--out", default="results/r1_hole_vs_whole")
    args = ap.parse_args(argv)
    released = _released(ROOT / args.released)
    files = sorted((ROOT / args.per_patch).glob("*_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no round-1 per-patch scores under {args.per_patch}")
    rows, residual = [], {}
    for npz in files:
        cell, seed = npz.stem.rsplit("_seed", 1)
        fam, lvl = cell.split("_")
        s = dict(np.load(npz))
        M = np.load(ROOT / args.sweep / cell / f"seed{seed}" / "mask.npy")
        n_hole = (M == 0).reshape(M.shape[0], -1).sum(1)
        n_pix = int(M[0].size)
        for m in (METHOD, *BASELINES):
            k, worst = shift_residual(s[f"{m}/psnr"], s[f"{m}/psnr_masked"], n_hole, n_pix)
            n0, w0 = residual.get(m, (0, 0.0))
            residual[m] = (n0 + k, max(w0, worst))
        for b in BASELINES:
            d = decompose(s[f"{METHOD}/psnr"], s[f"{METHOD}/psnr_masked"], s[f"{b}/psnr"], s[f"{b}/psnr_masked"])
            for metric, key in (("psnr", "whole_diff"), ("psnr_masked", "hole_diff")):
                rel = released[(fam, lvl, int(seed), metric, b)]
                if abs(rel - d[key]) > 1e-9:
                    raise ValueError(f"{cell} seed {seed} {b} {metric}: recomputed {d[key]} != released {rel}")
            rows.append({"family": fam, "level": lvl, "seed": int(seed), "baseline": b,
                         "hole_pixels_min": int(n_hole.min()), "hole_pixels_max": int(n_hole.max()),
                         "n_pixels": n_pix, **d})
    out = ROOT / args.out
    _write(out / "comparisons.csv", rows)
    _write(out / "shift_residual.csv",
           [{"method": m, "n_patches_checked": k, "max_abs_residual_db": w} for m, (k, w) in residual.items()])
    worst = max(rows, key=lambda r: abs(r["change"]))
    print(f"[r1] {len(rows)} comparisons; largest change {worst['change']:+.4f} dB "
          f"({worst['family']} {worst['level']} seed {worst['seed']} vs {worst['baseline']}, "
          f"{worst['n_capped']} capped patches); largest rest_part "
          f"{max(abs(r['rest_part']) for r in rows):.2e} dB; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
