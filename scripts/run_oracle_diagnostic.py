"""Oracle-support diagnostic (E8): is the sparse model identifiable from the
observed pixels of each corruption family?

    python scripts/run_oracle_diagnostic.py

For every patch, cell and seed of the DFRWS and GovDocs1 corpora (masks
regenerated exactly as the sweeps generate them, or read from a sweep cell when
present): S = the kappa largest-magnitude orthonormal 2-D DCT coefficients of the
ground truth (kappa = 5 percent of the pixels, 205 for 64 x 64); Phi_S = the
corresponding basis images; the oracle reconstruction is Phi_S a with a the least
squares fit of Phi_S on the observed pixels. Recorded per patch: oracle hole PSNR,
the condition number and smallest singular value of the observed rows of Phi_S,
and the oracle's relative approximation error on the full patch (how sparse the
patch really is). The oracle knows the support, which no reconstruction method
does; it measures what the observed pixels can determine at best under the
sparse model, not what any method achieves.

Outputs (results/oracle_diagnostic): oracle.csv per corpus, family and level
(median oracle hole PSNR, median and 90th percentile condition number, fraction
with condition number above 1e3, median smallest singular value), and
coherence.json with the mutual coherence of the pixel and DCT bases, which is the
same for every mask.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from checkpoint import UnitStore, digests, write_text_atomic  # noqa: E402
from csfr_objective import dct_matrix  # noqa: E402
from masks import SEVERITY, make_mask  # noqa: E402
from quality import psnr_hole  # noqa: E402

CORPORA = {"dfrws2006": ("data/patches.npy", "results/sweep_v2_dfrws"),
           "govdocs1_v2": ("data/patches_govdocs1_v2.npy", "results/sweep_v2_govdocs1")}
FAMILIES = ["CF1", "CF2", "CF3", "CF4"]
COND_FLAG = 1e3
# the code a cell's row depends on besides this script: a change invalidates the saved cells
CODE = [ROOT / "src" / "csfr_objective.py", ROOT / "src" / "masks.py", ROOT / "src" / "quality.py"]


def dct_basis(n: int) -> np.ndarray:
    """The solver's orthonormal DCT-II matrix D (rows are basis vectors): A = D X D^T."""
    return dct_matrix(n, torch.device("cpu"), torch.float64).numpy()


def pixel_dct_matrix(n: int) -> np.ndarray:
    """Phi with vec(X) = Phi vec(A) (row-major), columns the 2-D DCT basis images."""
    D = dct_basis(n)
    return np.kron(D, D).T


def mutual_coherence(n: int) -> float:
    """max |<e_i, psi_j>| between the pixel basis and the 2-D DCT basis."""
    D = dct_basis(n)
    return float(np.abs(D).max() ** 2)


def oracle_patch(X: np.ndarray, M: np.ndarray, kappa: int, Phi: np.ndarray) -> dict:
    x = X.astype(np.float64).reshape(-1)
    a_true = Phi.T @ x
    S = np.argsort(-np.abs(a_true), kind="stable")[:kappa]
    PhiS = Phi[:, S]
    obs = M.reshape(-1) == 1
    a, _, rank, sv = np.linalg.lstsq(PhiS[obs], x[obs], rcond=None)
    rec = (PhiS @ a).reshape(X.shape)
    p, _, _ = psnr_hole(X, rec, M)
    smax, smin = float(sv[0]), float(sv[-1]) if len(sv) else 0.0
    best = (PhiS @ a_true[S])
    return {"psnr_hole": p, "cond": smax / smin if smin > 0 else float("inf"), "smin": smin,
            "rank": int(rank), "kappa": kappa,
            "sparse_approx_rel_err": float(np.linalg.norm(best - x) / max(np.linalg.norm(x), 1e-300))}


def _q(v, p):
    v = np.asarray(v, dtype=np.float64)
    return float(np.percentile(v[np.isfinite(v)], p)) if np.isfinite(v).any() else float("inf")


def cell_row(X: np.ndarray, Phi: np.ndarray, kappa: int, corpus: str, fam: str, lvl: str, frac: float,
             seeds: list[int], masks: list[Path]) -> dict:
    """oracle.csv's row for one cell: every patch at every seed, with the sweep's
    mask where the sweep saved one and the regenerated mask otherwise."""
    per = []
    for seed, mpath in zip(seeds, masks):
        M = np.load(mpath) if mpath.exists() else make_mask(X.shape, fam, frac, np.random.default_rng(seed))
        per += [oracle_patch(X[i], M[i], kappa, Phi) for i in range(len(X))]
    cond = [r["cond"] for r in per]
    return {"corpus": corpus, "family": fam, "level": lvl, "frac": frac,
            "n_patches": len(per), "seeds": " ".join(map(str, seeds)), "kappa": kappa,
            "oracle_psnr_hole_median": _q([r["psnr_hole"] for r in per], 50),
            "cond_median": _q(cond, 50), "cond_p90": _q(cond, 90),
            "frac_cond_above_1e3": float(np.mean([c > COND_FLAG for c in cond])),
            "frac_rank_deficient": float(np.mean([r["rank"] < kappa for r in per])),
            "smin_median": _q([r["smin"] for r in per], 50),
            "sparse_approx_rel_err_median": _q([r["sparse_approx_rel_err"] for r in per], 50)}


def _csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")   # written as text: CRLF on Windows
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--kappa-frac", type=float, default=0.05)
    ap.add_argument("--out", default="results/oracle_diagnostic")
    args = ap.parse_args(argv)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for corpus in args.corpora:
        ppath, sweep = CORPORA[corpus]
        X = np.load(ROOT / ppath).astype(np.float32)
        n = X.shape[-1]
        Phi = pixel_dct_matrix(n)
        kappa = int(round(args.kappa_frac * n * n))
        for fam in FAMILIES:
            for lvl, frac in SEVERITY.items():
                masks = [ROOT / sweep / f"{fam}_{lvl}" / f"seed{seed}" / "mask.npy" for seed in args.seeds]
                store = UnitStore(out / "units" / corpus, {
                    "seeds": args.seeds, "kappa": kappa,
                    "inputs": digests([ROOT / ppath, *(m for m in masks if m.exists())]),
                    "code": digests([Path(__file__).resolve(), *CODE])})
                rows += store.rows(f"{fam}_{lvl}", lambda fam=fam, lvl=lvl, frac=frac, masks=masks: [
                    cell_row(X, Phi, kappa, corpus, fam, lvl, frac, args.seeds, masks)])
                r = rows[-1]
                print(f"[oracle] {corpus} {fam}_{lvl}: PSNR {r['oracle_psnr_hole_median']:.2f}, "
                      f"cond median {r['cond_median']:.3g}, >1e3 {r['frac_cond_above_1e3']:.2f}", flush=True)
                write_text_atomic(out / "oracle.csv", _csv_text(rows))
    n = 64
    (out / "coherence.json").write_text(json.dumps(
        {"n": n, "mutual_coherence": mutual_coherence(n), "sqrt_N_times_mu": mutual_coherence(n) * n,
         "definition": "max |<e_i, psi_j>| over pixel basis vectors e_i and 2-D DCT-II basis images psi_j",
         "note": "the same for every mask; the per-mask identifiability is the condition number in oracle.csv"},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
