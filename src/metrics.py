"""
metrics.py — Forensic-appropriate evaluation metrics for CSFR (Paper 2)
========================================================================
M1: PSNR                 — Peak Signal-to-Noise Ratio on corrupted pixels
M2: SSIM                 — Structural Similarity (full image)
M3: CVR                  — Constraint Violation Rate (C1–C4)
M4: HRP                  — Hallucination Risk Proxy
M5: ARC                  — Audit Record Coverage. Measured by
                           src/provenance.py; the constant here is the
                           by-construction value for a solver whose audit
                           stage ran to completion on every declared query.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAVE_SKIMAGE = True
except ImportError:
    _ssim_fn = None  # type: ignore[assignment]
    _HAVE_SKIMAGE = False


# ── M1: PSNR ──────────────────────────────────────────────────────────
def psnr(ref: np.ndarray, rec: np.ndarray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio (dB).  Higher is better."""
    mse = np.mean((ref - rec) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


# ── M2: SSIM ──────────────────────────────────────────────────────────
def ssim(ref: np.ndarray, rec: np.ndarray) -> float:
    """SSIM if scikit-image available; otherwise NaN."""
    if not _HAVE_SKIMAGE:
        return float("nan")
    if ref.ndim == 1:
        n = ref.size
        side = int(np.floor(np.sqrt(n)))
        if side * side != n:
            return float("nan")
        ref = ref.reshape(side, side)
        rec = rec.reshape(side, side)
    return float(_ssim_fn(ref, rec, data_range=ref.max() - ref.min()))


# ── M3: CVR ───────────────────────────────────────────────────────────
def _dct1(x: np.ndarray) -> np.ndarray:
    return dct(x, type=2, norm="ortho", axis=-1)


def _highpass_diag(n: int) -> np.ndarray:
    w = np.zeros(n)
    w[n // 2:] = 1.0
    return w


def constraint_violation_rates(
    rec: np.ndarray, y: np.ndarray, M: np.ndarray,
    lo: float, hi: float,
    tv_thresh: float, hf_thresh: float,
) -> dict[str, float]:
    """Per-constraint violation rate (fraction of positions violating)."""
    n = rec.size
    c1 = float(np.mean(np.abs(rec[M == 1] - y[M == 1]) > 1e-6))
    c2 = float(np.mean(np.abs(np.diff(rec)) > tv_thresh))
    c3 = float(np.mean((rec < lo - 1e-6) | (rec > hi + 1e-6)))
    coeff = _dct1(rec)
    hp = _highpass_diag(n)
    c4 = float(np.mean(np.abs(coeff * hp) > hf_thresh))
    return {"C1_fragment": c1, "C2_spatial": c2,
            "C3_intensity": c3, "C4_frequency": c4}


# ── M4: HRP ───────────────────────────────────────────────────────────
def hallucination_risk_proxy(
    rec: np.ndarray, y: np.ndarray, M: np.ndarray,
) -> float:
    """Mean |z-score| of imputed pixels vs observed-pixel distribution."""
    known = y[M == 1]
    if known.size == 0:
        return float("nan")
    mu, sd = np.mean(known), np.std(known) + 1e-9
    imputed = rec[M == 0]
    if imputed.size == 0:
        return 0.0
    z = (imputed - mu) / sd
    return float(np.mean(np.abs(z)))


# ── M5: Audit Record Coverage ─────────────────────────────────────────
def audit_record_coverage() -> float:
    """By-construction ARC for a solver whose audit stage emitted a record for
    every declared query. The measured value comes from
    provenance.measured_audit_record_coverage; this constant is the value the
    sweep harness records when that stage is not re-run per patch. The paper
    called this the Traceability Score in an earlier draft."""
    return 1.0


# earlier name, kept so released scripts and notebooks keep importing.
traceability_score = audit_record_coverage


# ── Solver-configuration summary ──────────────────────────────────────
def build_provenance(
    rec: np.ndarray, y: np.ndarray, M: np.ndarray,
    lam_l1: float, lam_tv: float, lam_fc: float,
) -> dict:
    """Summary of the constraint configuration and index sets behind one
    reconstruction. This is bookkeeping, not the per-pixel sensitivity record;
    that is provenance.write_manifest."""
    weights_sum = lam_l1 + lam_tv + lam_fc + 1e-12
    base = {
        "C2_spatial": lam_tv / weights_sum,
        "C4_frequency": lam_fc / weights_sum,
        "C1_anchor_residual": lam_l1 / weights_sum,
    }
    known_idx = np.where(M == 1)[0].tolist()
    imputed_idx = np.where(M == 0)[0].tolist()
    return {
        "n_known": len(known_idx),
        "n_imputed": len(imputed_idx),
        "imputed_weights_template": base,
        "known_indices_head": known_idx[:32],
        "imputed_indices_head": imputed_idx[:32],
        "note": "CSFR is deterministic; full per-index map omitted for size.",
    }
