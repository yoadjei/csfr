"""Unit tests for src/metrics.py."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import psnr, ssim, constraint_violation_rates, hallucination_risk_proxy, traceability_score


def test_psnr_identical():
    x = np.random.rand(64) * 255
    assert psnr(x, x) == float("inf")


def test_psnr_nonzero():
    x = np.ones(64) * 128
    y = np.ones(64) * 130
    p = psnr(x, y)
    assert 30 < p < 60, f"Unexpected PSNR: {p}"


def test_cvr_all_known():
    """All pixels known → C1 should be 0."""
    n = 64
    y = np.random.rand(n) * 255
    M = np.ones(n, dtype=np.int32)
    cvr = constraint_violation_rates(y, y, M, lo=0, hi=255, tv_thresh=64, hf_thresh=32)
    assert cvr["C1_fragment"] == 0.0


def test_hrp_all_known():
    """All pixels known → HRP should be 0."""
    y = np.random.rand(64) * 255
    M = np.ones(64, dtype=np.int32)
    assert hallucination_risk_proxy(y, y, M) == 0.0


def test_traceability():
    assert traceability_score() == 1.0


if __name__ == "__main__":
    test_psnr_identical(); print("psnr_identical OK")
    test_psnr_nonzero(); print("psnr_nonzero OK")
    test_cvr_all_known(); print("cvr_all_known OK")
    test_hrp_all_known(); print("hrp_all_known OK")
    test_traceability(); print("traceability OK")
    print("all tests passed")
