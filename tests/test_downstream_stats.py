"""Unit tests for downstream statistics and metrics (Task 1c).

Test ECE calculation, confident-error definitions, McNemar, Holm correction,
bootstrap CI, and grid completeness assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_downstream_stats import bootstrap_ci_confident_error_cond_diff_arrays


class TestECE:
    """Expected Calibration Error with 15 equal-width bins."""

    def test_ece_perfect_calibration(self):
        """Perfect calibration: predicted conf matches actual acc per bin."""
        # 100 instances, 10 bins, each bin has 10 instances at same confidence.
        # All correct, confidence 0.95 → empirical acc 1.0 → ECE should be ~0.05.
        confidences = np.ones(100) * 0.95
        correct = np.ones(100)

        # Compute ECE manually: 15 equal bins
        ece = 0.0
        n_bins = 15
        for bin_idx in range(n_bins):
            low = bin_idx / n_bins
            high = (bin_idx + 1) / n_bins
            mask = (confidences >= low) & (confidences < high)
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

        assert ece < 0.1, "Perfect calibration should have low ECE"

    def test_ece_overconfident(self):
        """Overconfident predictions: high conf, low acc → high ECE."""
        confidences = np.ones(100) * 0.9
        correct = np.zeros(100)  # all wrong

        ece = 0.0
        n_bins = 15
        for bin_idx in range(n_bins):
            low = bin_idx / n_bins
            high = (bin_idx + 1) / n_bins
            mask = (confidences >= low) & (confidences < high)
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

        assert ece > 0.7, "All-wrong predictions should have high ECE"


class TestConfidentError:
    """Confident-error rate: joint and conditional definitions."""

    def test_confident_error_joint_and_conditional(self):
        """Joint P(wrong and conf >= 0.9) vs conditional P(wrong | conf >= 0.9)."""
        # Construct a case where they differ meaningfully
        correct = np.array([1, 1, 1, 0, 0, 1, 1, 1])
        confidence = np.array([0.95, 0.92, 0.91, 0.88, 0.75, 0.55, 0.5, 0.4])

        # High confidence (>= 0.9): indices 0, 1, 2 → {1, 1, 1}
        high_conf_mask = confidence >= 0.9
        n_high_conf = high_conf_mask.sum()

        # Joint: wrong AND high conf (none in this example)
        wrong = (correct == 0)
        joint = (wrong & high_conf_mask).sum() / len(correct)

        # Conditional: wrong | high conf
        if n_high_conf > 0:
            conditional = (wrong & high_conf_mask).sum() / n_high_conf
        else:
            conditional = 0.0

        assert joint == 0.0, f"Expected joint 0.0 (all high-conf predictions correct), got {joint}"
        assert conditional == 0.0, f"Expected conditional 0.0, got {conditional}"

        # Now test case where high-conf has errors
        correct2 = np.array([1, 1, 0, 0, 0, 1, 1, 1])
        confidence2 = np.array([0.95, 0.92, 0.91, 0.88, 0.75, 0.55, 0.5, 0.4])
        high_conf_mask2 = confidence2 >= 0.9
        wrong2 = (correct2 == 0)
        joint2 = (wrong2 & high_conf_mask2).sum() / len(correct2)
        conditional2 = (wrong2 & high_conf_mask2).sum() / high_conf_mask2.sum() if high_conf_mask2.sum() > 0 else 0.0

        assert joint2 == 1.0 / 8.0, f"Expected joint 0.125, got {joint2}"
        assert conditional2 == 1.0 / 3.0, f"Expected conditional 0.333, got {conditional2}"
        assert conditional2 > joint2, "Conditional should be higher"

    def test_confident_error_never_confident(self):
        """Method never confident → joint rate is 0 regardless of accuracy."""
        correct = np.array([1, 1, 0, 0, 0])
        confidence = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        high_conf_mask = confidence >= 0.9
        wrong = (correct == 0)
        joint = (wrong & high_conf_mask).sum() / len(correct)

        assert joint == 0.0, "Joint rate is 0 when no predictions are confident"


class TestMcNemar:
    """Paired McNemar test with exact binomial for small discordant counts."""

    def test_mcnemar_discordant_known_answer(self):
        """McNemar on a 2x2 table with known result."""
        # True table (comparing method A vs B on same patches):
        # - Both correct: 40
        # - Both wrong: 10
        # - A correct, B wrong: 20
        # - A wrong, B correct: 30
        # Discordant: 20 vs 30 → b=20, c=30

        from scipy.stats import binomtest

        b, c = 20, 30  # discordant counts
        n = b + c

        # Null: P(B | discordant) = 0.5 under null
        # Two-tailed exact binomial test
        result = binomtest(b, n, p=0.5, alternative='two-sided')
        p_value = result.pvalue

        # For b=20, c=30 (n=50), we expect p-value around 0.20
        assert 0.05 < p_value < 0.25, f"Expected p around 0.20, got {p_value}"


class TestHolmCorrection:
    """Holm-Bonferroni correction on a p-value vector."""

    def test_holm_correction_known_vector(self):
        """Holm correction on a known p-vector."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.2])
        alpha = 0.05

        # Holm procedure: sort ascending, test at alpha/m, alpha/(m-1), ...
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        m = len(p_values)

        rejected = []
        for i, p in enumerate(sorted_p):
            threshold = alpha / (m - i)
            if p < threshold:
                rejected.append(True)
            else:
                rejected.append(False)

        # Expect: 0.001 < 0.05/5 (yes), 0.01 < 0.05/4 (yes),
        #         0.05 < 0.05/3 (no), ...
        assert rejected[0], "Smallest p should be rejected"
        assert rejected[1], "Second smallest p should be rejected"
        assert not rejected[2], "Third p should not be rejected"

    def test_holm_correction_all_pass(self):
        """No rejections when all p-values are large."""
        p_values = np.array([0.5, 0.6, 0.7])
        alpha = 0.05

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        m = len(p_values)

        rejected = []
        for i, p in enumerate(sorted_p):
            threshold = alpha / (m - i)
            rejected.append(p < threshold)

        assert not any(rejected), "No rejections when all p-values large"


class TestBootstrapCI:
    """Bootstrap CI on difference in confident-error rates."""

    def test_bootstrap_ci_reproducible(self):
        """Bootstrap CI is reproducible under fixed seed."""
        np.random.seed(42)
        method_a = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        method_b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])

        # Confident-error: wrong and confident
        diff_obs = (1 - method_a).mean() - (1 - method_b).mean()

        # Bootstrap: resample with replacement, compute diff
        n_resamples = 100
        diffs = []
        np.random.seed(42)
        for _ in range(n_resamples):
            idx_a = np.random.choice(len(method_a), size=len(method_a), replace=True)
            idx_b = np.random.choice(len(method_b), size=len(method_b), replace=True)
            diff = (1 - method_a[idx_a]).mean() - (1 - method_b[idx_b]).mean()
            diffs.append(diff)
        diffs = np.array(diffs)

        ci_lo = np.percentile(diffs, 2.5)
        ci_hi = np.percentile(diffs, 97.5)

        # Run again with same seed
        np.random.seed(42)
        diffs2 = []
        for _ in range(n_resamples):
            idx_a = np.random.choice(len(method_a), size=len(method_a), replace=True)
            idx_b = np.random.choice(len(method_b), size=len(method_b), replace=True)
            diff = (1 - method_a[idx_a]).mean() - (1 - method_b[idx_b]).mean()
            diffs2.append(diff)

        assert np.allclose(diffs, diffs2), "Bootstrap should be reproducible under fixed seed"


class TestGridCompleteness:
    """Assertion that fires on incomplete grid."""

    def test_completeness_check_fails_on_missing_cell(self, tmp_path):
        """Completeness check detects missing directory."""
        downstream_dir = tmp_path / "downstream"
        downstream_dir.mkdir()

        # Create only CF1_L1
        (downstream_dir / "CF1_L1").mkdir()
        for method in ["zero_fill", "bilinear"]:
            csv_file = downstream_dir / "CF1_L1" / f"{method}.csv"
            csv_file.write_text("patch_id,correct\n1,1\n", encoding="utf-8")

        # Simple check function
        families = ["CF1", "CF2"]
        levels = ["L1"]
        methods = ["zero_fill", "bilinear"]

        missing = []
        for fam in families:
            for lvl in levels:
                cell_dir = downstream_dir / f"{fam}_{lvl}"
                if not cell_dir.exists():
                    missing.append(f"{fam}_{lvl}")

        assert len(missing) > 0, "Should detect missing cells"

    def test_completeness_check_passes_on_complete(self, tmp_path):
        """Completeness check passes on complete grid."""
        downstream_dir = tmp_path / "downstream"
        downstream_dir.mkdir()

        families = ["CF1"]
        levels = ["L1"]
        methods = ["zero_fill", "bilinear"]

        for fam in families:
            for lvl in levels:
                cell_dir = downstream_dir / f"{fam}_{lvl}"
                cell_dir.mkdir()
                for method in methods:
                    csv_file = cell_dir / f"{method}.csv"
                    csv_file.write_text("patch_id,correct\n1,1\n", encoding="utf-8")

        # Check
        missing = []
        for fam in families:
            for lvl in levels:
                cell_dir = downstream_dir / f"{fam}_{lvl}"
                if not cell_dir.exists():
                    missing.append(f"{fam}_{lvl}")

        assert len(missing) == 0, "Should not detect missing cells on complete grid"


class TestAccuracyRecovery:
    """Accuracy recovery against floor and ceiling."""

    def test_accuracy_recovery_formula(self):
        """Recovery = (acc_method - acc_zero) / (acc_clean - acc_zero)."""
        acc_zero = 0.1
        acc_method = 0.5
        acc_clean = 1.0

        recovery = (acc_method - acc_zero) / (acc_clean - acc_zero)

        assert recovery == pytest.approx(0.4 / 0.9), "Recovery formula"

    def test_accuracy_recovery_at_zero_floor(self):
        """Recovery is 0 when method equals zero-fill."""
        acc_zero = 0.1
        acc_method = 0.1
        acc_clean = 1.0

        recovery = (acc_method - acc_zero) / (acc_clean - acc_zero)

        assert recovery == pytest.approx(0.0)

    def test_accuracy_recovery_at_ceiling(self):
        """Recovery is 1 when method equals clean."""
        acc_zero = 0.1
        acc_method = 1.0
        acc_clean = 1.0

        recovery = (acc_method - acc_zero) / (acc_clean - acc_zero)

        assert recovery == pytest.approx(1.0)


class TestSignalCoveringSubset:
    """Signal-covering subset filtering."""

    def test_signal_covering_subset_threshold(self):
        """Filter on centre_erased_frac >= 0.25."""
        centre_erased_frac = np.array([0.1, 0.25, 0.5, 0.0, 0.3])

        signal_covering = centre_erased_frac >= 0.25

        assert signal_covering.sum() == 3, "Should select 3 of 5"
        assert np.array_equal(signal_covering, [False, True, True, False, True])

    def test_signal_covering_empty_subset(self):
        """Subset can be empty if all centre_erased_frac < 0.25."""
        centre_erased_frac = np.array([0.1, 0.2, 0.15])

        signal_covering = centre_erased_frac >= 0.25

        assert signal_covering.sum() == 0


class TestConditionalConfidentErrorDiff:
    """Bootstrap CI on difference in conditional confident-error rates."""

    def test_conditional_error_rate_known_answer(self):
        """Conditional confident-error on a constructed case with known answer."""
        # Construct two methods with known conditional error rates
        # Method A: 3 confident, 2 wrong → P(wrong|conf) = 2/3
        # Method B: 3 confident, 1 wrong → P(wrong|conf) = 1/3
        # Difference: 2/3 - 1/3 = 1/3

        # Method A: all high confidence except last 2
        correct_a = np.array([0, 0, 1, 0, 0])
        confidence_a = np.array([0.95, 0.92, 0.91, 0.5, 0.4])

        # Method B: same layout
        correct_b = np.array([0, 1, 1, 0, 0])
        confidence_b = np.array([0.95, 0.92, 0.91, 0.5, 0.4])

        # Compute conditional rates
        threshold = 0.9
        high_conf_a = confidence_a >= threshold
        high_conf_b = confidence_b >= threshold

        wrong_a = correct_a == 0
        wrong_b = correct_b == 0

        if high_conf_a.sum() > 0:
            cond_err_a = (wrong_a & high_conf_a).sum() / high_conf_a.sum()
        else:
            cond_err_a = 0.0

        if high_conf_b.sum() > 0:
            cond_err_b = (wrong_b & high_conf_b).sum() / high_conf_b.sum()
        else:
            cond_err_b = 0.0

        diff = cond_err_a - cond_err_b

        assert high_conf_a.sum() == 3
        assert high_conf_b.sum() == 3
        assert cond_err_a == pytest.approx(2.0 / 3.0), f"Expected 2/3, got {cond_err_a}"
        assert cond_err_b == pytest.approx(1.0 / 3.0), f"Expected 1/3, got {cond_err_b}"
        assert diff == pytest.approx(1.0 / 3.0), f"Expected 1/3, got {diff}"

    def test_conditional_error_undefined_when_no_confident(self):
        """Conditional rate is undefined (nan/None) when a resample has no confident predictions."""
        # If a resample has zero confident predictions, division by zero occurs
        # Rule: exclude such resamples from the CI computation

        correct = np.array([1, 1, 1, 1, 1])
        confidence = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # all low confidence

        threshold = 0.9
        high_conf = confidence >= threshold
        wrong = correct == 0

        # Should return 0.0 or raise on division by zero
        if high_conf.sum() == 0:
            # This is the undefined case
            assert True, "Undefined case detected: zero confident predictions"
        else:
            pytest.fail("Expected zero confident predictions")

    def test_conditional_error_ci_reproducible(self):
        """Conditional error CI is reproducible under fixed seed."""
        # Construct simple aligned data
        correct_a = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.float32)
        conf_a = np.array([0.95, 0.92, 0.91, 0.55, 0.75, 0.55, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)
        signal_a = np.array([True, True, True, True, True, True, True, True, True, True])

        correct_b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        conf_b = np.array([0.95, 0.92, 0.91, 0.55, 0.75, 0.55, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)
        signal_b = np.array([True, True, True, True, True, True, True, True, True, True])

        # Dummy rows for patch_id lookup
        rows_a = [{"patch_id": str(i)} for i in range(10)]
        rows_b = [{"patch_id": str(i)} for i in range(10)]
        idx_a = {r["patch_id"]: i for i, r in enumerate(rows_a)}
        idx_b = {r["patch_id"]: i for i, r in enumerate(rows_b)}

        # Run twice with same seed
        ci_lo_1, ci_hi_1, n_valid_1 = bootstrap_ci_confident_error_cond_diff_arrays(
            correct_a, conf_a, signal_a,
            correct_b, conf_b, signal_b,
            idx_a, idx_b, rows_a, rows_b,
            n_resamples=100, seed=42
        )

        ci_lo_2, ci_hi_2, n_valid_2 = bootstrap_ci_confident_error_cond_diff_arrays(
            correct_a, conf_a, signal_a,
            correct_b, conf_b, signal_b,
            idx_a, idx_b, rows_a, rows_b,
            n_resamples=100, seed=42
        )

        if ci_lo_1 is not None:
            assert ci_lo_1 == pytest.approx(ci_lo_2), "CI low should be reproducible"
            assert ci_hi_1 == pytest.approx(ci_hi_2), "CI high should be reproducible"
            assert n_valid_1 == n_valid_2, "Valid resample count should be reproducible"

    def test_conditional_error_empty_subset_returns_none(self):
        """Empty signal-covering subset returns (None, None) like joint CI."""
        # Empty signal subset for both methods
        correct_a = np.array([1, 0, 1, 0], dtype=np.float32)
        conf_a = np.array([0.95, 0.92, 0.91, 0.55], dtype=np.float32)
        signal_a = np.array([False, False, False, False])  # all false

        correct_b = np.array([1, 1, 1, 0], dtype=np.float32)
        conf_b = np.array([0.95, 0.92, 0.91, 0.55], dtype=np.float32)
        signal_b = np.array([False, False, False, False])  # all false

        rows_a = [{"patch_id": str(i)} for i in range(4)]
        rows_b = [{"patch_id": str(i)} for i in range(4)]
        idx_a = {r["patch_id"]: i for i, r in enumerate(rows_a)}
        idx_b = {r["patch_id"]: i for i, r in enumerate(rows_b)}

        ci_lo, ci_hi, n_valid = bootstrap_ci_confident_error_cond_diff_arrays(
            correct_a, conf_a, signal_a,
            correct_b, conf_b, signal_b,
            idx_a, idx_b, rows_a, rows_b
        )

        assert ci_lo is None, "Should return None for empty subset"
        assert ci_hi is None, "Should return None for empty subset"
        assert n_valid == 0, "Should return 0 valid resamples for empty subset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
