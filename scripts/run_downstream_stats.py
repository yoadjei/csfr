"""Compute downstream utility metrics and statistics.

Reads the downstream results grid (results/downstream/<FAM>_<LVL>/<method>.csv),
joins with clean.csv, computes metrics per cell (accuracy recovery, ECE,
confident-error rate, abstention), and statistical tests (McNemar with Holm
correction for accuracy; bootstrap CIs on confident-error differences).

CRITICAL SEPARATION OF CONCERNS:
  Accuracy differences are tested by McNemar exact test with Holm-Bonferroni
  correction across baselines per cell. Columns: mcnemar_statistic_accuracy,
  mcnemar_p_raw_accuracy, mcnemar_p_holm_accuracy.

  Confident-error differences are assessed by percentile bootstrap CI (95%,
  10k resamples, seeded for reproducibility) on the signal-covering subset
  (centre_erased_frac >= 0.25). Columns: ci_lo_conferr_diff_signal,
  ci_hi_conferr_diff_signal.

  Do NOT use accuracy p-values to claim confident-error significance, or
  vice versa. Each column names its own metric.

McNemar statistic sign convention (verified against the released data, do not
restate it from memory):
  statistic = b - c, where
    b = patches CSFR gets right and the baseline gets wrong
    c = patches CSFR gets wrong and the baseline gets right
  - Negative: CSFR is BEHIND the baseline on accuracy
  - Positive: CSFR is AHEAD of the baseline on accuracy
  Worked example from results/downstream, CF3_L5, CSFR vs telea: b=92, c=183,
  so statistic = -91, and the accuracies are CSFR 0.222 against telea 0.313.
  CSFR is behind, which is what the negative sign means.

Known properties (documented a priori; no tuning after results):
  - Bounded-Unknown boolean saturates by (family, level): 0.00 for CF1 all
    levels; 0.88-1.0 for CF2 (L1-L5); 0 for CF3_L1, 1.0 for CF3_L2+. The
    continuous isolated_missing_frac carries per-patch information; use that
    for reporting, not the saturated flag.
  - Signal-covering subset (centre_erased_frac >= 0.25) only discriminates
    CF2 (~352 at L1, ~760 at L2+). For CF1, CF3, CF4 it is empty, complete,
    or near-complete per cell by mask geometry alone. Metrics report n_signal
    per cell; if n_signal == 0 or n_signal == len(cell), subset is degenerate
    and CI is meaningless for that cell.

Usage:
    python scripts/run_downstream_stats.py

Outputs:
    results/downstream/metrics.csv — one row per family, level, method
    results/downstream/stats.csv — one row per family, level, method_a, method_b
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "downstream"
OUT_METRICS = RESULTS_DIR / "metrics.csv"
OUT_STATS = RESULTS_DIR / "stats.csv"

# Configuration (from downstream.yaml)
FAMILIES = ["CF1", "CF2", "CF3", "CF4"]
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
# lama is the external-prior generative inpainter. it is optional: it needs
# external weights and a gpu, so a grid without it is still analysable. every
# method present in the results is analysed, rather than a fixed list, so a new
# baseline cannot be silently dropped from the tables.
CORE_METHODS = ["zero_fill", "bilinear", "telea", "ns", "dictlearn", "csfr"]
OPTIONAL_METHODS = ["lama"]


def discover_methods() -> list:
    """Core methods, plus any optional method actually present in every cell.

    An optional method that is present in some cells but not others is dropped
    with a warning: a metric averaged over a different set of cells per method
    would not be comparable.
    """
    found = []
    for m in OPTIONAL_METHODS:
        present = [
            (RESULTS_DIR / f"{fam}_{lvl}" / f"{m}.csv").exists()
            for fam in FAMILIES for lvl in LEVELS
        ]
        if all(present):
            found.append(m)
        elif any(present):
            print(f"[downstream_stats] {m} present in only "
                  f"{sum(present)}/{len(present)} cells; excluded")
    return CORE_METHODS + found


METHODS = discover_methods()
SIGNAL_COVERING_THRESHOLD = 0.25
BOUNDED_UNKNOWN_THRESHOLD = 0.5
CONFIDENT_THRESHOLD = 0.9
N_BINS_ECE = 15
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


def assert_grid_complete() -> None:
    """Raise if grid is incomplete; silently pass otherwise."""
    families = FAMILIES
    levels = LEVELS
    methods = METHODS

    missing = []
    for fam in families:
        for lvl in levels:
            cell_dir = RESULTS_DIR / f"{fam}_{lvl}"
            if not cell_dir.exists():
                missing.append(f"{fam}_{lvl}")
                continue
            for method in methods:
                csv_file = cell_dir / f"{method}.csv"
                if not csv_file.exists():
                    missing.append(f"{fam}_{lvl}/{method}.csv")
                else:
                    # Check row count
                    with open(csv_file) as f:
                        rows = len(f.readlines()) - 1  # subtract header
                        if rows != 1000:
                            missing.append(f"{fam}_{lvl}/{method}.csv ({rows} rows, expected 1000)")

    clean = RESULTS_DIR / "clean.csv"
    if not clean.exists():
        missing.append("clean.csv")
    else:
        with open(clean) as f:
            rows = len(f.readlines()) - 1
            if rows != 1000:
                missing.append(f"clean.csv ({rows} rows, expected 1000)")

    if missing:
        raise RuntimeError(
            f"Grid incomplete. Missing or malformed:\n  " + "\n  ".join(missing)
        )


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dict_to_arrays(rows: list[dict], *keys: str) -> dict[str, np.ndarray]:
    """Convert list of dicts to dict of numpy arrays, one per key."""
    result = {}
    for key in keys:
        values = []
        for r in rows:
            val = r.get(key, 0)
            # Handle boolean strings
            if isinstance(val, str):
                if val.lower() in ("true", "1"):
                    values.append(1)
                elif val.lower() in ("false", "0"):
                    values.append(0)
                else:
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(0)
            else:
                values.append(val)
        try:
            result[key] = np.array(values, dtype=np.float32)
        except (ValueError, TypeError):
            result[key] = np.array(values, dtype=np.int32)
    return result


def merge_on_patch_id(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge two lists of dicts on patch_id, returning aligned lists."""
    idx_a = {r["patch_id"]: r for r in rows_a}
    idx_b = {r["patch_id"]: r for r in rows_b}
    common_ids = set(idx_a.keys()) & set(idx_b.keys())

    merged_a = [idx_a[pid] for pid in sorted(common_ids)]
    merged_b = [idx_b[pid] for pid in sorted(common_ids)]

    return merged_a, merged_b


def accuracy(correct: np.ndarray) -> float:
    """Top-1 accuracy from correct vector."""
    return float((correct == 1).sum()) / len(correct)


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS_ECE) -> float:
    """Expected Calibration Error with equal-width bins."""
    ece_val = 0.0
    for bin_idx in range(n_bins):
        low = bin_idx / n_bins
        high = (bin_idx + 1) / n_bins
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            bin_acc = (correct[mask] == 1).mean()
            bin_conf = confidences[mask].mean()
            ece_val += mask.sum() / len(correct) * abs(bin_acc - bin_conf)

    return float(ece_val)


def confident_error_joint(correct: np.ndarray, confidences: np.ndarray,
                          threshold: float = CONFIDENT_THRESHOLD) -> tuple[float, int]:
    """P(wrong and confidence >= threshold) and count of confident predictions.

    Returns:
        (joint_rate, n_confident)
    """
    high_conf = confidences >= threshold
    wrong = correct == 0
    joint_rate = float((wrong & high_conf).sum()) / len(correct)
    n_confident = int(high_conf.sum())

    return joint_rate, n_confident


def confident_error_conditional(correct: np.ndarray, confidences: np.ndarray,
                                threshold: float = CONFIDENT_THRESHOLD) -> float:
    """P(wrong | confidence >= threshold).

    Returns:
        conditional_rate (0 if no confident predictions)
    """
    high_conf = confidences >= threshold
    if high_conf.sum() == 0:
        return 0.0

    wrong = correct == 0
    return float((wrong & high_conf).sum()) / high_conf.sum()


def isolated_missing_frac_stats(frac: np.ndarray) -> dict:
    """Distribution of isolated_missing_frac for CSFR abstention."""
    return {
        "mean": float(frac.mean()),
        "sd": float(frac.std()),
        "q25": float(np.percentile(frac, 25)),
        "median": float(np.percentile(frac, 50)),
        "q75": float(np.percentile(frac, 75)),
    }


def compute_metrics(
    families: list[str] = FAMILIES,
    levels: list[str] = LEVELS,
    methods: list[str] = METHODS,
) -> list[dict]:
    """Compute metrics per (family, level, method)."""
    assert_grid_complete()

    # Load clean baseline
    clean_rows = read_csv(RESULTS_DIR / "clean.csv")
    clean_data = dict_to_arrays(clean_rows, "confidence", "correct")
    clean_acc = accuracy(clean_data["correct"])
    clean_ece = ece(clean_data["confidence"], clean_data["correct"])

    rows = []
    for family in families:
        for level in levels:
            cell_dir = RESULTS_DIR / f"{family}_{level}"

            # Load zero_fill baseline for recovery calculation
            zero_fill_rows = read_csv(cell_dir / "zero_fill.csv")
            zf_data = dict_to_arrays(zero_fill_rows, "confidence", "correct")
            zero_fill_acc = accuracy(zf_data["correct"])

            # Process each method
            for method in methods:
                csv_file = cell_dir / f"{method}.csv"
                method_rows = read_csv(csv_file)

                # Align with clean (on patch_id)
                method_rows, clean_aligned = merge_on_patch_id(method_rows, clean_rows)

                method_data = dict_to_arrays(
                    method_rows, "confidence", "correct", "centre_erased_frac", "isolated_missing_frac",
                    "bounded_unknown"
                )
                clean_aligned_data = dict_to_arrays(clean_aligned, "correct")

                method_acc = accuracy(method_data["correct"])
                method_ece = ece(method_data["confidence"], method_data["correct"])

                # Accuracy recovery
                if clean_acc > zero_fill_acc:
                    recovery = (method_acc - zero_fill_acc) / (clean_acc - zero_fill_acc)
                else:
                    recovery = 0.0

                # Confident-error: joint and conditional
                conf_err_joint, n_confident = confident_error_joint(
                    method_data["correct"], method_data["confidence"]
                )
                conf_err_cond = confident_error_conditional(
                    method_data["correct"], method_data["confidence"]
                )

                # Signal-covering subset (centre_erased_frac >= 0.25)
                signal_mask = method_data["centre_erased_frac"] >= SIGNAL_COVERING_THRESHOLD
                n_signal = int(signal_mask.sum())
                if n_signal > 0:
                    signal_acc = accuracy(method_data["correct"][signal_mask])
                    signal_conf_err_joint, signal_n_confident = confident_error_joint(
                        method_data["correct"][signal_mask], method_data["confidence"][signal_mask]
                    )
                    signal_conf_err_cond = confident_error_conditional(
                        method_data["correct"][signal_mask], method_data["confidence"][signal_mask]
                    )
                else:
                    signal_acc = None
                    signal_conf_err_joint = None
                    signal_conf_err_cond = None
                    signal_n_confident = 0

                # Bounded-Unknown (CSFR only)
                if method == "csfr":
                    bu_rate = float((method_data["bounded_unknown"] == 1).sum()) / len(method_data["bounded_unknown"])
                    # Accuracy on non-abstained patches
                    non_abstained = method_data["bounded_unknown"] == 0
                    if non_abstained.sum() > 0:
                        non_abstained_acc = accuracy(method_data["correct"][non_abstained])
                    else:
                        non_abstained_acc = None
                    # Distribution of isolated_missing_frac
                    imf_stats = isolated_missing_frac_stats(method_data["isolated_missing_frac"])
                else:
                    bu_rate = None
                    non_abstained_acc = None
                    imf_stats = None

                rows.append({
                    "family": family,
                    "level": level,
                    "method": method,
                    "accuracy": method_acc,
                    "recovery": recovery,
                    "ece": method_ece,
                    "conf_err_joint": conf_err_joint,
                    "conf_err_cond": conf_err_cond,
                    "n_confident": n_confident,
                    "signal_n": n_signal,
                    "signal_accuracy": signal_acc,
                    "signal_conf_err_joint": signal_conf_err_joint,
                    "signal_conf_err_cond": signal_conf_err_cond,
                    "signal_n_confident": signal_n_confident,
                    "bu_rate": bu_rate,
                    "non_abstained_acc": non_abstained_acc,
                    "imf_mean": imf_stats["mean"] if imf_stats else None,
                    "imf_sd": imf_stats["sd"] if imf_stats else None,
                    "imf_q25": imf_stats["q25"] if imf_stats else None,
                    "imf_median": imf_stats["median"] if imf_stats else None,
                    "imf_q75": imf_stats["q75"] if imf_stats else None,
                })

    return rows


def mcnemar_test_arrays(
    correct_a: np.ndarray, correct_b: np.ndarray,
    idx_a: dict[str, int], idx_b: dict[str, int],
) -> tuple[float, float]:
    """Paired McNemar test on correctness arrays, aligned on patch IDs.

    Args:
        correct_a: Correctness array for method A.
        correct_b: Correctness array for method B.
        idx_a: Mapping from patch_id to index in correct_a.
        idx_b: Mapping from patch_id to index in correct_b.

    Returns:
        (test_statistic, p_value)
    """
    # Align on common patch IDs
    common_ids = set(idx_a.keys()) & set(idx_b.keys())
    indices_a = [idx_a[pid] for pid in sorted(common_ids)]
    indices_b = [idx_b[pid] for pid in sorted(common_ids)]

    correct_a_aligned = correct_a[indices_a]
    correct_b_aligned = correct_b[indices_b]

    # Count discordant pairs
    a_correct_b_wrong = (correct_a_aligned == 1) & (correct_b_aligned == 0)
    a_wrong_b_correct = (correct_a_aligned == 0) & (correct_b_aligned == 1)

    b = int(a_correct_b_wrong.sum())  # a wins
    c = int(a_wrong_b_correct.sum())  # b wins
    n_discordant = b + c

    if n_discordant == 0:
        # No discordance; cannot compute test
        return 0.0, 1.0

    # Exact binomial test: under null, P(b | discordant) = 0.5
    result = binomtest(b, n_discordant, p=0.5, alternative="two-sided")

    return float(b - c), result.pvalue


def holm_bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Holm-Bonferroni correction to a p-value vector.

    Args:
        p_values: Array of p-values.

    Returns:
        Corrected p-values (capped at 1.0).
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    corrected = np.zeros(m)
    for i, p in enumerate(sorted_p):
        corrected[sorted_indices[i]] = min(1.0, p * (m - i))

    return corrected


def bootstrap_ci_confident_error_diff_arrays(
    correct_csfr: np.ndarray, conf_csfr: np.ndarray, signal_mask_csfr: np.ndarray,
    correct_baseline: np.ndarray, conf_baseline: np.ndarray, signal_mask_baseline: np.ndarray,
    idx_csfr: dict[str, int], idx_baseline: dict[str, int],
    rows_csfr: list[dict], rows_baseline: list[dict],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap CI on difference in confident-error rate (CSFR - baseline).

    Args:
        correct_csfr, conf_csfr, signal_mask_csfr: CSFR arrays
        correct_baseline, conf_baseline, signal_mask_baseline: Baseline arrays
        idx_csfr, idx_baseline: Patch ID mappings
        rows_csfr, rows_baseline: Original rows for patch_id lookup
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (ci_low, ci_high) on the difference in confident-error rate.
    """
    np.random.seed(seed)

    # Align on common patch IDs
    common_ids = (set(idx_csfr.keys()) & set(idx_baseline.keys()))
    sorted_ids = sorted(common_ids)

    indices_csfr = [idx_csfr[pid] for pid in sorted_ids]
    indices_baseline = [idx_baseline[pid] for pid in sorted_ids]

    correct_csfr_aligned = correct_csfr[indices_csfr]
    conf_csfr_aligned = conf_csfr[indices_csfr]
    signal_csfr_aligned = signal_mask_csfr[indices_csfr]

    correct_baseline_aligned = correct_baseline[indices_baseline]
    conf_baseline_aligned = conf_baseline[indices_baseline]
    signal_baseline_aligned = signal_mask_baseline[indices_baseline]

    # Filter to signal-covering subset
    conf_csfr_sig = conf_csfr_aligned[signal_csfr_aligned]
    correct_csfr_sig = correct_csfr_aligned[signal_csfr_aligned]
    conf_baseline_sig = conf_baseline_aligned[signal_baseline_aligned]
    correct_baseline_sig = correct_baseline_aligned[signal_baseline_aligned]

    if len(correct_csfr_sig) == 0 or len(correct_baseline_sig) == 0:
        return None, None

    # Bootstrap
    diffs = []
    for _ in range(n_resamples):
        idx_csfr_boot = np.random.choice(len(correct_csfr_sig), size=len(correct_csfr_sig), replace=True)
        idx_baseline_boot = np.random.choice(len(correct_baseline_sig), size=len(correct_baseline_sig), replace=True)

        conf_csfr_boot = conf_csfr_sig[idx_csfr_boot]
        correct_csfr_boot = correct_csfr_sig[idx_csfr_boot]
        conf_baseline_boot = conf_baseline_sig[idx_baseline_boot]
        correct_baseline_boot = correct_baseline_sig[idx_baseline_boot]

        wrong_csfr_boot = correct_csfr_boot == 0
        high_conf_csfr_boot = conf_csfr_boot >= CONFIDENT_THRESHOLD
        wrong_baseline_boot = correct_baseline_boot == 0
        high_conf_baseline_boot = conf_baseline_boot >= CONFIDENT_THRESHOLD

        diff_boot = (wrong_csfr_boot & high_conf_csfr_boot).sum() / len(correct_csfr_boot) - \
                   (wrong_baseline_boot & high_conf_baseline_boot).sum() / len(correct_baseline_boot)
        diffs.append(diff_boot)

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    return ci_lo, ci_hi


def bootstrap_ci_confident_error_cond_diff_arrays(
    correct_csfr: np.ndarray, conf_csfr: np.ndarray, signal_mask_csfr: np.ndarray,
    correct_baseline: np.ndarray, conf_baseline: np.ndarray, signal_mask_baseline: np.ndarray,
    idx_csfr: dict[str, int], idx_baseline: dict[str, int],
    rows_csfr: list[dict], rows_baseline: list[dict],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Optional[float], Optional[float], int]:
    """Bootstrap CI on difference in conditional confident-error rate (CSFR - baseline).

    Conditional confident-error is P(wrong | confidence >= threshold), computed only
    on predictions with confidence >= 0.9. When a bootstrap resample contains zero
    confident predictions for a method, the conditional rate is undefined and that
    resample is excluded from the CI computation (silently; count returned separately).

    Args:
        correct_csfr, conf_csfr, signal_mask_csfr: CSFR arrays
        correct_baseline, conf_baseline, signal_mask_baseline: Baseline arrays
        idx_csfr, idx_baseline: Patch ID mappings
        rows_csfr, rows_baseline: Original rows for patch_id lookup
        n_resamples: Number of bootstrap resamples (target; actual may be lower due exclusion).
        seed: Random seed.

    Returns:
        (ci_low, ci_high, n_valid) where n_valid is the count of resamples with
        confident predictions in both methods. If signal subset is empty, returns
        (None, None, 0).
    """
    np.random.seed(seed)

    # Align on common patch IDs
    common_ids = (set(idx_csfr.keys()) & set(idx_baseline.keys()))
    sorted_ids = sorted(common_ids)

    indices_csfr = [idx_csfr[pid] for pid in sorted_ids]
    indices_baseline = [idx_baseline[pid] for pid in sorted_ids]

    correct_csfr_aligned = correct_csfr[indices_csfr]
    conf_csfr_aligned = conf_csfr[indices_csfr]
    signal_csfr_aligned = signal_mask_csfr[indices_csfr]

    correct_baseline_aligned = correct_baseline[indices_baseline]
    conf_baseline_aligned = conf_baseline[indices_baseline]
    signal_baseline_aligned = signal_mask_baseline[indices_baseline]

    # Filter to signal-covering subset
    conf_csfr_sig = conf_csfr_aligned[signal_csfr_aligned]
    correct_csfr_sig = correct_csfr_aligned[signal_csfr_aligned]
    conf_baseline_sig = conf_baseline_aligned[signal_baseline_aligned]
    correct_baseline_sig = correct_baseline_aligned[signal_baseline_aligned]

    if len(correct_csfr_sig) == 0 or len(correct_baseline_sig) == 0:
        return None, None, 0

    # Bootstrap
    diffs = []
    for _ in range(n_resamples):
        idx_csfr_boot = np.random.choice(len(correct_csfr_sig), size=len(correct_csfr_sig), replace=True)
        idx_baseline_boot = np.random.choice(len(correct_baseline_sig), size=len(correct_baseline_sig), replace=True)

        conf_csfr_boot = conf_csfr_sig[idx_csfr_boot]
        correct_csfr_boot = correct_csfr_sig[idx_csfr_boot]
        conf_baseline_boot = conf_baseline_sig[idx_baseline_boot]
        correct_baseline_boot = correct_baseline_sig[idx_baseline_boot]

        # Conditional error rate: P(wrong | conf >= threshold)
        wrong_csfr_boot = correct_csfr_boot == 0
        high_conf_csfr_boot = conf_csfr_boot >= CONFIDENT_THRESHOLD
        wrong_baseline_boot = correct_baseline_boot == 0
        high_conf_baseline_boot = conf_baseline_boot >= CONFIDENT_THRESHOLD

        n_conf_csfr = high_conf_csfr_boot.sum()
        n_conf_baseline = high_conf_baseline_boot.sum()

        # Skip this resample if either method has no confident predictions
        if n_conf_csfr == 0 or n_conf_baseline == 0:
            continue

        cond_err_csfr = (wrong_csfr_boot & high_conf_csfr_boot).sum() / n_conf_csfr
        cond_err_baseline = (wrong_baseline_boot & high_conf_baseline_boot).sum() / n_conf_baseline

        diff_boot = cond_err_csfr - cond_err_baseline
        diffs.append(diff_boot)

    n_valid = len(diffs)
    if n_valid == 0:
        return None, None, 0

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    return ci_lo, ci_hi, n_valid


def compute_statistics(
    metrics_rows: list[dict],
    families: list[str] = FAMILIES,
    levels: list[str] = LEVELS,
    methods: list[str] = METHODS,
) -> list[dict]:
    """Compute paired McNemar tests and bootstrap CIs.

    Tests CSFR against each baseline per cell. Computes two bootstrap CIs:
    - Joint confident-error difference: P(wrong and confident >= 0.9)
    - Conditional confident-error difference: P(wrong | confident >= 0.9)
    """
    stats_rows = []

    for family in families:
        for level in levels:
            cell_dir = RESULTS_DIR / f"{family}_{level}"

            # Load CSFR
            csfr_rows = read_csv(cell_dir / "csfr.csv")
            csfr_data = dict_to_arrays(csfr_rows, "confidence", "correct", "centre_erased_frac")
            csfr_signal_mask = csfr_data["centre_erased_frac"] >= SIGNAL_COVERING_THRESHOLD

            # Create patch_id lookup for csfr
            csfr_idx = {r["patch_id"]: i for i, r in enumerate(csfr_rows)}

            # Test CSFR against each baseline
            baseline_methods = [m for m in methods if m != "csfr"]
            baseline_p_values = []

            for baseline_method in baseline_methods:
                baseline_rows = read_csv(cell_dir / f"{baseline_method}.csv")
                baseline_data = dict_to_arrays(baseline_rows, "confidence", "correct", "centre_erased_frac")
                baseline_signal_mask = baseline_data["centre_erased_frac"] >= SIGNAL_COVERING_THRESHOLD

                # Create patch_id lookup for baseline
                baseline_idx = {r["patch_id"]: i for i, r in enumerate(baseline_rows)}

                # McNemar test (aligned on patch_id)
                test_stat, p_value = mcnemar_test_arrays(
                    csfr_data["correct"], baseline_data["correct"], csfr_idx, baseline_idx
                )
                baseline_p_values.append(p_value)

                # Bootstrap CI on joint confident-error difference (signal-covering subset)
                try:
                    ci_lo, ci_hi = bootstrap_ci_confident_error_diff_arrays(
                        csfr_data["correct"], csfr_data["confidence"], csfr_signal_mask,
                        baseline_data["correct"], baseline_data["confidence"], baseline_signal_mask,
                        csfr_idx, baseline_idx, csfr_rows, baseline_rows
                    )
                except Exception as e:
                    print(f"Warning: bootstrap CI (joint) failed for {family}_{level} CSFR vs {baseline_method}: {e}")
                    ci_lo, ci_hi = None, None

                # Bootstrap CI on conditional confident-error difference (signal-covering subset)
                try:
                    ci_lo_cond, ci_hi_cond, n_valid_cond = bootstrap_ci_confident_error_cond_diff_arrays(
                        csfr_data["correct"], csfr_data["confidence"], csfr_signal_mask,
                        baseline_data["correct"], baseline_data["confidence"], baseline_signal_mask,
                        csfr_idx, baseline_idx, csfr_rows, baseline_rows
                    )
                except Exception as e:
                    print(f"Warning: bootstrap CI (conditional) failed for {family}_{level} CSFR vs {baseline_method}: {e}")
                    ci_lo_cond, ci_hi_cond, n_valid_cond = None, None, None

                stats_rows.append({
                    "family": family,
                    "level": level,
                    "method_a": "csfr",
                    "method_b": baseline_method,
                    "mcnemar_statistic_accuracy": test_stat,
                    "mcnemar_p_raw_accuracy": p_value,
                    "mcnemar_p_holm_accuracy": None,  # filled in Holm correction step
                    "ci_lo_conferr_diff_signal": ci_lo,
                    "ci_hi_conferr_diff_signal": ci_hi,
                    "ci_lo_conferr_cond_diff_signal": ci_lo_cond,
                    "ci_hi_conferr_cond_diff_signal": ci_hi_cond,
                    "n_valid_cond_resamples": n_valid_cond,
                })

            # Holm correction across baseline methods
            if baseline_p_values:
                corrected_p = holm_bonferroni_correction(np.array(baseline_p_values))
                for i, baseline_method in enumerate(baseline_methods):
                    # Find the row and update mcnemar_p_holm_accuracy
                    for row in stats_rows:
                        if (row["family"] == family and row["level"] == level and
                            row["method_b"] == baseline_method):
                            row["mcnemar_p_holm_accuracy"] = corrected_p[i]

    return stats_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write list of dicts to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Main entry point."""
    print("[downstream_stats] Computing metrics...")
    metrics_rows = compute_metrics()
    write_csv(OUT_METRICS, metrics_rows)
    print(f"[downstream_stats] Wrote {len(metrics_rows)} rows to {OUT_METRICS}")

    print("[downstream_stats] Computing statistics...")
    stats_rows = compute_statistics(metrics_rows)
    write_csv(OUT_STATS, stats_rows)
    print(f"[downstream_stats] Wrote {len(stats_rows)} rows to {OUT_STATS}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
