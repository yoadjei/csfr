"""Smoke tests for the downstream utility harness (Task 1b)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_downstream import (  # noqa: E402
    apply_smoke_overrides,
    compute_bounded_unknown_stats,
    main,
    mask_overlap_centre,
    select_class_balanced,
)


def test_bounded_unknown_all_missing():
    M = np.zeros((8, 8), dtype=np.float32)
    flag, frac = compute_bounded_unknown_stats(M, threshold=0.5)
    assert flag is True
    assert frac == pytest.approx(1.0)  # all 64 pixels are isolated


def test_bounded_unknown_fully_observed():
    M = np.ones((8, 8), dtype=np.float32)
    flag, frac = compute_bounded_unknown_stats(M, threshold=0.5)
    assert flag is False
    assert frac == pytest.approx(0.0)  # no missing pixels


def test_bounded_unknown_well_observed_hole():
    """A single interior hole has four observed neighbours → not flagged."""
    from run_downstream import compute_bounded_unknown_stats
    M = np.ones((8, 8), dtype=np.float32)
    M[3, 3] = 0.0
    flag, frac = compute_bounded_unknown_stats(M, threshold=0.5)
    assert flag is False
    assert frac == pytest.approx(0.0)  # one isolated pixel out of one missing


def test_mask_overlap_centre_all_in_box():
    M = np.ones((10, 10), dtype=np.float32)
    M[3:7, 3:7] = 0.0  # erased block sits in the centre half-box
    overlap = mask_overlap_centre(M, centre_frac=0.5)
    assert overlap == pytest.approx(1.0)


def test_select_class_balanced_counts():
    import torch

    n = 500
    labels = torch.arange(n) % 10
    images = torch.zeros(n, 1, 4, 4)
    imgs, labs, idx = select_class_balanced(images, labels, n_test=100, seed=0)
    assert imgs.shape[0] == 100
    assert labs.shape[0] == 100
    assert idx.shape[0] == 100
    counts = np.bincount(labs.numpy(), minlength=10)
    assert counts.min() >= 10
    assert counts.max() <= 10


def test_smoke_cli(tmp_path):
    """1 family × 1 level × 2 methods × ~20 patches writes CSVs."""
    import yaml

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg_path = tmp_path / "downstream_smoke.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    rc = main(["--config", str(cfg_path), "--smoke", "--force", "--device", "cpu"])
    assert rc == 0
    assert (out / "summary.csv").exists()
    assert (out / "environment.json").exists()
    zero_csv = out / "CF1_L1" / "zero_fill.csv"
    bilin_csv = out / "CF1_L1" / "bilinear.csv"
    assert zero_csv.exists()
    assert bilin_csv.exists()
    rows = list(csv.DictReader(zero_csv.open(encoding="utf-8")))
    assert len(rows) == 20
    assert {"true_label", "prediction", "confidence", "correct",
            "bounded_unknown", "mask_overlap_centre"} <= set(rows[0])
    # non-CSFR methods record bounded_unknown as False
    assert rows[0]["bounded_unknown"] in ("False", "false", "0")


def test_smoke_overrides_shrink_grid():
    cfg = {
        "n_test": 200,
        "families": ["CF1", "CF2", "CF3", "CF4"],
        "levels": {"L1": 0.05, "L2": 0.15},
        "methods": ["zero_fill", "bilinear", "telea", "ns", "dictlearn", "csfr"],
        "seeds": [0, 1],
    }
    sm = apply_smoke_overrides(cfg)
    assert sm["n_test"] == 20
    assert sm["families"] == ["CF1"]
    assert list(sm["levels"]) == ["L1"]
    assert sm["methods"] == ["zero_fill", "bilinear"]
    assert sm["seeds"] == [0]


def test_clean_csv_is_written(tmp_path):
    """Fix 1: clean.csv is written with one row per subset patch."""
    import torch
    import yaml

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    from run_downstream import run_pipeline
    run_pipeline(cfg, force=True)

    clean_csv = out / "clean.csv"
    assert clean_csv.exists(), f"clean.csv not found at {clean_csv}"
    rows = list(csv.DictReader(clean_csv.open(encoding="utf-8")))
    assert len(rows) == 20, f"expected 20 rows in clean.csv, got {len(rows)}"
    assert set(rows[0].keys()) >= {"patch_id", "true_label", "prediction", "confidence", "correct"}
    # Verify predictions match direct inference on clean data
    from downstream import load_digit_dataset, load_classifier, predict
    images, labels, _ = load_digit_dataset(prefer="svhn", split="test")
    from run_downstream import select_class_balanced, _to_uint_scale, _to_classifier_tensor
    imgs, labs, patch_ids = select_class_balanced(images, labels, 20, 0)
    patches = _to_uint_scale(imgs)
    model = load_classifier(ckpt, device=torch.device("cpu"), freeze_weights=True)
    pred, conf = predict(model, _to_classifier_tensor(patches), device=torch.device("cpu"))
    for i, row in enumerate(rows):
        assert int(row["prediction"]) == int(pred[i]), f"row {i}: prediction mismatch"
        assert int(row["correct"]) == int(pred[i] == labs[i]), f"row {i}: correct mismatch"


def test_resume_preserves_other_seeds(tmp_path):
    """Fix 2: resuming with a subset of seeds preserves rows from other seeds."""
    import yaml

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0, 1]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    from run_downstream import run_pipeline
    run_pipeline(cfg, force=True)

    # First CSV should have both seeds
    csv_path = out / "CF1_L1" / "zero_fill.csv"
    rows_both = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows_both) == 40, f"expected 40 rows (2 seeds × 20), got {len(rows_both)}"
    n_seed0 = sum(1 for r in rows_both if int(r["seed"]) == 0)
    n_seed1 = sum(1 for r in rows_both if int(r["seed"]) == 1)
    assert n_seed0 == 20 and n_seed1 == 20, f"seed counts: {n_seed0}, {n_seed1}"

    # Now rerun with only seed 0; seed 1 rows should be preserved
    cfg["seeds"] = [0]
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    run_pipeline(cfg, force=False)  # resume, not force

    rows_resume = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows_resume) == 40, f"expected 40 rows after resume with seed=0, got {len(rows_resume)}"
    n_seed1_after = sum(1 for r in rows_resume if int(r["seed"]) == 1)
    assert n_seed1_after == 20, f"seed 1 rows were destroyed in resume; expected 20, got {n_seed1_after}"


def test_incomplete_csv_not_accepted_as_complete(tmp_path):
    """Fix 2: a CSV with fewer rows than n_test is not treated as complete."""
    import yaml

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    from run_downstream import run_pipeline
    # Write a partial CSV to simulate an incomplete run
    csv_path = out / "CF1_L1" / "zero_fill.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "seed", "true_label", "prediction"])
        w.writeheader()
        # Only 5 rows instead of 20
        for i in range(5):
            w.writerow({"patch_id": i, "seed": 0, "true_label": 5, "prediction": 5})

    # Run without --force; should recompute because CSV is incomplete
    run_pipeline(cfg, force=False)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 20, f"expected recompute with 20 rows, got {len(rows)}"


def test_centre_erased_frac_correct(tmp_path):
    """Fix 3: centre_erased_frac is computed correctly and is monotone in frac."""
    import torch
    import yaml
    from run_downstream import centre_box_mask

    # Hand-computed test: 10×10 image, centre_frac=0.5 (5×5 centre box)
    M = np.ones((10, 10), dtype=np.float32)
    M[2:7, 2:7] = 0.0  # 5×5 erased block (the entire centre 5×5 box)

    # centre_erased_frac should be 25 / 25 = 1.0
    from run_downstream import centre_erased_frac_stat
    frac = centre_erased_frac_stat(M, centre_frac=0.5)
    assert frac == pytest.approx(1.0), f"expected 1.0, got {frac}"

    # Test monotonicity: CF1 with increasing frac should increase centre_erased_frac
    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 10
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05, "L2": 0.15}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    from run_downstream import run_pipeline
    run_pipeline(cfg, force=True)

    # Check both CSVs have the new column
    for lvl in ["L1", "L2"]:
        csv_path = out / f"CF1_{lvl}" / "zero_fill.csv"
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert len(rows) > 0
        assert "centre_erased_frac" in rows[0], f"centre_erased_frac not in {lvl} CSV"
        # L2 (frac=0.15) should have higher mean centre_erased_frac than L1 (frac=0.05)
        # due to more erasure
        mean_l1 = np.mean([float(r["centre_erased_frac"]) for r in rows])
        if lvl == "L2":
            rows_l1 = list(csv.DictReader((out / "CF1_L1" / "zero_fill.csv").open(encoding="utf-8")))
            mean_l2 = mean_l1
            mean_l1 = np.mean([float(r["centre_erased_frac"]) for r in rows_l1])
            assert mean_l2 > mean_l1, f"L2 mean {mean_l2} should be > L1 mean {mean_l1}"


def test_scale_inference_explicit_or_assert(tmp_path):
    """Fix 4: scale inference is explicit, allows small overruns, clips to valid range."""
    import torch
    from run_downstream import _to_uint_scale, _to_classifier_tensor

    # Valid input in [0, 1]
    x = torch.rand(2, 4, 4)
    rec255 = _to_uint_scale(x)
    assert float(rec255.max()) <= 255.0, "scaled value exceeds 255"

    # Valid input in [0, 255]
    x = torch.randint(0, 256, (2, 4, 4)).float()
    rec255 = _to_uint_scale(x)
    assert float(rec255.max()) <= 255.0, "should preserve [0, 255]"

    # Small overshoot [255, 256] should be clipped, not rejected
    x_overshoot = torch.ones(2, 4, 4) * 255.5
    rec255 = _to_uint_scale(x_overshoot, scale="uint8")
    assert float(rec255.max()) <= 255.0, "should clip overshoot to [0, 255]"

    # Genuinely wrong scale (far out of range) should fail loudly
    x_bad = torch.ones(2, 4, 4) * 1000  # way out of range
    with pytest.raises((AssertionError, ValueError)):
        _to_uint_scale(x_bad, scale="uint8")

    # Round-trip test: [0, 1] -> [0, 255] -> [0, 1]
    x = torch.rand(2, 4, 4)
    rec255 = _to_uint_scale(x)
    x_back = _to_classifier_tensor(rec255)
    assert x_back.shape == (2, 1, 4, 4), f"shape mismatch: {x_back.shape}"
    assert float(x_back.max()) <= 1.0, "classifier input should be in [0, 1]"


def test_resume_idempotence_multiple_seeds(tmp_path):
    """Fix 2 (revised): idempotence - running 3 times yields same row counts."""
    import yaml
    from run_downstream import run_pipeline

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0, 1]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    # Run 1: initial
    run_pipeline(cfg, force=True)
    csv_path = out / "CF1_L1" / "zero_fill.csv"
    rows1 = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    count1 = len(rows1)
    n_seed0_r1 = sum(1 for r in rows1 if int(r["seed"]) == 0)
    n_seed1_r1 = sum(1 for r in rows1 if int(r["seed"]) == 1)

    # Run 2: resume (should be identical)
    run_pipeline(cfg, force=False)
    rows2 = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    count2 = len(rows2)
    n_seed0_r2 = sum(1 for r in rows2 if int(r["seed"]) == 0)
    n_seed1_r2 = sum(1 for r in rows2 if int(r["seed"]) == 1)

    # Run 3: resume again (should still be identical)
    run_pipeline(cfg, force=False)
    rows3 = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    count3 = len(rows3)
    n_seed0_r3 = sum(1 for r in rows3 if int(r["seed"]) == 0)
    n_seed1_r3 = sum(1 for r in rows3 if int(r["seed"]) == 1)

    # All runs should have same row counts
    assert count1 == 40, f"run 1: expected 40, got {count1}"
    assert count2 == 40, f"run 2: expected 40, got {count2}"
    assert count3 == 40, f"run 3: expected 40, got {count3}"
    assert n_seed0_r1 == 20 and n_seed1_r1 == 20
    assert n_seed0_r2 == 20 and n_seed1_r2 == 20
    assert n_seed0_r3 == 20 and n_seed1_r3 == 20


def test_recompute_preserves_other_seeds(tmp_path):
    """Fix 2 (revised): when seed-0 is incomplete and recomputed, seed-1 rows survive."""
    import yaml
    from run_downstream import run_pipeline

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg_path = tmp_path / "cfg.yaml"

    # Step 1: run with n_test=200, both seeds
    cfg["n_test"] = 200
    cfg["seeds"] = [0, 1]
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    run_pipeline(cfg, force=True)

    csv_path = out / "CF1_L1" / "zero_fill.csv"
    rows_after_200 = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n_seed1_after_200 = sum(1 for r in rows_after_200 if int(r["seed"]) == 1)
    assert n_seed1_after_200 == 200, f"seed 1 should have 200 rows after initial run"

    # Step 2: re-run with n_test=1000, only seed 0 (seed 0 incomplete at 200, will recompute)
    cfg["n_test"] = 1000
    cfg["seeds"] = [0]
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    run_pipeline(cfg, force=False)

    rows_after_recompute = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n_seed0_final = sum(1 for r in rows_after_recompute if int(r["seed"]) == 0)
    n_seed1_final = sum(1 for r in rows_after_recompute if int(r["seed"]) == 1)

    # Seed 0 should have recomputed to 1000 rows, seed 1 should still have 200
    assert n_seed0_final == 1000, f"seed 0 should recompute to 1000, got {n_seed0_final}"
    assert n_seed1_final == 200, f"seed 1 should survive with 200, got {n_seed1_final}"


def test_patch_seed_uniqueness(tmp_path):
    """Fix 2 (revised): (patch_id, seed) pairs are unique in written CSV."""
    import yaml
    from run_downstream import run_pipeline

    ckpt = ROOT / "results" / "downstream" / "classifier.pt"
    if not ckpt.exists():
        pytest.skip("classifier.pt missing")

    cfg = yaml.safe_load((ROOT / "configs" / "downstream.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    cfg["output_dir"] = str(out)
    cfg["classifier_path"] = str(ckpt)
    cfg["n_test"] = 20
    cfg["families"] = ["CF1"]
    cfg["levels"] = {"L1": 0.05}
    cfg["methods"] = ["zero_fill"]
    cfg["seeds"] = [0, 1]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    run_pipeline(cfg, force=True)

    csv_path = out / "CF1_L1" / "zero_fill.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    # Check uniqueness of (patch_id, seed)
    pairs = set()
    for row in rows:
        pair = (int(row["patch_id"]), int(row["seed"]))
        assert pair not in pairs, f"duplicate (patch_id, seed) = {pair}"
        pairs.add(pair)
    assert len(pairs) == len(rows), "uniqueness check failed"
