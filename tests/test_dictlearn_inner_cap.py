"""The inner-cap study of the dictionary-learning baseline: it reproduces the
sweep's output at the released cap and compares it with a larger cap."""
import csv
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baselines import baseline_dictlearn, limit_threads  # noqa: E402
from masks import make_mask  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("dic", ROOT / "scripts" / "dictlearn_inner_cap.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_compare_signs_the_difference_released_minus_reference():
    m = _mod()
    ref = np.full((2, 8, 8), 100.0)
    M = np.ones((2, 8, 8))
    M[:, :4] = 0
    near, far = ref + 1.0, ref + 2.0                  # 1 and 2 grey levels of error in the hole
    r = m.compare(ref, far, near, M)
    assert r["diff_mean"] < 0 and abs(r["diff_mean"] - 20 * np.log10(0.5)) < 1e-9
    assert abs(r["diff_abs_max"] + r["diff_mean"]) < 1e-12


def test_reproduces_the_sweep_at_the_released_cap(tmp_path):
    m = _mod()
    patches = np.load(ROOT / "data" / "patches.npy")[:3].astype(np.float32)
    np.save(tmp_path / "p.npy", patches)
    sweep = tmp_path / "sweep_tiny"
    cell = sweep / "CF2_L3" / "seed0"
    cell.mkdir(parents=True)
    M = make_mask(patches.shape, "CF2", 0.3, np.random.default_rng(0))
    y = (patches * M).astype(np.float32)
    np.save(cell / "y.npy", y)
    np.save(cell / "mask.npy", M)
    # saved as the sweep saves it: in a worker initialised by limit_threads (the
    # BLAS thread count at start-up can move one float32 ulp; run_sweep_v2.run_baseline)
    with ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=limit_threads) as pool:
        saved = pool.submit(baseline_dictlearn, y, M, 0).result()
    np.save(cell / "x_hat_dictlearn.npy", saved.astype(np.float32))
    (sweep / "run_metadata.json").write_text(json.dumps(
        {"config": {"patches_path": str(tmp_path / "p.npy"), "families": ["CF1", "CF2"],
                    "levels": {"L3": 0.3}}}))
    assert m.main(["--sweep-dir", str(sweep), "--patches", "3", "--reference-factor", "2",
                   "--workers", "1", "--out-dir", str(tmp_path / "out")]) == 0
    (row,) = csv.DictReader((tmp_path / "out" / "sweep_tiny.csv").open())   # CF1 is left out
    assert (row["family"], row["level"], row["matches_sweep"]) == ("CF2", "L3", "True")
    assert int(row["cap_reference"]) == 2 * int(row["cap_released"]) == 400
    assert 0 <= int(row["n_learned"]) <= int(row["n_patches"]) == 3


def test_an_interrupted_study_resumes_at_the_cell_it_stopped_on(tmp_path, monkeypatch):
    m = _mod()
    sweep = tmp_path / "sweep_tiny"
    for lvl in ("L1", "L2"):
        cell = sweep / f"CF2_{lvl}" / "seed0"
        cell.mkdir(parents=True)
        for f in ("y.npy", "mask.npy", "x_hat_dictlearn.npy"):
            np.save(cell / f, np.zeros((1, 4, 4), np.float32))
    np.save(tmp_path / "p.npy", np.zeros((1, 4, 4), np.float32))
    (sweep / "run_metadata.json").write_text(json.dumps(
        {"config": {"patches_path": str(tmp_path / "p.npy"), "families": ["CF2"],
                    "levels": {"L1": 0.1, "L2": 0.2}}}))
    args = ["--sweep-dir", str(sweep), "--workers", "1", "--out-dir", str(tmp_path / "out")]
    calls = []

    def row(pool, cell_dir, ref, n, factor, workers):
        calls.append(cell_dir.parent.name)
        if calls == ["CF2_L1", "CF2_L2"]:
            raise RuntimeError("killed")
        return {"family": "CF2", "level": cell_dir.parent.name[-2:], "diff_mean": 0.0,
                "diff_abs_max": 0.0, "matches_sweep": True}

    monkeypatch.setattr(m, "cell_row", row)
    with pytest.raises(RuntimeError):
        m.main(args)
    assert m.main(args) == 0
    assert calls == ["CF2_L1", "CF2_L2", "CF2_L2"]           # CF2_L1 was not run again
    rows = list(csv.DictReader((tmp_path / "out" / "sweep_tiny.csv").open()))
    assert [r["level"] for r in rows] == ["L1", "L2"]
