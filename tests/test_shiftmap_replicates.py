"""The replicate statistics of scripts/shiftmap_replicates.py."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("reps", ROOT / "scripts" / "shiftmap_replicates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spread_counts_differing_patches_and_the_spread_of_cell_means():
    m = _mod()
    ref = np.full((3, 4, 4), 100.0)
    M = np.ones((3, 4, 4))
    M[:, :2] = 0
    a = ref.copy()
    a[M == 0] += 1.0                                # every patch at 1 grey level error
    b = a.copy()
    b[1][M[1] == 0] += 1.0                          # patch 1 differs in the second run
    s = m.spread(ref, [a, a.copy(), b], M)
    assert s["n_realisations"] == 3 and s["frac_patches_differ"] == 1 / 3
    p1 = 10 * np.log10(255 ** 2 / 1.0)
    p2 = 10 * np.log10(255 ** 2 / 4.0)
    assert abs(s["patch_range_max"] - (p1 - p2)) < 1e-9
    assert abs(s["cell_mean_range"] - (p1 - p2) / 3) < 1e-9
    assert s["cell_mean_sd"] > 0


def test_an_existing_measurement_is_kept(tmp_path):
    """Shift-map is not deterministic, so measuring again gives new numbers; a
    resumed campaign must keep the measurement the paper quotes."""
    m = _mod()
    sweep = tmp_path / "sweep_v2_dfrws"
    sweep.mkdir()
    (tmp_path / "out").mkdir()
    kept = tmp_path / "out" / "sweep_v2_dfrws.csv"
    kept.write_text("family,level\nCF1,L1\n")
    assert m.main(["--sweep-dir", str(sweep), "--out-dir", str(tmp_path / "out")]) == 0
    assert kept.read_text() == "family,level\nCF1,L1\n"      # no run_metadata.json was needed either


def _tiny_sweep(tmp_path, cells=("CF1_L1", "CF1_L2")):
    sweep = tmp_path / "sweep_v2_tiny"
    ref = np.full((2, 4, 4), 100.0, dtype=np.float32)
    np.save(tmp_path / "p.npy", ref)
    M = np.ones((2, 4, 4), dtype=np.float32)
    M[:, :2] = 0
    for c in cells:
        d = sweep / c / "seed0"
        d.mkdir(parents=True)
        np.save(d / "y.npy", ref * M)
        np.save(d / "mask.npy", M)
        np.save(d / "x_hat_shiftmap.npy", ref + (1 - M))
    (sweep / "run_metadata.json").write_text(json.dumps(
        {"config": {"patches_path": str(tmp_path / "p.npy"), "families": ["CF1"],
                    "levels": {c.split("_")[1]: 0.1 for c in cells}}}))
    return sweep


def test_an_interrupted_measurement_resumes_at_the_cell_it_stopped_on(tmp_path, monkeypatch):
    m = _mod()
    sweep = _tiny_sweep(tmp_path)
    args = ["--sweep-dir", str(sweep), "--runs", "1", "--workers", "1", "--out-dir", str(tmp_path / "out")]
    calls = []

    def dies_on_second_cell(y, M):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("killed")
        return y + 2 * (1 - M)

    monkeypatch.setattr(m, "baseline_shiftmap", dies_on_second_cell)
    with pytest.raises(RuntimeError):
        m.main(args)
    assert not (tmp_path / "out" / "sweep_v2_tiny.csv").exists()

    calls.clear()
    monkeypatch.setattr(m, "baseline_shiftmap", lambda y, M: (calls.append(1), y + 2 * (1 - M))[1])
    assert m.main(args) == 0
    assert len(calls) == 1                                    # only the second cell ran again
    rows = list(csv.DictReader((tmp_path / "out" / "sweep_v2_tiny.csv").open()))
    assert [r["level"] for r in rows] == ["L1", "L2"]
    assert rows[0]["cell_means"] == rows[1]["cell_means"]     # same fake output in both cells


def test_force_measures_every_cell_again(tmp_path, monkeypatch):
    m = _mod()
    sweep = _tiny_sweep(tmp_path, cells=("CF1_L1",))
    args = ["--sweep-dir", str(sweep), "--runs", "1", "--workers", "1", "--out-dir", str(tmp_path / "out")]
    calls = []
    monkeypatch.setattr(m, "baseline_shiftmap", lambda y, M: (calls.append(1), y + 2 * (1 - M))[1])
    m.main(args)
    m.main(args + ["--force"])
    assert len(calls) == 2
