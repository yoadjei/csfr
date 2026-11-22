"""Downstream harness, round-2 options: the convergent solver, the registry
baselines, a reconstruction cache shared across classifiers (LaMa enters
through it), and temperature-scaled confidence fitted on SVHN test images
disjoint from the evaluation subset. The released configuration is unchanged."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CNN = ROOT / "results" / "downstream" / "classifier.pt"


def _harness():
    spec = importlib.util.spec_from_file_location("downstream_h", ROOT / "scripts" / "run_downstream.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def cfg(tmp_path):
    if not CNN.exists():
        pytest.skip("released classifier checkpoint missing")
    c = {"experiment": "t", "seeds": [0, 1], "device": "cpu", "corpus": "svhn",
         "classifier_path": str(CNN), "output_dir": str(tmp_path / "cnn"), "n_test": 20,
         "families": ["CF2"], "levels": {"L3": 0.3},
         "methods": ["bilinear", "guleryuz", "csfr", "lama"],
         "solver": {"solver": "v2"}, "cases_dir": str(tmp_path / "cases"), "calibration_n": 300,
         "bounded_unknown_threshold": 0.5, "centre_box_frac": 0.5}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c))
    return p, c, tmp_path


def test_v2_methods_cache_and_temperature(cfg):
    p, c, tmp = cfg
    h = _harness()
    assert h.main(["--config", str(p)]) == 0
    cell = tmp / "cnn" / "CF2_L3"
    for m in ["bilinear", "guleryuz", "csfr"]:
        rows = list(csv.DictReader((cell / f"{m}.csv").open()))
        assert len(rows) == 40 and all(r["confidence_ts"] != "" for r in rows)
        for s in (0, 1):
            assert (tmp / "cases" / "CF2_L3" / f"seed{s}" / f"x_hat_{m}.npy").exists()
    assert not (cell / "lama.csv").exists()                      # pending until the venv writes it
    env = json.loads((tmp / "cnn" / "environment.json").read_text())
    assert env["temperature"] > 0 and env["calibration"]["n"] == 300
    assert env["calibration"]["disjoint_from_evaluation"] is True


def test_cached_reconstructions_are_reused_and_lama_scored(cfg):
    p, c, tmp = cfg
    h = _harness()
    h.main(["--config", str(p)])
    cache = tmp / "cases" / "CF2_L3" / "seed0"
    t = (cache / "x_hat_csfr.npy").stat().st_mtime_ns
    for s in (0, 1):
        d = tmp / "cases" / "CF2_L3" / f"seed{s}"
        np.save(d / "x_hat_lama.npy", np.load(d / "y.npy").astype(np.float32))
    c2 = dict(c, output_dir=str(tmp / "second"))
    p2 = tmp / "cfg2.yaml"
    p2.write_text(yaml.safe_dump(c2))
    h.main(["--config", str(p2)])
    assert (cache / "x_hat_csfr.npy").stat().st_mtime_ns == t        # read, not recomputed
    a = list(csv.DictReader((tmp / "cnn" / "CF2_L3" / "csfr.csv").open()))
    b = list(csv.DictReader((tmp / "second" / "CF2_L3" / "csfr.csv").open()))
    assert [r["prediction"] for r in a] == [r["prediction"] for r in b]
    assert len(list(csv.DictReader((tmp / "second" / "CF2_L3" / "lama.csv").open()))) == 40


def test_a_cell_csv_interrupted_while_written_leaves_the_previous_file(tmp_path, monkeypatch):
    # a stage stopped at its time budget (the Kaggle run) must not leave half a
    # CSV that the resumed run would read as finished rows
    h = _harness()
    path = tmp_path / "CF1_L1" / "csfr.csv"
    h.write_csv(path, [{"a": 1}], ["a"])
    before = path.read_text(encoding="utf-8")

    real = csv.DictWriter.writerow

    def dies_on_the_second_row(self, row):
        if row.get("a") == 3:
            raise KeyboardInterrupt
        return real(self, row)

    monkeypatch.setattr(csv.DictWriter, "writerow", dies_on_the_second_row)
    with pytest.raises(KeyboardInterrupt):
        h.write_csv(path, [{"a": 2}, {"a": 3}], ["a"])
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in path.parent.iterdir()] == ["csfr.csv"]
