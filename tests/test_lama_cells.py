"""scripts/run_lama_cells.py: a directory counts as done only when its output was
written in full, and then it carries its metadata."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("lama_cells", ROOT / "scripts" / "run_lama_cells.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cells(root: Path, names=("CF1_L1/seed0", "CF1_L2/seed0")):
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        np.save(d / "y.npy", np.zeros((2, 8, 8), np.float32))
        np.save(d / "mask.npy", np.ones((2, 8, 8), np.float32))
    return root


def test_a_directory_killed_before_its_output_is_written_runs_again(tmp_path, monkeypatch):
    m = _mod()
    root = _cells(tmp_path / "cases")
    monkeypatch.setattr(m, "_model", lambda: None)
    calls = []

    def inpaint(model, y, M):
        calls.append(1)
        return y + 1

    monkeypatch.setattr(m, "inpaint_any", inpaint)
    real_save = m.save_npy_atomic

    def killed_on_second(path, arr):
        if len(calls) == 2:
            raise KeyboardInterrupt
        real_save(path, arr)

    monkeypatch.setattr(m, "save_npy_atomic", killed_on_second)
    with pytest.raises(KeyboardInterrupt):
        m.main(["--root", str(root)])
    first, second = root / "CF1_L1" / "seed0", root / "CF1_L2" / "seed0"
    assert (first / "x_hat_lama.npy").exists() and json.loads((first / "lama_meta.json").read_text())["n"] == 2
    assert not (second / "x_hat_lama.npy").exists()

    monkeypatch.setattr(m, "save_npy_atomic", real_save)
    calls.clear()
    assert m.main(["--root", str(root)]) == 0
    assert len(calls) == 1                                   # only the unfinished directory
    assert np.array_equal(np.load(second / "x_hat_lama.npy"), np.ones((2, 8, 8), np.float32))
    assert (second / "lama_meta.json").exists()
