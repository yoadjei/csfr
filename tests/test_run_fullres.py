"""Full-resolution stage (E11): a method that raises does not stop the others;
its failure is recorded with the error, kept out of the scores, listed in
failures.csv, and retried on the next run."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("run_fullres", ROOT / "scripts" / "run_fullres.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_a_failure_in_the_traced_call_stops_tracing(monkeypatch):
    """The memory-traced second call can fail on its own (it runs under the
    tracing overhead); tracing must not stay on for the timings that follow."""
    import tracemalloc

    import pytest
    m = _mod()
    calls = []

    def flaky(y, M, seed=0):
        calls.append(tracemalloc.is_tracing())
        if len(calls) == 2:
            raise MemoryError("bad allocation")
        return y

    monkeypatch.setitem(m.BASELINES, "flaky", flaky)
    y3 = np.ones((3, 8, 8), dtype=np.float32)
    with pytest.raises(MemoryError):
        m.run_method("flaky", y3, np.ones_like(y3), trace_memory=True, max_iter=10)
    assert calls == [False, True]
    assert not tracemalloc.is_tracing()


def test_a_failing_method_is_recorded_and_retried(tmp_path, monkeypatch):
    m = _mod()
    img = np.random.default_rng(0).random((32, 48, 3))
    monkeypatch.setattr(m, "image_list", lambda: [("tiny", "kodak", tmp_path / "tiny.png")])
    monkeypatch.setattr(m, "load_rgb", lambda path, corpus: img)
    fail = {"shiftmap": True}

    def fake_run_method(name, y3, M3, trace_memory, max_iter):
        if fail.get(name):
            raise RuntimeError("INPAINT_SHIFTMAP child failed: cv2.error: bad allocation")
        x = np.where(M3 == 1, y3, y3[M3 == 1].mean()).astype(np.float64)
        return x, {"seconds": 0.01, "rss_peak_mb": 1.0}

    monkeypatch.setattr(m, "run_method", fake_run_method)
    out = tmp_path / "out"
    argv = ["--methods", "bilinear", "shiftmap", "--corruptions", "CF1_L3", "--out", str(out)]

    assert m.main(argv) == 0
    metrics = json.loads((out / "cases" / "CF1_L3" / "tiny" / "metrics.json").read_text())
    assert metrics["methods"]["bilinear"]["status"] == "done"
    assert metrics["methods"]["shiftmap"]["status"] == "failed"
    assert "RuntimeError" in metrics["methods"]["shiftmap"]["error"]
    assert "bad allocation" in metrics["methods"]["shiftmap"]["error"]
    assert {r["method"] for r in _rows(out / "per_image.csv")} == {"bilinear"}
    assert {r["method"] for r in _rows(out / "cost.csv")} == {"bilinear"}
    failures = _rows(out / "failures.csv")
    assert [(r["corruption"], r["image"], r["method"]) for r in failures] == [("CF1_L3", "tiny", "shiftmap")]
    assert "bad allocation" in failures[0]["error"]

    fail["shiftmap"] = False
    assert m.main(argv) == 0
    metrics = json.loads((out / "cases" / "CF1_L3" / "tiny" / "metrics.json").read_text())
    assert metrics["methods"]["shiftmap"]["status"] == "done"
    assert "error" not in metrics["methods"]["shiftmap"]
    assert {r["method"] for r in _rows(out / "per_image.csv")} == {"bilinear", "shiftmap"}
    assert not (out / "failures.csv").exists()
