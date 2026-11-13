"""Unit checkpoints (src/checkpoint.py): a finished unit is computed once, a rerun
with the same parameters reuses it, and the rows a caller gets are the same
whether they were just computed or read back."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from checkpoint import UnitStore, digests, save_npy_atomic, write_text_atomic  # noqa: E402

PARAMS = {"sweep": "results/sweep_v2_dfrws", "patches": 10, "deltas": (0.25, 1.0)}


class Counter:
    def __init__(self, rows):
        self.calls, self._rows = 0, rows

    def __call__(self):
        self.calls += 1
        return self._rows


def test_a_finished_unit_is_reused_by_a_rerun(tmp_path):
    compute = Counter([{"cell": "CF1_L1", "err": 0.5}])
    first = UnitStore(tmp_path, PARAMS).rows("CF1_L1", compute)
    again = UnitStore(tmp_path, dict(PARAMS)).rows("CF1_L1", compute)
    assert compute.calls == 1
    assert first == again == [{"cell": "CF1_L1", "err": 0.5}]


def test_other_parameters_recompute_the_unit(tmp_path):
    compute = Counter([{"x": 1}])
    UnitStore(tmp_path, PARAMS).rows("u", compute)
    UnitStore(tmp_path, {**PARAMS, "patches": 20}).rows("u", compute)
    assert compute.calls == 2
    assert UnitStore(tmp_path, {**PARAMS, "patches": 20}).load("u") == [{"x": 1}]
    assert UnitStore(tmp_path, PARAMS).load("u") is None       # overwritten by the second run


def test_computed_rows_are_the_rows_a_resumed_run_reads(tmp_path):
    """numpy scalars, NaN and tuples come back as the plain values a reload gives,
    so a run that was interrupted writes the same output as one that was not."""
    rows = [{"f32": np.float32(0.1), "f64": np.float64(1 / 3), "b": np.bool_(True), "i": np.int64(7),
             "nan": float("nan"), "pix": (3, 4), "s": "CF2_L3"}]
    got = UnitStore(tmp_path, PARAMS).rows("u", lambda: rows)
    back = UnitStore(tmp_path, PARAMS).load("u")
    assert got[0]["f32"] == back[0]["f32"] == float(np.float32(0.1))
    assert got[0]["f64"] == 1 / 3 and got[0]["b"] is True and got[0]["i"] == 7
    assert math.isnan(got[0]["nan"]) and math.isnan(back[0]["nan"])
    assert got[0]["pix"] == back[0]["pix"] == [3, 4]
    assert {k: v for k, v in got[0].items() if k != "nan"} == {k: v for k, v in back[0].items() if k != "nan"}


def test_an_unreadable_unit_file_is_recomputed(tmp_path):
    store = UnitStore(tmp_path, PARAMS)
    store.rows("u", lambda: [{"x": 1}])
    store.path("u").write_text("{ truncated", encoding="utf-8")
    compute = Counter([{"x": 2}])
    assert store.rows("u", compute) == [{"x": 2}] and compute.calls == 1


def test_a_failed_unit_leaves_nothing_behind(tmp_path):
    store = UnitStore(tmp_path, PARAMS)

    def boom():
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        store.rows("u", boom)
    assert store.load("u") is None
    assert not list(tmp_path.iterdir())


def test_unit_names_must_be_file_names(tmp_path):
    with pytest.raises(ValueError):
        UnitStore(tmp_path, PARAMS).path("../escape")


def test_a_unit_whose_other_output_is_missing_is_recomputed(tmp_path):
    side = tmp_path / "manifest.json"
    compute = Counter([{"x": 1}])
    store = UnitStore(tmp_path / "units", PARAMS)
    store.rows("u", compute, outputs=[side])          # writing side is the compute's job
    side.write_text("{}")
    store.rows("u", compute, outputs=[side])
    assert compute.calls == 1
    side.unlink()
    store.rows("u", compute, outputs=[side])
    assert compute.calls == 2


def test_digests_identify_inputs_and_code(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.zeros(3))
    np.save(b, np.zeros(3))
    assert digests([a, b]) == {"a.npy": digests([b])["b.npy"], "b.npy": digests([a])["a.npy"]}
    np.save(b, np.ones(3))
    assert digests([a])["a.npy"] != digests([b])["b.npy"]


def test_digests_refuse_two_files_of_one_name(tmp_path):
    (tmp_path / "d").mkdir()
    for f in (tmp_path / "x.py", tmp_path / "d" / "x.py"):
        f.write_text("")
    with pytest.raises(ValueError):
        digests([tmp_path / "x.py", tmp_path / "d" / "x.py"])


def test_the_key_names_the_parameters(tmp_path):
    a, b = UnitStore(tmp_path, PARAMS), UnitStore(tmp_path, {**PARAMS, "deltas": [0.25, 1.0]})
    assert a.key() == b.key() and len(a.key()) == 12              # a tuple and a list store alike
    assert UnitStore(tmp_path, {**PARAMS, "patches": 11}).key() != a.key()


def test_atomic_writes_leave_no_partial_file_that_a_glob_would_find(tmp_path, monkeypatch):
    save_npy_atomic(tmp_path / "x_hat_lama.npy", np.arange(4, dtype=np.float32))
    write_text_atomic(tmp_path / "meta.json", '{"a": 1}')
    assert np.array_equal(np.load(tmp_path / "x_hat_lama.npy"), np.arange(4, dtype=np.float32))
    assert (tmp_path / "meta.json").read_text(encoding="utf-8") == '{"a": 1}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json", "x_hat_lama.npy"]

    def killed(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(np, "save", killed)
    with pytest.raises(KeyboardInterrupt):
        save_npy_atomic(tmp_path / "x_hat_csfr.npy", np.zeros(2))
    assert not list(tmp_path.glob("x_hat_csfr*"))           # the method still counts as not run
