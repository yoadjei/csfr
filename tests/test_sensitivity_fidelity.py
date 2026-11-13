"""Query-pixel selection of the fidelity campaign: deterministic, at the stated
Chebyshev distances, on the right side of the mask."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("fid", ROOT / "scripts" / "run_sensitivity_fidelity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block_mask():
    Mb = np.ones((32, 32), bool)
    Mb[10:20, 4:14] = False
    return Mb


def test_observed_queries_are_observed_at_the_stated_distances():
    m = _mod()
    Mb = _block_mask()
    qs = m.observed_queries(Mb)
    dist = distance_transform_cdt(Mb, metric="chessboard")
    assert [d for d, _ in qs] == [1, 3, 8]
    for d, (r, c) in qs:
        assert Mb[r, c]
        assert dist[r, c] == d if d < 8 else dist[r, c] >= 8
    # first in raster order
    d1 = np.argwhere(Mb & (dist == 1))[0]
    assert qs[0][1] == (int(d1[0]), int(d1[1]))


def test_missing_distance_is_skipped():
    m = _mod()
    Mb = np.ones((8, 8), bool)
    Mb[::2, ::2] = False          # scattered loss: no observed pixel is 3 away from the hole
    assert [d for d, _ in m.observed_queries(Mb)] == [1]


def test_imputed_queries_are_hole_pixels_including_the_deepest():
    m = _mod()
    Mb = _block_mask()
    qs = m.imputed_queries(Mb)
    dist = distance_transform_cdt(~Mb, metric="chessboard")
    assert all(not Mb[p] for p in qs)
    assert [int(dist[p]) for p in qs] == [1, 3, int(dist[~Mb].max())]
    assert m.imputed_queries(np.ones((4, 4), bool)) == []


def _tiny_sweep(tmp_path, cells=("CF1_L1", "CF1_L2")):
    sweep = tmp_path / "sweep"
    for c in cells:
        d = sweep / c / "seed0"
        d.mkdir(parents=True)
        np.save(d / "y.npy", np.zeros((1, 8, 8), np.float32))
        np.save(d / "mask.npy", np.ones((1, 8, 8), np.float32))
    return sweep


def _fake_rows(cell, tol):
    return [{"cell": cell, "family": cell[:3], "level": cell[4:], "delta": d, "tol": tol, "query_distance": 1,
             "rel_l2_error_ref": 0.05, "sign_agreement_topk": 1.0, "ref_converged": True,
             "record_seconds": 0.1, "reconstruction_seconds": 0.2} for d in (0.25, 1.0)]


def test_an_interrupted_fidelity_run_resumes_at_the_unit_it_stopped_on(tmp_path, monkeypatch):
    m = _mod()
    sweep = _tiny_sweep(tmp_path)
    out = tmp_path / "out"
    args = ["--sweep-dir", str(sweep), "--cells", "CF1_L1", "CF1_L2", "--out", str(out)]
    calls = []

    def unit(args_, cell, tol, inputs, manifest, rc):
        calls.append(cell)
        if calls == ["CF1_L1", "CF1_L2"]:
            raise RuntimeError("killed")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        return _fake_rows(cell, tol)

    monkeypatch.setattr(m, "fidelity_unit", unit)
    with pytest.raises(RuntimeError):
        m.main(args)
    assert not (out / "fidelity.csv").exists()
    assert m.main(args) == 0
    assert calls == ["CF1_L1", "CF1_L2", "CF1_L2"]
    rows = list(csv.DictReader((out / "fidelity.csv").open()))
    assert [r["cell"] for r in rows] == ["CF1_L1", "CF1_L1", "CF1_L2", "CF1_L2"]
    assert {r["level"] for r in csv.DictReader((out / "summary.csv").open())} == {"L1", "L2"}

    (out / "manifests" / "CF1_L1_patch0.json").unlink()     # a unit whose manifest is gone runs again
    m.main(args)
    assert calls[3:] == ["CF1_L1"]


def test_an_interrupted_record_comparison_resumes_at_the_cell_it_stopped_on(tmp_path, monkeypatch):
    m = _mod()
    sweep = _tiny_sweep(tmp_path)
    out = tmp_path / "out"
    args = ["--records-compare", "--sweep-dir", str(sweep), "--cells", "CF1_L1", "CF1_L2", "--out", str(out)]
    calls = []

    def unit(args_, cell):
        calls.append(cell)
        if calls == ["CF1_L1", "CF1_L2"]:
            raise RuntimeError("killed")
        return [{"cell": cell, "method": "csfr", "l1_norm": 1.0}]

    monkeypatch.setattr(m, "records_unit", unit)
    with pytest.raises(RuntimeError):
        m.main(args)
    assert m.main(args) == 0
    assert calls == ["CF1_L1", "CF1_L2", "CF1_L2"]
    assert [r["cell"] for r in csv.DictReader((out / "records_compare.csv").open())] == ["CF1_L1", "CF1_L2"]
