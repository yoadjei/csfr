"""Solver comparison (E5): every arm scored on the released masks, the released
Adam outputs read back verbatim, convergent arms carry their certificates, and
the hole-PSNR traces of both solvers are written for the trace cell."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from masks import make_mask  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("solver_cmp", ROOT / "scripts" / "run_solver_comparison.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _released(tmp_path, n=3):
    ref = np.load(ROOT / "data" / "patches.npy")[:n].astype(np.float32)
    np.save(tmp_path / "patches.npy", ref.astype(np.uint8))
    for cell, (fam, frac) in {"CF1_L3": ("CF1", 0.3), "CF2_L5": ("CF2", 0.7)}.items():
        d = tmp_path / "released" / cell / "seed0"
        d.mkdir(parents=True)
        M = make_mask(ref.shape, fam, frac, np.random.default_rng(0))
        np.save(d / "y.npy", ref * M)
        np.save(d / "mask.npy", M)
        np.save(d / "x_hat.npy", np.where(M == 1, ref, 128.0).astype(np.float32))
        (d / "metrics.json").write_text(json.dumps({"device": "cuda", "runtime_s": 1.5}))
    return tmp_path


def test_arms_traces_and_summary(tmp_path):
    base = _released(tmp_path)
    out = tmp_path / "out"
    rc = _mod().main(["--released", str(base / "released"), "--patches", str(base / "patches.npy"),
                      "--cells", "CF1_L3", "CF2_L5", "--seeds", "0", "--adam-budgets", "20", "40",
                      "--tols", "1e-3", "--retime-seed", "0", "--trace-cell", "CF2_L5",
                      "--trace-iters", "10", "20", "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "per_cell.csv").open()))
    arms = {r["arm"] for r in rows}
    # adam600_cpu is the re-time at --retime-seed, with its distance to the released output
    assert arms == {"adam600_released", "adam600_cpu", "adam20_cpu", "adam40_cpu", "fista_gap1e-03"}
    retime = next(r for r in rows if r["arm"] == "adam600_cpu" and r["cell"] == "CF1_L3")
    assert float(retime["max_abs_diff_vs_released"]) >= 0.0
    rel = next(r for r in rows if r["arm"] == "adam600_released" and r["cell"] == "CF1_L3")
    assert rel["device"] == "cuda" and float(rel["seconds"]) == 1.5
    conv = next(r for r in rows if r["arm"] == "fista_gap1e-03" and r["cell"] == "CF1_L3")
    assert int(conv["n_converged"]) == 3 and conv["device"] == "cpu"
    q = list(csv.DictReader((out / "quality_vs_compute.csv").open()))
    assert {r["family"] for r in q} == {"CF1", "CF2"}
    tr = list(csv.DictReader((out / "trace_CF2_L5.csv").open()))
    assert {r["arm"] for r in tr} == {"adam", "fista"}
    assert {int(r["iter"]) for r in tr if r["arm"] == "fista"} == {10, 20}
