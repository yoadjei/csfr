"""Round-2 sweep harness: every method's output is retained, CSFR certificates
are written, per-patch metrics feed the summary, external methods are scored
from saved outputs, runs resume, and parallel baselines equal serial ones."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _harness():
    spec = importlib.util.spec_from_file_location("sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def tiny(tmp_path):
    patches = np.load(ROOT / "data" / "patches.npy")[:4]
    np.save(tmp_path / "p.npy", patches)
    np.save(tmp_path / "c.npy", np.array([0, 0, 1, 1]))
    cfg = {"experiment": "tiny", "corpus": "dfrws-test", "patches_path": str(tmp_path / "p.npy"),
           "cluster_path": str(tmp_path / "c.npy"), "output_dir": str(tmp_path / "out"),
           "seeds": [0], "families": ["CF1"], "levels": {"L3": 0.30},
           "solver": {"max_iter": 400, "tol": 1e-6},
           "csfr_variants": ["csfr"], "ablation_variants": ["csfr_tv_only"], "ablation_seeds": [0],
           "baselines": ["zero_fill", "bilinear", "inpaint_ns"], "external": ["lama"]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, tmp_path / "out"


def test_cell_holds_outputs_certificates_and_per_patch_metrics(tiny):
    cfg, out = tiny
    assert _harness().main(["--config", str(cfg), "--workers", "1"]) == 0
    cell = out / "CF1_L3" / "seed0"
    for f in ["y.npy", "mask.npy", "realised_frac.npy", "x_hat_csfr.npy", "x_hat_csfr_tv_only.npy",
              "x_hat_zero_fill.npy", "x_hat_bilinear.npy", "x_hat_inpaint_ns.npy",
              "per_patch_csfr.npz", "certificate_csfr.json", "metrics.json"]:
        assert (cell / f).exists(), f
    pp = np.load(cell / "per_patch_csfr.npz")
    for k in ["psnr_hole", "ssim_hole", "gmsd_hole", "hrp", "psnr_whole", "exact", "hole_frac",
              "iterations", "converged", "gap"]:
        assert k in pp.files and pp[k].shape == (4,)
    m = json.loads((cell / "metrics.json").read_text())
    assert m["methods"]["lama"]["status"] == "pending"
    assert m["methods"]["csfr"]["status"] == "done"
    y, M, x = np.load(cell / "y.npy"), np.load(cell / "mask.npy"), np.load(cell / "x_hat_csfr.npy")
    assert np.array_equal(x[M == 1], y[M == 1])


def test_summary_and_metadata(tiny):
    cfg, out = tiny
    _harness().main(["--config", str(cfg), "--workers", "1"])
    rows = list(csv.DictReader((out / "summary.csv").open()))
    methods = {r["method"] for r in rows}
    assert methods == {"csfr", "csfr_tv_only", "zero_fill", "bilinear", "inpaint_ns"}
    r = next(r for r in rows if r["method"] == "csfr")
    for k in ["psnr_hole_mean", "psnr_hole_seedsd", "ssim_hole_mean", "gmsd_hole_mean", "hrp_mean",
              "n_exact", "n_at_cap", "n_converged", "iters_median", "runtime_s"]:
        assert k in r
    for r in rows:   # every exact patch is at the cap
        assert int(r["n_exact"]) <= int(r["n_at_cap"]) <= int(r["n_patches"])
    meta = json.loads((out / "run_metadata.json").read_text())
    for k in ["environment", "patches_sha256", "config", "git_commit", "objective"]:
        assert k in meta


def test_metadata_keeps_one_entry_per_invocation(tiny):
    """A resumed sweep is computed by several invocations, possibly at different
    commits. The metadata keeps every invocation, not only the last one."""
    cfg, out = tiny
    h = _harness()
    h.main(["--config", str(cfg), "--workers", "1"])
    h.main(["--config", str(cfg), "--workers", "1", "--only-method", "bilinear", "--force"])
    meta = json.loads((out / "run_metadata.json").read_text())
    first, second = meta["invocations"]                    # start and end writes of one run are one entry
    for e in (first, second):
        assert {"id", "git_commit", "git_dirty", "started", "finished", "args"} <= set(e)
        assert e["finished"] is not None
    assert first["args"]["only_method"] is None and first["args"]["force"] is False
    assert second["args"]["only_method"] == ["bilinear"] and second["args"]["force"] is True
    assert meta["git_commit"] == second["git_commit"]


def test_metadata_written_before_invocation_history_is_carried_forward(tiny):
    cfg, out = tiny
    out.mkdir()
    (out / "run_metadata.json").write_text(json.dumps({"git_commit": "abc", "git_dirty": True,
                                                       "written": "2026-11-09T21:35:15"}))
    _harness().main(["--config", str(cfg), "--workers", "1", "--only-method", "zero_fill"])
    legacy, current = json.loads((out / "run_metadata.json").read_text())["invocations"]
    assert legacy == {"git_commit": "abc", "git_dirty": True, "written": "2026-11-09T21:35:15",
                      "recorded_before_invocation_history": True}
    assert current["args"]["only_method"] == ["zero_fill"]


def test_external_outputs_are_scored_on_a_second_pass(tiny):
    cfg, out = tiny
    h = _harness()
    h.main(["--config", str(cfg), "--workers", "1"])
    cell = out / "CF1_L3" / "seed0"
    y, M = np.load(cell / "y.npy"), np.load(cell / "mask.npy")
    np.save(cell / "x_hat_lama.npy", np.where(M == 1, y, 100.0).astype(np.float32))
    h.main(["--config", str(cfg), "--workers", "1"])
    m = json.loads((cell / "metrics.json").read_text())
    assert m["methods"]["lama"]["status"] == "done"
    assert (cell / "per_patch_lama.npz").exists()
    assert "lama" in {r["method"] for r in csv.DictReader((out / "summary.csv").open())}


def test_resume_skips_finished_methods(tiny):
    cfg, out = tiny
    h = _harness()
    h.main(["--config", str(cfg), "--workers", "1"])
    f = out / "CF1_L3" / "seed0" / "per_patch_bilinear.npz"
    t = f.stat().st_mtime_ns
    h.main(["--config", str(cfg), "--workers", "1"])
    assert f.stat().st_mtime_ns == t


def test_parallel_baselines_equal_serial():
    h = _harness()
    from masks import make_mask
    ref = np.load(ROOT / "data" / "patches.npy")[:6].astype(np.float32)
    M = make_mask(ref.shape, "CF2", 0.3, np.random.default_rng(0))
    y = ref * M
    for name in ["dictlearn", "criminisi", "guleryuz"]:
        a, ia = h.run_baseline(name, y, M, seed=0, workers=1)
        b, ib = h.run_baseline(name, y, M, seed=0, workers=3)
        assert np.array_equal(a, b), name
        assert ia == ib, name
    _, info = h.run_baseline("criminisi", y, M, seed=0, workers=3)
    assert len(info["fallback"]) == 6 and info["fallback_patches"] == sum(info["fallback"])


def test_outputs_are_stored_losslessly(tmp_path):
    h = _harness()
    ints = np.round(np.random.default_rng(0).random((2, 4, 4)) * 255).astype(np.float32)
    h.save_output(tmp_path / "a.npy", ints)
    assert np.load(tmp_path / "a.npy").dtype == np.uint8
    floats = ints + 0.25
    h.save_output(tmp_path / "b.npy", floats)
    back = np.load(tmp_path / "b.npy")
    assert back.dtype == np.float32 and np.array_equal(back, floats)


def test_json_write_survives_a_transient_lock_on_the_target(tmp_path):
    """Windows refuses os.replace onto a file another process holds open (a virus
    scanner or indexer does this to fresh files); the write must wait it out."""
    import threading
    import time
    h = _harness()
    target = tmp_path / "metrics.json"
    target.write_text("{}")
    f = target.open("r")
    threading.Thread(target=lambda: (time.sleep(0.6), f.close()), daemon=True).start()
    h._write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_resume_recomputes_a_method_whose_metrics_entry_is_missing(tiny):
    """A crash between the per-patch file and metrics.json must not leave a method
    counted as done without its entry."""
    cfg, out = tiny
    h = _harness()
    h.main(["--config", str(cfg), "--workers", "1"])
    mpath = out / "CF1_L3" / "seed0" / "metrics.json"
    m = json.loads(mpath.read_text())
    del m["methods"]["bilinear"]
    mpath.write_text(json.dumps(m))
    h.main(["--config", str(cfg), "--workers", "1"])
    m = json.loads(mpath.read_text())
    assert m["methods"]["bilinear"]["status"] == "done"
