"""Byte-level JPEG corruption (E9): scan location, the three corruption kinds,
tolerant decoding, and the three missing-pixel masks a carver could build."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jpeg_corruption import (KINDS, corrupt, decode_tolerant, decoder_mask, locate_scan,  # noqa: E402
                             mcu_height, offset_mask, oracle_mask)


def _jpeg(mode="L", side=256, subsampling=None, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    g = (128 + 60 * np.sin(xx / 7.0) + 40 * np.cos(yy / 5.0) + rng.normal(0, 8, (side, side)))
    g = g.clip(0, 255).astype(np.uint8)
    arr = g if mode == "L" else np.stack([g, g[::-1], g.T], -1)
    buf = io.BytesIO()
    kw = {} if subsampling is None else {"subsampling": subsampling}
    Image.fromarray(arr).save(buf, "JPEG", quality=85, **kw)
    return buf.getvalue()


def test_locate_scan():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    assert 0 < s0 < s1 == len(b) - 2
    assert b[s1:s1 + 2] == b"\xff\xd9"


def test_mcu_height_follows_chroma_subsampling():
    assert mcu_height(_jpeg("L")) == 8
    assert mcu_height(_jpeg("RGB", subsampling=0)) == 8      # 4:4:4
    assert mcu_height(_jpeg("RGB", subsampling=2)) == 16     # 4:2:0


@pytest.mark.parametrize("kind", KINDS)
def test_corrupt_changes_only_the_scan_and_decodes(kind):
    b = _jpeg()
    scan = locate_scan(b)
    out, off = corrupt(b, kind, np.random.default_rng(1), scan)
    assert scan[0] <= off < scan[1]
    assert out[:scan[0]] == b[:scan[0]]                       # headers untouched
    assert out != b and out.endswith(b"\xff\xd9")
    img, used = decode_tolerant(out)
    assert img.shape == (256, 256) and img.dtype == np.float32 and used in ("pil", "cv2", "grey")


def test_truncate_keeps_the_stated_fraction():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    out, off = corrupt(b, "truncate", np.random.default_rng(0), (s0, s1))
    frac = (off - s0) / (s1 - s0)
    assert 0.3 <= frac <= 0.9 and len(out) == off + 2


def test_masks_for_a_half_truncation():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    cut = s0 + (s1 - s0) // 2
    t = b[:cut] + b"\xff\xd9"
    clean, _ = decode_tolerant(b)
    dec, _ = decode_tolerant(t)
    Mo = oracle_mask(clean, dec)
    lost = 1.0 - Mo.mean()
    assert 0.3 <= lost <= 0.7                                  # roughly the lower half
    Md = decoder_mask(dec)
    assert ((Md == 0) <= (Mo == 0)).mean() > 0.99             # grey fill blocks are truly lost
    Mf = offset_mask(clean.shape, s0, s1, cut, mcu_h=8)
    rows_ok = [((Mf[r] == 0) >= (Mo[r] == 0)).all() for r in range(clean.shape[0])]
    assert np.mean(rows_ok) >= 0.9                             # offset mask covers the loss
    for M in (Mo, Md, Mf):
        assert M.dtype == np.float32 and set(np.unique(M)) <= {0.0, 1.0}


def test_masks_handle_sizes_that_are_not_block_multiples():
    clean = np.full((37, 45), 100, np.float32)
    dec = clean.copy()
    dec[30:, :] = 128.0
    Mo = oracle_mask(clean, dec)
    assert Mo.shape == clean.shape and Mo[36, 44] == 0 and Mo[0, 0] == 1
    Md = decoder_mask(dec)
    assert Md[36, 44] == 0


def _small_run(tmp_path) -> Path:
    """Config of a one-file, one-kind run; returns its path."""
    import json

    import yaml
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(_jpeg(side=96))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"files": [{"file": "a.jpg", "corpus": "synthetic", "mode": "L"}]}))
    cfg = {"experiment": "t", "manifest": str(tmp_path / "manifest.json"), "src_dir": str(src),
           "output_dir": str(tmp_path / "out"), "kinds": ["truncate"], "repeats": 1, "seed": 0,
           "max_side": 64, "conditions": ["oracle", "decoder", "offset"], "offset_margin_rows": 2,
           "solver": {"max_iter": 3000}, "csfr_variants": ["csfr"],
           "baselines": ["bilinear", "inpaint_ns"], "external": ["lama"]}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "cfg.yaml"


def test_a_csfr_only_shard_solves_its_files_in_order_and_nothing_else(tmp_path):
    import csv
    import json
    mod = _driver()
    cfg_path = _small_run(tmp_path)
    src = tmp_path / "src"
    (src / "b.jpg").write_bytes(_jpeg(side=96, seed=1))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"files": [{"file": "a.jpg", "corpus": "synthetic", "mode": "L"},
                   {"file": "b.jpg", "corpus": "synthetic", "mode": "L"}]}))
    gpu = tmp_path / "gpu" / "shard0"
    # files 1 then 0; shard 1 of 2 takes the second of them: file 0
    assert mod.main(["--config", str(cfg_path), "--csfr-only", "--output-dir", str(gpu), "--files", "1,0",
                     "--shard", "1/2", "--workers", "2", "--bootstrap", "200"]) == 0
    assert sorted(p.name for p in (gpu / "cases").iterdir()) == ["a_truncate_r0"]
    assert not (tmp_path / "out").exists()
    rows = list(csv.DictReader((gpu / "per_image.csv").open()))
    assert {r["method"] for r in rows} == {"csfr"} and {r["status"] for r in rows} == {"done"}
    m = json.loads((gpu / "cases" / "a_truncate_r0" / "oracle" / "metrics.json").read_text())
    assert set(m["inputs_sha256"]) == {"y", "mask"}
    with pytest.raises(SystemExit):
        mod.main(["--config", str(cfg_path), "--files", "0", "--limit", "1"])


def test_records_solved_on_other_inputs_are_refused(tmp_path):
    import json
    mod = _driver()
    cfg_path = _small_run(tmp_path)
    args = ["--config", str(cfg_path), "--workers", "2", "--bootstrap", "200"]
    assert mod.main(args) == 0
    mp = tmp_path / "out" / "cases" / "a_truncate_r0" / "decoder" / "metrics.json"
    m = json.loads(mp.read_text())
    m["inputs_sha256"]["y"] = "0" * 64
    mp.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="other inputs"):
        mod.main(args)


def test_driver_end_to_end(tmp_path):
    import csv
    mod = _driver()
    assert mod.main(["--config", str(_small_run(tmp_path)), "--workers", "2", "--bootstrap", "200"]) == 0
    out = tmp_path / "out"
    rows = list(csv.DictReader((out / "per_image.csv").open()))
    done = [r for r in rows if r["status"] == "done"]
    assert {r["method"] for r in done} == {"csfr", "bilinear", "inpaint_ns"}
    assert {r["condition"] for r in done} == {"oracle", "decoder", "offset"}
    assert any(r["method"] == "lama" and r["status"] == "pending" for r in rows)
    assert {r["kind"] for r in rows} == {"truncate"}
    assert {r["method"]: r["method_kind"] for r in rows} == {
        "csfr": "csfr", "bilinear": "classical", "inpaint_ns": "classical", "lama": "external"}
    case = out / "cases" / "a_truncate_r0" / "oracle"
    assert np.load(case / "y.npy").shape == (1, 64, 64)
    mq = list(csv.DictReader((out / "masks_quality.csv").open()))
    assert len(mq) == 3 and all(float(r["coverage_of_oracle_loss"]) >= 0 for r in mq)
    oracle = next(r for r in mq if r["condition"] == "oracle")
    assert float(oracle["false_pos_frac"]) == 0.0 and float(oracle["false_neg_frac"]) == 0.0
    summ = list(csv.DictReader((out / "summary.csv").open()))
    assert {r["method"] for r in summ} == {"csfr", "bilinear", "inpaint_ns"}
    assert {r["kind"] for r in summ} == {"truncate"}
    csfr = [r for r in summ if r["method"] == "csfr"]
    assert len(csfr) == 3 and all(r["vs_best_classical"] in ("bilinear", "inpaint_ns") for r in csfr)


def _driver():
    import importlib.util
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("jpeg_drv", root / "scripts" / "run_jpeg_corruption.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakePool:
    def shutdown(self, **kw):
        pass


def _spawn_pool():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    return ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))


def test_a_dead_worker_gets_a_new_pool_and_a_rerun():
    import os
    mod = _driver()
    made, waits, seen = [], [], []

    def make():
        made.append(_spawn_pool())
        return made[-1]

    def fn(pool):
        seen.append(pool)
        if len(seen) == 1:
            return pool.submit(os._exit, 1).result()     # the worker dies: the pool is broken for good
        return pool.submit(abs, -3).result()

    try:
        out, live = mod.on_live_pool(fn, make(), make, retries=2, wait=7, sleep=waits.append)
    finally:
        for p in made:
            p.shutdown()
    assert out == 3
    assert len(made) == 2 and seen == made and live is made[1]
    assert waits == [7]


def test_a_pool_that_keeps_breaking_is_given_up_after_the_retries():
    from concurrent.futures.process import BrokenProcessPool
    mod = _driver()
    calls = []

    def fn(pool):
        calls.append(pool)
        raise BrokenProcessPool("a worker died")

    with pytest.raises(BrokenProcessPool):
        mod.on_live_pool(fn, _FakePool(), _FakePool, retries=2, wait=0, sleep=lambda s: None)
    assert len(calls) == 3


def test_other_errors_get_no_new_pool():
    mod = _driver()
    made = []

    def fn(pool):
        raise ValueError("a bug, not a dead worker")

    with pytest.raises(ValueError):
        mod.on_live_pool(fn, _FakePool(), lambda: made.append(1), retries=2, wait=0, sleep=lambda s: None)
    assert made == []


def test_driver_finishes_when_a_worker_dies_mid_run(tmp_path, monkeypatch):
    import csv
    import os
    mod = _driver()
    real, broke = mod.start_condition, []

    def flaky(d, clean, dec, M, Mo, cfg, pool, *rest):
        if not broke:
            broke.append(d.name)
            pool.submit(os._exit, 1).result()
        return real(d, clean, dec, M, Mo, cfg, pool, *rest)

    monkeypatch.setattr(mod, "start_condition", flaky)
    monkeypatch.setattr(mod, "POOL_WAIT_S", 0)
    assert mod.main(["--config", str(_small_run(tmp_path)), "--workers", "2", "--bootstrap", "200"]) == 0
    assert broke == ["oracle"]
    rows = list(csv.DictReader((tmp_path / "out" / "per_image.csv").open()))
    done = {(r["condition"], r["method"]) for r in rows if r["status"] == "done"}
    assert done == {(c, m) for c in ("oracle", "decoder", "offset") for m in ("csfr", "bilinear", "inpaint_ns")}


class _InlinePool:
    """Runs each task at submit and logs submits and collections in order."""

    def __init__(self, log):
        self.log = log

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        self.log.append("submit")
        return f

    def shutdown(self, **kw):
        pass


def test_every_condition_of_a_file_is_submitted_before_any_is_collected(tmp_path, monkeypatch):
    import json

    import yaml
    mod = _driver()
    cfg_path = _small_run(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["kinds"], cfg["repeats"] = ["truncate", "overwrite"], 1
    log = []
    real = mod.finish_condition
    monkeypatch.setattr(mod, "finish_condition", lambda job: (log.append("collect"), real(job))[1])
    rec = json.loads(Path(cfg["manifest"]).read_text())["files"][0]
    raw = (Path(cfg["src_dir"]) / rec["file"]).read_bytes()
    import torch
    before = torch.get_num_threads()
    try:
        rows, mq = mod.run_file(0, 1, rec, raw, cfg, tmp_path / "out", _InlinePool(log), False)
    finally:
        torch.set_num_threads(before)
    n_cond = len({(r["case"], r["condition"]) for r in rows})
    n_fill = len({(r["case"], r["condition"]) for r in rows if not r["nothing_to_fill"]})
    assert n_cond >= 2 and log.count("collect") == n_cond
    assert log.index("collect") == log.count("submit")    # every submit came before the first collect
    assert log.count("submit") == n_fill * 3              # csfr and two baselines per condition with a hole
    assert len(mq) == 2 * 3


def test_pooled_csfr_is_the_solve_the_main_process_gave(tmp_path):
    """The pool runs one-thread solves; the record must be what a direct solve of
    the same input gives (the driver solved in the main process before)."""
    import json

    import torch
    import yaml
    from csfr_solver import solve
    mod = _driver()
    cfg_path = _small_run(tmp_path)
    assert mod.main(["--config", str(cfg_path), "--workers", "2", "--bootstrap", "200"]) == 0
    cfg = yaml.safe_load(cfg_path.read_text())
    d = tmp_path / "out" / "cases" / "a_truncate_r0" / "decoder"
    y, M = np.load(d / "y.npy"), np.load(d / "mask.npy")
    rec = json.loads((d / "metrics.json").read_text())["methods"]["csfr"]
    before = torch.get_num_threads()
    torch.set_num_threads(mod.SOLVE_THREADS)       # what the main process ran with
    try:
        x, cert = solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(M),
                        mod._sweep.solver_config("csfr", cfg["solver"]))
    finally:
        torch.set_num_threads(before)
    assert rec["iterations"] == int(cert.iterations[0]) and rec["gap"] == float(cert.gap[0])
    ref = np.load(d.parent / "clean.npy")[None].astype(np.float32)
    Mo = np.load(d.parent / "oracle" / "mask.npy")
    assert rec["psnr_hole"] == float(mod._sweep.score(ref, x.numpy(), y, Mo)["psnr_hole"][0])


def test_rows_keep_the_corruption_kind():
    # each method's metrics carry their own "kind" (csfr, classical, external);
    # flattened into the row it must not replace the corruption kind
    mod = _driver()
    case = {"case": "a_flip_r0", "kind": "flip", "condition": "oracle"}
    metrics = {"nothing_to_fill": False, "methods": {
        "csfr": {"kind": "csfr", "status": "done", "psnr_hole": 20.0},
        "lama": {"kind": "external", "status": "pending"}}}
    rows = mod.method_rows(case, metrics)
    assert [(r["kind"], r["method"], r["method_kind"]) for r in rows] == [
        ("flip", "csfr", "csfr"), ("flip", "lama", "external")]
    assert rows[0]["psnr_hole"] == 20.0 and rows[0]["nothing_to_fill"] is False


def test_summary_is_per_corruption_kind_with_paired_comparisons():
    mod = _driver()
    rows = []
    for kind, shift in (("truncate", 0.0), ("flip", 5.0)):
        for c in range(4):
            case = {"case": f"f{c}_{kind}_r0", "kind": kind, "condition": "oracle"}
            psnr = {"csfr": 20 + shift + c, "bilinear": 18 + shift + c, "fsr": 19 + shift + c,
                    "lama": 21 + shift + c}
            metrics = {"nothing_to_fill": False, "methods": {
                m: {"kind": "external" if m == "lama" else "csfr" if m == "csfr" else "classical",
                    "status": "done", "psnr_hole": v, "ssim_hole": 0.5, "gmsd_hole": 0.1, "psnr_whole": v}
                for m, v in psnr.items()}}
            rows += mod.method_rows(case, metrics)
    summ = mod.summarise(rows, ["bilinear", "fsr"], n_boot=200, seed=0)
    assert {(r["kind"], r["condition"]) for r in summ} == {("truncate", "oracle"), ("flip", "oracle")}
    csfr = {r["kind"]: r for r in summ if r["method"] == "csfr"}
    for kind, shift in (("truncate", 0.0), ("flip", 5.0)):
        r = csfr[kind]
        assert r["n_cases"] == 4 and r["psnr_hole_mean"] == 20 + shift + 1.5
        assert r["vs_best_classical"] == "fsr" and r["diff_psnr_hole_vs_best_classical"] == 1.0
        assert r["vs_lama"] == "lama" and r["diff_psnr_hole_vs_lama"] == -1.0
        assert r["n_paired_vs_best_classical"] == 4 and r["n_paired_vs_lama"] == 4


def test_summary_counts_but_does_not_compare_cases_without_information():
    # a mask that marks nothing missing leaves every method the decode; one that
    # marks everything missing leaves them no data, and their scores are only
    # their fallback constants: both are counted, neither enters the means or
    # the paired tests
    mod = _driver()
    rows = []
    for c, (nothing_to_fill, nothing_observed, shift) in enumerate(
            [(False, False, 0.0), (False, False, 1.0), (True, False, 50.0), (False, True, -50.0)]):
        case = {"case": f"f{c}_flip_r0", "kind": "flip", "condition": "offset", "nothing_observed": nothing_observed}
        metrics = {"nothing_to_fill": nothing_to_fill, "methods": {
            m: {"kind": "csfr" if m == "csfr" else "classical", "status": "done", "psnr_hole": v + shift,
                "ssim_hole": 0.5, "gmsd_hole": 0.1, "psnr_whole": v}
            for m, v in {"csfr": 20.0, "bilinear": 19.0}.items()}}
        rows += mod.method_rows(case, metrics)
    summ = mod.summarise(rows, ["bilinear"], n_boot=200, seed=0)
    csfr = next(r for r in summ if r["method"] == "csfr")
    assert csfr["n_cases"] == 2 and csfr["psnr_hole_mean"] == 20.5
    assert (csfr["n_nothing_to_fill"], csfr["n_nothing_observed"]) == (1, 1)
    assert csfr["n_paired_vs_best_classical"] == 2 and csfr["diff_psnr_hole_vs_best_classical"] == 1.0
