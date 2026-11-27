"""kaggle/jpeg_gpu_kernel.py and scripts/merge_jpeg_gpu.py: the GPU run solves
every file once, in the stated order, one shard per GPU; it rebuilds the source
JPEGs only when they match the manifest; it judges completeness from the config;
it brings back no arrays but the figure's; and the merge keeps the inputs, the
CPU records and the one-device rule."""
import csv
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def jk():
    return _load("jpeg_gpu_kernel", "kaggle/jpeg_gpu_kernel.py")


@pytest.fixture(scope="module")
def mg():
    return _load("merge_jpeg_gpu", "scripts/merge_jpeg_gpu.py")


def test_a_first_session_without_kaggle_input_runs_the_kernel(jk, tmp_path, monkeypatch):
    monkeypatch.setattr(jk, "KAGGLE_INPUT", tmp_path / "input")  # absent: no input attached
    monkeypatch.setattr(jk, "KAGGLE_OUT", tmp_path / "working")
    monkeypatch.setattr(jk, "kernel", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["jpeg_gpu_kernel.py"])
    assert jk.main() == 2  # not on Kaggle: no /kaggle/working
    (tmp_path / "working").mkdir()
    assert jk.main() == 0
    assert jk.restore_state(tmp_path / "input", tmp_path / "work") == 1


def test_the_order_solves_every_manifest_file_once(jk):
    drv = _load("jpeg_drv_order", "scripts/run_jpeg_corruption.py")
    files = json.loads((ROOT / jk.MANIFEST).read_text(encoding="utf-8"))["files"]
    order = drv.parse_files(jk.ORDER)
    assert sorted(order) == list(range(len(files)))
    assert drv.parse_files("3-5,0,1-2") == [3, 4, 5, 0, 1, 2]


def test_each_gpu_runs_its_own_shard_csfr_only(jk):
    cmds = [jk.shard_cmd("py", k, 2, f"cuda:{k}") for k in range(2)]
    for k, cmd in enumerate(cmds):
        s = " ".join(map(str, cmd))
        assert "--csfr-only" in cmd and f"--output-dir {jk.GPU_DIR}/shard{k}" in s
        assert f"--files {jk.ORDER}" in s and f"--shard {k}/2" in s and f"--device cuda:{k}" in s


def test_the_figure_file_is_the_figure_scripts(jk):
    src = (ROOT / "scripts" / "render_figures_v2.py").read_text(encoding="utf-8")
    m = re.search(r"^JPEG_EX_FILE, JPEG_EX_REPEAT, JPEG_EX_MASK = (\d+),", src, re.M)
    assert m and int(m.group(1)) == jk.FIGURE_FILE


# ── the DFTT JPEGs ───────────────────────────────────────────────────


def _jpeg(seed, side=96):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (side, side), dtype=np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _dftt_work(tmp_path, blobs, sha_of_second=None):
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    for s in ("fetch_corpora.py", "extract_patches.py"):
        shutil.copy(ROOT / "scripts" / s, work / "scripts" / s)
    (work / "data/jpeg_corruption").mkdir(parents=True)
    recs = [{"file": f"mikus_11-carve-fat_{i}.jpg", "source": f"11-carve-fat.dd#carved{i:04d}",
             "sha256": hashlib.sha256(b).hexdigest(), "corpus": "nick_mikus_dftt"} for i, b in enumerate(blobs)]
    if sha_of_second:
        recs[1]["sha256"] = sha_of_second
    (work / "data/jpeg_corruption/manifest.json").write_text(json.dumps({"files": recs}))
    disk = b"\x00" * 4096 + blobs[0] + b"\x00" * 4096 + blobs[1] + b"\x00" * 4096
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("11-carve-fat/11-carve-fat.dd", disk)
    return work, zbuf.getvalue()


def test_dftt_jpegs_are_carved_from_the_test_image_and_checked(jk, tmp_path):
    blobs = [_jpeg(0), _jpeg(1)]
    assert all(len(b) > 2000 for b in blobs)                 # the carver's minimum size
    work, zbytes = _dftt_work(tmp_path, blobs)
    urls = []

    def download(url, dest):
        urls.append(url)
        Path(dest).write_bytes(zbytes)

    assert jk.fetch_dftt_jpegs(work, cache=tmp_path / "cache", download=download) == 2
    assert urls == [jk.DFTT_URL.format(jk.DFTT_ARCHIVES["11-carve-fat"])]
    for i, b in enumerate(blobs):
        assert (work / jk.SRC / f"mikus_11-carve-fat_{i}.jpg").read_bytes() == b
    assert jk.fetch_dftt_jpegs(work, cache=tmp_path / "cache", download=download) == 0   # present: no download
    assert len(urls) == 1


def test_a_dftt_jpeg_that_differs_from_its_manifest_is_refused(jk, tmp_path):
    work, zbytes = _dftt_work(tmp_path, [_jpeg(0), _jpeg(1)], sha_of_second="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        jk.fetch_dftt_jpegs(work, cache=tmp_path / "cache", download=lambda u, d: Path(d).write_bytes(zbytes))
    assert not (work / jk.SRC / "mikus_11-carve-fat_1.jpg").exists()


# ── completeness ─────────────────────────────────────────────────────


def _gpu_tree(root, *, drop=None):
    """A small repo: one config, two manifest files, one shard holding every
    visible condition with both variants done (except `drop`)."""
    cfg = {"kinds": ["truncate"], "repeats": 1, "conditions": ["oracle", "decoder"],
           "csfr_variants": ["csfr", "csfr_tv_only"]}
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "jpeg_corruption.yaml").write_text(yaml.safe_dump(cfg))
    (root / "data/jpeg_corruption").mkdir(parents=True)
    (root / "data/jpeg_corruption/manifest.json").write_text(json.dumps({"files": [{"file": "a.jpg"},
                                                                                 {"file": "b.jpg"}]}))
    shard = root / "results/jpeg_corruption_gpu/shard0"
    rows = []
    for stem, visible in (("a", True), ("b", False)):
        for cond in cfg["conditions"]:
            rows.append({"case": f"{stem}_truncate_r0", "condition": cond, "visible_damage": visible})
            if visible:
                d = shard / "cases" / f"{stem}_truncate_r0" / cond
                d.mkdir(parents=True)
                methods = {v: {"status": "done", "psnr_hole": 20.0} for v in cfg["csfr_variants"]
                           if (cond, v) != drop}
                (d / "metrics.json").write_text(json.dumps({"methods": methods}))
    with (shard / "masks_quality.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def test_complete_when_every_visible_condition_has_both_variants(jk, tmp_path):
    assert jk.missing(_gpu_tree(tmp_path)) == []


def test_a_missing_variant_or_case_is_reported(jk, tmp_path):
    root = _gpu_tree(tmp_path / "x", drop=("decoder", "csfr_tv_only"))
    assert jk.missing(root) == ["a_truncate_r0/decoder: lacks csfr_tv_only"]
    root = _gpu_tree(tmp_path / "y")
    (root / "data/jpeg_corruption/manifest.json").write_text(json.dumps({"files": [{"file": "a.jpg"},
                                                                                 {"file": "c.jpg"}]}))
    assert jk.missing(root) == ["c_truncate_r0/oracle: not run", "c_truncate_r0/decoder: not run"]


def test_status(jk):
    assert jk.status_of([0, 0], False, []) == "complete"
    assert jk.status_of([None, 0], True, ["x"]) == "incomplete (time budget)"
    assert jk.status_of([0, 1], False, []).startswith("failed: shard1 exited 1")
    assert jk.status_of([0, 0], False, ["x"]).startswith("failed: the shards finished, 1")
    # a crash is not hidden behind the other shard's time budget
    assert jk.status_of([None, 1], True, ["x"]).startswith(
        "failed: shard1 exited 1 (the others stopped at the time budget)")


def test_a_second_shard_layout_neither_hides_nor_doubles_a_record(jk, tmp_path):
    root = _gpu_tree(tmp_path)
    other = root / "results/jpeg_corruption_gpu/shard1/cases/a_truncate_r0/oracle"
    other.mkdir(parents=True)
    (other / "metrics.json").write_text(json.dumps({"methods": {"csfr": {"status": "done"}}}))   # incomplete
    assert jk.missing(root) == []
    done = jk.solved(root / "results/jpeg_corruption_gpu", ["csfr", "csfr_tv_only"])
    assert sorted(done) == [("a_truncate_r0", "decoder"), ("a_truncate_r0", "oracle")]
    both = {"methods": {v: {"status": "done"} for v in ("csfr", "csfr_tv_only")}, "inputs_sha256": {"y": "1"}}
    (other / "metrics.json").write_text(json.dumps(both))                    # complete, other inputs
    with pytest.raises(RuntimeError, match="different inputs"):
        jk.missing(root)


def test_completeness_is_judged_on_the_given_gpu_tree(jk, tmp_path):
    root = _gpu_tree(tmp_path)
    moved = tmp_path / "elsewhere"
    shutil.move(str(root / "results/jpeg_corruption_gpu"), str(moved))
    assert jk.missing(root, moved) == []
    assert jk.missing(root) != []


def test_shards_run_together_and_stop_at_the_budget(jk, tmp_path):
    py = sys.executable
    logs = [tmp_path / "a.log", tmp_path / "b.log"]
    rcs, timed_out = jk.run_all_with_budget([[py, "-c", "pass"], [py, "-c", "raise SystemExit(3)"]],
                                            seconds=120, logs=logs)
    assert (rcs, timed_out) == ([0, 3], False)
    rcs, timed_out = jk.run_all_with_budget([[py, "-c", "pass"], [py, "-c", "import time; time.sleep(300)"]],
                                            seconds=30, logs=logs)
    assert (rcs, timed_out) == ([0, None], True)


def test_outputs_keep_only_the_figures_arrays(jk, tmp_path):
    work = tmp_path / "work"
    (work / "data/jpeg_corruption").mkdir(parents=True)
    files = [{"file": f"f{i}.jpg"} for i in range(jk.FIGURE_FILE + 1)]
    (work / jk.MANIFEST).write_text(json.dumps({"files": files}))
    stem = f"f{jk.FIGURE_FILE}"
    for case in (f"{stem}_flip_r0", "f0_flip_r0"):
        d = work / jk.GPU_DIR / "shard0" / "cases" / case / "oracle"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
        for name in ("x_hat_csfr.npy", "x_hat_csfr_tv_only.npy", "y.npy", "mask.npy", ".partial_x.npy"):
            np.save(d / name, np.zeros(2))
    out = tmp_path / "out"
    jk.save_outputs(work, out, status="incomplete (time budget)", record={"session": 1}, lacking=["x"])
    with tarfile.open(out / "results.tar.gz") as t:
        names = {Path(n).as_posix() for n in t.getnames()}
    arrays = {n for n in names if n.endswith(".npy")}
    base = f"{jk.GPU_DIR}/shard0/cases/{stem}_flip_r0/oracle"
    assert arrays == {f"{base}/x_hat_csfr.npy", f"{base}/x_hat_csfr_tv_only.npy"}
    assert f"{jk.GPU_DIR}/shard0/cases/f0_flip_r0/oracle/metrics.json" in names
    state = out / jk.STATE_DIR
    kept = {p.relative_to(state).as_posix() for p in state.rglob("*.npy")}
    assert kept == {f"{base}/x_hat_csfr.npy", f"{base}/x_hat_csfr_tv_only.npy"}   # a later session keeps them
    assert json.loads((state / jk.STATE_FILE).read_text())["session"] == 1
    assert (out / "STATUS").read_text().strip() == "incomplete (time budget)"
    assert jk.restore_state(out, tmp_path / "next") == 2
    assert (tmp_path / "next" / jk.GPU_DIR / "shard0" / "cases" / "f0_flip_r0" / "oracle" / "metrics.json").exists()


# ── the merge ────────────────────────────────────────────────────────


def _sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def _merge_tree(root, *, other_inputs=False):
    y = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    m = np.ones((1, 2, 3), dtype=np.float32)
    hashes = {"y": _sha(y), "mask": _sha(m)}
    gpu = root / "results/jpeg_corruption_gpu/shard0/cases"
    for case in ("fig_truncate_r0", "z_truncate_r0"):
        d = gpu / case / "oracle"
        d.mkdir(parents=True)
        rec = {"status": "done", "kind": "csfr", "iterations": 40, "gap": 1e-7, "psnr_hole": 21.5,
               "ssim_hole": 0.8, "gmsd_hole": 0.1, "psnr_whole": 30.0, "exact": False, "converged": True}
        (d / "metrics.json").write_text(json.dumps({"methods": {"csfr": rec}, "nothing_to_fill": False,
                                                    "inputs_sha256": hashes}))
        np.save(d / "x_hat_csfr.npy", np.full((1, 2, 3), 7.0, dtype=np.float32))
    local = root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle"   # the workstation reached this one
    local.mkdir(parents=True)
    np.save(local / "y.npy", y + 1 if other_inputs else y)
    np.save(local / "mask.npy", m)
    np.save(local / "x_hat_csfr.npy", np.full((1, 2, 3), 6.0, dtype=np.float32))
    cpu = {"status": "done", "kind": "csfr", "iterations": 40, "gap": 1.1e-7, "psnr_hole": 21.5000001,
           "ssim_hole": 0.8, "gmsd_hole": 0.1, "psnr_whole": 30.0, "exact": False, "converged": True}
    (local / "metrics.json").write_text(json.dumps({"methods": {"csfr": cpu, "bilinear": {"status": "done"}},
                                                    "nothing_to_fill": False}))
    return root


def test_merge_puts_the_gpu_records_in_and_keeps_the_cpu_ones(mg, tmp_path):
    root = _merge_tree(tmp_path)
    pairs = mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")
    lm = json.loads((root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle/metrics.json").read_text())
    assert list(lm["methods"]) == ["csfr", "bilinear"]                  # order kept, baselines untouched
    assert lm["methods"]["csfr"]["device"] == "cuda" and lm["methods"]["csfr"]["gap"] == 1e-7
    assert lm["csfr_cpu"]["csfr"]["gap"] == 1.1e-7
    new = json.loads((root / "results/jpeg_corruption/cases/z_truncate_r0/oracle/metrics.json").read_text())
    assert list(new["methods"]) == ["csfr"] and "csfr_cpu" not in new and new["inputs_sha256"]
    assert [(p["case"], p["cpu_gap"], p["gpu_gap"]) for p in pairs] == [("fig_truncate_r0", 1.1e-7, 1e-7)]
    d = root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle"
    assert float(np.load(d / "x_hat_csfr.npy").max()) == 7.0 and float(np.load(d / "x_hat_csfr_cpu.npy").max()) == 6.0
    diffs = mg.largest_differences(pairs)
    assert diffs["n"] == 1 and diffs["max_abs_diff_iterations"] == 0
    again = mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")   # idempotent
    assert again == pairs
    assert float(np.load(d / "x_hat_csfr_cpu.npy").max()) == 6.0


def test_merge_refuses_a_solve_of_other_inputs(mg, tmp_path):
    root = _merge_tree(tmp_path, other_inputs=True)
    with pytest.raises(RuntimeError, match="other inputs"):
        mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")


def test_merge_dry_run_writes_nothing(mg, tmp_path):
    root = _merge_tree(tmp_path)
    before = (root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle/metrics.json").read_text()
    pairs = mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig", dry_run=True)
    assert len(pairs) == 1
    assert (root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle/metrics.json").read_text() == before
    assert not (root / "results/jpeg_corruption/cases/z_truncate_r0").exists()


def test_merge_refuses_records_of_other_inputs_even_when_the_files_match(mg, tmp_path):
    root = _merge_tree(tmp_path)
    lm = root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle/metrics.json"
    m = json.loads(lm.read_text())
    m["inputs_sha256"] = {"y": "0" * 64, "mask": "0" * 64}
    lm.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="other inputs"):
        mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")


def test_merge_refuses_a_figure_condition_without_its_gpu_outputs(mg, tmp_path):
    root = _merge_tree(tmp_path)
    (root / "results/jpeg_corruption_gpu/shard0/cases/fig_truncate_r0/oracle/x_hat_csfr.npy").unlink()
    with pytest.raises(RuntimeError, match="figure's GPU outputs"):
        mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")


def test_merge_neither_marks_nor_compares_a_condition_with_nothing_to_fill(mg, tmp_path):
    root = _merge_tree(tmp_path)
    for side in ("results/jpeg_corruption_gpu/shard0", "results/jpeg_corruption"):
        mp = root / side / "cases/fig_truncate_r0/oracle/metrics.json"
        m = json.loads(mp.read_text())
        m["nothing_to_fill"] = True
        mp.write_text(json.dumps(m))
    (root / "results/jpeg_corruption_gpu/shard0/cases/fig_truncate_r0/oracle/x_hat_csfr.npy").unlink()
    pairs = mg.merge(root, root / "results/jpeg_corruption_gpu", ["csfr"], "fig")
    assert pairs == []
    lm = json.loads((root / "results/jpeg_corruption/cases/fig_truncate_r0/oracle/metrics.json").read_text())
    assert "device" not in lm["methods"]["csfr"] and "csfr_cpu" not in lm
    assert lm["methods"]["csfr"]["gap"] == 1.1e-7                           # the workstation's record stays


def test_merge_refuses_an_incomplete_gpu_run(mg, monkeypatch):
    seen = []

    class Kernel:
        FIGURE_FILE = 2

        @staticmethod
        def missing(root, gpu_dir):
            seen.append(gpu_dir)
            return ["a_truncate_r0/oracle: not run"]

    monkeypatch.setattr(mg, "_gpu_kernel", lambda: Kernel)
    assert mg.main(["--gpu", "results/elsewhere"]) == 1
    assert seen == [mg.ROOT / "results/elsewhere"]
