"""kaggle/downstream_v2_kernel.py: the Kaggle session runs the workstation's
downstream stage within its time budget and keeps what a later session or the
workstation needs; completeness is judged from the configs that define the
stage; the dataset holds only the committed tree and checked GovDocs1 JPEGs."""
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import psutil
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("kaggle_kernel", ROOT / "kaggle" / "downstream_v2_kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _complete_tree(root: Path) -> Path:
    """A downstream results tree, complete for a shrunken grid: 2 patches, seeds
    0 and 1, one cell, two methods; the OCR grid is the real one over 2 crops."""
    for clf in ("cnn", "resnet"):
        cfg = yaml.safe_load((ROOT / "configs" / f"downstream_v2_{clf}.yaml").read_text(encoding="utf-8"))
        cfg.update(n_test=2, seeds=[0, 1], families=["CF1"], levels={"L1": 0.05}, methods=["csfr", "lama"])
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "configs" / f"downstream_v2_{clf}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        d = root / cfg["output_dir"]
        _write_csv(d / "clean.csv", [{"patch_id": i} for i in range(2)])
        for m in cfg["methods"]:
            _write_csv(d / "CF1_L1" / f"{m}.csv", [{"patch_id": i, "seed": s} for s in (0, 1) for i in range(2)])
    for rel in ("results/downstream_v2/classifier_resnet.pt", "results/downstream_v2/stats/metrics.csv",
                "results/downstream_v2/stats/stats.csv", "results/downstream_v2/stats/coverage.csv",
                "data/ocr/crops.npy", "results/ocr/summary.csv", "results/ocr/per_crop.csv"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (root / "data/ocr/manifest.json").write_text(json.dumps({"crops": [{"crop": 0}, {"crop": 1}]}))
    (root / "scripts").mkdir(exist_ok=True)
    shutil.copy(ROOT / "scripts" / "run_ocr_task.py", root / "scripts")
    ocr = _mod()._ocr_constants(root)
    for cell in ocr.CELLS:
        for seed in ocr.SEEDS:
            p = root / "results/ocr/cases" / cell / f"seed{seed}" / "texts.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({k: ["a", "b"] for k in [*ocr.METHODS, *ocr.EXTERNAL, "corrupted_input"]}))
    return root


def test_a_complete_tree_lacks_nothing(tmp_path):
    assert _mod().missing(_complete_tree(tmp_path)) == []


def test_what_a_tree_lacks_is_named(tmp_path):
    m, root = _mod(), _complete_tree(tmp_path)
    (root / "results/downstream_v2/cnn/CF1_L1/lama.csv").unlink()
    _write_csv(root / "results/downstream_v2/resnet/CF1_L1/csfr.csv", [{"patch_id": i, "seed": 0} for i in range(2)])
    texts = root / "results/ocr/cases/CF2_L3/seed1/texts.json"
    t = json.loads(texts.read_text())
    del t["lama"]
    t["csfr"] = ["a"]
    texts.write_text(json.dumps(t))
    (root / "results/downstream_v2/stats/coverage.csv").unlink()
    assert m.missing(root) == [
        "results/downstream_v2/cnn/CF1_L1/lama.csv: absent",
        "results/downstream_v2/resnet/CF1_L1/csfr.csv: 2 rows, seeds ['0']",
        "results/downstream_v2/stats/coverage.csv: absent",
        "results/ocr/cases/CF2_L3/seed1/texts.json: lacks csfr, lama",
    ]


# the stage writes to a log file, as in the kernel: a child inheriting pytest's
# captured stdout gets an invalid handle on Windows
def test_a_stage_that_ends_in_time_returns_its_code(tmp_path):
    with (tmp_path / "stage.log").open("w") as log:
        assert _mod().run_with_budget([sys.executable, "-c", "import sys; sys.exit(3)"], seconds=60,
                                      stdout=log) == (3, False)


def test_the_budget_stops_the_stage_and_everything_it_started(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = ("import subprocess, sys, time\n"
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
              f"open(r'{pid_file}', 'w').write(str(c.pid))\n"
              "time.sleep(600)\n")
    with (tmp_path / "stage.log").open("w") as log:
        assert _mod().run_with_budget([sys.executable, "-c", script], seconds=8, stdout=log) == (None, True)
    child = int(pid_file.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("rc, timed_out, lacking, expected", [
    (0, False, [], "complete"),
    (None, True, ["a"], "incomplete (time budget)"),
    (1, False, ["a"], "failed: the stage exited 1"),
    (0, False, ["a: absent"], "failed: the stage finished, 1 outputs incomplete: a: absent"),
])
def test_the_status_says_how_the_session_ended(rc, timed_out, lacking, expected):
    assert _mod().status_of(rc, timed_out, lacking).startswith(expected)


def test_a_mounted_input_is_found_at_any_depth_and_two_are_refused(tmp_path):
    m = _mod()
    nested = tmp_path / "datasets" / "user" / "slug"
    (nested / "scripts").mkdir(parents=True)
    (nested / "scripts" / "run_round2_campaign.sh").write_text("")
    assert m.find_marked(tmp_path, "scripts/run_round2_campaign.sh") == nested
    assert m.find_marked(tmp_path, "state/STATE.json") is None
    (tmp_path / "other" / "scripts").mkdir(parents=True)
    (tmp_path / "other" / "scripts" / "run_round2_campaign.sh").write_text("")
    with pytest.raises(RuntimeError, match="more than one"):
        m.find_marked(tmp_path, "scripts/run_round2_campaign.sh")


def _work(root: Path) -> Path:
    for rel in ("results/downstream_v2/cnn/CF1_L1/csfr.csv", "results/downstream_v2/classifier_resnet.pt",
                "results/downstream_v2/classifier_resnet_training.pt", "results/downstream_v2/run_cnn.log",
                "results/downstream_v2/cases/CF1_L1/seed0/x_hat_csfr.npy",
                "results/ocr/cases/CF1_L1/seed0/texts.json", "results/ocr/cases/CF1_L1/seed0/.partial_x_hat_fsr.npy",
                "data/ocr/crops.npy", "data/ocr/manifest.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    return root


def _files_in(tar_path: Path) -> set[str]:
    with tarfile.open(tar_path) as t:
        return {m.name for m in t.getmembers() if m.isfile()}


def test_an_unfinished_session_keeps_its_state_and_the_results_hold_no_arrays(tmp_path):
    m = _mod()
    work, out = _work(tmp_path / "work"), tmp_path / "out"
    m.save_outputs(work, out, status="incomplete (time budget)", record={"session": 2}, lacking=["a"])
    assert _files_in(out / "results.tar.gz") == {
        "results/downstream_v2/cnn/CF1_L1/csfr.csv", "results/downstream_v2/classifier_resnet.pt",
        "results/downstream_v2/run_cnn.log", "results/ocr/cases/CF1_L1/seed0/texts.json",
        "data/ocr/crops.npy", "data/ocr/manifest.json"}
    assert (out / "state/results/downstream_v2/cases/CF1_L1/seed0/x_hat_csfr.npy").exists()
    assert (out / "state/results/downstream_v2/classifier_resnet_training.pt").exists()
    assert json.loads((out / "state/STATE.json").read_text())["session"] == 2
    assert (out / "STATUS").read_text().strip() == "incomplete (time budget)"
    assert json.loads((out / "kaggle_env.json").read_text())["status"] == "incomplete (time budget)"

    # the next session continues from that state, counting on
    work2 = tmp_path / "work2"
    assert m.restore_state(out.parent, work2) == 3
    assert (work2 / "results/downstream_v2/cases/CF1_L1/seed0/x_hat_csfr.npy").exists()


def test_a_finished_session_saves_no_state(tmp_path):
    m = _mod()
    out = tmp_path / "out"
    m.save_outputs(_work(tmp_path / "work"), out, status="complete", record={"session": 1}, lacking=[])
    assert not (out / "state").exists() and (out / "results.tar.gz").exists()


def test_a_session_with_no_work_tree_still_reports(tmp_path):
    m = _mod()
    m.save_outputs(tmp_path / "absent", tmp_path / "out", status="failed: x", record={}, lacking=[])
    assert (tmp_path / "out" / "STATUS").read_text().strip() == "failed: x"


def test_an_environment_without_a_module_or_a_working_gpu_is_refused():
    m = _mod()
    ok = {"python": "3.12", "numpy": "2.5.1", "cuda_ok": True, "gpu": "Tesla T4", "cv2_xphoto": True}
    assert m._probe_failures("main", ok) == []
    assert m._probe_failures("main", {**ok, "numpy": "MISSING: no module"}) == ["main: numpy MISSING: no module"]
    assert m._probe_failures("lama", {**ok, "cuda_ok": False, "gpu": "no kernel image"}) == \
        ["lama: no working CUDA (no kernel image)"]
    assert m._probe_failures("main", {**ok, "cv2_xphoto": False}) == ["main: cv2 without xphoto (shift-map, FSR)"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)


def _repo(tmp_path: Path) -> Path:
    import hashlib
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_round2_campaign.sh").write_text("echo\n")
    jpg = b"\xff\xd8 a govdocs jpeg"
    src = repo / "data/jpeg_corruption/src"
    src.mkdir(parents=True)
    (src / "govdocs_000.jpg").write_bytes(jpg)
    (src / "dfrws_000.jpg").write_bytes(b"not wanted")
    (repo / "data/jpeg_corruption/manifest.json").write_text(json.dumps({"files": [
        {"file": "govdocs_000.jpg", "corpus": "govdocs1", "sha256": hashlib.sha256(jpg).hexdigest()},
        {"file": "dfrws_000.jpg", "corpus": "dfrws", "sha256": "0"}]}))
    (repo / ".gitignore").write_text("data/jpeg_corruption/src/\nsecret.txt\n")
    (repo / "secret.txt").write_text("not for upload")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    (repo / "untracked.txt").write_text("not committed")
    return repo


def test_the_dataset_is_the_committed_tree_and_the_checked_govdocs_jpegs(tmp_path):
    m = _mod()
    path = m.stage_dataset(_repo(tmp_path), tmp_path / "ds", "someone")
    assert _files_in(path) == {".gitignore", "scripts/run_round2_campaign.sh", "data/jpeg_corruption/manifest.json",
                               "data/jpeg_corruption/src/govdocs_000.jpg"}
    meta = json.loads((tmp_path / "ds" / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/csfr-round2-code"


def test_a_jpeg_that_does_not_match_its_manifest_is_refused(tmp_path):
    m, repo = _mod(), _repo(tmp_path)
    (repo / "data/jpeg_corruption/src/govdocs_000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="govdocs_000.jpg"):
        m.stage_dataset(repo, tmp_path / "ds", "someone")


def test_uncommitted_changes_are_refused_because_they_would_not_reach_kaggle(tmp_path):
    m, repo = _mod(), _repo(tmp_path)
    (repo / "scripts" / "run_round2_campaign.sh").write_text("changed\n")
    with pytest.raises(RuntimeError, match="commit first"):
        m.stage_dataset(repo, tmp_path / "ds", "someone")


def test_a_later_session_takes_the_previous_output_as_input(tmp_path):
    m = _mod()
    one = json.loads((m.kernel_dir(tmp_path / "k1", "someone", 1) / "kernel-metadata.json").read_text())
    two = json.loads((m.kernel_dir(tmp_path / "k2", "someone", 2) / "kernel-metadata.json").read_text())
    assert one["id"] == "someone/csfr-downstream-v2-s1" and one["kernel_sources"] == []
    assert two["kernel_sources"] == ["someone/csfr-downstream-v2-s1"]
    assert one["enable_gpu"] and one["enable_internet"] and one["is_private"]
    assert one["dataset_sources"] == ["someone/csfr-round2-code"]
    assert (tmp_path / "k1" / one["code_file"]).exists()


def test_a_cloned_work_tree_is_used_as_it_is(tmp_path):
    # the notebook route clones the repository to the work tree before the driver runs
    m = _mod()
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    (work / "scripts" / "run_round2_campaign.sh").write_text("clone\n")
    assert m.locate_source(tmp_path / "no_input", work) == "clone"
    assert (work / "scripts" / "run_round2_campaign.sh").read_text() == "clone\n"


def _govdocs_work(tmp_path: Path, jpg: bytes, sha: str) -> Path:
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "fetch_corpora.py", work / "scripts")
    (work / "data/jpeg_corruption").mkdir(parents=True)
    (work / "data/jpeg_corruption/manifest.json").write_text(json.dumps({"files": [
        {"file": "govdocs_000.jpg", "corpus": "govdocs1", "source": "001.zip:001/001234.jpg", "sha256": sha},
        {"file": "mikus_000.jpg", "corpus": "nick_mikus_dftt", "source": "x", "sha256": "0"}]}))
    return work


def _zip_with(path: Path, member: str, data: bytes) -> Path:
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, data)
    return path


def test_the_govdocs_jpegs_come_from_their_archives_checked_by_hash(tmp_path):
    import hashlib
    m, jpg = _mod(), b"\xff\xd8 a govdocs jpeg"
    work = _govdocs_work(tmp_path, jpg, hashlib.sha256(jpg).hexdigest())
    archive = _zip_with(tmp_path / "local.zip", "001/001234.jpg", jpg)
    urls = []

    def download(url, dest):
        urls.append(url)
        shutil.copy(archive, dest)

    assert m.fetch_govdocs_jpegs(work, tmp_path / "cache", download=download) == 1
    assert urls == ["https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/001.zip"]
    assert (work / "data/jpeg_corruption/src/govdocs_000.jpg").read_bytes() == jpg
    # present and matching: nothing to fetch
    assert m.fetch_govdocs_jpegs(work, tmp_path / "cache", download=download) == 0 and len(urls) == 1


def test_a_govdocs_jpeg_that_does_not_match_its_manifest_is_not_written(tmp_path):
    m = _mod()
    work = _govdocs_work(tmp_path, b"", "0" * 64)
    archive = _zip_with(tmp_path / "local.zip", "001/001234.jpg", b"something else")
    with pytest.raises(ValueError, match="govdocs_000.jpg"):
        m.fetch_govdocs_jpegs(work, tmp_path / "cache", download=lambda url, dest: shutil.copy(archive, dest))
    assert not (work / "data/jpeg_corruption/src/govdocs_000.jpg").exists()


@pytest.mark.parametrize("system_site", [False, True])
def test_a_venv_gets_pip_from_the_image_pip_not_ensurepip(tmp_path, monkeypatch, system_site):
    m = _mod()
    calls = []
    monkeypatch.setattr(m, "sh", lambda cmd, **kw: calls.append([str(c) for c in cmd]))
    py = m._venv(tmp_path / ".venv-x", system_site=system_site)
    assert py == tmp_path / ".venv-x" / "bin" / "python"
    create, bootstrap = calls
    assert create[:4] == [sys.executable, "-m", "venv", "--without-pip"]
    assert ("--system-site-packages" in create) == system_site
    assert bootstrap[:5] == [sys.executable, "-m", "pip", "--python", str(py)]
    assert "--ignore-installed" in bootstrap and bootstrap[-1] == "pip"


def _fake_main_env(m, monkeypatch, probes):
    calls = []
    monkeypatch.setattr(m, "_venv", lambda path, system_site: Path(path) / "bin" / "python")
    monkeypatch.setattr(m, "sh", lambda cmd, **kw: calls.append([str(c) for c in cmd]))
    monkeypatch.setattr(m, "_probe", lambda py, modules: probes.pop(0))
    monkeypatch.setattr(m, "capture", lambda cmd, **kw: "torch==2.13.0\nnumpy==2.3.5\n")
    return calls


def test_the_main_environment_alone_gets_a_cuda_torch(tmp_path, monkeypatch):
    m = _mod()
    (tmp_path / "requirements.txt").write_text("numpy==2.3.5\ntorch==2.13.0\n")
    ok = {"numpy": "2.3.5", "torch": "2.13.0", "cuda_ok": True, "gpu": "Tesla T4", "cv2_xphoto": True}
    calls = _fake_main_env(m, monkeypatch, [{**ok, "cuda_ok": False, "gpu": "no kernel image"}, dict(ok)])
    py, probe = m.make_main_environment(tmp_path)
    assert py == tmp_path / ".venv-main" / "bin" / "python"
    assert probe["cuda_ok"] and probe["freeze"] == ["torch==2.13.0", "numpy==2.3.5"]
    assert any("--index-url" in c and m.TORCH_INDEXES[0] in c and "torch==2.13.0" in c for c in calls)


def test_the_main_environment_alone_refuses_a_missing_module(tmp_path, monkeypatch):
    m = _mod()
    (tmp_path / "requirements.txt").write_text("torch==2.13.0\n")
    bad = {"numpy": "MISSING: no module", "cuda_ok": True, "gpu": "Tesla T4"}
    _fake_main_env(m, monkeypatch, [bad])
    with pytest.raises(RuntimeError, match="numpy MISSING"):
        m.make_main_environment(tmp_path)
