#!/usr/bin/env python3
"""Round-2 downstream stage (E12 SVHN classification, E13 OCR) in a Kaggle GPU session.

A Kaggle script kernel (accelerator GPU, internet on) that runs the stage the
workstation would run, `bash scripts/run_round2_campaign.sh downstream`, with
DOWNSTREAM_DEVICE=cuda and OCR_GPU=1: the ResNet training, LaMa and EasyOCR
use the GPU, and every other step runs as the stage defines it (the classifiers
and the reconstructions on the CPU, as configured). Three environments, as on
the workstation:

  .venv-main  requirements.txt exactly, isolated from the Kaggle image: the
              reconstructions and classifiers run under the package versions of
              the round-2 workstation runs (torch 2.13.0 in its CUDA build), and
              the image's diffusers cannot be picked up as an inpainting backend
  .venv-lama  the image's CUDA torch + simple-lama-inpainting 0.1.2 without its pins
  .venv-ocr   the image's CUDA torch + easyocr 1.7.2

The repository reaches the session either as a clone in /tmp/paper2_recon
(the notebook kaggle/downstream_v2.ipynb clones it from GitHub at a pinned
commit before running this script) or as a private dataset built by `stage`
below. The GovDocs1 JPEGs the OCR crops come from are not in git: when absent
they are downloaded from their Digital Corpora archives and each is kept only if
it matches its manifest SHA-256. From the second session on, the previous
session's output (its state/) is an input (/kaggle/input).

Before the stage starts: every environment runs a CUDA operation, the round-1
CNN classifier scores as on the workstation, the evaluation and calibration
subsets hash as on the workstation (EXPECTED), and every GovDocs1 JPEG matches
its manifest SHA-256. Each step of the stage resumes where it stopped, so the
stage is stopped at the session's time budget, state/ is saved, and the next
session, given this output, continues.

Outputs (/kaggle/working):
  STATUS           complete | incomplete (time budget) | failed: <reason>
  results.tar.gz   results/downstream_v2, results/ocr and data/ocr without the
                   arrays and the training checkpoint: what the workstation needs
  state/           everything a later session needs, arrays included (only
                   while the stage is not complete)
  kaggle_env.json  the GPU, CPU count, package versions per environment, timings
  stage.log        the stage's own output; each step logs to results/*/run_*.log

Locally:
  python kaggle/downstream_v2_kernel.py stage <dir> --user <kaggle user>   # the dataset to upload
  python kaggle/downstream_v2_kernel.py kernel-dir <dir> --user <u> --session <n>
  python kaggle/downstream_v2_kernel.py check [<repo>]   # what a results tree still lacks
  python kaggle/downstream_v2_kernel.py expected         # recompute EXPECTED
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUT = Path("/kaggle/working")
WORK = Path("/tmp/paper2_recon")
SRC_TARBALL = "paper2_recon_src.tar.gz"
MARKER = "scripts/run_round2_campaign.sh"
STATE_DIR = "state"
STATE_FILE = "STATE.json"
DATASET_SLUG = "csfr-round2-code"
KERNEL_SLUG = "csfr-downstream-v2-s{}"
SESSION_SECONDS = 12 * 3600
# saving state/ (the arrays, up to ~14 GB) and the results tarball
RESERVE_SECONDS = 45 * 60
# what a later session needs; SVHN and the model weights download again
CARRIED = ("results/downstream_v2", "results/ocr", "data/ocr")
STAGE = ["bash", "scripts/run_round2_campaign.sh", "downstream"]

# the inputs as the workstation computes them (`expected`); a mismatch stops the session
EXPECTED = {
    "corpus": "svhn",
    "n_test_images": 26032,
    "eval_ids_sha256": "1d7bcb926678db0de111c6a87ba63f5cda426891bcf3b160a3765b98109e4af8",
    "calibration_sha256": "70b94340708aeeb4f3e438060274e7a50c7592cde126c4e52086b3324f833905",
    "cnn_clean_test_top1": 0.8783,
}

# computes the EXPECTED fields with the harness's own selection (run from the repo root)
INPUT_CHECK = r"""
import hashlib, importlib.util, json, sys
import numpy as np, yaml
sys.path.insert(0, "src")
from downstream import load_classifier, load_digit_dataset, predict
spec = importlib.util.spec_from_file_location("harness", "scripts/run_downstream.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
cfg = yaml.safe_load(open("configs/downstream_v2_cnn.yaml", encoding="utf-8"))
imgs, labs, corpus = load_digit_dataset(prefer=cfg["corpus"], split="test")
seed = int(cfg["seeds"][0])
_, _, ids = h.select_class_balanced(imgs, labs, int(cfg["n_test"]), seed)
ids = np.asarray(ids, dtype=np.int64)
rest = np.setdiff1d(np.arange(imgs.shape[0]), ids)
cal = np.sort(np.random.default_rng(seed + 1000).choice(rest, int(cfg["calibration_n"]), replace=False))
model = load_classifier(cfg["classifier_path"])
pred, _ = predict(model, imgs if imgs.ndim == 4 else imgs.unsqueeze(1))
print(json.dumps({"corpus": corpus, "n_test_images": int(imgs.shape[0]),
                  "eval_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                  "calibration_sha256": hashlib.sha256(cal.tobytes()).hexdigest(),
                  "cnn_clean_test_top1": round(float((pred.numpy() == labs.numpy()).mean()), 4)}))
"""

# environment probe: versions, and a CUDA convolution (a GPU the build has no kernels for fails here)
PROBE = r"""
import importlib, json, platform, sys
out = {"python": platform.python_version()}
for name in sys.argv[1:]:
    try:
        out[name] = getattr(importlib.import_module(name), "__version__", "present")
    except Exception as exc:
        out[name] = f"MISSING: {exc}"
import torch
out["cuda_build"] = torch.version.cuda
try:
    x = torch.nn.functional.conv2d(torch.ones(1, 3, 8, 8, device="cuda"), torch.ones(4, 3, 3, 3, device="cuda"))
    out["cuda_ok"] = float(x.sum().item()) == 4 * 36 * 27.0
    out["gpu"] = torch.cuda.get_device_name(0)
except Exception as exc:
    out["cuda_ok"], out["gpu"] = False, f"{type(exc).__name__}: {exc}"
if "cv2" in sys.argv[1:]:
    import cv2
    out["cv2_xphoto"] = hasattr(cv2, "xphoto")
print(json.dumps(out))
"""
MAIN_MODULES = ["numpy", "scipy", "skimage", "sklearn", "cv2", "torch", "yaml", "PIL"]
LAMA_MODULES = ["numpy", "torch", "PIL", "simple_lama_inpainting"]
OCR_MODULES = ["numpy", "torch", "PIL", "easyocr"]
# the PyPI torch wheel for Linux is a CUDA build; if it does not run on the
# session's driver, the same version from the PyTorch indexes for older CUDA
TORCH_INDEXES = ("https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu126")


def log(msg: str) -> None:
    print(f"[kaggle] {time.strftime('%F %T')} {msg}", flush=True)


# children get no stdin: one inheriting a captured stdin gets an invalid handle on Windows
def sh(cmd, **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd], check=True, stdin=subprocess.DEVNULL, **kw)


def capture(cmd, *, cwd=None, text: bool = True, check: bool = True):
    """The standard output of cmd."""
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, stdin=subprocess.DEVNULL,
                          text=text, check=check).stdout


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


# ── completeness ─────────────────────────────────────────────────────


def _csv_rows(path: Path) -> list[dict] | None:
    import csv
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ocr_constants(root: Path):
    return _script(root / "scripts" / "run_ocr_task.py", "_ocr_task")


def missing(root: Path) -> list[str]:
    """What the downstream stage has not yet written in full under root, from
    the configs and constants that define it: per classifier the clean CSV and
    every cell x method CSV with n_test rows per seed, the statistics, the
    trained ResNet, and per OCR case directory the texts of every method and of
    the corrupted input, plus the OCR scores. Empty when the stage is complete."""
    import yaml
    root = Path(root)
    out: list[str] = []
    for clf in ("cnn", "resnet"):
        cfg = yaml.safe_load((root / "configs" / f"downstream_v2_{clf}.yaml").read_text(encoding="utf-8"))
        d = root / cfg["output_dir"]
        n, seeds = int(cfg["n_test"]), {str(s) for s in cfg["seeds"]}
        clean = _csv_rows(d / "clean.csv")
        if clean is None or len(clean) != n:
            out.append(f"{cfg['output_dir']}/clean.csv: {'absent' if clean is None else f'{len(clean)} rows'}")
        for fam in cfg["families"]:
            for lvl in cfg["levels"]:
                for method in cfg["methods"]:
                    rel = f"{cfg['output_dir']}/{fam}_{lvl}/{method}.csv"
                    rows = _csv_rows(root / rel)
                    if rows is None:
                        out.append(f"{rel}: absent")
                    elif len(rows) != n * len(seeds) or {r["seed"] for r in rows} != seeds:
                        out.append(f"{rel}: {len(rows)} rows, seeds {sorted({r['seed'] for r in rows})}")
    for rel in ("results/downstream_v2/classifier_resnet.pt", "results/downstream_v2/stats/metrics.csv",
                "results/downstream_v2/stats/stats.csv", "results/downstream_v2/stats/coverage.csv",
                "data/ocr/crops.npy", "data/ocr/manifest.json", "results/ocr/summary.csv",
                "results/ocr/per_crop.csv"):
        if not (root / rel).exists():
            out.append(f"{rel}: absent")
    manifest = root / "data" / "ocr" / "manifest.json"
    if manifest.exists():
        ocr = _ocr_constants(root)
        n_crops = len(json.loads(manifest.read_text(encoding="utf-8"))["crops"])
        names = [*ocr.METHODS, *ocr.EXTERNAL, "corrupted_input"]
        for cell in ocr.CELLS:
            for seed in ocr.SEEDS:
                rel = f"results/ocr/cases/{cell}/seed{seed}/texts.json"
                p = root / rel
                texts = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                lacking = [k for k in names if len(texts.get(k, [])) != n_crops]
                if lacking:
                    out.append(f"{rel}: lacks {', '.join(lacking)}")
    return out


# ── running the stage within the session ────────────────────────────


def run_with_budget(cmd, *, seconds: float, cwd=None, env=None, stdout=None) -> tuple[int | None, bool]:
    """Run cmd; at `seconds`, stop it and everything it started. Returns (exit
    code, None when stopped; whether the budget stopped it). Stopping is safe
    because every step of the stage writes its outputs atomically and resumes."""
    import psutil
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=stdout,
                            stderr=subprocess.STDOUT if stdout is not None else None,
                            stdin=subprocess.DEVNULL)
    try:
        return proc.wait(timeout=max(seconds, 0.0)), False
    except subprocess.TimeoutExpired:
        parent = psutil.Process(proc.pid)
        tree = [parent, *parent.children(recursive=True)]
        for p in tree:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(tree, timeout=60)
        proc.wait(timeout=60)
        return None, True


def status_of(rc: int | None, timed_out: bool, lacking: list[str]) -> str:
    if rc == 0 and not lacking:
        return "complete"
    if timed_out:
        return "incomplete (time budget)"
    if rc != 0:
        return f"failed: the stage exited {rc}; see stage.log and results/*/run_*.log"
    return f"failed: the stage finished, {len(lacking)} outputs incomplete: {'; '.join(lacking[:5])}"


# ── inputs, environments, outputs ───────────────────────────────────


def find_marked(base: Path, rel: str) -> Path | None:
    """The directory under base that holds the file rel (Kaggle mounts a dataset
    at a nested path). None when there is none; an error when there are two."""
    rel_parts = Path(rel).parts
    hits = []
    for p in Path(base).rglob(rel_parts[-1]):
        top = p.parents[len(rel_parts) - 1]
        if (top / rel).is_file():
            hits.append(top)
    if len(hits) > 1:
        raise RuntimeError(f"{rel} found in more than one input: {hits}")
    return hits[0] if hits else None


def locate_source(base: Path, work: Path) -> str:
    """Put the repository in the work tree; how it came: a clone already there
    (the notebook route), or a dataset under base, extracted or not."""
    if (work / MARKER).is_file():
        return "clone"
    tree = find_marked(base, MARKER)
    if tree is not None:
        shutil.copytree(tree, work, dirs_exist_ok=True)
        return "dataset"
    tarball = find_marked(base, SRC_TARBALL)
    if tarball is None:
        raise RuntimeError(f"no repository in {work} or {base}: neither {MARKER} nor {SRC_TARBALL}")
    with tarfile.open(tarball / SRC_TARBALL) as t:
        t.extractall(work, filter="data")
    return "dataset archive"


def _sha256_of(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def fetch_govdocs_jpegs(work: Path, cache: Path = Path("/tmp/govdocs1"), download=None) -> int:
    """Write the manifest's GovDocs1 JPEGs that are absent or differ, each read
    from its Digital Corpora archive by its manifest path and written only if it
    matches its manifest SHA-256 (the download is untrusted data). Returns how
    many were written. download(url, dest) defaults to fetch_corpora's."""
    fetch = _script(work / "scripts" / "fetch_corpora.py", "_fetch_corpora")
    download = download or fetch._download
    manifest = json.loads((work / "data/jpeg_corruption/manifest.json").read_text(encoding="utf-8"))["files"]
    src = work / "data/jpeg_corruption/src"
    todo: dict[str, list[dict]] = {}
    for rec in manifest:
        if rec["corpus"] == "govdocs1" and _sha256_of(src / rec["file"]) != rec["sha256"]:
            todo.setdefault(rec["source"].split(":", 1)[0], []).append(rec)
    import zipfile
    for archive, recs in sorted(todo.items()):
        cache.mkdir(parents=True, exist_ok=True)
        download(fetch.GOVDOCS_URL.format(int(archive.split(".")[0])), cache / archive)
        with zipfile.ZipFile(cache / archive) as z:
            for rec in recs:
                data = z.read(rec["source"].split(":", 1)[1])
                if hashlib.sha256(data).hexdigest() != rec["sha256"]:
                    raise ValueError(f"{rec['file']} from {rec['source']} does not match its manifest SHA-256")
                src.mkdir(parents=True, exist_ok=True)
                (src / rec["file"]).write_bytes(data)
    return sum(len(r) for r in todo.values())


def restore_state(base: Path, work: Path) -> int:
    """Copy the previous session's state into the work tree; this session's number."""
    top = find_marked(base, f"{STATE_DIR}/{STATE_FILE}")
    if top is None:
        return 1
    state = top / STATE_DIR
    for rel in CARRIED:
        if (state / rel).exists():
            shutil.copytree(state / rel, work / rel, dirs_exist_ok=True)
    return int(json.loads((state / STATE_FILE).read_text(encoding="utf-8"))["session"]) + 1


def _probe(py: Path, modules: list[str]) -> dict:
    return _last_json(capture([py, "-c", PROBE, *modules]))


def _probe_failures(name: str, probe: dict) -> list[str]:
    bad = [f"{name}: {k} {v}" for k, v in probe.items() if isinstance(v, str) and v.startswith("MISSING")]
    if not probe.get("cuda_ok"):
        bad.append(f"{name}: no working CUDA ({probe.get('gpu')})")
    if probe.get("cv2_xphoto") is False:
        bad.append(f"{name}: cv2 without xphoto (shift-map, FSR)")
    return bad


def _venv(path: Path, system_site: bool) -> Path:
    sh([sys.executable, "-m", "venv", path, *(["--system-site-packages"] if system_site else [])])
    py = path / "bin" / "python"
    sh([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    return py


def _main_venv(work: Path) -> tuple[Path, dict]:
    """.venv-main (requirements.txt exactly, isolated) with a torch that runs CUDA; (python, probe)."""
    main = _venv(work / ".venv-main", system_site=False)
    sh([main, "-m", "pip", "install", "-q", "-r", work / "requirements.txt"])
    probe = _probe(main, MAIN_MODULES)
    for index in TORCH_INDEXES:
        if probe.get("cuda_ok"):
            break
        log(f"torch from PyPI has no working CUDA here ({probe.get('gpu')}); trying {index}")
        torch_pin = next(ln.strip() for ln in (work / "requirements.txt").read_text().splitlines()
                         if ln.startswith("torch=="))
        sh([main, "-m", "pip", "install", "-q", "--force-reinstall", torch_pin, "--index-url", index])
        probe = _probe(main, MAIN_MODULES)
    return main, probe


def make_main_environment(work: Path) -> tuple[Path, dict]:
    """The main environment alone (kaggle/jpeg_gpu_kernel.py): (python, probe with pip freeze)."""
    main, probe = _main_venv(work)
    bad = _probe_failures("main", probe)
    if bad:
        raise RuntimeError("environment: " + "; ".join(bad))
    probe["freeze"] = capture([main, "-m", "pip", "freeze"], check=False).splitlines()
    return main, probe


def make_environments(work: Path) -> dict:
    main, probe = _main_venv(work)
    lama = _venv(work / ".venv-lama", system_site=True)
    sh([lama, "-m", "pip", "install", "-q", "--no-deps", "simple-lama-inpainting==0.1.2"])
    sh([lama, "-m", "pip", "install", "-q", "fire"])
    ocr = _venv(work / ".venv-ocr", system_site=True)
    sh([ocr, "-m", "pip", "install", "-q", "easyocr==1.7.2", "PyYAML"])
    probes = {"main": probe, "lama": _probe(lama, LAMA_MODULES), "ocr": _probe(ocr, OCR_MODULES)}
    bad = [b for name, p in probes.items() for b in _probe_failures(name, p)]
    if bad:
        raise RuntimeError("environment: " + "; ".join(bad))
    for name, py in (("main", main), ("lama", lama), ("ocr", ocr)):
        probes[name]["freeze"] = capture([py, "-m", "pip", "freeze"], check=False).splitlines()
    return probes


def check_inputs(work: Path) -> dict:
    """Stop unless the inputs are the workstation's: the SVHN subsets, the CNN
    classifier, and the GovDocs1 JPEGs the OCR crops come from."""
    got = _last_json(capture([work / ".venv-main" / "bin" / "python", "-c", INPUT_CHECK], cwd=work))
    wrong = {k: (got.get(k), v) for k, v in EXPECTED.items() if got.get(k) != v}
    manifest = json.loads((work / "data/jpeg_corruption/manifest.json").read_text(encoding="utf-8"))["files"]
    for rec in manifest:
        if rec["corpus"] == "govdocs1":
            p = work / "data/jpeg_corruption/src" / rec["file"]
            digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"
            if digest != rec["sha256"]:
                wrong[rec["file"]] = (digest, rec["sha256"])
    if wrong:
        raise RuntimeError(f"inputs differ from the workstation's (got, expected): {wrong}")
    return got


def keep_in_results(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """results.tar.gz holds what the workstation needs: no arrays (except the OCR
    crop set), no training checkpoint, no partial file."""
    name = Path(member.name)
    if name.name.startswith(".partial_") or name.name == "classifier_resnet_training.pt":
        return None
    if name.suffix == ".npy" and name.as_posix() != "data/ocr/crops.npy":
        return None
    return member


def save_outputs(work: Path, out: Path, *, status: str, record: dict, lacking: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    present = [rel for rel in CARRIED if (work / rel).exists()]
    with tarfile.open(out / "results.tar.gz", "w:gz") as t:
        for rel in present:
            t.add(work / rel, arcname=rel, filter=keep_in_results)
    if status != "complete" and present:
        for rel in present:
            shutil.copytree(work / rel, out / STATE_DIR / rel, dirs_exist_ok=True)
        (out / STATE_DIR / STATE_FILE).write_text(json.dumps(
            {"session": record.get("session", 1), "status": status, "lacking": lacking[:200]}, indent=2))
    record.update(status=status, finished=time.strftime("%F %T"), n_lacking=len(lacking))
    (out / "kaggle_env.json").write_text(json.dumps(record, indent=2, default=str))
    (out / "STATUS").write_text(status + "\n")
    log(f"STATUS {status}")


def kernel() -> int:
    t0 = time.time()
    KAGGLE_OUT.mkdir(parents=True, exist_ok=True)
    record: dict = {"started": time.strftime("%F %T"), "cpu_count": os.cpu_count()}
    try:
        record["nvidia_smi"] = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader"], check=False).strip()
    except OSError as exc:
        record["nvidia_smi"] = f"unavailable: {exc}"
    status, lacking = "failed: before the stage", []
    try:
        record["source"] = locate_source(KAGGLE_INPUT, WORK)
        record["commit"] = capture(["git", "-C", WORK, "rev-parse", "HEAD"], check=False).strip() or "unknown"
        record["govdocs_jpegs_fetched"] = fetch_govdocs_jpegs(WORK)
        record["session"] = restore_state(KAGGLE_INPUT, WORK)
        log(f"session {record['session']}")
        record["environments"] = make_environments(WORK)
        record["inputs"] = check_inputs(WORK)
        record["setup_seconds"] = time.time() - t0
        env = {**os.environ, "PATH": f"{WORK / '.venv-main' / 'bin'}{os.pathsep}{os.environ['PATH']}",
               "DOWNSTREAM_DEVICE": "cuda", "OCR_GPU": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
               "TORCH_HOME": "/tmp/torch", "HF_HOME": "/tmp/hf"}
        env.pop("VIRTUAL_ENV", None)
        budget = SESSION_SECONDS - RESERVE_SECONDS - (time.time() - t0)
        log(f"stage with a budget of {budget / 3600:.2f} h")
        with (KAGGLE_OUT / "stage.log").open("a", encoding="utf-8") as f:
            rc, timed_out = run_with_budget(STAGE, seconds=budget, cwd=WORK, env=env, stdout=f)
        record["stage_seconds"], record["stage_rc"] = time.time() - t0 - record["setup_seconds"], rc
        lacking = missing(WORK)
        status = status_of(rc, timed_out, lacking)
    except Exception as exc:  # the output must be saved whatever failed
        status = f"failed: {type(exc).__name__}: {exc}"
        log(traceback.format_exc())
        try:
            lacking = missing(WORK)
        except Exception:
            lacking = []
    finally:
        save_outputs(WORK, KAGGLE_OUT, status=status, record=record, lacking=lacking)
    return 0


# ── local side ───────────────────────────────────────────────────────


def stage_dataset(repo: Path, out: Path, user: str) -> Path:
    """The private dataset: the committed tree (git archive HEAD: nothing
    uncommitted, nothing gitignored) plus the GovDocs1 JPEGs of the
    JPEG-corruption manifest, the OCR crops' source, each checked against its
    SHA-256. Writes <out>/paper2_recon_src.tar.gz and dataset-metadata.json."""
    repo, out = Path(repo), Path(out)
    dirty = capture(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"]).strip()
    if dirty:
        raise RuntimeError(f"uncommitted changes would not reach Kaggle; commit first:\n{dirty}")
    raw = capture(["git", "-C", repo, "archive", "--format=tar", "HEAD"], text=False)
    manifest = json.loads((repo / "data/jpeg_corruption/manifest.json").read_text(encoding="utf-8"))["files"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / SRC_TARBALL
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, tarfile.open(path, "w:gz") as dst:
        for m in src.getmembers():
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        for rec in manifest:
            if rec["corpus"] != "govdocs1":
                continue
            data = (repo / "data/jpeg_corruption/src" / rec["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != rec["sha256"]:
                raise ValueError(f"{rec['file']} does not match its manifest SHA-256")
            info = tarfile.TarInfo(f"data/jpeg_corruption/src/{rec['file']}")
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            dst.addfile(info, io.BytesIO(data))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "csfr round2 code", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}, indent=2))
    return path


def kernel_dir(out: Path, user: str, session: int) -> Path:
    """A kernel push directory for session n: this script and its metadata; from
    session 2 on, the previous session's output is an input."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / Path(__file__).name)
    slug = KERNEL_SLUG.format(session)
    meta = {"id": f"{user}/{slug}", "title": slug.replace("-", " "), "code_file": Path(__file__).name,
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [f"{user}/{DATASET_SLUG}"],
            "kernel_sources": [f"{user}/{KERNEL_SLUG.format(session - 1)}"] if session > 1 else [],
            "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main(argv=None) -> int:
    if argv is None and len(sys.argv) == 1:
        if not KAGGLE_INPUT.is_dir():
            print("no /kaggle/input: this is the Kaggle kernel; locally use stage, kernel-dir, check or expected",
                  file=sys.stderr)
            return 2
        return kernel()
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("out")
    s.add_argument("--user", required=True)
    k = sub.add_parser("kernel-dir")
    k.add_argument("out")
    k.add_argument("--user", required=True)
    k.add_argument("--session", type=int, default=1)
    c = sub.add_parser("check")
    c.add_argument("root", nargs="?", default=str(repo))
    sub.add_parser("expected")
    args = ap.parse_args(argv)
    if args.cmd == "stage":
        print(stage_dataset(repo, Path(args.out), args.user))
    elif args.cmd == "kernel-dir":
        print(kernel_dir(Path(args.out), args.user, args.session))
    elif args.cmd == "check":
        lacking = missing(Path(args.root))
        print("\n".join(lacking) if lacking else "complete")
        return 1 if lacking else 0
    else:
        print(capture([sys.executable, "-c", INPUT_CHECK], cwd=repo).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
