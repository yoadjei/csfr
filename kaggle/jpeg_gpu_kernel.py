#!/usr/bin/env python3
"""The CSFR solves of the byte-level JPEG corruption experiment (E9) on Kaggle GPUs.

A Kaggle kernel (accelerator GPU T4 x2, internet on) that runs the workstation's
JPEG driver, scripts/run_jpeg_corruption.py, with --csfr-only on CUDA: both CSFR
variants, float64, for every case and condition of all 105 manifest files, one
shard of the files per GPU. The classical baselines and LaMa stay on the
workstation. The CSFR solve is bound by memory bandwidth, which a T4 has several
times more of than the workstation; every file is solved here, so that no table
row mixes devices, and the files the workstation solved on its CPU become a
measured CPU-versus-GPU comparison (scripts/merge_jpeg_gpu.py).

The files run in ORDER: first the ones the workstation had not reached when this
was written, then the rest, so that a session that ends early has solved the
files the workstation needs most. Every condition resumes from its metrics.json:
at the session's time budget the shards stop, state/ is saved, and the next
session, given this output as an input, continues.

Before the shards start: the main environment (requirements.txt exactly, CUDA
torch; see downstream_v2_kernel.py) runs a CUDA operation, and every source JPEG
matches its manifest SHA-256: the GovDocs1 ones are read from their Digital
Corpora archives, the five DFTT ones carved from the DFTT test images (public,
SourceForge) exactly as scripts/fetch_corpora.py carved them.

Outputs (/kaggle/working):
  STATUS           complete | incomplete (time budget) | failed: <reason>
  results.tar.gz   results/jpeg_corruption_gpu without arrays, except the CSFR
                   outputs of the figure's file (render_figures_v2.JPEG_EX_FILE)
  state/           what a later session needs (while not complete)
  kaggle_env.json  GPUs, CPU count, package versions, timings
  shard<k>.log     each shard's driver output

Locally:
  python kaggle/jpeg_gpu_kernel.py check [<repo>]   # what a results tree still lacks
"""
from __future__ import annotations

import argparse
import csv
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
import zipfile
from pathlib import Path


def _downstream_kernel():
    spec = importlib.util.spec_from_file_location("_downstream_v2_kernel",
                                                  Path(__file__).resolve().with_name("downstream_v2_kernel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dk = _downstream_kernel()
KAGGLE_INPUT, KAGGLE_OUT, WORK = _dk.KAGGLE_INPUT, _dk.KAGGLE_OUT, _dk.WORK
log, capture = _dk.log, _dk.capture

GPU_DIR = "results/jpeg_corruption_gpu"
CONFIG = "configs/jpeg_corruption.yaml"
MANIFEST = "data/jpeg_corruption/manifest.json"
SRC = "data/jpeg_corruption/src"
# manifest indices; the workstation was at file 49 (index 48) when this was written
ORDER = "48-104,0-47"
WORKERS_PER_GPU = 2         # two processes per GPU hide each other's launch latency
FIGURE_FILE = 2             # render_figures_v2.JPEG_EX_FILE: its CSFR outputs come back
STATE_DIR, STATE_FILE = "state", "STATE.json"
SESSION_SECONDS, RESERVE_SECONDS = _dk.SESSION_SECONDS, 30 * 60
DFTT_URL = "https://downloads.sourceforge.net/project/dftt/Test%20Images/{}"
DFTT_ARCHIVES = {"11-carve-fat": "11_%20Basic%20Data%20Carving%20%231/11-carve-fat.zip",
                 "12-carve-ext2": "12_%20Basic%20Data%20Carving%20%232/12-carve-ext2.zip"}


def shard_cmd(py, k: int, n: int, device: str) -> list:
    return [py, "-I", "scripts/run_jpeg_corruption.py", "--config", CONFIG, "--csfr-only",
            "--output-dir", f"{GPU_DIR}/shard{k}", "--files", ORDER, "--shard", f"{k}/{n}",
            "--device", device, "--workers", str(WORKERS_PER_GPU), "--solve-threads", "1", "--bootstrap", "200"]


# ── completeness ─────────────────────────────────────────────────────


def missing(root: Path) -> list[str]:
    """What the GPU run has not yet solved under root: every case of every
    manifest file must be in some shard's masks_quality.csv, and every condition
    with visible damage must hold both CSFR variants done. Empty when complete."""
    import yaml
    root = Path(root)
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    files = json.loads((root / MANIFEST).read_text(encoding="utf-8"))["files"]
    seen: dict[tuple[str, str], bool] = {}
    for mq in sorted((root / GPU_DIR).glob("shard*/masks_quality.csv")):
        with mq.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                seen[(r["case"], r["condition"])] = r["visible_damage"] == "True"
    out = []
    for rec in files:
        for kind in cfg["kinds"]:
            for rep in range(cfg["repeats"]):
                case = f"{Path(rec['file']).stem}_{kind}_r{rep}"
                for cond in cfg["conditions"]:
                    if (case, cond) not in seen:
                        out.append(f"{case}/{cond}: not run")
                        continue
                    if not seen[(case, cond)]:
                        continue
                    hits = list((root / GPU_DIR).glob(f"shard*/cases/{case}/{cond}/metrics.json"))
                    methods = json.loads(hits[0].read_text(encoding="utf-8"))["methods"] if hits else {}
                    lacking = [v for v in cfg["csfr_variants"] if methods.get(v, {}).get("status") != "done"]
                    if lacking:
                        out.append(f"{case}/{cond}: lacks {', '.join(lacking)}")
    return out


# ── inputs ───────────────────────────────────────────────────────────


def fetch_dftt_jpegs(work: Path, cache: Path = Path("/tmp/dftt"), download=None) -> int:
    """Write the manifest's DFTT JPEGs that are absent or differ: each DFTT test
    image is downloaded, its disk image carved with the carver that built the
    manifest (scripts/extract_patches.carve_jpegs), and the blob the manifest
    names (<image>.dd#carved<j>) written only if it matches its SHA-256 (the
    download is untrusted data). Returns how many were written."""
    fetch = _dk._script(work / "scripts" / "fetch_corpora.py", "_fetch_corpora")
    download = download or fetch._download
    sys.path.insert(0, str(work / "scripts"))
    from extract_patches import carve_jpegs
    files = json.loads((work / MANIFEST).read_text(encoding="utf-8"))["files"]
    src = work / SRC
    todo: dict[str, list[dict]] = {}
    for rec in files:
        if rec["corpus"] == "nick_mikus_dftt" and _dk._sha256_of(src / rec["file"]) != rec["sha256"]:
            todo.setdefault(rec["source"].split(".dd#", 1)[0], []).append(rec)
    for disk, recs in sorted(todo.items()):
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / f"{disk}.zip"
        download(DFTT_URL.format(DFTT_ARCHIVES[disk]), archive)
        with zipfile.ZipFile(archive) as z:
            dd = next(n for n in z.namelist() if n.endswith(".dd"))
            blobs = list(carve_jpegs(z.read(dd)))
        for rec in recs:
            j = int(rec["source"].rsplit("#carved", 1)[1])
            data = blobs[j] if j < len(blobs) else b""
            if hashlib.sha256(data).hexdigest() != rec["sha256"]:
                raise ValueError(f"{rec['file']} from {rec['source']} does not match its manifest SHA-256")
            src.mkdir(parents=True, exist_ok=True)
            (src / rec["file"]).write_bytes(data)
    return sum(len(r) for r in todo.values())


def check_sources(work: Path) -> None:
    """Stop unless every source JPEG is the workstation's (manifest SHA-256)."""
    files = json.loads((work / MANIFEST).read_text(encoding="utf-8"))["files"]
    wrong = [r["file"] for r in files if _dk._sha256_of(work / SRC / r["file"]) != r["sha256"]]
    if wrong:
        raise RuntimeError(f"source JPEGs differ from the manifest: {wrong}")


def restore_state(base: Path, work: Path) -> int:
    """Copy the previous session's state into the work tree; this session's number."""
    top = _dk.find_marked(base, f"{STATE_DIR}/{STATE_FILE}")
    if top is None:
        return 1
    if (top / STATE_DIR / GPU_DIR).exists():
        shutil.copytree(top / STATE_DIR / GPU_DIR, work / GPU_DIR, dirs_exist_ok=True)
    return int(json.loads((top / STATE_DIR / STATE_FILE).read_text(encoding="utf-8"))["session"]) + 1


# ── running ──────────────────────────────────────────────────────────


def run_all_with_budget(cmds: list[list], *, seconds: float, cwd=None, env=None,
                        logs: list[Path]) -> tuple[list[int | None], bool]:
    """Run the commands concurrently; at `seconds`, stop every one and everything
    it started. Returns (exit codes, None where stopped; whether the budget
    stopped them). Stopping is safe: every condition resumes from metrics.json."""
    import psutil
    procs = []
    for cmd, path in zip(cmds, logs):
        f = path.open("a", encoding="utf-8")
        procs.append((subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=f,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), f))
    deadline = time.time() + max(seconds, 0.0)
    try:
        while time.time() < deadline and any(p.poll() is None for p, _ in procs):
            time.sleep(min(10.0, max(deadline - time.time(), 0.0)))
        running = [p.poll() is None for p, _ in procs]
        for (p, _), stop in zip(procs, running):
            if stop:
                parent = psutil.Process(p.pid)
                tree = [parent, *parent.children(recursive=True)]
                for q in tree:
                    try:
                        q.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(tree, timeout=60)
            p.wait(timeout=60)
        return [None if stop else p.returncode for (p, _), stop in zip(procs, running)], any(running)
    finally:
        for _, f in procs:
            f.close()


def status_of(rcs: list[int | None], timed_out: bool, lacking: list[str]) -> str:
    if not lacking and all(rc == 0 for rc in rcs):
        return "complete"
    if timed_out:
        return "incomplete (time budget)"
    bad = [f"shard{k} exited {rc}" for k, rc in enumerate(rcs) if rc != 0]
    if bad:
        return f"failed: {', '.join(bad)}; see shard<k>.log"
    return f"failed: the shards finished, {len(lacking)} conditions incomplete: {'; '.join(lacking[:5])}"


# ── outputs ──────────────────────────────────────────────────────────


def figure_stem(work: Path) -> str:
    return Path(json.loads((work / MANIFEST).read_text(encoding="utf-8"))["files"][FIGURE_FILE]["file"]).stem


def result_filter(stem: str):
    """results.tar.gz: no arrays except the CSFR outputs of the figure's file, no partial file."""
    def keep(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(member.name)
        if name.name.startswith(".partial_"):
            return None
        if name.suffix == ".npy":
            in_fig = any(part.startswith(f"{stem}_") for part in name.parts)
            return member if in_fig and name.name.startswith("x_hat_csfr") else None
        return member
    return keep


def state_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if Path(member.name).suffix == ".npy" or Path(member.name).name.startswith(".partial_") else member


def save_outputs(work: Path, out: Path, *, status: str, record: dict, lacking: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gpu = work / GPU_DIR
    if gpu.exists():
        with tarfile.open(out / "results.tar.gz", "w:gz") as t:
            t.add(gpu, arcname=GPU_DIR, filter=result_filter(figure_stem(work)))
        if status != "complete":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                t.add(gpu, arcname=GPU_DIR, filter=state_filter)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as t:
                t.extractall(out / STATE_DIR, filter="data")
            (out / STATE_DIR / STATE_FILE).write_text(json.dumps(
                {"session": record.get("session", 1), "status": status, "lacking": lacking[:200]}, indent=2))
    record.update(status=status, finished=time.strftime("%F %T"), n_lacking=len(lacking))
    (out / "kaggle_env.json").write_text(json.dumps(record, indent=2, default=str))
    (out / "STATUS").write_text(status + "\n")
    log(f"STATUS {status}")


def kernel() -> int:
    t0 = time.time()
    KAGGLE_OUT.mkdir(parents=True, exist_ok=True)
    record: dict = {"started": time.strftime("%F %T"), "cpu_count": os.cpu_count(), "order": ORDER}
    try:
        record["nvidia_smi"] = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader"], check=False).strip()
    except OSError as exc:
        record["nvidia_smi"] = f"unavailable: {exc}"
    status, lacking = "failed: before the shards", []
    try:
        record["source"] = _dk.locate_source(KAGGLE_INPUT, WORK)
        record["commit"] = capture(["git", "-C", WORK, "rev-parse", "HEAD"], check=False).strip() or "unknown"
        record["govdocs_jpegs_fetched"] = _dk.fetch_govdocs_jpegs(WORK)
        record["dftt_jpegs_fetched"] = fetch_dftt_jpegs(WORK)
        check_sources(WORK)
        record["session"] = restore_state(KAGGLE_INPUT, WORK)
        log(f"session {record['session']}")
        py, record["environment"] = _dk.make_main_environment(WORK)
        n_gpu = int(capture([py, "-c", "import torch; print(torch.cuda.device_count())"]).strip())
        if n_gpu < 1:
            raise RuntimeError("no CUDA device")
        record["n_gpu"] = n_gpu
        record["setup_seconds"] = time.time() - t0
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TORCH_HOME": "/tmp/torch"}
        env.pop("VIRTUAL_ENV", None)
        budget = SESSION_SECONDS - RESERVE_SECONDS - (time.time() - t0)
        log(f"{n_gpu} shards with a budget of {budget / 3600:.2f} h")
        rcs, timed_out = run_all_with_budget([shard_cmd(py, k, n_gpu, f"cuda:{k}") for k in range(n_gpu)],
                                             seconds=budget, cwd=WORK, env=env,
                                             logs=[KAGGLE_OUT / f"shard{k}.log" for k in range(n_gpu)])
        record["shards_seconds"], record["shard_rcs"] = time.time() - t0 - record["setup_seconds"], rcs
        lacking = missing(WORK)
        status = status_of(rcs, timed_out, lacking)
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


def main(argv=None) -> int:
    if argv is None and len(sys.argv) == 1:
        if not KAGGLE_INPUT.is_dir():
            print("no /kaggle/input: this is the Kaggle kernel; locally use check", file=sys.stderr)
            return 2
        return kernel()
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("root", nargs="?", default=str(repo))
    args = ap.parse_args(argv)
    lacking = missing(Path(args.root))
    print("\n".join(lacking[:50]) + (f"\n... {len(lacking)} in all" if len(lacking) > 50 else "")
          if lacking else "complete")
    return 1 if lacking else 0


if __name__ == "__main__":
    sys.exit(main())
