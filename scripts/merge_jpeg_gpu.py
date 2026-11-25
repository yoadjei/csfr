"""Merge the GPU CSFR solves of the JPEG corruption experiment (E9) into the workstation's results.

    python -I scripts/merge_jpeg_gpu.py [--gpu results/jpeg_corruption_gpu] [--dry-run]

kaggle/jpeg_gpu_kernel.py solves both CSFR variants of every case and condition
on GPUs; unpack its results.tar.gz at the repository root first. Run this with
the JPEG stage stopped: both write results/jpeg_corruption/cases/*/*/metrics.json.
It refuses unless the GPU run is complete, so that the CSFR rows of E9 are all
from one device. For every condition the GPU solved:

  - the inputs it solved (the SHA-256 of y and mask the driver records) must be
    the workstation's, where the workstation has written them;
  - the GPU record replaces the CSFR record in metrics.json, marked device cuda;
    a CPU record the workstation made is kept under "csfr_cpu" in the same file
    (no per_image row comes from it); a condition the workstation has not reached
    gets a metrics.json with the GPU records, and the driver adds the rest;
  - for the figure's file (render_figures_v2.JPEG_EX_FILE) the GPU outputs
    x_hat_<variant>.npy replace the workstation's, kept as x_hat_<variant>_cpu.npy.

Then results/jpeg_corruption/csfr_cpu_vs_gpu.csv lists, per condition and
variant solved on both, the two records side by side, and the largest
differences are printed. Running it again changes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from checkpoint import write_csv_atomic, write_json_atomic  # noqa: E402

LOCAL = "results/jpeg_corruption"
COMPARED = ["iterations", "converged", "gap", "psnr_hole", "ssim_hole", "gmsd_hole", "psnr_whole", "exact"]


def _gpu_kernel():
    """kaggle/jpeg_gpu_kernel.py of this repository (its completeness rules and record choice)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_jpeg_gpu_kernel", ROOT / "kaggle" / "jpeg_gpu_kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs_sha256(cond_dir: Path) -> dict | None:
    """The SHA-256 of a condition's y and mask as the driver records them, from its files; None when absent."""
    if not ((cond_dir / "y.npy").exists() and (cond_dir / "mask.npy").exists()):
        return None
    return {k: hashlib.sha256(np.load(cond_dir / f"{k}.npy").tobytes()).hexdigest() for k in ("y", "mask")}


def merge(root: Path, gpu_dir: Path, variants: list[str], figure_stem: str, dry_run: bool = False) -> list[dict]:
    """Merge every GPU condition under gpu_dir into root/LOCAL; returns the CPU-versus-GPU rows.

    A condition whose mask keeps every pixel has nothing to solve: its records
    are the input on either device, so they are neither marked nor compared."""
    pairs = []
    for (case, cond), (gm, g) in sorted(_gpu_kernel().solved(gpu_dir, variants).items()):
        ld = root / LOCAL / "cases" / case / cond
        lm = ld / "metrics.json"
        local = json.loads(lm.read_text(encoding="utf-8")) if lm.exists() else {"methods": {}}
        for mine in (inputs_sha256(ld), local.get("inputs_sha256")):
            if mine is not None and mine != g["inputs_sha256"]:
                raise RuntimeError(f"{case}/{cond}: the GPU solved other inputs than the workstation's")
        local["nothing_to_fill"] = g["nothing_to_fill"]
        local["inputs_sha256"] = g["inputs_sha256"]
        solved_here = not g["nothing_to_fill"]
        for v in variants:
            if not solved_here:
                local["methods"].setdefault(v, g["methods"][v])
                continue
            old = local["methods"].get(v)
            if old is not None and old.get("device") != "cuda":
                local.setdefault("csfr_cpu", {})[v] = old
            local["methods"][v] = {**g["methods"][v], "device": "cuda"}
        for v, cpu in local.get("csfr_cpu", {}).items():
            pairs.append({"case": case, "condition": cond, "variant": v,
                          **{f"cpu_{k}": cpu.get(k) for k in COMPARED},
                          **{f"gpu_{k}": local["methods"][v].get(k) for k in COMPARED}})
        figure = case.startswith(f"{figure_stem}_") and solved_here
        absent = [v for v in variants if figure and not (gm.parent / f"x_hat_{v}.npy").exists()]
        if absent:
            raise RuntimeError(f"{case}/{cond}: the figure's GPU outputs {absent} are not in {gm.parent}")
        if dry_run:
            continue
        ld.mkdir(parents=True, exist_ok=True)
        if figure:
            for v in variants:
                if (ld / f"x_hat_{v}.npy").exists() and not (ld / f"x_hat_{v}_cpu.npy").exists():
                    shutil.copy2(ld / f"x_hat_{v}.npy", ld / f"x_hat_{v}_cpu.npy")
                shutil.copy2(gm.parent / f"x_hat_{v}.npy", ld / f"x_hat_{v}.npy")
        write_json_atomic(lm, local)
    return pairs


def largest_differences(pairs: list[dict]) -> dict:
    def gap(k):
        d = [abs(float(p[f"cpu_{k}"]) - float(p[f"gpu_{k}"])) for p in pairs
             if p[f"cpu_{k}"] is not None and p[f"gpu_{k}"] is not None]
        return max(d) if d else None
    return {"n": len(pairs), **{f"max_abs_diff_{k}": gap(k) for k in ("psnr_hole", "ssim_hole", "gmsd_hole",
                                                                      "psnr_whole", "iterations")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="results/jpeg_corruption_gpu")
    ap.add_argument("--dry-run", action="store_true", help="check and compare; write nothing")
    args = ap.parse_args(argv)
    lacking = _gpu_kernel().missing(ROOT, ROOT / args.gpu)
    if lacking:
        print(f"the GPU run is not complete ({len(lacking)} conditions), e.g. {lacking[:3]}; nothing merged")
        return 1
    cfg = yaml.safe_load((ROOT / "configs" / "jpeg_corruption.yaml").read_text(encoding="utf-8"))
    files = json.loads((ROOT / cfg["manifest"]).read_text(encoding="utf-8"))["files"]
    stem = Path(files[_gpu_kernel().FIGURE_FILE]["file"]).stem
    pairs = merge(ROOT, ROOT / args.gpu, cfg["csfr_variants"], stem, args.dry_run)
    if pairs and not args.dry_run:
        write_csv_atomic(ROOT / LOCAL / "csfr_cpu_vs_gpu.csv", pairs, list(pairs[0]))
    print(json.dumps(largest_differences(pairs), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
