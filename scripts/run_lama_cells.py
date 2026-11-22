"""Fill in the external-prior (LaMa) reconstructions for a results tree.

Runs inside the isolated environment (scripts/setup_lama_venv.sh):

    .venv-lama/Scripts/python scripts/run_lama_cells.py --root results/sweep_v2_govdocs1

On the CPU unless --device says otherwise (the Kaggle downstream run passes
cuda); lama_meta.json records the device.

For every directory under --root holding y.npy and mask.npy but no
x_hat_lama.npy, reconstructs with LaMa (big-lama via simple-lama-inpainting) at
the native resolution, composites the observed pixels back, and writes
x_hat_lama.npy (float32) and lama_meta.json. The main environment's harness then
scores the saved outputs, so LaMa enters the same paired analysis as every other
method. Grey patches are (N, H, W); colour images are (N, H, W, 3) or (H, W, 3)
and are passed to the model as RGB.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from checkpoint import save_npy_atomic, write_text_atomic  # noqa: E402


def _model(device):
    from simple_lama_inpainting import SimpleLama
    try:
        return SimpleLama(device=device)
    except TypeError:
        return SimpleLama()


def inpaint_any(model, y: np.ndarray, M: np.ndarray) -> np.ndarray:
    from PIL import Image
    single = y.ndim == 3 and y.shape[-1] == 3 and M.ndim == 2
    if single:
        y, M = y[None], M[None]
    colour = y.ndim == 4
    out = np.empty_like(y, dtype=np.float32)
    for i in range(y.shape[0]):
        patch = np.clip(y[i], 0, 255).astype(np.uint8)
        img = Image.fromarray(patch, mode="RGB" if colour else "L").convert("RGB")
        m2 = M[i] if M[i].ndim == 2 else M[i][..., 0]
        hole = Image.fromarray(((m2 == 0).astype(np.uint8)) * 255, mode="L")
        pred = model(img, hole)
        if not isinstance(pred, Image.Image):
            pred = Image.fromarray(np.asarray(pred).astype(np.uint8))
        pred = pred.convert("RGB" if colour else "L")
        h, w = m2.shape
        if pred.size != (w, h):
            pred = pred.resize((w, h), Image.LANCZOS)
        p = np.asarray(pred, dtype=np.float32)
        mm = m2[..., None] if colour else m2
        out[i] = y[i] * mm + p * (1.0 - mm)
    return out[0] if single else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="torch device, or auto (the GPU when there is one)")
    args = ap.parse_args(argv)

    import torch
    from csfr_reconstruct_2d import pick_device
    torch.set_num_threads(args.threads)
    requested = pick_device(args.device)
    model = _model(requested)
    # the device the model reports, where it reports one: an old SimpleLama picks its own
    device = str(getattr(model, "device", requested))
    dirs = sorted({p.parent for p in Path(args.root).rglob("y.npy") if (p.parent / "mask.npy").exists()})
    todo = [d for d in dirs if args.force or not (d / "x_hat_lama.npy").exists()]
    print(f"[lama] {len(todo)} of {len(dirs)} directories to run", flush=True)
    for d in todo:
        y = np.load(d / "y.npy")
        M = np.load(d / "mask.npy")
        t0 = time.time()
        x = inpaint_any(model, y, M)
        dt = time.time() - t0
        n = y.shape[0] if y.ndim in (3, 4) and not (y.ndim == 3 and y.shape[-1] == 3) else 1
        meta = {"backend": "big-lama (simple-lama-inpainting 0.1.2)", "resolution": "native",
                "input": "grey replicated to RGB" if y.ndim == 3 and y.shape[-1] != 3 else "RGB",
                "observed_pixels": "composited back exactly", "device": device,
                "torch": torch.__version__, "numpy": np.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "seconds": dt, "seconds_per_item": dt / max(n, 1), "n": int(n)}
        # x_hat_lama.npy marks the directory done, so it is written last and whole
        write_text_atomic(d / "lama_meta.json", json.dumps(meta, indent=2))
        save_npy_atomic(d / "x_hat_lama.npy", x.astype(np.float32))
        print(f"[lama] {d.relative_to(Path(args.root))}: {n} items, {dt:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
