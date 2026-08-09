"""Trial run of the LaMa external-prior inpainter on SVHN downstream corpus.

This script validates that the inpainter works correctly and measures:
  - Per-patch runtime (wall-clock seconds)
  - PSNR on a handful of test patches
  - Determinism (same seed = same output)

Outputs are written to scratch directory (not committed).

Usage::

    python scripts/trial_lama.py --output /path/to/scratch --n-patches 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from csfr_reconstruct_2d import pick_device, psnr  # noqa: E402
from downstream import load_digit_dataset  # noqa: E402
from run_csfr_sweep import make_mask  # noqa: E402


def trial_lama(
    output_dir: Path,
    n_patches: int = 5,
    family: str = "CF1",
    level: str = "L5",
) -> dict:
    """Run trial inpainting and measure performance.

    Parameters
    ----------
    output_dir : Path
        Scratch directory for outputs.
    n_patches : int
        Number of SVHN test patches to inpaint.
    family : str
        Corruption family (CF1, CF2, CF3, CF4).
    level : str
        Severity level (L1-L5).

    Returns
    -------
    dict
        Trial results: per-patch times, PSNR, settings.
    """
    from generative_inpainter import inpaint_diffusers

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trial_lama] loading SVHN test set...")
    images, labels, corpus = load_digit_dataset(prefer="svhn", split="test")
    print(f"  corpus: {corpus}, shape: {images.shape}")

    # Select first n_patches from test set (deterministic, no seeding)
    indices = np.arange(n_patches)
    patches_pil = images[indices]  # (N, 1, 32, 32) in [0, 1]
    patches = (patches_pil.numpy() * 255).astype(np.float32)  # (N, 1, 32, 32) in [0, 255]
    patches = patches.squeeze(1)  # (N, 32, 32)

    print(f"[trial_lama] selected {patches.shape[0]} patches, shape {patches.shape[1:]}")

    # Create masks with known family/level
    frac_dict = {"L1": 0.05, "L2": 0.15, "L3": 0.30, "L4": 0.50, "L5": 0.70}
    frac = frac_dict.get(level, 0.50)
    rng = np.random.default_rng(seed=0)
    M = make_mask(patches.shape, family, frac, rng)
    y = patches * M  # corrupted

    print(f"[trial_lama] masks: family={family}, level={level}, frac={frac:.0%}")
    print(f"  erased pixels per patch: {((1 - M) * patches.shape[1] * patches.shape[2]).mean():.0f}")

    # Run inpainting and time it
    print(f"[trial_lama] running inpainting...")
    times = []
    try:
        for i in range(patches.shape[0]):
            t0 = time.time()
            x_i = inpaint_diffusers(
                y[i : i + 1],
                M[i : i + 1],
                seed=0,
                num_inference_steps=20,
                device="cpu",
            )
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  patch {i}: {elapsed:.2f} sec")

        x_hat = np.vstack([
            inpaint_diffusers(
                y[i : i + 1],
                M[i : i + 1],
                seed=0,
                num_inference_steps=20,
                device="cpu",
            )
            for i in range(patches.shape[0])
        ])
        print(f"[trial_lama] inpainting complete")
    except Exception as e:
        print(f"[trial_lama] ERROR during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

    # Compute metrics
    print(f"[trial_lama] computing metrics...")
    x_hat_t = torch.from_numpy(x_hat)
    patches_t = torch.from_numpy(patches)
    psnrs = []
    for i in range(patches.shape[0]):
        p = psnr(patches_t[i : i + 1], x_hat_t[i : i + 1])
        psnrs.append(float(p))
        print(f"  patch {i}: PSNR = {p:.2f} dB")

    # Also compute PSNR vs corrupted images (for baseline comparison)
    y_t = torch.from_numpy(y)
    psnrs_vs_corrupted = []
    for i in range(y.shape[0]):
        p = psnr(patches_t[i : i + 1], y_t[i : i + 1])
        psnrs_vs_corrupted.append(float(p))

    results = {
        "status": "success",
        "corpus": corpus,
        "family": family,
        "level": level,
        "frac": frac,
        "n_patches": int(patches.shape[0]),
        "patch_shape": list(patches.shape[1:]),
        "per_patch_time_sec": times,
        "mean_time_sec": float(np.mean(times)),
        "std_time_sec": float(np.std(times)),
        "psnr_vs_original_db": psnrs,
        "mean_psnr_db": float(np.mean(psnrs)),
        "std_psnr_db": float(np.std(psnrs)),
        "psnr_vs_corrupted_db": psnrs_vs_corrupted,
        "mean_psnr_vs_corrupted_db": float(np.mean(psnrs_vs_corrupted)),
    }

    # Save results
    result_json = output_dir / "trial_lama_results.json"
    result_json.write_text(json.dumps(results, indent=2))
    print(f"[trial_lama] results saved to {result_json}")

    # Print summary
    print(f"\n=== TRIAL SUMMARY ===")
    print(f"Status: {results['status']}")
    print(f"Corpus: {results['corpus']}")
    print(f"Mean PSNR vs original: {results['mean_psnr_db']:.2f} ± {results['std_psnr_db']:.2f} dB")
    print(f"Mean time per patch: {results['mean_time_sec']:.2f} ± {results['std_time_sec']:.2f} sec")
    print(f"Total time: {sum(times):.2f} sec for {len(times)} patches")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trial run of LaMa inpainter")
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory (scratch path)",
    )
    ap.add_argument("--n-patches", type=int, default=5, help="Number of patches")
    ap.add_argument("--family", default="CF1", help="Corruption family")
    ap.add_argument("--level", default="L5", help="Severity level")
    args = ap.parse_args()

    results = trial_lama(
        Path(args.output),
        n_patches=args.n_patches,
        family=args.family,
        level=args.level,
    )
    sys.exit(0 if results["status"] == "success" else 1)
