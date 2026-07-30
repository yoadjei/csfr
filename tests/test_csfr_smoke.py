"""Smoke test: 16×16 synthetic patch, 30% missing, CSFR should recover well."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from csfr_reconstruct_2d import reconstruct, pick_device, psnr


def test_csfr_smoke():
    np.random.seed(42)
    ref = np.random.rand(1, 16, 16).astype(np.float32) * 200 + 20
    M = np.ones_like(ref)
    idx = np.random.choice(16 * 16, int(0.3 * 16 * 16), replace=False)
    M.reshape(1, -1)[0, idx] = 0.0
    y = ref * M

    dev = pick_device("cpu")
    x_hat, info = reconstruct(
        torch.from_numpy(y), torch.from_numpy(M),
        lam_l1=0.02, lam_tv=0.10, lam_fc=0.005,
        max_iter=100, lr=2.0, device=dev,
        ckpt_path=None, ckpt_every=999, resume=False,
    )
    p = psnr(torch.from_numpy(ref[0]), x_hat[0])
    print(f"PSNR = {p:.2f} dB")
    assert p > 15.0, f"PSNR too low: {p:.2f}"
    # C3: intensity bounds (clamp enforced)
    x_np = x_hat.cpu().numpy()
    assert x_np.min() >= -0.01, f"Below lower bound: {x_np.min()}"
    assert x_np.max() <= 255.01, f"Above upper bound: {x_np.max()}"
    print("smoke test passed")


if __name__ == "__main__":
    test_csfr_smoke()
