"""Generative inpainting: LaMa (preferred) or diffusers StableDiffusion.

Two external-prior backends behind one interface. LaMa is preferred: it is
purpose-built for inpainting rather than repurposed text-to-image, needs no
prompt, and is roughly two orders of magnitude faster, which is the difference
between a full grid that fits in one GPU session and one that does not. Stable
Diffusion remains as the fallback. Either way the model is external-prior and
carries no provenance, so its traceability score is 0 by construction.

The interface handles:
  - Input: (N, H, W) grayscale float [0, 255]
  - Mask: (N, H, W) float where 1=observed, 0=erased
  - Output: (N, H, W) float [0, 255]

Observed pixels are preserved exactly. The model output is composited onto the
observed data, so the only thing this method contributes is what it invents in
the erased region. Every other baseline in the harness leaves observed pixels
intact and CSFR enforces that as the hard C1 constraint, so without compositing
the comparison would measure resampling loss rather than hallucination.

Methodological caveats (to be noted in the paper):
  - The native input is 32x32 or 64x64 grayscale; the model expects
    512x512 RGB. We resize, replicate channels, run inference, then
    resize back. This is a real limitation that applies to all external
    generative priors.
  - The model is non-deterministic without seeding; we seed the generator.
  - CPU inference is slow (~5-15 sec per patch depending on step count).
    On an L4 GPU in float16 it is roughly a second per patch at 20 steps.
  - The model was trained on RGB natural images, not digits; the domain
    mismatch may affect performance on SVHN downstream.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# pipelines are cached by (model_id, device, dtype). loading stable diffusion
# costs tens of seconds and several gb, and the harness calls this once per
# cell and seed, so reloading each time would dominate the run.
_PIPELINE_CACHE: dict = {}


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
# same weights under the community mirror, verified to resolve to the identical
# revision (8a4288a76071) as the primary. use it if the primary is withdrawn.
FALLBACK_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def resolve_dtype(device: str) -> torch.dtype:
    """float16 on cuda, float32 elsewhere. float16 on cpu is slower, not faster."""
    return torch.float16 if str(device).startswith("cuda") else torch.float32


# ── LaMa backend ──────────────────────────────────────────────────────
# None means run at the patch's native resolution, which is what LaMa should do.
# it is fully convolutional and only needs sides divisible by 8, which 32 and 64
# both are. an earlier version upscaled to 256 on the assumption that a model
# trained on natural-image crops would degrade on tiny inputs. measured on a T4
# over 8 SVHN patches, that assumption was wrong and expensive:
#
#   cell        zero   bilin   telea   lama@32   lama@256
#   CF1 0.30   11.74   32.79   34.55     36.57      14.50
#   CF2 0.30   10.96   25.09   24.36     26.39      14.16
#   CF3 0.70    7.90    7.90   22.81     22.03      13.14
#
# the round trip smears zeros from erased pixels into observed ones on the way
# up and discards detail on the way down, costing 8 to 22 db. at native size
# LaMa is competitive with the strongest classical baseline, which is what the
# paper needs: the external prior must be strong for the calibration argument
# against it to carry any weight.
LAMA_WORK_SIZE = None

_LAMA_CACHE: dict = {}


def lama_available() -> bool:
    """Whether the LaMa backend can be imported."""
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
        return False


def _load_lama(device: str):
    """Load and cache the LaMa model. Weights download on first use."""
    key = str(device)
    if key not in _LAMA_CACHE:
        from simple_lama_inpainting import SimpleLama

        try:
            model = SimpleLama(device=torch.device(device))
        except TypeError:
            # older releases take no device argument and pick cuda themselves
            model = SimpleLama()
        _LAMA_CACHE[key] = model
    return _LAMA_CACHE[key]


def inpaint_lama(
    y: np.ndarray,
    M: np.ndarray,
    device: str = "cpu",
    work_size: Optional[int] = LAMA_WORK_SIZE,
) -> np.ndarray:
    """Inpaint with LaMa. Observed pixels are preserved exactly.

    Parameters
    ----------
    y : (N, H, W) float [0, 255]
        Corrupted images, erased pixels already zeroed.
    M : (N, H, W) float {0, 1}
        1 = observed, 0 = erased.
    device : str
        Torch device string.
    work_size : int or None
        Side length used for inference. None runs at the patch's native size,
        which is the default and the measured-best choice. See LAMA_WORK_SIZE.

    Returns
    -------
    (N, H, W) float [0, 255]

    LaMa is deterministic: it is a single forward pass with no sampling, so no
    seed is needed and repeated runs are bit-identical on the same device.
    """
    from PIL import Image

    model = _load_lama(device)
    src_h, src_w = y.shape[1:]
    resize = work_size is not None and (work_size, work_size) != (src_h, src_w)
    x_hat = np.zeros_like(y)

    for i in range(y.shape[0]):
        patch = np.clip(y[i], 0, 255).astype(np.uint8)
        # LaMa's convention is 255 where the image must be filled in
        hole = ((M[i] == 0).astype(np.uint8)) * 255

        img_rgb = Image.fromarray(patch, mode="L").convert("RGB")
        mask_img = Image.fromarray(hole, mode="L")
        if resize:
            img_rgb = img_rgb.resize((work_size, work_size), Image.BICUBIC)
            # nearest keeps the mask binary and hole boundaries aligned
            mask_img = mask_img.resize((work_size, work_size), Image.NEAREST)

        out = model(img_rgb, mask_img)
        if not isinstance(out, Image.Image):
            out = Image.fromarray(np.asarray(out).astype(np.uint8))

        out = out.convert("L")
        if out.size != (src_w, src_h):
            out = out.resize((src_w, src_h), Image.LANCZOS)
        pred = np.asarray(out, dtype=np.float32)

        # composite: observed pixels bit-exact, model only inside the hole
        x_hat[i] = y[i] * M[i] + pred * (1.0 - M[i])

    return x_hat


_SELECTED_BACKEND: Optional[str] = None


def inpaint(
    y: np.ndarray,
    M: np.ndarray,
    *,
    backend: str = "auto",
    seed: int = 0,
    device: str = "cpu",
    num_inference_steps: int = 20,
) -> np.ndarray:
    """Run the external-prior inpainter. ``auto`` prefers LaMa, falls back to SD.

    The harness calls this. Whichever backend runs, record which one in the
    environment capture: the paper must name the external prior it compared
    against, and the two are different models with different failure modes.
    """
    global _SELECTED_BACKEND
    if backend == "auto":
        backend = "lama" if lama_available() else "diffusers"
    _SELECTED_BACKEND = backend
    if backend == "lama":
        return inpaint_lama(y, M, device=device)
    if backend == "diffusers":
        return inpaint_diffusers(
            y, M, seed=seed, num_inference_steps=num_inference_steps, device=device
        )
    raise ValueError(f"unknown backend: {backend}")


def get_model_info(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu") -> dict:
    """Metadata about the inpainting model, for the environment capture.

    The paper has to name the external prior it ran against, so this records the
    resolved model identity and the device the run actually used. When called after
    inpaint(), reflects the backend that was actually selected.
    """
    try:
        import diffusers  # noqa: F401
    except ImportError:
        return {"status": "unavailable", "reason": "diffusers not installed"}

    # use the backend that was actually selected if inpaint has been called
    selected = _SELECTED_BACKEND if _SELECTED_BACKEND else ("lama" if lama_available() else "diffusers")

    if selected == "lama":
        return {
            "backend": "lama",
            "model_id": "big-lama (simple-lama-inpainting)",
            "library": "simple_lama_inpainting",
            "torch_version": torch.__version__,
            "deterministic": "yes (single forward pass, no sampling)",
            "input_resolution": "native (no resize)" if LAMA_WORK_SIZE is None else f"native, resized to {LAMA_WORK_SIZE}x{LAMA_WORK_SIZE}",
            "input_type": "RGB (replicated from grayscale)",
            "observed_pixels": "preserved exactly (output composited onto observed data)",
            "device": str(device),
        }

    info = {
        "backend": "diffusers",
        "model_id": model_id,
        "library": "diffusers",
        "torch_version": torch.__version__,
        "dtype": str(resolve_dtype(device)),
        "deterministic": "yes (with generator seed)",
        "input_resolution": "flexible (resized to 512x512 internally)",
        "input_type": "RGB (replicated from grayscale)",
        "observed_pixels": "preserved exactly (output composited onto observed data)",
        "device": str(device),
    }
    try:
        from huggingface_hub import model_info as hub_info
        info["revision"] = hub_info(model_id).sha
    except Exception as exc:  # network or hub failure is not fatal for metadata
        info["revision"] = f"unresolved: {type(exc).__name__}"
    return info


def composite_observed(y_i: np.ndarray, M_i: np.ndarray,
                       pred: np.ndarray) -> np.ndarray:
    """Keep observed pixels bit-exact, take the model only in the erased region.

    Separated out so the contract is testable without loading a diffusion model.
    """
    return y_i * M_i + pred * (1.0 - M_i)


def _load_pipeline(model_id: str, device: str, dtype: torch.dtype):
    """Load and cache one inpainting pipeline."""
    key = (model_id, str(device), str(dtype))
    if key not in _PIPELINE_CACHE:
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        _PIPELINE_CACHE[key] = pipe
    return _PIPELINE_CACHE[key]


def inpaint_diffusers(
    y: np.ndarray,
    M: np.ndarray,
    seed: int = 0,
    model_id: str = "runwayml/stable-diffusion-inpainting",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    device: str = "cpu",
) -> np.ndarray:
    """Run Stable Diffusion inpainting on a batch.

    Parameters
    ----------
    y : (N, H, W) float [0, 255]
        Corrupted (masked) images.
    M : (N, H, W) float {0, 1}
        Mask where 1=observed, 0=erased.
    seed : int
        RNG seed for deterministic output (default 0).
    model_id : str
        Hugging Face model identifier (default runwayml/stable-diffusion-inpainting).
    num_inference_steps : int
        Diffusion steps; higher = slower but potentially better. Default 20 (fast).
    guidance_scale : float
        Classifier-free guidance scale (default 7.5).
    device : str
        Device ("cpu", "cuda", etc.). Default "cpu".

    Returns
    -------
    x_hat : (N, H, W) float [0, 255]
        Inpainted images, with observed pixels preserved exactly.
    """
    from PIL import Image

    batch_size = y.shape[0]
    src_h, src_w = y.shape[1:]
    model_size = 512  # Stable Diffusion input size

    pipe = _load_pipeline(model_id, device, resolve_dtype(device))

    x_hat = np.zeros_like(y)
    generator = torch.Generator(device=device)

    for i in range(batch_size):
        generator.manual_seed(seed + i)  # deterministic per-sample variation

        # Convert grayscale to RGB (replicate channels)
        y_uint8 = np.clip(y[i], 0, 255).astype(np.uint8)  # (H, W)
        y_rgb = Image.fromarray(
            np.stack([y_uint8, y_uint8, y_uint8], axis=-1), mode="RGB"
        )
        y_rgb_resized = y_rgb.resize((model_size, model_size), Image.LANCZOS)

        # Invert mask: diffusers expects 1=erased, 0=observed (opposite convention)
        m_inv = (1 - M[i]) * 255
        m_inv_uint8 = np.clip(m_inv, 0, 255).astype(np.uint8)
        m_pil = Image.fromarray(m_inv_uint8, mode="L")
        m_pil_resized = m_pil.resize((model_size, model_size), Image.NEAREST)

        # Inpaint with generic prompt (not digit-specific; the model is domain-agnostic)
        with torch.no_grad():
            output = pipe(
                prompt="a detailed image",  # generic, non-specific prompt
                image=y_rgb_resized,
                mask_image=m_pil_resized,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=model_size,
                width=model_size,
            )

        # Convert output back to grayscale
        result_pil = output.images[0]
        result_rgb = np.array(result_pil, dtype=np.float32)  # (512, 512, 3)
        result_gray = result_rgb.mean(axis=2)  # average RGB to gray

        # Resize back to original size
        result_pil_gray = Image.fromarray(
            np.clip(result_gray, 0, 255).astype(np.uint8), mode="L"
        )
        result_resized = result_pil_gray.resize((src_w, src_h), Image.LANCZOS)
        pred = np.array(result_resized, dtype=np.float32)

        # without this the 32-512-32 round trip would also resample the observed
        # data, and the resulting psnr loss would have nothing to do with what
        # the prior invented.
        x_hat[i] = composite_observed(y[i], M[i], pred)

    return x_hat


def compute_weights_checksum(weights_path: Path) -> str:
    """Compute SHA256 of model weights for auditability."""
    sha256 = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
