"""graphical abstract for ieee access: 660 x 295 px jpg under 45 kb.

left: a real dfrws patch under cf1 l3 corruption, the csfr reconstruction and
the approximate local sensitivity of one imputed pixel, all computed live from
the released code. right: what the method does and does not claim.

usage: python scripts/render_graphical_abstract.py [output.jpg]
default output: submission/ieee_access/04_graphical_abstract.jpg (gitignored)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from csfr_reconstruct_2d import reconstruct  # noqa: E402
from provenance import provenance_row  # noqa: E402
from run_csfr_sweep import make_mask  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission" / "ieee_access" / "04_graphical_abstract.jpg"
W, H = 660, 295
SOLVER = dict(lam_l1=0.02, lam_tv=0.10, lam_fc=0.005)
PATCH, LOSS, SEED = 7, 0.30, 0  # cf1 at 30 % loss (l3): the regime the paper supports


def compute():
    patches = np.load(ROOT / "data" / "patches.npy").astype(np.float32)
    x = patches[PATCH]
    rng = np.random.default_rng(SEED)
    mask = make_mask((1, 64, 64), "CF1", LOSS, rng)[0]
    y = x * mask
    xh, _ = reconstruct(torch.from_numpy(y[None]), torch.from_numpy(mask[None]),
                        max_iter=600, lr=2.0, device=torch.device("cpu"),
                        ckpt_path=None, ckpt_every=0, resume=False, **SOLVER)
    xh = xh[0]
    holes = np.argwhere(mask == 0)
    r, c = (int(v) for v in holes[len(holes) // 2])
    w, info = provenance_row(xh, torch.from_numpy(y), torch.from_numpy(mask), (r, c), **SOLVER)
    psnr = 10 * np.log10(255 ** 2 / np.mean((xh.numpy() - x) ** 2))
    print(f"patch psnr {psnr:.1f} dB, query ({r},{c}) support {info['support']} cg {info['cg_iters']}")
    return x, y, mask, xh.numpy(), w, (r, c), psnr


def tile(arr, size):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "L").resize((size, size), Image.NEAREST).convert("RGB")


def sens_tile(wt, mask, query, size):
    r, c = query
    lim = np.abs(wt).max() or 1.0
    v = wt / lim
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    # blue for negative, red for positive, white at zero
    rgb[..., 0] = 255 * (1 - np.clip(-v, 0, 1))
    rgb[..., 2] = 255 * (1 - np.clip(v, 0, 1))
    rgb[..., 1] = 255 * (1 - np.abs(v))
    rgb[mask == 0] = [235, 235, 235]
    img = Image.fromarray(rgb.astype(np.uint8), "RGB").resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    s = size / 64
    d.rectangle([c * s - 3, r * s - 3, c * s + s + 3, r * s + s + 3], outline=(200, 0, 0), width=2)
    return img


def font(sz, bold=False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(x, y, mask, xh, w, query, psnr) -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    size, gap = 122, 18
    x0, ytop = 12, 48
    corrupted = y.copy()
    corrupted[mask == 0] = 20  # dark grey marks erased pixels
    panels = [(tile(corrupted, size), f"Carved fragment\n{LOSS:.0%} pixels lost"),
              (tile(xh, size), f"CSFR output\n{psnr:.1f} dB, no learned prior"),
              (sens_tile(w * (mask == 1), mask, query, size), "Sensitivity record\nfor one imputed pixel")]
    for i, (im, label) in enumerate(panels):
        px = x0 + i * (size + gap)
        img.paste(im, (px, ytop))
        d.rectangle([px - 1, ytop - 1, px + size, ytop + size], outline=(60, 60, 60))
        d.multiline_text((px + size / 2, ytop + size + 7), label, fill=(30, 30, 30), font=font(11),
                         anchor="ma", align="center", spacing=2)
        if i < 2:
            ax = px + size + 3
            d.polygon([(ax, ytop + size / 2 - 6), (ax + 12, ytop + size / 2), (ax, ytop + size / 2 + 6)],
                      fill=(0, 51, 102))

    d.text((x0, 12), "Constraint-driven forensic image reconstruction with an audit record",
           fill=(0, 51, 102), font=font(15, bold=True))

    tx = x0 + 3 * (size + gap) + 6
    lines = [
        ("Deterministic, training-free", True),
        ("penalised objective in a DCT basis", False),
        ("Observed pixels kept exactly", True),
        ("terminal projection, global clipping", False),
        ("Leads under scattered loss", True),
        ("+1.0 to +1.4 dB, cluster-robust,", False),
        ("reproduced on GovDocs1", False),
        ("Trails on burst and large gaps", True),
        ("reported, not hidden", False),
        ("Audit record, not admissibility", True),
        ("approximate local sensitivity;", False),
        ("a court decides admissibility", False),
    ]
    yy = ytop
    limit = W - tx - 6
    for text, bold in lines:
        f = font(12 if bold else 11, bold)
        width = d.textlength(text, font=f)
        assert width <= limit, f"line too wide for the column: {text!r} ({width:.0f} > {limit})"
        d.text((tx, yy), text, fill=(20, 20, 20) if bold else (80, 80, 80), font=f)
        yy += 19 if bold else 15
    assert yy <= H - 10, f"text column overflows: {yy}"
    return img


def main() -> int:
    img = draw(*compute())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for q in (88, 80, 72, 65, 58, 50, 42):
        img.save(OUT, "JPEG", quality=q, optimize=True, subsampling=2)
        size = OUT.stat().st_size
        if size < 45 * 1024:
            break
    else:
        print(f"warning: {size / 1024:.1f} kB still above the 45 kB limit at quality {q}")
        return 1
    print(f"wrote {OUT} {img.size} {size / 1024:.1f} kB at quality {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
