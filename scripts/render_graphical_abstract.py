"""Graphical abstract for IEEE Access: a 660 x 295 px JPEG under 45 kB.

    python scripts/render_graphical_abstract.py [output.jpg]

Left: the GovDocs1 patch of the record walkthrough (render_figures_v2.
walkthrough_data: contiguous loss, CF3 at L5): the observed patch, CSFR's
certified reconstruction, and the sensitivity record of one imputed pixel,
computed here with the configuration the sweep recorded. Right: what the method
gives and how accurate it was. Every number on the right is read from
paper/numbers_v2.tex or from the code, so the abstract cannot drift from the
paper.

Default output: submission/ieee_access/round2/04_graphical_abstract.jpg.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "ieee_access" / "round2" / "04_graphical_abstract.jpg"
NUMBERS = ROOT / "paper" / "numbers_v2.tex"
W, H = 660, 295
MAX_BYTES = 45 * 1024
NAVY, INK, MUTED = (16, 52, 96), (20, 20, 20), (82, 81, 78)
HOLE_RGB = (201, 199, 193)      # the hole, darker than the record's zero so the two stay apart
QUERY_DEPTH = 3                 # the imputed pixel of the record: this many rows inside the hole, mid-row


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro_values(path: Path = NUMBERS) -> dict[str, str]:
    return dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{(.*)\}$", path.read_text(encoding="utf-8"), re.M))


def power_of_ten(tex: str) -> str:
    """10^{-6} -> "-6": the exponent of a power of ten written in TeX."""
    m = re.fullmatch(r"10\^\{(-?\d+)\}", tex)
    if not m:
        raise ValueError(f"not a power of ten: {tex}")
    return m.group(1).replace("-", "\u2212")


def text_lines(v: dict[str, str], srf_max_err: float) -> list[tuple[str, bool, str]]:
    """The right column: (text, is_heading, superscript drawn after the text).
    Refuses a number that is missing or pending."""
    need = ("csfrTol", "recSrfPooled", "recNQueries")
    bad = [k for k in need if k not in v or "pending" in v[k]]
    if bad:
        raise ValueError(f"numbers of the graphical abstract not generated: {bad}")
    return [("One certified reconstruction", True, ""),
            ("unique minimiser, Frank\u2013Wolfe gap \u2264 10", False, power_of_ten(v["csfrTol"])),
            ("Observed pixels kept exactly", True, ""),
            ("hard constraints, with the 8-bit range", False, ""),
            ("Sensitivity record, checked", True, ""),
            (f"{v['recSrfPooled']}% of {v['recNQueries']} queries within {100 * srf_max_err:.0f}%", False, ""),
            ("of a re-solved perturbation", False, ""),
            ("Accuracy reported as measured", True, ""),
            ("block-DCT ahead on scattered loss,", False, ""),
            ("LaMa ahead on burst loss", False, ""),
            ("Suited to contiguous loss", True, ""),
            ("where the result must be audited", False, "")]


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise OSError("no TrueType font found (Arial or DejaVu Sans)")


def query_pixel(observed: np.ndarray) -> tuple[int, int]:
    """QUERY_DEPTH rows inside the hole from its lowest row, in the middle column
    (the walkthrough patch loses its top rows)."""
    hole_rows = np.flatnonzero((~observed).all(axis=1))
    if hole_rows.size == 0:
        raise ValueError("the patch has no fully lost row")
    return int(hole_rows.max()) - QUERY_DEPTH + 1, observed.shape[1] // 2


def grey_tile(img: np.ndarray, hole: np.ndarray, size: int) -> Image.Image:
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[hole] = HOLE_RGB
    return Image.fromarray(rgb, "RGB").resize((size, size), Image.NEAREST)


def record_tile(row: np.ndarray, hole: np.ndarray, query: tuple[int, int], cmap, size: int) -> Image.Image:
    """The row of one imputed pixel over the observed pixels on a symmetric
    diverging scale, signed square root of the weight so that the reach of the
    row shows beside its largest entry (the key gives only the signs); the hole
    in grey and the queried pixel boxed."""
    w = np.where(hole, 0.0, row)
    lim = float(np.abs(w).max()) or 1.0
    v = np.sign(w) * np.sqrt(np.abs(w) / lim)
    rgb = (np.asarray(cmap(0.5 + 0.5 * v))[..., :3] * 255).astype(np.uint8)
    rgb[hole] = HOLE_RGB
    tile = Image.fromarray(rgb, "RGB").resize((size, size), Image.NEAREST)
    s = size / hole.shape[0]
    r, c = query
    ImageDraw.Draw(tile).rectangle([c * s - 3, r * s - 3, (c + 1) * s + 2, (r + 1) * s + 2], outline=INK, width=2)
    return tile


def colour_key(cmap, width: int, height: int) -> Image.Image:
    v = np.linspace(0.0, 1.0, width)
    rgb = (np.asarray(cmap(v))[:, :3] * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(rgb[None], height, axis=0), "RGB")


def draw(data: dict, row: np.ndarray, query: tuple[int, int], lines: list[tuple[str, bool, str]],
         cmap) -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((12, 12), "Certified reconstruction of lost image regions, with a checked record",
           fill=NAVY, font=font(15, bold=True))
    hole = ~data["observed"]
    size, gap, x0, top = 118, 20, 12, 52
    panels = [(grey_tile(data["truth"] * data["observed"], hole, size), f"Observed: {100 * hole.mean():.0f}% lost"),
              (grey_tile(data["reconstruction"], np.zeros_like(hole), size), "CSFR output, certified"),
              (record_tile(row, hole, query, cmap, size), "Record of the boxed pixel")]
    for i, (tile, label) in enumerate(panels):
        px = x0 + i * (size + gap)
        img.paste(tile, (px, top))
        d.rectangle([px - 1, top - 1, px + size, top + size], outline=(60, 60, 60))
        d.text((px + size / 2, top + size + 7), label, fill=INK, font=font(11), anchor="ma")
        if i < len(panels) - 1:
            ax, ay = px + size + 4, top + size / 2
            d.polygon([(ax, ay - 6), (ax + 12, ay), (ax, ay + 6)], fill=NAVY)
    kx, ky, kw = x0 + 2 * (size + gap) + 14, top + size + 25, size - 28
    img.paste(colour_key(cmap, kw, 7), (kx, ky))
    d.rectangle([kx - 1, ky - 1, kx + kw, ky + 7], outline=(120, 120, 120))
    small = font(10)
    d.text((kx - 4, ky + 3), "\u2212", fill=INK, font=small, anchor="rm")
    d.text((kx + kw + 4, ky + 3), "+", fill=INK, font=small, anchor="lm")
    d.multiline_text((kx + kw / 2, ky + 12), "its dependence on each\nobserved pixel", fill=MUTED, font=small,
                     anchor="ma", align="center", spacing=1)
    d.multiline_text((x0, H - 34), "A GovDocs1 document patch whose top rows are lost (contiguous loss). The record is "
                     "the derivative of\nthe boxed pixel's value with respect to the observed pixels: red raises it, "
                     "blue lowers it.", fill=MUTED, font=small, spacing=2)

    tx = x0 + len(panels) * (size + gap) + 4
    limit = W - tx - 8
    y = top - 2
    for text, heading, sup in lines:
        f = font(12 if heading else 11, heading)
        if heading and y > top:
            y += 4
        d.text((tx, y), text, fill=INK if heading else MUTED, font=f)
        width = d.textlength(text, font=f)
        if sup:
            d.text((tx + width + 1, y - 3), sup, fill=INK if heading else MUTED, font=font(8, heading))
            width += 1 + d.textlength(sup, font=font(8, heading))
        if width > limit:
            raise ValueError(f"line too wide for the column: {text!r} ({width:.0f} > {limit})")
        y += 16 if heading else 14
    if y > H - 40:
        raise ValueError(f"the text column runs into the note: {y} > {H - 40}")
    return img


def save_jpeg(img: Image.Image, out: Path) -> int:
    """The highest quality under MAX_BYTES; returns the quality used."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for q in (92, 88, 84, 80, 75, 70, 65, 60):
        img.save(out, "JPEG", quality=q, optimize=True, subsampling=0)
        if out.stat().st_size < MAX_BYTES:
            return q
    raise ValueError(f"{out.name} is {out.stat().st_size / 1024:.1f} kB at quality 60, over the 45 kB limit")


def record_row(figs, data: dict, query: tuple[int, int]) -> np.ndarray:
    """The record row of `query`, with the solver configuration the sweep recorded."""
    import torch
    cfg = figs.SolverConfig(**data["solver"])
    Mb = torch.from_numpy(data["observed"])
    y = torch.from_numpy((data["truth"] * data["observed"]).astype(np.float64))
    x = torch.from_numpy(data["reconstruction"].astype(np.float64))
    w, info = figs.sensitivity_row(x, y, Mb, query, cfg, figs.RecordConfig())
    print(f"[graphical abstract] record of {query}: CG {info['cg_iters']} it., support {info['support']}")
    return w.numpy()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else OUT
    figs = _load("_figures_v2", ROOT / "scripts" / "render_figures_v2.py")
    nums = _load("_numbers_v2", ROOT / "scripts" / "render_numbers_v2.py")
    data = figs.walkthrough_data()
    if data is None:
        print("[graphical abstract] the walkthrough patch's sweep cell is not available")
        return 1
    q = query_pixel(data["observed"])
    img = draw(data, record_row(figs, data, q), q, text_lines(macro_values(), nums.SRF_MAX_ERR), figs.SIGNED)
    quality = save_jpeg(img, out)
    print(f"[graphical abstract] wrote {out} {img.size} {out.stat().st_size / 1024:.1f} kB at quality {quality}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
