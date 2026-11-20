"""Every figure the manuscript includes prints at publication size as drawn
(R4.5): its page is as wide as the column or text block it is placed in, so it
is not scaled; its fonts are embedded and none is a Type 3 bitmap font; no text
is smaller than 7 pt except the sub- and superscripts of larger text; nothing
is drawn past the page edge, where it would be cut off; and every
image panel is resampled finely enough that a patch pixel prints as a square
of equal cells."""
import re
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MM = 72 / 25.4
MIN_TEXT_PT = 7.0
MIN_PPI = 300
WIDTH_TOL_PT = 0.5


def class_widths() -> dict[str, float]:
    cls = (PAPER / "ieeeaccess" / "ieeeaccess.cls").read_text(encoding="utf-8", errors="replace")
    out = {}
    for name in ("textwidth", "columnwidth"):
        m = re.search(rf"^\\{name} ([\d.]+)mm", cls, re.M)
        assert m, f"\\{name} not found in ieeeaccess.cls"
        out[name] = float(m.group(1)) * MM
    return out


def included_figures() -> list[tuple[str, str]]:
    tex = (PAPER / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    found = re.findall(r"\\includegraphics\[width=\\(textwidth|columnwidth)\]\{(figures/[^}]+\.pdf)\}", tex)
    assert len(found) == len(re.findall(r"\\includegraphics", tex)), "an \\includegraphics with another width"
    assert found, "no figure found"
    return [(path, width) for width, path in found]


def _spans_by_line(page):
    # without the mediabox clip, so that text drawn past the page edge is seen
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_MEDIABOX_CLIP
    for b in page.get_text("dict", flags=flags)["blocks"]:
        for line in b.get("lines", []):
            yield [s for s in line["spans"] if s["text"].strip()]


FIGS = included_figures()


@pytest.mark.parametrize("path,width", FIGS, ids=[p for p, _ in FIGS])
def test_figure_prints_at_publication_size(path, width):
    doc = pymupdf.open(PAPER / path)
    page = doc[0]
    expected = class_widths()[width]
    assert abs(page.rect.width - expected) <= WIDTH_TOL_PT, (
        f"{path} is {page.rect.width:.1f} pt wide but placed at \\{width} = {expected:.1f} pt, so it is scaled")

    fonts = page.get_fonts(full=True)
    assert fonts, f"{path} has no text"
    type3 = [f[3] for f in fonts if f[2] == "Type3"]
    assert not type3, f"{path} has Type 3 fonts {type3}"
    unembedded = [f[3] for f in fonts if not doc.extract_font(f[0])[3]]
    assert not unembedded, f"{path} has fonts that are not embedded: {unembedded}"

    # text must keep clear of the edge: a glyph past it is dropped from the
    # extraction, so a cut line shows only as a span that reaches the edge
    margin = page.rect + (1.0, 1.0, -1.0, -1.0)
    cut = [s["text"] for spans in _spans_by_line(page) for s in spans if not margin.contains(pymupdf.Rect(s["bbox"]))]
    assert not cut, f"{path} has text at or past the page edge: {cut[:5]}"
    inside = page.rect + (-WIDTH_TOL_PT, -WIDTH_TOL_PT, WIDTH_TOL_PT, WIDTH_TOL_PT)
    drawn_out = [d["rect"] for d in page.get_drawings() if not inside.contains(d["rect"])]
    assert not drawn_out, f"{path} has drawings past the page edge: {drawn_out[:3]}"

    small = []
    for spans in _spans_by_line(page):
        if not any(s["size"] >= MIN_TEXT_PT - 0.05 for s in spans):
            small += [(s["text"], round(s["size"], 2)) for s in spans]
    assert not small, f"{path} has text under {MIN_TEXT_PT} pt that is not a sub- or superscript: {small[:8]}"

    for im in page.get_image_info():
        r = pymupdf.Rect(im["bbox"])
        ppi = min(im["width"] / (r.width / 72), im["height"] / (r.height / 72))
        assert ppi >= MIN_PPI, (f"{path}: an image of {im['width']}x{im['height']} px is drawn over "
                                f"{r.width:.0f}x{r.height:.0f} pt, {ppi:.0f} ppi")
