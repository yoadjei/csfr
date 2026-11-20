"""Highlighted-changes manuscript: sentences split only where a colour group
cannot break the LaTeX, unchanged text stays uncoloured, new sentences and
changed blocks are coloured, and the result still converts for the venue."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("hl", ROOT / "scripts" / "build_highlighted.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DOC = r"""\documentclass{article}
\begin{document}
\section{One}
Old sentence stays. Another old one.
\begin{equation}
x = 1
\end{equation}
\end{document}
"""


def test_split_respects_braces_and_math():
    m = _mod()
    line = r"A $x. Y$ b. \textbf{C. D} e. F~\cite{k}. G"
    parts = m.split_sentences(line)
    assert "".join(parts) == line
    assert [p.strip() for p in parts] == [r"A $x. Y$ b.", r"\textbf{C. D} e.", r"F~\cite{k}.", "G"]


def test_split_after_a_closing_brace_and_not_after_abbreviations():
    m = _mod()
    parts = m.split_sentences(r"\textbf{Rule.} Every step. Use e.g.\ this and et al.~\cite{a} too.")
    assert [p.strip() for p in parts] == [r"\textbf{Rule.}", "Every step.",
                                          r"Use e.g.\ this and et al.~\cite{a} too."]


def test_identical_sources_are_not_coloured():
    m = _mod()
    out = m.highlight(DOC, DOC)
    assert m.COLOUR not in out.split(r"\begin{document}", 1)[1]


def test_new_sentence_and_changed_block_are_coloured():
    m = _mod()
    new = DOC.replace("Another old one.", "A new claim.").replace("x = 1", "x = 2")
    out = m.highlight(DOC, new)
    assert "Old sentence stays. {" + m.COLOUR + " A new claim.}" in out
    assert r"\begin{equation}" + m.COLOUR in out
    assert m.COLOUR_DEF in out


def test_new_keywords_are_coloured_one_by_one():
    """The keyword list is one line: colouring it as a sentence put the opening
    and the closing brace in different keywords once the venue conversion
    sorted them."""
    m = _mod()
    old = DOC.replace(r"\section{One}", "\\begin{keyword}\nAlpha \\sep Beta \\sep Delta\n\\end{keyword}\n\\section{One}")
    new = old.replace(r"Alpha \sep Beta \sep Delta", r"Alpha \sep Gamma \sep Beta")
    out = m.highlight(old, new)
    assert "Alpha \\sep {" + m.COLOUR + " Gamma} \\sep Beta\n" in out
    assert m.highlight(old, old).count(m.COLOUR) == 0


def test_keyword_sort_ignores_the_colour_group():
    m = _mod()
    bi = m._build_ieee()
    kws = ["Zeta", "{" + m.COLOUR + " Alpha}", "beta"]
    assert sorted(kws, key=bi.keyword_key) == ["{" + m.COLOUR + " Alpha}", "beta", "Zeta"]


def _pdf_with_two_colours(path: Path, m) -> None:
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    blue = tuple(c / 255 for c in m.REVISION_RGB)
    page.insert_text((72, 100), "Old text in black.", fontsize=10, color=(0, 0, 0))
    page.insert_text((72, 130), "A new sentence in blue.", fontsize=10, color=blue)
    page.insert_text((72, 160), "Black again.", fontsize=10, color=(0, 0, 0))
    doc.save(path)


def test_annotate_highlights_only_the_revision_colour(tmp_path):
    import pymupdf
    m = _mod()
    pdf = tmp_path / "t.pdf"
    _pdf_with_two_colours(pdf, m)
    n, pages = m.annotate(pdf)
    assert (n, pages) == (1, 1)
    page = pymupdf.open(pdf)[0]
    annots = list(page.annots())
    assert [a.type[1] for a in annots] == ["Highlight"]
    assert tuple(annots[0].colors["stroke"]) == m.HIGHLIGHT_RGB
    blue = page.search_for("A new sentence in blue.")[0]
    black = page.search_for("Old text in black.")[0]
    assert annots[0].rect.contains(blue + (0.5, 0.5, -0.5, -0.5))
    assert not annots[0].rect.intersects(black)


def test_annotate_refuses_a_pdf_without_the_revision_colour(tmp_path):
    """A colour mismatch (another RGB, or a changed colour model) would leave the
    PDF without highlights and still look like success."""
    import pymupdf
    import pytest
    m = _mod()
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "All black.", fontsize=10)
    pdf = tmp_path / "b.pdf"
    doc.save(pdf)
    with pytest.raises(ValueError):
        m.annotate(pdf)


def test_current_manuscript_highlights_and_converts():
    """The real sources: the coloured manuscript stays brace-balanced and goes
    through the venue conversion, and stays balanced after it."""
    m = _mod()
    coloured = m.highlight(m.submitted_source(), m.SRC.read_text(encoding="utf-8"))
    assert coloured.count(m.COLOUR) > 50
    body = coloured.split(r"\begin{document}", 1)[1]
    assert m._balanced(body.replace(r"\$", ""))
    bi = m._build_ieee()
    out, _ = bi.convert(coloured, bi.default_class())
    assert m.COLOUR in out
    assert m._balanced(out.split(r"\begin{document}", 1)[1].replace(r"\$", ""))
    keywords = out.split(r"\begin{keywords}", 1)[1].split(r"\end{keywords}", 1)[0]
    assert "{" + m.COLOUR + " Convex Optimisation}" in keywords


BBL_OLD = r"""\begin{thebibliography}{10}
\providecommand{\url}[1]{#1}

\bibitem{kept}
A.~Author, ``Kept,'' \emph{J.}, 2001.

\bibitem{edited}
B.~Author, ``Edited,'' \emph{J.}, 2002.

\end{thebibliography}
"""

BBL_NEW = r"""\begin{thebibliography}{10}
\providecommand{\url}[1]{#1}

\bibitem{kept}
A.~Author, ``Kept,''
  \emph{J.}, 2001.

\bibitem{added}
C.~Author, ``Added,'' \emph{J.}, 2025.

\bibitem{edited}
B.~Author, ``Edited,'' \emph{J.}, vol.~3, 2002.

\end{thebibliography}
"""


def test_cite_keys_in_first_citation_order_without_comments():
    m = _mod()
    tex = ("As shown~\\cite{b,a} and \\citep[p.~3]{c}; again \\cite{a}.\n"
           "% \\cite{commented}\n"
           "Kept 50\\% \\nocite{d} % \\cite{tail}\n"
           "Split over lines~\\cite{e,\n  f} \\citestyle{plain}\n")
    assert m.cite_keys(tex) == ["b", "a", "c", "d", "e", "f"]


def test_bbl_items_span_each_entry_text():
    m = _mod()
    items = m.bbl_items(BBL_NEW)
    assert list(items) == ["kept", "added", "edited"]
    a, b = items["added"]
    assert BBL_NEW[a:b].strip() == r"C.~Author, ``Added,'' \emph{J.}, 2025."
    assert BBL_NEW[b:].startswith("\n\n\\bibitem{edited}")


def test_revised_references_are_new_or_changed_in_print():
    """Rewrapped lines are not a change; a new entry and an edited one are."""
    m = _mod()
    assert m.revised_references(BBL_OLD, BBL_NEW) == {"added", "edited"}
    assert m.revised_references(BBL_NEW, BBL_NEW) == set()


def test_colour_references_wraps_only_the_given_entries():
    m = _mod()
    out = m.colour_references(BBL_NEW, {"added", "edited"})
    assert out.count(m.COLOUR) == 2
    assert m._balanced(out.replace(r"\$", ""))
    assert "\\bibitem{added}{\\leavevmode" + m.COLOUR + "C.~Author" in out
    assert "2025.%\n}\n\n\\bibitem{edited}" in out
    # the unchanged entry and the list around the entries are untouched
    a, b = m.bbl_items(BBL_NEW)["kept"]
    assert BBL_NEW[:b] in out
    assert out.endswith("%\n}\n\n\\end{thebibliography}\n")
    assert m.colour_references(BBL_NEW, set()) == BBL_NEW


def test_build_rewrites_the_bibliography_before_the_passes_that_typeset_it(tmp_path, monkeypatch):
    """The hook sees bibtex's output and the two pdflatex runs after bibtex
    read what it returns."""
    m = _mod()
    bi = m._build_ieee()
    monkeypatch.setattr(bi, "PAPER", tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        tool = Path(cmd[0]).stem
        if tool == "bibtex":
            (tmp_path / "doc.bbl").write_text("bibtex output\n", encoding="utf-8", newline="\n")
        if tool == "pdflatex":
            calls.append(("pdflatex", (tmp_path / "doc.bbl").read_text(encoding="utf-8")
                          if (tmp_path / "doc.bbl").exists() else None))
            (tmp_path / "doc.log").write_text("Output written on doc.pdf (3 pages, 1 bytes).\n",
                                              encoding="utf-8")
        else:
            calls.append((tool, None))

    monkeypatch.setattr(bi.subprocess, "run", fake_run)

    def hook(text):
        calls.append(("hook", text))
        return text.upper()

    assert bi.build(tmp_path / "doc.tex", bbl_hook=hook) == 0
    assert calls == [("pdflatex", None), ("bibtex", None), ("hook", "bibtex output\n"),
                     ("pdflatex", "BIBTEX OUTPUT\n"), ("pdflatex", "BIBTEX OUTPUT\n")]


def test_build_never_hands_the_hook_a_stale_bibliography(tmp_path, monkeypatch):
    """If bibtex writes no .bbl, the build stops instead of colouring the .bbl
    an earlier run left behind."""
    import pytest
    m = _mod()
    bi = m._build_ieee()
    monkeypatch.setattr(bi, "PAPER", tmp_path)
    (tmp_path / "doc.bbl").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(bi.subprocess, "run", lambda cmd, **kw: None)
    seen = []
    with pytest.raises(RuntimeError, match="no doc.bbl"):
        bi.build(tmp_path / "doc.tex", bbl_hook=lambda text: seen.append(text) or text)
    assert seen == []


def test_submitted_bibliography_matches_the_submitted_pdf():
    """Independent of cite_keys: the rebuilt submitted bibliography has as many
    entries as the reference list of the PDF typeset from the submitted source
    in the journal's class (kept in the gitignored submission folder)."""
    import re

    import pymupdf
    import pytest
    m = _mod()
    bi = m._build_ieee()
    pdf = m.ROOT / "submission" / "ieee_access" / "reference_only" / "manuscript_ieee_class.pdf"
    if not pdf.exists() or not (bi.MIKTEX / "bibtex.exe").exists():
        pytest.skip("submitted class PDF or bibtex not available")
    with pymupdf.open(pdf) as d:
        text = "".join(p.get_text() for p in d)
    refs = text[text.upper().rindex("REFERENCES"):]
    labels = [int(x) for x in re.findall(r"^\[(\d+)\]", refs, re.M)]
    assert labels == list(range(1, len(labels) + 1))
    assert len(m.bbl_items(m.submitted_bbl(m.submitted_source(), bi))) == len(labels)


def test_submitted_bibliography_has_every_submitted_citation():
    """bibtex on the submitted citations and .bib gives one entry per cited key."""
    import pytest
    m = _mod()
    bi = m._build_ieee()
    if not (bi.MIKTEX / "bibtex.exe").exists():
        pytest.skip("bibtex not installed")
    old = m.submitted_source()
    keys = m.cite_keys(bi.convert(old, bi.default_class())[0])
    old_bbl = m.submitted_bbl(old, bi)
    assert list(m.bbl_items(old_bbl)) == keys
    assert m.revised_references(old_bbl, old_bbl) == set()
