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


def test_current_manuscript_highlights_and_converts():
    """The real sources: the coloured manuscript stays brace-balanced and goes
    through the venue conversion."""
    m = _mod()
    coloured = m.highlight(m.submitted_source(), m.SRC.read_text(encoding="utf-8"))
    assert coloured.count(m.COLOUR) > 50
    body = coloured.split(r"\begin{document}", 1)[1]
    assert m._balanced(body.replace(r"\$", ""))
    bi = m._build_ieee()
    out, _ = bi.convert(coloured, bi.default_class())
    assert m.COLOUR in out
