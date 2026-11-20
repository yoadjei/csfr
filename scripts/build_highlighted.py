"""Build the highlighted-changes manuscript for the resubmission.

    python scripts/build_highlighted.py [--build]

Compares the manuscript as submitted (git tag ieee-access-r1-submitted) with the
current source and colours what is new: every sentence of running text that
does not occur in the submitted version, and every equation, table, figure or
algorithm whose source changed. Deleted text is not shown; the response to the
reviewers says what was removed and why. The coloured source goes through the
same venue conversion as the clean manuscript (scripts/build_ieee.py), so the
two PDFs differ only in colour.

latexdiff was the first choice. It needs the Perl module Algorithm::Diff,
which this machine's Perl lacks, and on a revision that rewrites most
paragraphs its strike-through output is mostly noise; sentence-level colouring
is the fallback the plan names.

Sentences are split only at brace depth zero and outside inline math, so a
colour group never crosses a brace or a math boundary. Lines that open or
close an environment, headings, labels and comments are left untouched.

With --build, the reference list is coloured too: an entry is new or changed
when its printed text is not in the bibliography of the submitted version
(bibtex on the submitted citations and .bib; submitted_bbl). The compiled PDF
then gets a yellow highlight annotation over the coloured text (annotate), as
the decision letter asks.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "paper2_reconstruction.tex"
TAG = "ieee-access-r1-submitted"
OUT_TEX = PAPER / "paper2_ieeeaccess_highlighted.tex"
OUT_PDF = ROOT / "submission" / "ieee_access" / "round2" / "02_highlighted.pdf"
COLOUR = r"\color{revisioncolour}"
# saturated blue: RGB(0,70,190) was indistinguishable from black in thin text
REVISION_RGB = (0, 0, 235)
COLOUR_DEF = r"\definecolor{revisioncolour}{RGB}{%d,%d,%d}" % REVISION_RGB
# the decision letter asks for the PDF's yellow highlight tool
HIGHLIGHT_RGB = (1.0, 1.0, 0.0)
CITE = re.compile(r"\\(?:no)?cite(?!style)[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}")
BIB_END = r"\end{thebibliography}"

# environments compared and coloured as a whole
BLOCK_ENVS = ("equation", "equation*", "align", "align*", "table", "table*", "figure",
              "figure*", "algorithm")
# a line starting with one of these is structure, not running text
STRUCTURAL = re.compile(r"\s*(\\(section|subsection|subsubsection|label|begin|end|caption|"
                        r"bibliography|bibliographystyle|input|maketitle|title|author|"
                        r"address|ead|cortext|newcommand|renewcommand|providecommand|"
                        r"usepackage|documentclass|hypersetup|newif|anonymous|fi|else|"
                        r"ifanonymous|centering|toprule|midrule|bottomrule|FloatBarrier)\b|%|$)")
PREFIX = re.compile(r"\s*(\\item(\[[^\]]*\])?\s*|\\paragraph\{[^{}]*\}\s*)")


def split_sentences(text: str) -> list[str]:
    """Sentences of one line of running text, split at whitespace that follows
    '.', '!' or '?' (optionally closed by '}') at brace depth zero and outside
    inline math. Concatenating the pieces returns the line unchanged."""
    out, start, depth, math, i = [], 0, 0, False, 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "$":
            math = not math
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch.isspace() and depth == 0 and not math and i > start:
            prev = text[start:i].rstrip("}")
            nxt = text[i:].lstrip()
            starts_sentence = nxt[:1].isupper() or nxt.startswith(("\\", "$"))
            if prev.endswith((".", "!", "?")) and starts_sentence:
                out.append(text[start:i + 1])
                start = i + 1
        i += 1
    if start < len(text):
        out.append(text[start:])
    return out


def _norm(s: str) -> str:
    return " ".join(s.split())


def _balanced(s: str) -> bool:
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth < 0:
            return False
        i += 1
    return depth == 0 and s.replace("\\$", "").count("$") % 2 == 0


def _blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """(first line, last line, env) of every top-level block environment."""
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"\s*\\begin\{([^}]+)\}", lines[i])
        if m and m.group(1) in BLOCK_ENVS:
            env = m.group(1)
            j = i
            while not re.match(rf"\s*\\end\{{{re.escape(env)}\}}", lines[j]):
                j += 1
            out.append((i, j, env))
            i = j + 1
        else:
            i += 1
    return out


def _body(tex: str) -> tuple[str, str]:
    head, body = tex.split(r"\begin{document}", 1)
    return head + r"\begin{document}", body


def inventory(tex: str) -> tuple[set[str], set[str]]:
    """Normalised sentences of running text and normalised block sources."""
    lines = _body(tex)[1].split("\n")
    blocks = _blocks(lines)
    inside = {k for a, b, _ in blocks for k in range(a, b + 1)}
    sentences = set()
    for k, line in enumerate(lines):
        if k in inside or STRUCTURAL.match(line):
            continue
        m = PREFIX.match(line)
        rest = line[m.end():] if m else line
        sentences.update(_norm(s) for s in split_sentences(rest) if _norm(s))
    return sentences, {_norm("\n".join(lines[a:b + 1])) for a, b, _ in blocks}


def _keyword_lines(lines: list[str]) -> list[int]:
    """Indices of the lines between \\begin{keyword} and \\end{keyword}."""
    out, inside = [], False
    for k, line in enumerate(lines):
        if re.match(r"\s*\\begin\{keyword\}", line):
            inside = True
        elif re.match(r"\s*\\end\{keyword\}", line):
            inside = False
        elif inside:
            out.append(k)
    return out


def keyword_set(tex: str) -> set[str]:
    lines = _body(tex)[1].split("\n")
    return {_norm(t) for k in _keyword_lines(lines) for t in lines[k].split(r"\sep") if _norm(t)}


def highlight(old: str, new: str) -> str:
    """The new source with what is not in the old one coloured. Keywords are
    coloured one by one: the venue conversion sorts them, so a colour group
    around several would be split between them."""
    old_sentences, old_blocks = inventory(old)
    old_keywords = keyword_set(old)
    head, body = _body(new)
    lines = body.split("\n")
    blocks = _blocks(lines)
    inside = {k for a, b, _ in blocks for k in range(a, b + 1)}
    for a, b, env in blocks:
        if _norm("\n".join(lines[a:b + 1])) not in old_blocks:
            lines[a] = re.sub(rf"(\\begin\{{{re.escape(env)}\}}(\[[^\]]*\])?)", lambda m: m.group(1) + COLOUR,
                              lines[a], count=1)
    for k in _keyword_lines(lines):
        terms = [t.strip() for t in lines[k].split(r"\sep")]
        lines[k] = r" \sep ".join(t if not t or _norm(t) in old_keywords else "{" + COLOUR + " " + t + "}"
                                  for t in terms)
        inside.add(k)
    for k, line in enumerate(lines):
        if k in inside or STRUCTURAL.match(line) or not _balanced(line):
            continue
        m = PREFIX.match(line)
        cut = m.end() if m else 0
        pieces = []
        for s in split_sentences(line[cut:]):
            core = s.rstrip()
            if _norm(core) and _norm(core) not in old_sentences and _balanced(core):
                pieces.append("{" + COLOUR + " " + core + "}" + s[len(core):])
            else:
                pieces.append(s)
        lines[k] = line[:cut] + "".join(pieces)
    head = head.replace(r"\begin{document}", COLOUR_DEF + "\n" + r"\begin{document}", 1)
    return head + "\n".join(lines)


def submitted_file(path: str) -> str:
    """A file of the repository as submitted (path relative to the root, with /)."""
    # stdin=DEVNULL: under a test runner on Windows the inherited stdin handle
    # can be invalid, and git needs none.
    return subprocess.run(["git", "show", f"{TAG}:{path}"], cwd=ROOT,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def submitted_source() -> str:
    return submitted_file("paper/paper2_reconstruction.tex")


def cite_keys(tex: str) -> list[str]:
    """The keys the source cites, in order of first citation (the order of the
    entries under an unsorted style). Comments are removed first and the rest
    is scanned as one text, so a key list broken over lines is read whole."""
    text = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in tex.splitlines())
    keys: list[str] = []
    for m in CITE.finditer(text):
        for k in m.group(1).split(","):
            if k.strip() and k.strip() not in keys:
                keys.append(k.strip())
    return keys


def bbl_items(bbl: str) -> dict[str, tuple[int, int]]:
    """Each entry's key -> (start, end) of its text in the bibliography: from
    after its \\bibitem to the next \\bibitem or the end of the list, trailing
    whitespace left out."""
    found = list(BIBITEM.finditer(bbl))
    list_end = bbl.index(BIB_END)
    items = {}
    for i, m in enumerate(found):
        stop = found[i + 1].start() if i + 1 < len(found) else list_end
        items[m.group(1)] = (m.end(), m.end() + len(bbl[m.end():stop].rstrip()))
    return items


def submitted_bbl(old_tex: str, bi) -> str:
    """The bibliography of the submitted manuscript: bibtex on its citations,
    in their order, with its .bib, the style of the venue conversion and the
    search paths of the build."""
    conv, _ = bi.convert(old_tex, bi.default_class())
    style = re.search(r"\\bibliographystyle\{([^}]*)\}", conv).group(1)
    bib = re.search(r"\\bibliography\{([^}]*)\}", conv).group(1)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "submitted.bib").write_text(submitted_file(f"paper/{bib}.bib"), encoding="utf-8")
        (d / "submitted.aux").write_text(
            "".join(f"\\citation{{{k}}}\n" for k in cite_keys(conv))
            + f"\\bibstyle{{{style}}}\n\\bibdata{{submitted}}\n", encoding="utf-8")
        r = subprocess.run([str(bi.MIKTEX / "bibtex.exe"), "submitted"], cwd=d, env=bi.tex_env(),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if r.returncode > 1 or not (d / "submitted.bbl").exists():  # 1 = warnings only
            raise RuntimeError(f"bibtex on the submitted bibliography failed:\n{r.stdout[-2000:]}")
        return (d / "submitted.bbl").read_text(encoding="utf-8")


def revised_references(old_bbl: str, new_bbl: str) -> set[str]:
    """Keys of the new bibliography whose printed entry the old one lacks: new
    references, and cited ones whose entry text changed."""
    old = {k: _norm(old_bbl[a:b]) for k, (a, b) in bbl_items(old_bbl).items()}
    return {k for k, (a, b) in bbl_items(new_bbl).items() if old.get(k) != _norm(new_bbl[a:b])}


def colour_references(bbl: str, keys: set[str]) -> str:
    """The bibliography with the text of each entry in keys in a colour group
    of its own; the labels keep the class's colour. \\leavevmode starts the
    item's paragraph (and so its label) before the colour is set; the group
    closes on a line of its own after a %, so a comment ending the entry
    cannot swallow it."""
    out, pos = [], 0
    for key, (a, b) in sorted(bbl_items(bbl).items(), key=lambda kv: kv[1][0]):
        if key in keys:
            out += [bbl[pos:a], "{\\leavevmode" + COLOUR, bbl[a:b].lstrip(), "%\n}"]
            pos = b
    out.append(bbl[pos:])
    return "".join(out)


def _build_ieee():
    spec = importlib.util.spec_from_file_location("build_ieee", ROOT / "scripts" / "build_ieee.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _coloured_runs(line: dict, colour: int) -> list:
    """Rectangles of the runs of consecutive spans of one text line drawn in
    `colour`; a span of another colour ends a run, a blank span does not."""
    import pymupdf
    runs, run = [], None
    for s in line["spans"]:
        if not s["text"].strip():
            continue
        if s["color"] == colour:
            r = pymupdf.Rect(s["bbox"])
            run = r if run is None else run | r
        elif run is not None:
            runs.append(run)
            run = None
    if run is not None:
        runs.append(run)
    return runs


def annotate(pdf: Path) -> tuple[int, int]:
    """Add a yellow highlight annotation over the text drawn in the revision
    colour, one annotation per text block (a paragraph, caption, table cell or
    displayed equation) covering its coloured runs line by line. The blue stays,
    so the changes also show where annotations are not displayed or printed. A
    changed figure's graphics are not text: the figure is marked by its
    highlighted caption. Returns (annotations, pages with one); refuses a PDF
    with no text in the revision colour, which would mean the colour did not
    reach the PDF as REVISION_RGB."""
    import pymupdf
    colour = (REVISION_RGB[0] << 16) | (REVISION_RGB[1] << 8) | REVISION_RGB[2]
    doc = pymupdf.open(pdf)
    total, pages = 0, 0
    for page in doc:
        added = 0
        for block in page.get_text("dict")["blocks"]:
            rects = [r for line in block.get("lines", []) for r in _coloured_runs(line, colour)]
            if rects:
                annot = page.add_highlight_annot(quads=rects)
                annot.set_colors(stroke=HIGHLIGHT_RGB)
                annot.update()
                added += 1
        total += added
        pages += added > 0
    if total == 0:
        doc.close()
        raise ValueError(f"{pdf.name}: no text in the revision colour {REVISION_RGB}")
    tmp = pdf.with_name(pdf.stem + ".annotating.pdf")
    doc.save(tmp, garbage=1, deflate=True)
    doc.close()
    tmp.replace(pdf)
    return total, pages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="compile and copy the PDF into the submission folder")
    args = ap.parse_args(argv)
    bi = _build_ieee()
    old = submitted_source()
    coloured = highlight(old, SRC.read_text(encoding="utf-8"))
    out, _ = bi.convert(coloured, bi.default_class())
    OUT_TEX.write_text(out, encoding="utf-8")
    n = out.count(COLOUR)
    print(f"[highlighted] {OUT_TEX.name}: {n} coloured spans")
    if not args.build:
        return 0
    old_bbl = submitted_bbl(old, bi)

    def colour_revised(bbl: str) -> str:
        keys = revised_references(old_bbl, bbl)
        print(f"[highlighted] {len(keys)} of {len(bbl_items(bbl))} references new or changed")
        return colour_references(bbl, keys)

    status = bi.build(OUT_TEX, bbl_hook=colour_revised)
    pdf = OUT_TEX.with_suffix(".pdf")
    if status == 0 and pdf.exists():
        n_annots, n_pages = annotate(pdf)
        print(f"[highlighted] {n_annots} yellow highlight annotations on {n_pages} pages")
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, OUT_PDF)
        print(f"[highlighted] copied to {OUT_PDF.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
