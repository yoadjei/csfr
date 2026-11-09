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
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "paper2_reconstruction.tex"
TAG = "ieee-access-r1-submitted"
OUT_TEX = PAPER / "paper2_ieeeaccess_highlighted.tex"
OUT_PDF = ROOT / "submission" / "ieee_access" / "round2" / "02_highlighted.pdf"
COLOUR = r"\color{revisioncolour}"
# saturated blue: RGB(0,70,190) was indistinguishable from black in thin text
COLOUR_DEF = r"\definecolor{revisioncolour}{RGB}{0,0,235}"

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


def highlight(old: str, new: str) -> str:
    """The new source with what is not in the old one coloured."""
    old_sentences, old_blocks = inventory(old)
    head, body = _body(new)
    lines = body.split("\n")
    blocks = _blocks(lines)
    inside = {k for a, b, _ in blocks for k in range(a, b + 1)}
    for a, b, env in blocks:
        if _norm("\n".join(lines[a:b + 1])) not in old_blocks:
            lines[a] = re.sub(rf"(\\begin\{{{re.escape(env)}\}}(\[[^\]]*\])?)", lambda m: m.group(1) + COLOUR,
                              lines[a], count=1)
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


def submitted_source() -> str:
    # stdin=DEVNULL: under a test runner on Windows the inherited stdin handle
    # can be invalid, and git needs none.
    return subprocess.run(["git", "show", f"{TAG}:paper/paper2_reconstruction.tex"], cwd=ROOT,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _build_ieee():
    spec = importlib.util.spec_from_file_location("build_ieee", ROOT / "scripts" / "build_ieee.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="compile and copy the PDF into the submission folder")
    args = ap.parse_args(argv)
    bi = _build_ieee()
    coloured = highlight(submitted_source(), SRC.read_text(encoding="utf-8"))
    out, _ = bi.convert(coloured, bi.default_class())
    OUT_TEX.write_text(out, encoding="utf-8")
    n = out.count(COLOUR)
    print(f"[highlighted] {OUT_TEX.name}: {n} coloured spans")
    if not args.build:
        return 0
    status = bi.build(OUT_TEX)
    pdf = OUT_TEX.with_suffix(".pdf")
    if status == 0 and pdf.exists():
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, OUT_PDF)
        print(f"[highlighted] copied to {OUT_PDF.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
