"""derive the ieee access manuscript from the elsarticle source, and build it.

one source of truth. paper/paper2_reconstruction.tex carries the content and
the AUTO table markers that render_stats_tables.py splices into. this script
rewrites only the venue shell: document class, front matter, keywords, the
back-matter headings ieee spells differently, bibliography style, and author
biographies. it writes paper/paper2_ieeeaccess.tex beside the source so the
figure and .bib paths resolve unchanged.

the document class is the journal's own ieeeaccess.cls when it is present in
paper/ieeeaccess/ (download the latex template from the ieee access author
page and unzip it there; the fonts it bundles are not redistributed here), and
plain ieeetran in journal mode otherwise, so the gate stays green on a clean
clone. `--build` runs pdflatex and bibtex on the result with the class
directory on the search path and exits non-zero on any latex error or
undefined reference.

the author biographies come from paper/authors_ieee.json. an author without a
biography is omitted from that section and named in the build output, so a
missing biography cannot ship as an empty block.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "paper2_reconstruction.tex"
OUT = PAPER / "paper2_ieeeaccess.tex"
AUTHORS = PAPER / "authors_ieee.json"
CLASS_DIR = PAPER / "ieeeaccess"
MIKTEX = Path(os.environ.get(
    "MIKTEX_BIN", r"C:\Users\adjei\AppData\Local\Programs\MiKTeX\miktex\bin\x64"))

SHORT_TITLE = "Constraint-Driven Sparse Reconstruction of Forensic Image Fragments"
ACCESS_CLASS = r"\documentclass{ieeeaccess}"
TRAN_CLASS = r"\documentclass[journal]{IEEEtran}"

# elsarticle-only preamble lines that must not reach ieeetran.
DROP_PREAMBLE = (
    r"\usepackage{lineno}",
)

# back-matter headings whose elsevier wording ieee spells or names differently.
HEADINGS = {
    r"\section*{Acknowledgements}": r"\section*{Acknowledgment}",
    r"\section*{CRediT Author Statement}": r"\section*{Author Contributions}",
    r"\section*{Declaration of Competing Interest}": r"\section*{Conflicts of Interest}",
}


def class_available() -> bool:
    return (CLASS_DIR / "ieeeaccess.cls").exists()


def default_class() -> str:
    return ACCESS_CLASS if class_available() else TRAN_CLASS


def between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[i:j]


def named_branch(front: str) -> str:
    """the \\else arm of the front-matter blinding switch: the real authors."""
    m = re.search(r"\\ifanonymous(.*?)\\else(.*?)\\fi", front, re.S)
    return m.group(2) if m else front


# an author argument may itself contain one braced macro, \corref{cor1}, so
# the capture must allow one level of nesting or it stops at the wrong brace.
AUTHOR_RE = re.compile(r"\\author\[[^\]]*\]\{((?:[^{}]|\{[^{}]*\})*)\}")


def parse_front(front: str) -> dict:
    title = re.search(r"\\title\{(.*?)\}\s*$", front, re.M | re.S).group(1).strip()
    branch = named_branch(front)
    raw = AUTHOR_RE.findall(branch)
    strip = lambda a: re.sub(r"\\corref\{[^}]*\}", "", a).strip()
    authors = [strip(a) for a in raw]
    corresponding = next((strip(a) for a in raw if r"\corref" in a),
                         authors[0] if authors else "")
    email = (re.search(r"\\ead\{([^}]*)\}", branch) or [None, ""])[1]
    address = (re.search(r"\\address\[[^\]]*\]\{([^}]*)\}", branch) or [None, ""])[1]
    abstract = between(front, r"\begin{abstract}", r"\end{abstract}").strip()
    keywords = [k.strip() for k in
                between(front, r"\begin{keyword}", r"\end{keyword}").split(r"\sep")]
    # ieee access asks for index terms in alphabetical order.
    keywords = sorted((k for k in keywords if k), key=str.lower)
    return {"title": title, "authors": authors, "corresponding": corresponding,
            "email": email, "address": address, "abstract": abstract,
            "keywords": keywords}


def running_head(f: dict, anonymous: bool) -> str:
    first = "Anonymous" if anonymous else f["authors"][0].split()[-1]
    return first


def access_front(f: dict, anonymous: bool, funding: str) -> str:
    """front matter in the ieeeaccess.cls dialect: uppercase author names with
    affiliation marks, \\address, \\corresp, and a keywords environment."""
    if anonymous:
        author = r"\author{\uppercase{Anonymous Authors}\authorrefmark{1}}"
        address = r"\address[1]{Author details withheld for review}"
        corresp = ""
    else:
        names = [rf"\uppercase{{{n}}}\authorrefmark{{1}}" for n in f["authors"]]
        joined = (", ".join(names[:-1]) + (", AND " if len(names) > 2 else " AND ")
                  + names[-1]) if len(names) > 1 else names[0]
        author = r"\author{" + joined + "}"
        address = rf"\address[1]{{{f['address']}}}"
        corresp = rf"\corresp{{Corresponding author: {f['corresponding']} (e-mail: {f['email']}).}}"
    head = rf"{running_head(f, anonymous)} \headeretal: {SHORT_TITLE}"
    lines = [
        # the class requires both; ieee replaces them at production.
        r"\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}",
        r"\doi{10.1109/ACCESS.0000.0000000}",
        "",
        rf"\title{{{f['title']}}}",
        author,
        address,
    ]
    if funding:
        lines.append(rf"\tfootnote{{{funding}}}")
    lines += [
        rf"\markboth{{{head}}}{{{head}}}",
        corresp,
        "",
        r"\begin{abstract}",
        f["abstract"],
        r"\end{abstract}",
        "",
        r"\begin{keywords}",
        ", ".join(f["keywords"]),
        r"\end{keywords}",
        "",
        r"\titlepgskip=-21pt",
        "",
        r"\maketitle",
        "",
    ]
    return "\n".join(lines)


def tran_front(f: dict, anonymous: bool) -> str:
    """front matter for the plain ieeetran fallback."""
    if anonymous:
        author_block = r"\author{Anonymous Authors\thanks{Author details withheld for review.}}"
    else:
        names = f["authors"]
        joined = (", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ")
                  + names[-1]) if len(names) > 1 else names[0]
        thanks = [rf"\thanks{{The authors are with the {f['address']}.}}"]
        if f["corresponding"]:
            thanks.append(
                rf"\thanks{{Corresponding author: {f['corresponding']} (e-mail: {f['email']}).}}")
        author_block = r"\author{" + joined + "%\n" + "%\n".join(thanks) + "}"
    head = running_head(f, anonymous) + r" \MakeLowercase{\textit{et al.}}"
    return "\n".join([
        rf"\title{{{f['title']}}}",
        author_block,
        rf"\markboth{{IEEE Access}}{{{head}: {SHORT_TITLE}}}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        f["abstract"],
        r"\end{abstract}",
        "",
        r"\begin{IEEEkeywords}",
        ", ".join(f["keywords"]),
        r"\end{IEEEkeywords}",
        "",
    ])


ORCID_PLACEHOLDER = "0000-0000-0000-0000"


def orcid_placeholders() -> list[str]:
    """authors whose orcid field still holds the placeholder. ieee collects the
    id in the submission system, so it never reaches the tex; this is a reminder."""
    if not AUTHORS.exists():
        return []
    entries = json.loads(AUTHORS.read_text(encoding="utf-8"))
    return [e["name"] for e in entries if e.get("orcid") == ORCID_PLACEHOLDER]


def biographies(anonymous: bool) -> tuple[str, list[str]]:
    """ieee biography blocks for every author with a biography; names of
    those without one, so the caller can say so."""
    if anonymous or not AUTHORS.exists():
        return "", []
    entries = json.loads(AUTHORS.read_text(encoding="utf-8"))
    blocks, missing = [], []
    for e in entries:
        if not e.get("bio"):
            missing.append(e["name"])
            continue
        photo = e.get("photo")
        if photo and (PAPER / photo).exists():
            head = (r"\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,"
                    rf"clip,keepaspectratio]{{{photo}}}}}]{{{e['name']}}}")
            tail = r"\end{IEEEbiography}"
        else:
            head = rf"\begin{{IEEEbiographynophoto}}{{{e['name']}}}"
            tail = r"\end{IEEEbiographynophoto}"
        blocks.append("\n".join([head, e["bio"].strip(), tail, ""]))
    return "\n".join(blocks), missing


FUNDING_RE = re.compile(r"\\section\*\{Funding\}\n(.+?)\n\n", re.S)


def convert(src: str, doc_class: str) -> tuple[str, list[str]]:
    preamble, rest = src.split(r"\begin{document}", 1)
    anonymous = re.search(r"^\s*\\anonymoustrue", preamble, re.M) is not None
    access = doc_class == ACCESS_CLASS

    # lambda, not a string: the class line contains backslashes that re.sub
    # would otherwise read as escapes.
    preamble = re.sub(r"\\documentclass\[[^\]]*\]\{elsarticle\}", lambda _: doc_class, preamble)
    for line in DROP_PREAMBLE:
        preamble = preamble.replace(line + "\n", "")

    front = between(rest, r"\begin{frontmatter}", r"\end{frontmatter}")
    body = rest.split(r"\end{frontmatter}", 1)[1]
    body, _, tail = body.partition(r"\bibliographystyle{elsarticle-num}")
    tail = tail.replace(r"\end{document}", "")

    for old, new in HEADINGS.items():
        body = body.replace(old, new)

    # run-in paragraph headings. ieeeaccess.cls defines \paragraph with a
    # small-caps font that leaks into the paragraph body and, two floats
    # later, unbalances a table; ieeetran numbers them "a)". an italic run-in
    # label is what ieee journals typeset, and it needs no class support.
    body = re.sub(r"\\paragraph\{((?:[^{}]|\{[^{}]*\})*)\}", r"\\textit{\1}", body)

    # ieeeaccess.cls redefines \textbf in a way that fails on an argument that
    # opens with inline math ("Extra }, or forgotten $" at the end of the
    # enclosing table). giving each $...$ its own brace group is invisible in
    # the output and sidesteps it; elsarticle never needed this.
    inner = r"(?:[^{}]|\{[^{}]*\})*"  # one level of nesting, as in $d_{z}$
    body = re.sub(r"\\textbf\{(" + inner + r"\$" + inner + r")\}",
                  lambda m: r"\textbf{" + re.sub(r"(\$[^$]*\$)", r"{\1}", m.group(1)) + "}",
                  body)

    # ieee access carries the funding statement as the title footnote, not as
    # a section; ieeetran journal mode keeps the section.
    funding = ""
    if access:
        m = FUNDING_RE.search(body)
        if m:
            funding = " ".join(m.group(1).split())
            body = body[:m.start()] + body[m.end():]

    fields = parse_front(front)
    front_tex = access_front(fields, anonymous, funding) if access else tran_front(fields, anonymous)
    bios, missing = biographies(anonymous)
    out = "".join([
        "%% GENERATED by scripts/build_ieee.py from paper2_reconstruction.tex.\n",
        "%% Edit the source, not this file.\n",
        preamble,
        r"\begin{document}", "\n\n",
        front_tex,
        body,
        r"\bibliographystyle{IEEEtran}", tail.rstrip(), "\n\n",
        bios,
        r"\EOD" + "\n\n" if access else "",
        r"\end{document}", "\n",
    ])
    return out, missing


def build() -> int:
    """pdflatex, bibtex, pdflatex, pdflatex in paper/, with the class directory
    on every search path the fonts and map files need."""
    env = os.environ.copy()
    if class_available():
        for var in ("TEXINPUTS", "TFMFONTS", "T1FONTS", "TEXFONTMAPS", "ENCFONTS"):
            env[var] = str(CLASS_DIR) + os.pathsep + env.get(var, "")
    pdflatex = [str(MIKTEX / "pdflatex.exe"), "-interaction=nonstopmode",
                "-include-directory=" + str(CLASS_DIR), OUT.stem]
    bibtex = [str(MIKTEX / "bibtex.exe"), OUT.stem]
    for cmd in (pdflatex, bibtex, pdflatex, pdflatex):
        subprocess.run(cmd, cwd=PAPER, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    log = (PAPER / f"{OUT.stem}.log").read_text(encoding="utf-8", errors="replace")
    errors = sum(1 for ln in log.splitlines() if ln.startswith("!"))
    undefined = len(re.findall(r"(Citation|Reference) .* undefined|There were undefined references", log))
    pages = re.search(r"Output written on .*\((\d+) pages", log)
    # ieeeaccess.cls builds its footer as \hbox to 0pc{\hbox to \textwidth{..}},
    # which reports one overfull box per page "while \output is active". those
    # are the class's own, so only the overfull lines in our text are counted.
    overfull = [m for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)(.*)", log)
                if float(m.group(1)) > 20 and "output is active" not in m.group(2)]
    blg = PAPER / f"{OUT.stem}.blg"
    bib_warn = len(re.findall(r"^Warning--", blg.read_text(encoding="utf-8", errors="replace"), re.M)) if blg.exists() else -1
    print(f"[ieee] class={'ieeeaccess' if class_available() else 'IEEEtran'} "
          f"errors={errors} undefined={undefined} overfull>20pt={len(overfull)} "
          f"bibtex_warnings={bib_warn} pages={pages.group(1) if pages else '?'}")
    return 0 if errors == 0 and undefined == 0 and pages else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="doc_class", default=None,
                    help="document class line; defaults to ieeeaccess when paper/ieeeaccess/ "
                         "holds the class, else the ieeetran journal fallback")
    ap.add_argument("--build", action="store_true", help="also run pdflatex and bibtex")
    args = ap.parse_args(argv)
    doc_class = args.doc_class or default_class()

    out, missing = convert(SRC.read_text(encoding="utf-8"), doc_class)
    OUT.write_text(out, encoding="utf-8")
    print(f"[ieee] wrote {OUT.relative_to(ROOT)} with {doc_class}")
    if missing:
        print(f"[ieee] no biography for: {', '.join(missing)} (omitted from the biography section)")
    for name in orcid_placeholders():
        print(f"[ieee] orcid placeholder for: {name} (enter the real id in the submission system)")
    return build() if args.build else 0


if __name__ == "__main__":
    sys.exit(main())
