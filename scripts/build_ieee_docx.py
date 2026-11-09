"""build the word manuscript for ieee access from the generated ieee tex.

pandoc turns paper/paper2_ieeeaccess.tex into docx. before that, this script
prepares a docx-only variant of the tex: cross references and equation numbers
baked in from the aux file, table and figure captions numbered, tables
unwrapped from \\resizebox, the anonymity switch resolved, front matter and
biographies as plain text. after pandoc it fixes the document with python-docx:
ieee heading numbering, table borders, fonts, the biographies moved after the
references, and, when the official word template is given as the reference
document, the journal's own paragraph styles (Paper Title, AU, PI, Abstract,
IT, H1/H2/H3, PARA, Fig Caption, Table Caption, References, AU_Bios). it then
compares every number in every table with the tex, so a dropped cell cannot
pass silently, and can print the file to pdf through word for a visual check.

usage:
  python scripts/build_ieee_docx.py [--out FILE.docx] [--csl ieee.csl]
                                    [--reference-doc TEMPLATE.docx] [--pdf]

defaults: the ieee csl style at submission/ieee.csl (numeric citations;
without it pandoc falls back to author-year and the script says so) and the
official template at paper/ieeeaccess/Access-Template-2024.docx when present
(download it from the ieee access author page; the folder is not committed).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SRC = PAPER / "paper2_ieeeaccess.tex"
AUX = PAPER / "paper2_ieeeaccess.aux"
BIB = PAPER / "paper2_reconstruction.bib"
BUILD = PAPER / ".docx_build"
DEFAULT_OUT = PAPER / "paper2_ieeeaccess.docx"
DEFAULT_CSL = ROOT / "submission" / "ieee.csl"
DEFAULT_TEMPLATE = PAPER / "ieeeaccess" / "Access-Template-2024.docx"

UNNUMBERED = {"Acknowledgment", "Acknowledgements", "Ethics Statement", "Conflicts of Interest",
              "Author Contributions", "Code and Data Availability", "References", "Biographies"}
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ---------------------------------------------------------------- tex helpers

def group_end(s: str, i: int) -> int:
    """s[i] == "{"; index just past the matching "}"."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def macro_arg(s: str, name: str, start: int = 0) -> tuple[int, int, str] | None:
    """first \\name{...} at or after start: (match start, match end, argument)."""
    k = s.find("\\" + name + "{", start)
    if k < 0:
        return None
    i = k + len(name) + 1
    j = group_end(s, i)
    return k, j, s[i + 1:j - 1]


def unwrap_resizebox(tex: str) -> str:
    out, pos = [], 0
    while True:
        k = tex.find(r"\resizebox{", pos)
        if k < 0:
            out.append(tex[pos:])
            return "".join(out)
        out.append(tex[pos:k])
        i = k + len(r"\resizebox")
        i = group_end(tex, i)          # width
        i = group_end(tex, i)          # height
        j = group_end(tex, i)          # content
        out.append(tex[i + 1:j - 1])
        pos = j


def aux_numbers(aux: Path) -> dict[str, str]:
    raw = re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="replace"))
    # section labels arrive as "\mbox  {IV-C"; keep the visible number only
    return {k: re.sub(r"\\mbox\s*\{?", "", v).strip("{} ") for k, v in raw}


def resolve_anonymous(tex: str) -> str:
    """the source sets \\anonymousfalse; keep the named branches. the
    declaration lines go first so the conditional regex cannot start there."""
    tex = re.sub(r"^\\newif\\ifanonymous\s*$|^\\anonymous(?:false|true)\s*$", "", tex, flags=re.M)
    tex = re.sub(r"\\ifanonymous(?![a-zA-Z])(.*?)\\else(?![a-zA-Z])(.*?)\\fi(?![a-zA-Z])",
                 lambda m: m.group(2), tex, flags=re.S)
    return re.sub(r"\\ifanonymous(?![a-zA-Z])(.*?)\\fi(?![a-zA-Z])", "", tex, flags=re.S)


def bake_refs(tex: str, labels: dict[str, str]) -> str:
    tex = re.sub(r"\\eqref\{([^}]*)\}",
                 lambda m: f"({labels[m.group(1)]})" if m.group(1) in labels else m.group(0), tex)
    return re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), m.group(0)), tex)


def number_equations(tex: str, labels: dict[str, str]) -> str:
    """every equation environment is numbered in the pdf; put the same number
    at the end of the display so word readers can follow eqref."""
    counter = 0

    def repl(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", body)
        if lab:
            assert labels.get(lab.group(1)) == str(counter), (lab.group(1), labels.get(lab.group(1)), counter)
            body = body.replace(lab.group(0), "")
        return "\\begin{equation}" + body.rstrip() + f"\\qquad ({counter})\n\\end{{equation}}"

    return re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", repl, tex, flags=re.S)


def wrap_bracket_cells(tex: str) -> str:
    """pandoc loses a table cell that starts with "[" (it reads it as an
    optional argument). brace such cells."""
    def fix_row(line: str) -> str:
        if "&" not in line or "\\\\" not in line:
            return line
        fixed = []
        for c in line.split("&"):
            s = c.strip()
            if s.startswith("["):
                body, tail = (s[:-2].rstrip(), " \\\\") if s.endswith("\\\\") else (s, "")
                c = " {" + body + "}" + tail
            fixed.append(c)
        return "&".join(fixed)

    return "\n".join(fix_row(ln) for ln in tex.split("\n"))


def number_floats(tex: str, labels: dict[str, str]) -> str:
    """table*/figure* floats become plain table/figure with numbered captions,
    the label, centering and sizing macros removed."""
    def convert(kind: str, prefix: str, text: str) -> str:
        pattern = re.compile(r"\\begin\{" + kind + r"\*?\}(?:\[[^\]]*\])?(.*?)\\end\{" + kind + r"\*?\}", re.S)
        seq = 0

        def repl(m: re.Match) -> str:
            nonlocal seq
            seq += 1
            body = m.group(1)
            lab = re.search(r"\\label\{([^}]*)\}", body)
            number = labels.get(lab.group(1), str(seq)) if lab else str(seq)
            cap = macro_arg(body, "caption")
            assert cap, f"{kind} without caption"
            body = body[:cap[0]] + body[cap[1]:]
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            body = re.sub(r"\\(centering|small|footnotesize|scriptsize)\b", "", body)
            body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
            caption = rf"\caption{{{prefix} {number}. {cap[2]}}}"
            if kind == "table":
                return "\\begin{table}\n" + caption + "\n" + body.strip() + "\n\\end{table}"
            return "\\begin{figure}\n" + body.strip() + "\n" + caption + "\n\\end{figure}"

        return pattern.sub(repl, text)

    tex = convert("table", "TABLE", tex)
    return convert("figure", "FIGURE", tex)


def algorithm_as_list(tex: str, labels: dict[str, str]) -> str:
    """pandoc has no algorithmic reader; the block becomes a numbered list
    under a numbered caption."""
    m = re.search(r"\\begin\{algorithm\}.*?\\caption\{(.*?)\}\s*\\label\{([^}]*)\}\s*"
                  r"\\begin\{algorithmic\}(?:\[\d\])?\n(.*?)\\end\{algorithmic\}\s*\\end\{algorithm\}",
                  tex, re.S)
    if not m:
        return tex
    caption, label, body = m.group(1), m.group(2), m.group(3)
    number = labels.get(label, "1")
    items = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\\Comment\{(.*?)\}", r"\\quad\\textit{(\1)}", line)
        line = line.replace(r"\Return", r"\textbf{return}")
        if line.startswith(r"\Require"):
            items.append(r"\item \textbf{Input:} " + line[len(r"\Require"):].strip())
        elif line.startswith(r"\Ensure"):
            items.append(r"\item \textbf{Output:} " + line[len(r"\Ensure"):].strip())
        elif line.startswith(r"\Statex"):
            items.append(r"\item " + line[len(r"\Statex"):].strip())
        elif line.startswith(r"\State"):
            items.append(r"\item " + line[len(r"\State"):].strip())
        elif line.startswith(r"\For{"):
            items.append(r"\item \textbf{for} " + line[len(r"\For{"):-1] + r" \textbf{do}")
        elif line.startswith(r"\EndFor"):
            items.append(r"\item \textbf{end for}")
        else:
            items.append(r"\item " + line)
    block = (f"\\noindent\\textbf{{ALGORITHM {number}.}} " + caption + "\n\n\\begin{enumerate}\n"
             + "\n".join(items) + "\n\\end{enumerate}\n")
    return tex[:m.start()] + block + tex[m.end():]


def front_matter(tex: str) -> str:
    """proper-case author line; affiliation, corresponding author and funding
    as paragraphs after the title; index terms as a labelled line."""
    tex = re.sub(r"\\uppercase\{([^}]*)\}", r"\1", tex)
    tex = tex.replace(r"\authorrefmark{1}", r"\textsuperscript{1}")
    tex = tex.replace(", AND ", ", and ").replace(" AND ", " and ")
    front = []
    for macro in (r"\\address\[\d\]\{(.*?)\}\s*\n", r"\\corresp\{(.*?)\}\s*\n", r"\\tfootnote\{(.*?)\}\s*\n"):
        m = re.search(macro, tex)
        if m:
            front.append(m.group(1).strip())
            tex = tex.replace(m.group(0), "")
    if front:
        front[0] = r"\textsuperscript{1}" + front[0]
    for macro in ("history", "doi", "markboth"):
        tex = re.sub(r"^\\" + macro + r"\{.*\n", "", tex, flags=re.M)
    tex = re.sub(r"^\\titlepgskip=.*\n", "", tex, flags=re.M)
    tex = tex.replace("\\EOD\n", "")
    tex = re.sub(r"\\begin\{keywords\}\n(.*?)\n\\end\{keywords\}",
                 lambda m: "\\noindent\\textbf{INDEX TERMS} " + m.group(1), tex, flags=re.S)
    if front:
        block = "\n\n".join("\\noindent " + f for f in front)
        tex = tex.replace("\\maketitle\n", "\\maketitle\n\n" + block + "\n\n", 1)
    return tex


def biographies_as_text(tex: str) -> str:
    """ieee biography environments become a Biographies section. pandoc puts
    the reference list last; the docx pass moves this section after it."""
    blocks = []

    def repl(m: re.Match) -> str:
        photo, name, body = m.group(1), m.group(2), m.group(3).strip()
        head = (photo + "\n\n" if photo else "") + rf"\noindent\textbf{{{name}}} " + body
        blocks.append(head)
        return ""

    tex = re.sub(r"\\begin\{IEEEbiography\}\[\{(\\includegraphics[^\]]*)\}\]\{([^}]*)\}(.*?)\\end\{IEEEbiography\}",
                 repl, tex, flags=re.S)
    tex = re.sub(r"\\begin\{IEEEbiographynophoto\}()\{([^}]*)\}(.*?)\\end\{IEEEbiographynophoto\}",
                 repl, tex, flags=re.S)
    if blocks:
        tex = tex.replace("\\end{document}", "\\section*{Biographies}\n\n" + "\n\n".join(blocks) + "\n\n\\end{document}")
    return tex


INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def inline_inputs(tex: str) -> str:
    """pandoc resolves \\input against its working directory, not the paper
    directory, and drops a file it cannot find with only a warning: the number
    macros of paper/numbers_v2.tex would vanish from the docx. Inline every
    \\input from paper/; a missing file is an error here."""
    def repl(m: re.Match) -> str:
        name = m.group(1)
        return (PAPER / (name if name.endswith(".tex") else name + ".tex")).read_text(encoding="utf-8")
    return INPUT_RE.sub(repl, tex)


def prepare(tex: str, labels: dict[str, str]) -> str:
    tex = inline_inputs(tex)
    tex = resolve_anonymous(tex)
    tex = bake_refs(tex, labels)
    tex = number_equations(tex, labels)
    tex = unwrap_resizebox(tex)
    tex = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}\s*", "", tex)  # pandoc reads these as cell text
    tex = wrap_bracket_cells(tex)
    tex = number_floats(tex, labels)
    tex = algorithm_as_list(tex, labels)
    tex = tex.replace(r"\textsc{AdamStep}", r"\mathrm{AdamStep}")
    tex = front_matter(tex)
    tex = biographies_as_text(tex)
    return tex


# ---------------------------------------------------------------- pandoc

def run_pandoc(prepared: Path, out: Path, csl: Path | None, reference_doc: Path | None) -> str:
    import pypandoc
    args = ["--from=latex", "--citeproc", f"--bibliography={BIB}", f"--resource-path={PAPER}",
            "--metadata=reference-section-title=References"]
    note = ""
    if csl and csl.exists():
        args.append(f"--csl={csl}")
    else:
        note = " (author-year citations: no csl file)"
    if reference_doc:
        args.append(f"--reference-doc={reference_doc}")
    pypandoc.convert_file(str(prepared), "docx", outputfile=str(out), extra_args=args)
    return "ok" + note


# ---------------------------------------------------------------- docx pass

def roman(n: int) -> str:
    out = ""
    for value, sym in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                       (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += sym
            n -= value
    return out


def set_font(style, name="Times New Roman", size=None, bold=None, italic=None, color=None):
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = RGBColor(*color)


def style_document_plain(doc) -> None:
    """no template: letter page, one-inch margins, times, black headings."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, Inches
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(1))
    styles = doc.styles
    body = dict(size=11)
    spec = {
        "Normal": body, "Body Text": body, "First Paragraph": body, "Compact": body, "Block Text": body,
        "Title": dict(size=18, bold=True), "Author": dict(size=11),
        "Abstract Title": dict(size=11, bold=True), "Abstract": dict(size=10),
        "Heading 1": dict(size=12, bold=True, italic=False, color=(0, 0, 0)),
        "Heading 2": dict(size=11, bold=True, italic=True, color=(0, 0, 0)),
        "Heading 3": dict(size=11, bold=False, italic=True, color=(0, 0, 0)),
        "Table Caption": dict(size=9, bold=True), "Image Caption": dict(size=9, bold=False, italic=False),
        "Bibliography": dict(size=9.5), "Source Code": dict(name="Consolas", size=9),
    }
    for name, kw in spec.items():
        try:
            set_font(styles[name], **({"name": "Times New Roman"} | kw))
        except KeyError:
            continue
    for name in ("Body Text", "First Paragraph", "Abstract"):
        try:
            pf = styles[name].paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.space_after = Pt(6)
            pf.line_spacing = 1.0
        except KeyError:
            continue
    for name in ("Title", "Author", "Table Caption", "Image Caption"):
        try:
            styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except KeyError:
            continue
    for name in ("Heading 1", "Heading 2", "Heading 3", "Table Caption"):
        try:
            styles[name].paragraph_format.keep_with_next = True
        except KeyError:
            continue


def prepend_run(p, text: str) -> None:
    """a new run with `text` at the very start of the paragraph, before any
    math object, copying the first run's formatting."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    import copy
    r = OxmlElement("w:r")
    first = p._p.find(qn("w:r"))
    if first is not None and first.find(qn("w:rPr")) is not None:
        r.append(copy.deepcopy(first.find(qn("w:rPr"))))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    ppr = p._p.find(qn("w:pPr"))
    p._p.insert(1 if ppr is not None else 0, r)


def disable_list_numbering(p) -> None:
    """numId 0 on the paragraph overrides any numbering the style carries."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    ppr = p._p.get_or_add_pPr()
    for old in ppr.findall(qn("w:numPr")):
        ppr.remove(old)
    numpr = OxmlElement("w:numPr")
    for tag, val in (("w:ilvl", "0"), ("w:numId", "0")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        numpr.append(el)
    ppr.append(numpr)


def number_headings(doc) -> None:
    """ieee numbering in the heading text itself: I. UPPERCASE, A. Title, 1) Title.
    any numbering the target style carries is switched off per paragraph so
    the letters cannot double up."""
    h1 = h2 = h3 = 0
    for p in doc.paragraphs:
        name = p.style.name
        if name not in ("Heading 1", "Heading 2", "Heading 3"):
            continue
        title = p.text.strip()
        if name == "Heading 1":
            if title in UNNUMBERED:
                prefix = ""
            else:
                h1 += 1
                h2 = h3 = 0
                prefix = f"{roman(h1)}. "
            for r in p.runs:
                r.text = r.text.upper()
        elif name == "Heading 2":
            h2 += 1
            h3 = 0
            prefix = f"{chr(ord('A') + h2 - 1)}. "
        else:
            h3 += 1
            prefix = f"{h3}) "
        disable_list_numbering(p)
        if prefix:
            prepend_run(p, prefix)


def has_style(doc, name: str) -> bool:
    try:
        doc.styles[name]
        return True
    except KeyError:
        return False


def is_equation_paragraph(p) -> bool:
    return p._p.find(f".//{{{M_NS}}}oMathPara") is not None and not p.text.strip()


def is_list_item(p) -> bool:
    ppr = p._p.pPr
    return ppr is not None and ppr.numPr is not None


def apply_template_styles(doc) -> list[str]:
    """map pandoc's output onto the journal template's styles. pandoc's title,
    author and abstract-label paragraphs come out as Normal when the template
    lacks those styles, so the front matter is found by position: title,
    authors, the word Abstract, the abstract, the index terms, then the
    affiliation lines up to the first heading. returns the template styles
    that were missing, so the caller can say so."""
    from docx.oxml.ns import qn
    needed = ("Paper Title", "AU", "PI", "Abstract", "IT", "H1", "H2_First", "H2_Cont", "H3", "PARA",
              "PARA_Indent", "Fig Caption", "Table Caption", "References", "AU_Bios", "Equation", "DOP", "DOI")
    missing = sorted(v for v in needed if not has_style(doc, v))

    def restyle(p, name):
        if has_style(doc, name):
            p.style = doc.styles[name]

    paras = doc.paragraphs
    first_heading = next((i for i, p in enumerate(paras) if p.style.name in ("Heading 1", "H1")), len(paras))
    title_p, author_p = paras[0], paras[1]
    restyle(title_p, "Paper Title")
    restyle(author_p, "AU")
    label = next((p for p in paras[2:first_heading] if p.text.strip() == "Abstract"), None)
    abstract_p = next((p for p in paras[2:first_heading] if p.style.name == "Abstract"), None)
    it_p = next((p for p in paras[2:first_heading] if p.text.startswith("INDEX TERMS")), None)
    if it_p is not None:
        restyle(it_p, "IT")
        # affiliation, corresponding author, funding: after the index terms in
        # pandoc's order; the journal wants them under the author line
        it_i = paras.index(it_p)
        anchor = author_p._p
        block = paras[it_i + 1:first_heading]
        for k, p in enumerate(block):
            # the template spaces only the last line of the block
            restyle(p, "PI" if k == len(block) - 1 or not has_style(doc, "PI_No Space") else "PI_No Space")
            anchor.addnext(p._p)
            anchor = p._p
    if abstract_p is not None:
        if label is not None:
            run = abstract_p._p.makeelement(qn("w:r"), {})
            abstract_p._p.insert(1 if abstract_p._p.pPr is not None else 0, run)
            from docx.text.run import Run
            r = Run(run, abstract_p)
            r.text = "ABSTRACT "
            r.bold = True
            label._p.getparent().remove(label._p)
    # the template's order: date line, then doi line, then the title
    if has_style(doc, "DOP"):
        title_p.insert_paragraph_before(
            "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.", style="DOP")
    if has_style(doc, "DOI"):
        title_p.insert_paragraph_before("Digital Object Identifier 10.1109/ACCESS.0000.0000000", style="DOI")

    body_styles = {"Body Text", "First Paragraph", "Compact", "Block Text", "Normal"}
    mapping = {"Heading 1": "H1", "Heading 3": "H3", "Image Caption": "Fig Caption",
               "Table Caption": "Table Caption", "Bibliography": "References"}
    prev_heading, prev_body = None, False
    paras = doc.paragraphs  # the front matter moved; find the first heading again
    first_heading = next((i for i, p in enumerate(paras) if p.style.name in ("Heading 1", "H1")), len(paras))
    for p in paras[first_heading:]:
        name = p.style.name
        target = None
        if name == "Heading 2":
            target = "H2_First" if prev_heading == "Heading 1" else "H2_Cont"
        elif name in mapping:
            target = mapping[name]
        elif is_equation_paragraph(p):
            target = "Equation"
        elif name in body_styles and is_list_item(p):
            target = "PARA"  # body font; the paragraph keeps its own list numbering
        elif name in body_styles:
            target = "PARA_Indent" if prev_body else "PARA"
        if name.startswith("Heading"):
            prev_heading = name
        prev_body = target in ("PARA", "PARA_Indent")
        if target:
            restyle(p, target)
    return missing


def set_table_borders(table) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")


def style_tables(doc) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
    for t in doc.tables:
        set_table_borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = True
        ncol = len(t.columns)
        size = 8 if ncol <= 8 else (7 if ncol <= 11 else 6.5)
        for ri, row in enumerate(t.rows):
            for tc in row._tr.tc_lst:
                for p_el in tc.findall(qn_w("p")):
                    from docx.text.paragraph import Paragraph
                    p = Paragraph(p_el, t)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.size = Pt(size)
                        r.font.name = "Times New Roman"
                        if ri == 0:
                            r.font.bold = True


def qn_w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def text_width_in(doc) -> float:
    s = doc.sections[0]
    if s.page_width is None or s.left_margin is None or s.right_margin is None:
        return 6.5
    return (s.page_width - s.left_margin - s.right_margin) / 914400


def scale_images(doc) -> None:
    from docx.shared import Inches
    max_w = text_width_in(doc)
    for shape in doc.inline_shapes:
        if shape.width is None:
            continue
        if shape.width.inches > max_w:
            ratio = max_w / shape.width.inches
            shape.width = Inches(max_w)
            shape.height = int(shape.height * ratio)


def fix_figure_paragraphs(doc) -> int:
    """a paragraph whose style fixes the line height clips an inline picture
    to one text line. give every picture paragraph single spacing, centring
    and a little air, whatever style it inherited."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml.ns import qn
    from docx.shared import Pt
    n = 0
    for p in doc.paragraphs:
        if p._p.find(".//" + qn("w:drawing")) is None:
            continue
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        pf.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        n += 1
    return n


def move_biographies(doc, template: bool) -> bool:
    """the Biographies section sits before the reference list pandoc appends;
    move it after. with the template the heading goes and the entries take
    the AU_Bios style, as in the journal."""
    body = doc.element.body
    paras = doc.paragraphs
    start = next((i for i, p in enumerate(paras) if p.style.name in ("Heading 1", "H1") and p.text.strip().upper() == "BIOGRAPHIES"), None)
    refs = next((i for i, p in enumerate(paras) if p.style.name in ("Heading 1", "H1") and p.text.strip().upper() == "REFERENCES"), None)
    if start is None or refs is None or refs < start:
        return False
    block = paras[start:refs]
    sect = body.find(qn_w("sectPr"))
    for p in block:
        body.remove(p._p)
        if sect is not None:
            sect.addprevious(p._p)
        else:
            body.append(p._p)
    if template and has_style(doc, "AU_Bios"):
        block[0]._p.getparent().remove(block[0]._p)
        for p in block[1:]:
            p.style = doc.styles["AU_Bios"]
    return True


def add_page_numbers(doc) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for section in doc.sections:
        if "PAGE" in section.footer._element.xml:
            continue
        p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)


def fix_docx(path: Path, template: bool) -> dict:
    from docx import Document
    doc = Document(str(path))
    number_headings(doc)
    missing: list[str] = []
    if template:
        missing = apply_template_styles(doc)
    else:
        style_document_plain(doc)
    style_tables(doc)
    scale_images(doc)
    fix_figure_paragraphs(doc)
    moved = move_biographies(doc, template)
    add_page_numbers(doc)
    doc.save(str(path))
    return {"tables": len(doc.tables), "images": len(doc.inline_shapes), "biographies_moved": moved,
            "headings": sum(p.style.name in ("Heading 1", "Heading 2", "Heading 3", "H1", "H2_First", "H2_Cont", "H3")
                            for p in doc.paragraphs),
            "missing_template_styles": missing}


# ---------------------------------------------------------------- checks

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def tex_table_numbers(prepared: str) -> list[list[list[str]]]:
    """numbers per row per tabular, in order, from the prepared tex."""
    tables = []
    for m in re.finditer(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", prepared, re.S):
        rows = []
        for line in m.group(1).split("\n"):
            s = line.strip()
            if not s or s.startswith("%") or s.startswith("\\toprule") or s.startswith("\\midrule") \
                    or s.startswith("\\bottomrule") or s.startswith("\\cmidrule") or "&" not in s:
                continue
            s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", s)  # keep the cell text, drop the span spec
            s = re.sub(r"\^\{\*+\}", "", s)  # significance stars
            s = s.replace("\\,", "").replace("$", "").replace("~", " ")
            rows.append(NUM.findall(re.sub(r"\\[a-zA-Z]+", " ", s)))
        tables.append(rows)
    return tables


def docx_table_numbers(path: Path) -> list[list[list[str]]]:
    from docx import Document
    doc = Document(str(path))
    out = []
    for t in doc.tables:
        rows = []
        for row in t.rows:
            # read the w:t and m:t nodes directly: python-docx overrides .text
            # on run elements, which makes itertext() repeat every value
            text = " ".join(" ".join((node.text or "") for node in tc.iter(qn_w("t"), f"{{{M_NS}}}t"))
                            for tc in row._tr.tc_lst)
            rows.append(NUM.findall(text.replace("\u2212", "-").replace("\u2006", "").replace("\u2009", "")))
        out.append(rows)
    return out


def check_tables(prepared: str, path: Path) -> list[str]:
    """every number in every tex table row must appear, in order, in the same
    docx row. returns human-readable problems; empty means the tables match."""
    tex_tabs, docx_tabs = tex_table_numbers(prepared), docx_table_numbers(path)
    problems = []
    if len(tex_tabs) != len(docx_tabs):
        problems.append(f"table count: tex {len(tex_tabs)} vs docx {len(docx_tabs)}")
    for ti, (tt, dt) in enumerate(zip(tex_tabs, docx_tabs), start=1):
        if len(tt) != len(dt):
            problems.append(f"table {ti}: row count tex {len(tt)} vs docx {len(dt)}")
        for ri, (tr, dr) in enumerate(zip(tt, dt), start=1):
            if tr != dr:
                problems.append(f"table {ti} row {ri}: tex {tr} vs docx {dr}")
    return problems


def word_to_pdf(docx_path: Path, pdf_path: Path) -> int:
    """print through word (com automation); returns the page count."""
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(docx_path.resolve()))
        try:
            doc.ExportAsFixedFormat(str(pdf_path.resolve()), 17)  # wdExportFormatPDF
            return int(doc.ComputeStatistics(2))  # wdStatisticPages
        finally:
            doc.Close(False)
    finally:
        word.Quit()


# ---------------------------------------------------------------- main

def build(out: Path, csl: Path | None, reference_doc: Path | None, pdf: bool) -> int:
    if not SRC.exists() or not AUX.exists():
        print("[docx] run `python scripts/build_ieee.py --build` first (need the tex and its aux)")
        return 1
    template = reference_doc if reference_doc and reference_doc.exists() else None
    labels = aux_numbers(AUX)
    prepared = prepare(SRC.read_text(encoding="utf-8"), labels)
    BUILD.mkdir(exist_ok=True)
    prepared_path = BUILD / "paper2_ieeeaccess_docx.tex"
    prepared_path.write_text(prepared, encoding="utf-8")
    status = run_pandoc(prepared_path, out, csl, template)
    info = fix_docx(out, template is not None)
    problems = check_tables(prepared, out)
    print(f"[docx] pandoc {status}; template={'yes' if template else 'no'} tables={info['tables']} "
          f"images={info['images']} headings={info['headings']} biographies_moved={info['biographies_moved']}")
    if info["missing_template_styles"]:
        print("[docx] template styles not found:", ", ".join(info["missing_template_styles"]))
    for p in problems:
        print("[docx] TABLE MISMATCH:", p)
    if pdf:
        pdf_path = out.with_suffix(".pdf")
        try:
            pages = word_to_pdf(out, pdf_path)
            print(f"[docx] word print: {pdf_path.name} {pages} pages")
        except Exception as e:  # noqa: BLE001
            print(f"[docx] word print failed: {type(e).__name__}: {e}")
    print(f"[docx] wrote {out}")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csl", type=Path, default=DEFAULT_CSL)
    ap.add_argument("--reference-doc", type=Path, default=DEFAULT_TEMPLATE,
                    help="official word template; pandoc takes its page setup and styles")
    ap.add_argument("--pdf", action="store_true", help="also print to pdf through word for checking")
    a = ap.parse_args(argv)
    return build(a.out, a.csl, a.reference_doc, a.pdf)


if __name__ == "__main__":
    sys.exit(main())
