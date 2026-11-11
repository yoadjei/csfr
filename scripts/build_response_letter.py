"""Build the response to the reviewers from responses.yaml.

    python scripts/build_response_letter.py [--draft] [--docx]

Reads submission/ieee_access/round1_review/responses.yaml (gitignored: the
reviews are confidential), one entry per reviewer comment with the concern
copied verbatim from the decision letter, and writes
submission/ieee_access/round2/01_response_to_reviewers.md and, with --docx,
the Word file through pandoc. Each comment is rendered as concern, response
and action; the labels in `manuscript_sections` are resolved to the numbers
they carry in the revised IEEE build (paper/paper2_ieeeaccess.aux), so a
reference to "Section IV-A" or "Table 2" cannot go stale.

The build refuses, and lists why, when a comment is not answered, has an empty
response or action, or names a label the revised paper does not define.
--draft renders anyway and marks every such entry, for review while the
responses are being written.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML = ROOT / "submission" / "ieee_access" / "round1_review" / "responses.yaml"
AUX = ROOT / "paper" / "paper2_ieeeaccess.aux"
OUT_DIR = ROOT / "submission" / "ieee_access" / "round2"
OUT_MD = OUT_DIR / "01_response_to_reviewers.md"
OUT_DOCX = OUT_DIR / "01_response_to_reviewers.docx"

NUMBERS = ROOT / "paper" / "numbers_v2.tex"
PLACEHOLDER = r"\textbf{[pending]}"

KIND_WORD = {"sec": "Section", "tab": "Table", "fig": "Fig.", "alg": "Algorithm", "prop": "Proposition"}

# In a response or action, [[label]] is the number the revised build gives the
# label, and {{macro}} is the value scripts/render_numbers_v2.py gave the macro.
# A reference or a result typed by hand would go stale when the paper moves.
INLINE_LABEL = re.compile(r"\[\[([A-Za-z]+:[^\]\s]+)\]\]")
INLINE_NUMBER = re.compile(r"\{\{([A-Za-z]+)\}\}")
TYPED_REF = re.compile(r"\b(?:Sections?|Tables?|Figs?\.|Figures?|Eqs?\.|Propositions?|Algorithms?)\s*\(?[0-9IVX]+")


def macro_values(path: Path) -> dict[str, str]:
    """macro name -> value, from the generated numbers file."""
    return dict(re.findall(r"^\\providecommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",
                           path.read_text(encoding="utf-8"), re.M))


def _math(value: str) -> str:
    """A macro value for Markdown: TeX notation goes in $...$ for pandoc."""
    return f"${value}$" if any(c in value for c in "\\^{") else value


def resolve(text: str, labels: dict[str, str], values: dict[str, str]) -> str:
    """Replace [[label]] and {{macro}}; an unresolvable item stays visible."""
    text = INLINE_LABEL.sub(lambda m: cite_label(m.group(1), labels) if m.group(1) in labels
                            else f"**[?{m.group(1)}]**", text)
    return INLINE_NUMBER.sub(lambda m: _math(values[m.group(1)])
                             if values.get(m.group(1), PLACEHOLDER) != PLACEHOLDER
                             else f"**[?{m.group(1)}]**", text)


def aux_labels(aux: Path) -> dict[str, str]:
    """label -> visible number, as LaTeX prints it in the revised build."""
    raw = re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="replace"))
    return {k: re.sub(r"\\mbox\s*\{?", "", v).strip("{} ") for k, v in raw}


def cite_label(label: str, labels: dict[str, str]) -> str:
    """'sec:problem' -> 'Section III'; 'eq:gap' -> 'Eq. (5)'."""
    number = labels[label]
    prefix = label.split(":", 1)[0]
    if prefix == "eq":
        return f"Eq. ({number})"
    return f"{KIND_WORD.get(prefix, prefix)} {number}"


def problems(entry: dict, labels: dict[str, str], values: dict[str, str] | None = None) -> list[str]:
    """Why an entry is not ready to send; empty when it is."""
    if entry["kind"] != "comment":
        return []
    values = values or {}
    out = []
    if entry.get("status") != "answered":
        out.append("status is not 'answered'")
    for field in ("response", "action"):
        text = str(entry.get(field) or "")
        if not text.strip():
            out.append(f"empty {field}")
        out += [f"unknown label {lab}" for lab in INLINE_LABEL.findall(text) if lab not in labels]
        for name in INLINE_NUMBER.findall(text):
            if name not in values:
                out.append(f"unknown number {name}")
            elif values[name] == PLACEHOLDER:
                out.append(f"pending number {name}")
        out += [f"hand-typed reference '{ref}'" for ref in TYPED_REF.findall(text)]
    out += [f"unknown label {lab}" for lab in entry.get("manuscript_sections") or [] if lab not in labels]
    if not entry.get("manuscript_sections"):
        out.append("no manuscript location")
    return out


def _quote(text: str) -> str:
    return "\n".join("> " + ln if ln.strip() else ">" for ln in text.strip().splitlines())


def render(doc: dict, labels: dict[str, str], draft: bool, values: dict[str, str] | None = None) -> str:
    values = values or {}
    entries = doc["entries"]
    lines = [f"# Response to the reviewers: manuscript {doc['manuscript']}", ""]
    preamble = str(doc.get("preamble") or "").strip()
    if not preamble and not draft:
        raise ValueError("responses.yaml has no preamble")
    lines += [preamble or "**[DRAFT: preamble not written]**", ""]
    for reviewer in sorted({e["reviewer"] for e in entries}):
        lines += [f"## Reviewer {reviewer}", ""]
        for e in (x for x in entries if x["reviewer"] == reviewer):
            if e["kind"] == "heading":
                lines += [f"### {e['concern'].strip()}", ""]
                continue
            title = {"summary": "Summary", "assessment": e.get("question", "Assessment")}.get(
                e["kind"], f"Comment {e['id']}")
            lines += [f"**{title}**", "", _quote(e["concern"]), ""]
            if e.get("response"):
                lines += [f"**Response.** {resolve(e['response'].strip(), labels, values)}", ""]
            if e.get("action"):
                where = [cite_label(lab, labels) for lab in e.get("manuscript_sections") or [] if lab in labels]
                suffix = f" ({'; '.join(where)})" if where else ""
                lines += [f"**Action.** {resolve(e['action'].strip(), labels, values)}{suffix}", ""]
            issues = problems(e, labels, values)
            if issues and draft:
                lines += [f"**[DRAFT: {'; '.join(issues)}]**", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true", help="render even with open entries, marking them")
    ap.add_argument("--docx", action="store_true", help="also write the Word file through pandoc")
    args = ap.parse_args(argv)
    doc = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    labels, values = aux_labels(AUX), macro_values(NUMBERS)
    open_items = {e["id"]: p for e in doc["entries"] if (p := problems(e, labels, values))}
    if open_items and not args.draft:
        for k, p in open_items.items():
            print(f"[response] {k}: {'; '.join(p)}")
        print(f"[response] {len(open_items)} comment(s) not ready; use --draft to render anyway")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(doc, labels, args.draft, values), encoding="utf-8")
    print(f"[response] wrote {OUT_MD.relative_to(ROOT)} ({len(open_items)} open)")
    if args.docx:
        import pypandoc
        pypandoc.convert_file(str(OUT_MD), "docx", outputfile=str(OUT_DOCX))
        print(f"[response] wrote {OUT_DOCX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
