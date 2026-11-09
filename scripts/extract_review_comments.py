"""Extract every reviewer comment of the decision letter into responses.yaml.

    python scripts/extract_review_comments.py

Reads submission/ieee_access/round1_review/decision_letter_2026-10-28.txt
(gitignored: the reviews are confidential) and writes the skeleton
submission/ieee_access/round1_review/responses.yaml with one entry per comment,
the concern copied verbatim, and empty response, action, evidence and section
fields to fill. Lettered sub-points ((a), (b), ...) become their own entries so
that each receives its own response; the "Additional Questions" assessments are
kept as entries of kind "assessment". An existing file is never overwritten:
new comments are appended and filled entries are left untouched.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LETTER = ROOT / "submission/ieee_access/round1_review/decision_letter_2026-10-28.txt"
OUT = ROOT / "submission/ieee_access/round1_review/responses.yaml"
HEADINGS = {"MAJOR COMMENTS", "PRESENTATION", "MINOR COMMENTS"}


def _blocks(text: str) -> dict[int, tuple[str, str]]:
    """reviewer -> (comments text, additional-questions text)."""
    parts = re.split(r"^Reviewer: (\d+)\s*$", text, flags=re.M)
    out = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        c = body.split("Comments:", 1)[1] if "Comments:" in body else body
        comments, _, questions = c.partition("Additional Questions:")
        out[int(parts[i])] = (comments, questions)
    return out


def parse_comments(rev: int, text: str) -> list[dict]:
    entries, cur, section = [], None, "major"
    preamble: list[str] = []

    def close():
        if cur is not None:
            cur["concern"] = "\n".join(cur.pop("_lines")).strip()
            entries.append(cur)

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if s in HEADINGS:
            close()
            cur = None
            section = s.split()[0].lower()
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        sub = re.match(r"^\(([a-z])\)\s+(.*)$", s)
        if m and not raw.startswith("  "):
            close()
            cur = {"id": f"R{rev}.{m.group(1)}", "reviewer": rev, "number": int(m.group(1)),
                   "section": section, "kind": "comment", "_lines": [s]}
        elif sub and cur is not None:
            close()
            parent = re.match(r"R\d+\.(\d+)", cur["id"]).group(1) if "." in cur["id"] else "?"
            parent_id = f"R{rev}.{parent}"
            title = next((e["concern"].splitlines()[0] for e in entries if e["id"] == parent_id), "")
            cur = {"id": f"{parent_id}{sub.group(1)}", "reviewer": rev, "number": int(parent),
                   "section": section, "kind": "comment", "parent_title": title, "_lines": [s]}
        elif s.startswith("- ") and section == "minor":
            close()
            n = sum(1 for e in entries if e["section"] == "minor") + 1
            cur = {"id": f"R{rev}.m{n}", "reviewer": rev, "number": n, "section": "minor",
                   "kind": "comment", "_lines": [s[2:]]}
        elif cur is not None:
            cur["_lines"].append(s)
        elif s:
            preamble.append(s)
    close()
    # a numbered heading followed only by lettered sub-points is a title, not a comment
    ids = {e["id"] for e in entries}
    for e in entries:
        if any(i.startswith(e["id"]) and i != e["id"] and i[len(e["id"]):].isalpha() for i in ids):
            e["kind"] = "heading"
    if preamble:
        entries.insert(0, {"id": f"R{rev}.summary", "reviewer": rev, "number": 0, "section": "summary",
                           "kind": "summary", "concern": "\n".join(preamble)})
    return entries


def parse_questions(rev: int, text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"^(\d)\)\s+([^\n]*?\?):\s*(.*?)(?=^\d[a-z]?\)\s|\Z)", text, flags=re.M | re.S):
        answer = m.group(3).strip()
        if not answer or answer in ("No", "NA", "Yes"):
            continue
        out.append({"id": f"R{rev}.q{m.group(1)}", "reviewer": rev, "number": int(m.group(1)),
                    "section": "assessment", "kind": "assessment", "question": m.group(2).strip(),
                    "concern": answer})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letter", default=str(LETTER))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    text = Path(args.letter).read_text(encoding="utf-8")
    entries = []
    for rev, (comments, questions) in sorted(_blocks(text).items()):
        entries += parse_comments(rev, comments) + parse_questions(rev, questions)
    for e in entries:
        e.update({"response": "", "action": "", "manuscript_sections": [], "evidence": [], "status": "open"})
    out = Path(args.out)
    if out.exists():
        old = yaml.safe_load(out.read_text(encoding="utf-8"))["entries"]
        known = {e["id"] for e in old}
        entries = old + [e for e in entries if e["id"] not in known]
    doc = {"letter": Path(args.letter).name, "manuscript": "Access-2026-49478",
           "fields": {"response": "reply to the concern", "action": "what was changed, where",
                      "manuscript_sections": "section, table and figure labels in the revised paper",
                      "evidence": "result files (CSV/JSON) that carry the numbers quoted",
                      "status": "open | answered"},
           "entries": entries}
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
    kinds = {}
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"[responses] {len(entries)} entries {kinds} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
