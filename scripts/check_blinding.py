"""Fail if a blinded build leaks author identity.

Run after the manuscript builds. When the blinding switch is set to anonymous,
nothing that identifies an author may survive into the released PDF, and the
places it survives are not always the obvious ones: hyperref writes the author
list into PDF metadata, so a document whose visible title block is blank can
still carry the names in a field no one reads.

The leak vocabulary is not hardcoded. It is read out of the named branch of the
switch in the .tex, so adding an author or changing an affiliation extends the
check automatically instead of silently narrowing it.

Exit 0 when the document is in named mode (nothing to check) or when a blinded
document is clean. Exit 1 on any leak.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
PDFS = [ROOT / "paper" / "paper2_reconstruction.pdf",
        ROOT / "paper" / "paper2_supplementary.pdf"]

# tokens that identify the work regardless of who the authors are.
ALWAYS = ["github.com", "yoadjei"]

# an affiliation is mostly generic scaffolding. "Department", "University" and
# "Science" appear in any bibliography, so matching them reports the reference
# list rather than a leak. only the words that actually single out a place are
# worth checking, and these are the ones that never do.
GENERIC = {
    "department", "departments", "school", "faculty", "institute", "centre",
    "center", "laboratory", "lab", "university", "college", "computer",
    "computing", "science", "sciences", "technology", "engineering",
    "research", "and", "of", "the", "for", "national", "state", "city",
}


def named_branch(tex: str) -> str:
    """The \\else arm of the frontmatter blinding switch, which holds the real
    author block. Everything identifying lives here."""
    m = re.search(r"\\ifanonymous(.*?)\\else(.*?)\\fi", tex, re.S)
    return m.group(2) if m else ""


def strip_tex(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+", " ", s)).strip()


def identity_tokens(tex: str) -> list[str]:
    """Distinctive strings only. Whole-word matched later, so a surname cannot
    fire from inside an unrelated citation."""
    branch = named_branch(tex)
    tokens: set[str] = set(ALWAYS)

    def fields(macro: str) -> list[str]:
        return [strip_tex(f) for f in
                re.findall(rf"\\{macro}(?:\[[^\]]*\])?\{{([^}}]*)\}}", branch)]

    # every part of a personal name, and the email whole and in parts. the
    # author field capture stops at the first brace, so a trailing \corref{cor1}
    # leaves "cor" behind; requiring an initial capital drops that without
    # dropping any real name part.
    for name in fields("author"):
        tokens.update(w for w in re.split(r"[^A-Za-z-]+", name)
                      if len(w) >= 3 and w[:1].isupper())
    for mail in fields("ead"):
        tokens.add(mail)
        tokens.update(p for p in re.split(r"[@.]", mail) if len(p) >= 4)
    # from the affiliation, keep only what names a specific institution or place.
    for addr in fields("address"):
        tokens.update(w for w in re.split(r"[^A-Za-z-]+", addr)
                      if len(w) >= 4 and w.lower() not in GENERIC)
    return sorted(tokens)


def pdf_text_and_metadata(path: Path) -> tuple[str, str]:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    meta = " ".join(f"{k}={v}" for k, v in (reader.metadata or {}).items())
    return text, meta


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    if re.search(r"^\s*\\anonymousfalse", tex, re.M):
        print("[blinding] named mode; no check required")
        return 0
    if not re.search(r"^\s*\\anonymoustrue", tex, re.M):
        print("[blinding] switch set to neither value")
        return 1

    tokens = identity_tokens(tex)
    if not tokens:
        print("[blinding] could not read the author block; refusing to pass")
        return 1
    print(f"[blinding] anonymous mode, checking {len(tokens)} identity tokens")

    leaks = 0
    for pdf in PDFS:
        if not pdf.exists():
            print(f"[blinding] {pdf.name} not built")
            return 1
        text, meta = pdf_text_and_metadata(pdf)
        for where, blob in (("text", text), ("metadata", meta)):
            # collapse whitespace so a name broken across a line still matches,
            # and match whole words so "Tei" cannot fire inside "Bruckstein".
            blob = re.sub(r"\s+", " ", blob)
            for tok in tokens:
                m = re.search(rf"(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])",
                              blob, re.I)
                if m:
                    lo = max(0, m.start() - 50)
                    print(f"[blinding] LEAK {pdf.name} {where}: {tok!r} "
                          f"near ...{blob[lo:m.end() + 50]!r}...")
                    leaks += 1

    if leaks:
        print(f"[blinding] {leaks} leak(s); build is not safely anonymous")
        return 1
    print("[blinding] clean: no author identity in text or metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
