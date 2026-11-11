"""Response to the reviewers: labels resolve to the revised paper's numbers,
an unready comment blocks the build, and every reviewer comment appears
exactly once. The real responses.yaml is confidential and gitignored, so the
behavioural tests use a synthetic one; the coverage test runs only where the
real file exists."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("resp", ROOT / "scripts" / "build_response_letter.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AUX = r"""\newlabel{sec:problem}{{III}{3}{Problem Formulation}{section.8}{}}
\newlabel{sec:framework:solver}{{\mbox {IV-A}}{4}{Solver}{subsection.9}{}}
\newlabel{tab:constraints}{{1}{3}{caption}{table.19}{}}
\newlabel{eq:gap}{{5}{4}{x}{equation.24}{}}
"""


def _doc(**over):
    comment = {"id": "R1.1", "reviewer": 1, "number": 1, "section": "major", "kind": "comment",
               "concern": "1. The solver is not convergent.", "response": "It now is.",
               "action": "Rewrote the solver section.",
               "manuscript_sections": ["sec:framework:solver", "eq:gap"], "evidence": [],
               "status": "answered"}
    comment.update(over)
    return {"manuscript": "Access-2026-49478", "preamble": "We thank the reviewers.",
            "entries": [{"id": "R1.summary", "reviewer": 1, "number": 0, "kind": "summary",
                         "concern": "Summary text."}, comment]}


def test_labels_resolve_to_printed_numbers(tmp_path):
    m = _mod()
    (tmp_path / "a.aux").write_text(AUX, encoding="utf-8")
    labels = m.aux_labels(tmp_path / "a.aux")
    assert m.cite_label("sec:framework:solver", labels) == "Section IV-A"
    assert m.cite_label("tab:constraints", labels) == "Table 1"
    assert m.cite_label("eq:gap", labels) == "Eq. (5)"


def test_ready_comment_renders_concern_response_and_action(tmp_path):
    m = _mod()
    (tmp_path / "a.aux").write_text(AUX, encoding="utf-8")
    labels = m.aux_labels(tmp_path / "a.aux")
    out = m.render(_doc(), labels, draft=False)
    assert "> 1. The solver is not convergent." in out
    assert "**Response.** It now is." in out
    assert "**Action.** Rewrote the solver section. (Section IV-A; Eq. (5))" in out
    assert "DRAFT" not in out


@pytest.mark.parametrize("over,reason", [
    ({"status": "open"}, "status is not 'answered'"),
    ({"response": ""}, "empty response"),
    ({"manuscript_sections": ["sec:gone"]}, "unknown label sec:gone"),
    ({"manuscript_sections": []}, "no manuscript location"),
])
def test_unready_comment_is_reported(tmp_path, over, reason):
    m = _mod()
    (tmp_path / "a.aux").write_text(AUX, encoding="utf-8")
    labels = m.aux_labels(tmp_path / "a.aux")
    entry = _doc(**over)["entries"][1]
    assert reason in m.problems(entry, labels)
    assert "DRAFT" in m.render(_doc(**over), labels, draft=True)


NUMBERS = "\n".join([r"% GENERATED", r"\providecommand{\csfrTol}{10^{-6}}",
                     r"\providecommand{\buPooledMin}{0.80}",
                     r"\providecommand{\fallbackDictCFoneMin}{\textbf{[pending]}}"]) + "\n"


def _inputs(tmp_path):
    m = _mod()
    (tmp_path / "a.aux").write_text(AUX, encoding="utf-8")
    (tmp_path / "n.tex").write_text(NUMBERS, encoding="utf-8")
    return m, m.aux_labels(tmp_path / "a.aux"), m.macro_values(tmp_path / "n.tex")


def test_inline_labels_and_numbers_resolve_from_the_build(tmp_path):
    """A section number or a result typed into a response goes stale when the
    paper moves; [[label]] and {{macro}} are resolved at build time instead."""
    m, labels, values = _inputs(tmp_path)
    assert values["csfrTol"] == "10^{-6}" and values["fallbackDictCFoneMin"] == m.PLACEHOLDER
    doc = _doc(response="The solver of [[sec:framework:solver]] stops at {{csfrTol}}; AUROC {{buPooledMin}}.",
               action="Rewrote [[tab:constraints]].")
    out = m.render(doc, labels, draft=False, values=values)
    assert "**Response.** The solver of Section IV-A stops at $10^{-6}$; AUROC 0.80." in out
    assert "**Action.** Rewrote Table 1. (Section IV-A; Eq. (5))" in out
    assert m.problems(doc["entries"][1], labels, values) == []


@pytest.mark.parametrize("response,reason", [
    ("Rate {{fallbackDictCFoneMin}} percent.", "pending number fallbackDictCFoneMin"),
    ("Rate {{noSuchMacro}}.", "unknown number noSuchMacro"),
    ("See [[sec:gone]].", "unknown label sec:gone"),
    ("Section IV-A now states it.", "hand-typed reference 'Section IV'"),
    ("Proposition 2 adds an identity.", "hand-typed reference 'Proposition 2'"),
    ("The cap is shown in Eq. (3).", "hand-typed reference 'Eq. (3'"),
])
def test_unresolved_or_hand_typed_items_in_the_text_block(tmp_path, response, reason):
    m, labels, values = _inputs(tmp_path)
    entry = _doc(response=response)["entries"][1]
    assert reason in m.problems(entry, labels, values)


def test_every_comment_in_the_letter_has_exactly_one_entry():
    m = _mod()
    if not m.YAML.exists():
        pytest.skip("responses.yaml is confidential and not in this checkout")
    import re
    import yaml
    doc = yaml.safe_load(m.YAML.read_text(encoding="utf-8"))
    ids = [e["id"] for e in doc["entries"]]
    assert len(ids) == len(set(ids)), "duplicate entry ids"
    letter = (m.YAML.parent / doc["letter"]).read_text(encoding="utf-8")
    reviewers = {int(r) for r in re.findall(r"^Reviewer: (\d+)\s*$", letter, re.M)}
    assert reviewers == {e["reviewer"] for e in doc["entries"]}
    assert sum(e["kind"] == "comment" for e in doc["entries"]) == 62
