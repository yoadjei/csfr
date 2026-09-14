"""Internal consistency of the manuscript against itself and its generators.

test_manuscript_numbers.py checks that hand-typed numbers match their CSVs.
These tests check a different failure mode: statements that contradict other
statements in the same paper, or captions that describe a table other than the
one beneath them. Both have shipped in earlier drafts.

The specific regressions guarded here:

  * §5.2 said five baselines ran under the harness and that the deep-learning
    families appeared "for contrast only", while the external prior was a sixth
    same-harness baseline in four tables and in the abstract.
  * The downstream table's seventh column was headed "Cond. Conf-Err" and
    carried the baseline's *joint* rate, so the conditional rates the prose
    argues from were absent from the table it cites.
  * The frontier sensitivity swept six methods with no generative method among
    them, while concluding that positive weight on traceability lifts CSFR
    above the generative methods.
  * The ethics statement named a corpus that was never executed and omitted
    three that were.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
RESULTS = ROOT / "results"

# venue caps at FSI:Digital Investigation and IEEE TIFS.
ABSTRACT_WORD_CAP = 250


@pytest.fixture(scope="module")
def tex() -> str:
    return TEX.read_text(encoding="utf-8")


def section(tex: str, start: str, end: str) -> str:
    i, j = tex.index(start), tex.index(end)
    assert i < j, f"{start!r} does not precede {end!r}"
    return tex[i:j]


def test_abstract_within_venue_cap(tex):
    body = section(tex, r"\begin{abstract}", r"\end{abstract}")
    body = re.sub(r"\\[a-zA-Z]+", "", body)
    body = re.sub(r"[${}~\\]", " ", body)
    words = [w for w in body.split() if any(c.isalnum() for c in w)]
    assert len(words) <= ABSTRACT_WORD_CAP, (
        f"abstract is {len(words)} words, cap is {ABSTRACT_WORD_CAP}")


def test_baselines_section_agrees_with_the_tables(tex):
    """§5.2 must enumerate six same-harness baselines and name the sixth."""
    sec = section(tex, r"\subsection{Baselines}",
                  r"\subsection{Forensic-Appropriate Metrics}")
    assert "Six baselines are evaluated under the identical harness" in sec
    assert "LaMa" in sec, "the sixth same-harness baseline is never named"
    items = re.findall(r"\\item \\textbf\{", sec)
    assert len(items) == 6, f"§5.2 enumerates {len(items)} baselines, not six"
    # the withdrawn claim: it was contradicted by LaMa running under the harness.
    assert "Running them under this harness would require the learned priors" \
        not in sec


def test_abstract_and_baselines_agree_on_the_count(tex):
    body = section(tex, r"\begin{abstract}", r"\end{abstract}")
    assert "six same-harness baselines" in body


def test_lama_configuration_is_stated(tex):
    """Reviewers of a DL baseline always ask for the checkpoint and the
    grayscale-to-RGB path. Both are in the code; they must be in the paper."""
    sec = section(tex, r"\subsection{Baselines}",
                  r"\subsection{Forensic-Appropriate Metrics}")
    for fragment in ("big-lama", "simple-lama-inpainting", "native resolution",
                     "three channels", "composited"):
        assert fragment in sec, f"LaMa configuration omits {fragment!r}"


def test_downstream_table_header_matches_the_generator(tex):
    """Every column the header promises must be emitted, and vice versa."""
    block = section(tex, "% BEGIN AUTO DOWNSTREAM", "% END AUTO DOWNSTREAM")
    rows = [r for r in block.splitlines()
            if r.strip().endswith(r"\\") and "midrule" not in r]
    assert rows, "downstream table body is empty"
    for r in rows:
        assert len(r.split("&")) == 10, f"row has wrong column count: {r}"

    header = section(tex, r"\caption{Downstream utility", "% BEGIN AUTO DOWNSTREAM")
    # the caption must not promise markup or columns the body does not carry.
    assert "expected calibration error" not in header
    assert "dagger marks the best confident-error performance" not in header
    assert "zero-signal coverage" not in header
    assert r"$n_{\text{conf}}$" in header


def test_downstream_conditional_rates_are_in_the_table(tex):
    """The prose argues from CF3 L5 conditional rates. They must appear in the
    table that sentence cites, not only in the CSV."""
    met = {}
    with (RESULTS / "downstream" / "metrics.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            met[(r["family"], r["level"], r["method"])] = r
    block = section(tex, "% BEGIN AUTO DOWNSTREAM", "% END AUTO DOWNSTREAM")
    row = next(r for r in block.splitlines() if r.startswith("CF3 & L5"))
    for method in ("csfr", "lama"):
        cond = float(met[("CF3", "L5", method)]["conf_err_cond"])
        n = int(float(met[("CF3", "L5", method)]["n_confident"]))
        assert f"{cond:.3f}" in row, f"{method} conditional rate missing"
        assert f"({n:d})" in row, f"{method} confident count missing"


def test_downstream_recovery_definition_is_correct(tex):
    """Recovery runs from the zero-fill floor to the clean ceiling, and the
    ceiling is the 1000-digit subset, not the full test set."""
    header = section(tex, r"\caption{Downstream utility", "% BEGIN AUTO DOWNSTREAM")
    assert "zero-fill" in header and "0.8690" in header
    assert "relative to clean ceiling (0.8783)" not in header


def test_frontier_sensitivity_includes_the_generative_method(tex):
    d = json.loads((RESULTS / "sensitivity" / "frontier_sensitivity.json")
                   .read_text(encoding="utf-8"))
    scores = d["equal_weight_point"]["scores"]
    assert "lama" in scores, (
        "the sweep excludes the only measured TS=0 generative method, so it "
        "cannot support the claim made about generative methods")
    assert len(scores) == 7
    worst = d["weights_simplex"]["csfr_rank_max"]
    assert re.search(rf"never falls below \w+ of seven", tex), \
        "manuscript does not state CSFR's worst rank over seven methods"
    assert worst == 5, f"worst rank is now {worst}; update the manuscript"


def test_external_corpus_scope_is_stated(tex):
    """LaMa was not run on GovDocs1. 'Rank of six' means something different
    there than 'six baselines' does everywhere else, so it must be said."""
    sec = section(tex, r"\subsection{External Validation}",
                  r"\subsection{Iteration-Budget Sensitivity}")
    assert "not re-run on this corpus" in sec
    with (RESULTS / "csfr_sweep_govdocs1" / "summary.csv").open(encoding="utf-8") as f:
        methods = {r["method"] for r in csv.DictReader(f)}
    assert "lama" not in methods, \
        "LaMa now has GovDocs1 results; tabulate it and drop this caveat"


def test_ethics_statement_names_the_corpora_actually_used(tex):
    sec = section(tex, r"\section*{Ethics Statement}",
                  r"\section*{Declaration of Competing Interest}")
    for corpus in ("DFRWS 2006", "GovDocs1", "Nick Mikus", "SVHN"):
        assert corpus in sec, f"ethics statement omits {corpus}"
    assert "FFT-75" not in sec, \
        "FFT-75 is specified but never executed; it does not belong here"


def test_referenced_scripts_exist(tex):
    """Every scripts/*.py named in the manuscript must be a real file."""
    # strip the typesetting first: paths carry \allowbreak and escaped
    # underscores, so matching them in raw source is not worth the regex.
    plain = tex.replace(r"\allowbreak", "").replace(r"\_", "_")
    named = set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", plain))
    assert named, "no scripts named in the manuscript; the regex has rotted"
    missing = [n for n in named if not (ROOT / "scripts" / n).exists()]
    assert not missing, f"manuscript names scripts that do not exist: {missing}"
