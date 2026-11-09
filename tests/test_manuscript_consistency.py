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
    them, while concluding that positive weight on the audit record lifts CSFR
    above the generative methods.
  * The ethics statement named a corpus that was never executed and omitted
    three that were.
  * After the first rejection the revision withdrew the over-claiming terms
    (traceability score, provenance, forensic validity, refusal) and a set of
    absolute claims. Those must not creep back in through a later edit.
  * Two scripts rendered the same frontier figure file; the stale one shipped
    under the current caption.
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


# every method the primary sweep runs, and the words that name it in §V-C
METHOD_NAMES = {"zero_fill": "Zero-fill", "bilinear": "Linear interpolation",
                "inpaint_telea": "Telea", "inpaint_ns": "Navier--Stokes",
                "dictlearn": "Dictionary learning", "biharmonic": "Biharmonic",
                "guleryuz": "Guleryuz", "fsr": "Frequency-selective reconstruction",
                "shiftmap": "shift-map", "criminisi": "Criminisi", "csfr_tv_only": "TV-only",
                "lama": "LaMa"}


def _methods_section(tex: str) -> str:
    return section(tex, r"\subsection{Methods}", r"\subsection{Metrics}")


def test_methods_section_names_every_configured_method(tex):
    """The methods paragraph must name exactly the methods the primary sweep
    runs: a method added to the configuration without the paper, or described
    in the paper without being run, fails here."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "sweep_v2_govdocs1.yaml").read_text(encoding="utf-8"))
    configured = set(cfg["baselines"]) | set(cfg["external"]) | (set(cfg["csfr_variants"]) - {"csfr"})
    assert configured == set(METHOD_NAMES), f"update METHOD_NAMES: {configured ^ set(METHOD_NAMES)}"
    sec = _methods_section(tex)
    for key, name in METHOD_NAMES.items():
        assert name in sec, f"{key} is configured but the methods section never names it"
    assert "It is reported as the floor and is not counted as a baseline" in sec
    assert r"\methodsTotal\ methods" in sec


def test_methods_section_parameters_match_the_code(tex):
    """Every configuration number typed in the methods section is the default
    the registry runs with."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import baselines as b
    sec = _methods_section(tex)
    g = inspect.signature(b.baseline_guleryuz).parameters
    assert (g["n_iter"].default, g["tau0"].default, g["tau1"].default) == (100, 128.0, 2.0)
    assert "from $128$ to $2$ over $100$ iterations" in sec and "$16$ grid shifts" in sec
    assert inspect.signature(b._block_dct_threshold).parameters["shifts"].default == (0, 2, 4, 6)
    d = inspect.signature(b.baseline_dictlearn).parameters
    assert (d["n_atoms"].default, d["n_nonzero"].default) == (64, 8)
    assert "$64$ atoms of $8 \\times 8$ pixels" in sec and "$8$ non-zero coefficients" in sec
    assert inspect.signature(b.baseline_criminisi).parameters["patch"].default == 9
    assert "with $9 \\times 9$ patches, reduced to $7$, $5$ and $3$" in sec
    assert "inpaintRadius=3" in inspect.getsource(b.baseline_inpaint_cv)
    assert "via OpenCV with radius $3$" in sec


@pytest.mark.xfail(strict=True, reason="the abstract is rewritten when the round-2 results exist; "
                                       "remove this marker then")
def test_abstract_and_methods_agree_on_the_count(tex):
    body = section(tex, r"\begin{abstract}", r"\end{abstract}")
    assert r"\methodsTotal" in body or r"\methodsClassical" in body
    assert "six same-harness baselines" not in body


def test_lama_configuration_is_stated(tex):
    """Reviewers of a DL baseline always ask for the checkpoint and the
    grayscale-to-RGB path. Both are in the code; they must be in the paper."""
    sec = _methods_section(tex)
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
        "the sweep excludes the only generative method measured under the "
        "harness, so it cannot support the claim made about generative methods")
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


def test_device_claims_match_what_the_runs_recorded(tex):
    """The paper said the primary sweep ran on Windows CPU and that only the
    external prior used a GPU. All 60 per-cell metric files say device=cuda.
    The claim was not merely wrong, it was inverted: the external prior had no
    separate run at all. Pin the statement to the artefacts."""
    devices = set()
    for f in (RESULTS / "csfr_sweep_2d_v2").glob("*/seed*/metrics.json"):
        devices.add(json.loads(f.read_text(encoding="utf-8")).get("device"))
    assert devices == {"cuda"}, f"primary sweep devices are now {devices}"

    meta = json.loads((RESULTS / "csfr_sweep_2d_v2" / "run_metadata.json")
                      .read_text(encoding="utf-8"))
    assert meta["environment"]["python"].startswith("3.12")
    assert "cu128" in meta["environment"]["torch"]

    ext = json.loads((RESULTS / "csfr_sweep_govdocs1" / "run_metadata.json")
                     .read_text(encoding="utf-8"))
    assert ext["environment"]["python"].startswith("3.14")
    assert "cpu" in ext["environment"]["torch"]

    # the withdrawn claims must not reappear, in the paper or the readme.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for doc, name in ((tex, "manuscript"), (readme, "README")):
        assert "no table row mixes devices" not in doc, \
            f"{name} repeats the withdrawn device claim"
    assert "not recoverable from the artefacts" not in tex, \
        "the primary sweep's environment is recorded; it is recoverable"


def test_committed_corpora_match_the_hashes_the_runs_recorded(tex):
    """The strongest reproducibility link in the package: the array each sweep
    consumed, hashed as the harness hashes it (float32, not the uint8 file)."""
    import hashlib

    import numpy as np

    pairs = [("data/patches.npy", "csfr_sweep_2d_v2"),
             ("data/patches_govdocs1.npy", "csfr_sweep_govdocs1")]
    for corpus, run in pairs:
        arr = np.load(ROOT / corpus).astype(np.float32)
        got = hashlib.sha256(arr.tobytes()).hexdigest()
        rec = json.loads((RESULTS / run / "run_metadata.json")
                         .read_text(encoding="utf-8"))["patches_sha256"]
        assert got == rec, f"{corpus} no longer matches the hash {run} recorded"


def test_paper_hyperparameters_match_the_config(tex):
    """The CSFR weights are stated through generated macros (locked to
    SolverConfig by test_numbers_v2.py), and the paper says they are the
    released configuration's: the two must agree. The fixed-budget Adam arm's
    constants are typed and must match the same config."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from csfr_solver import SolverConfig

    cfg = (ROOT / "configs" / "csfr_sweep_2d.yaml").read_text(encoding="utf-8")

    def cfg_val(key):
        m = re.search(rf"^\s*{key}\s*:\s*([0-9.eE+-]+)", cfg, re.M)
        assert m, f"{key} absent from configs/csfr_sweep_2d.yaml"
        return float(m.group(1))

    s = SolverConfig()
    for macro, key, value in ((r"\lambda_{\ell_1} = \csfrLamLone", "lam_l1", s.lam_l1),
                              (r"\lambda_{\mathrm{TV}} = \csfrLamTV", "lam_tv", s.lam_tv),
                              (r"\lambda_{\mathrm{FC}} = \csfrLamFC", "lam_fc", s.lam_fc)):
        assert macro in tex, f"manuscript no longer states {macro!r}"
        assert cfg_val(key) == value, f"{key}: released config {cfg_val(key)}, solver {value}"
    assert "The weights are those of the released configuration" in tex
    for claim_text, key, value in (("Adam with learning rate $2.0$", "lr", 2.0),
                                   ("budget of $600$ iterations", "max_iter", 600)):
        assert claim_text in tex, f"manuscript no longer states {claim_text!r}"
        assert cfg_val(key) == value, f"{key} is {cfg_val(key)}, paper says {value}"


def test_supplement_title_matches_the_manuscript(tex):
    """The supplement carried a title from an earlier draft, "Constraint-Driven
    Sparse Fragment Reconstruction for Digital Forensics", long after the paper
    had been renamed. Two documents submitted together under different titles
    read as a packaging error, so pin them to each other."""
    supp = (ROOT / "paper" / "paper2_supplementary.tex").read_text(encoding="utf-8")

    def title_of(doc: str) -> str:
        m = re.search(r"\\title\{(.*?)\}", doc, re.S)
        assert m, "no \\title in document"
        t = m.group(1).replace(r"\\", " ")
        t = re.sub(r"Supplementary Material", "", t)
        return " ".join(t.split())

    assert title_of(supp) == title_of(tex), (
        f"supplement is titled {title_of(supp)!r}, "
        f"manuscript is titled {title_of(tex)!r}")


def test_supplement_carries_no_author_names(tex):
    """The supplement has no author block on purpose. If one is ever added it
    must follow the manuscript's blinding switch, otherwise a blinded
    submission leaks the authors through the supplementary file."""
    supp = (ROOT / "paper" / "paper2_supplementary.tex").read_text(encoding="utf-8")
    if re.search(r"\\author\{", supp):
        assert r"\ifanonymous" in supp, (
            "supplement gained an author block without a blinding switch; "
            "a blinded submission would leak the authors through it")


def test_ethics_statement_names_the_corpora_actually_used(tex):
    sec = section(tex, r"\section*{Ethics Statement}",
                  r"\section*{Declaration of Competing Interest}")
    for corpus in ("DFRWS 2006", "GovDocs1", "Nick Mikus", "SVHN"):
        assert corpus in sec, f"ethics statement omits {corpus}"
    assert "FFT-75" not in sec, \
        "FFT-75 is specified but never executed; it does not belong here"


def test_revised_terminology_is_used_throughout(tex):
    """The post-rejection revision renamed the metric and the record because
    the old names claimed more than the code delivers. Lock the new names in
    and the old claims out."""
    body = tex.split(r"\begin{document}", 1)[1]
    for phrase in ("Traceability Score", r"\mathrm{TS}", "ForensicValidity",
                   "Forensic-Validity", "their contribution shares",
                   "explicitly refuses", "per-pixel provenance record",
                   "certified by construction", "carries zero traceability",
                   "yields output with zero traceability"):
        assert phrase not in body, f"withdrawn wording is back: {phrase!r}"
    for phrase in ("Audit Record Coverage", "approximate local sensitivity",
                   "Sensitivity Record Fidelity", "auditability composite (AC)",
                   "not as proof of recoverability", "Adoption Scope"):
        assert phrase in body, f"revised wording missing: {phrase!r}"


def test_c3_is_stated_as_the_global_bound(tex):
    """The solver projects onto the global 8-bit box. The paper once defined C3
    as a neighbourhood-derived bound it never implemented."""
    table = section(tex, r"\label{tab:constraints}", r"\end{tabularx}")
    assert r"intensity range: $0 \le X \le 255$, the global 8-bit range & hard" in table
    note = section(tex, r"\label{sec:problem:c3}", r"\begin{proposition}[unique minimiser]")
    assert "not implemented or evaluated" in note
    solver = (ROOT / "src" / "csfr_solver.py").read_text(encoding="utf-8")
    assert "x = x.clamp(LO, HI)" in solver, "the solver no longer projects onto the global box"
    objective = (ROOT / "src" / "csfr_objective.py").read_text(encoding="utf-8")
    assert "LO, HI = 0.0, 255.0" in objective


def test_withdrawn_absolute_claims_stay_withdrawn(tex):
    """Each pattern is wording the revision replaced with a hedge. This is a
    regression lock on that pass, not a general style lint."""
    body = tex.split(r"\begin{document}", 1)[1]
    for pattern in (r"\bimpossible\b", r"\bunambiguous", r"\bproves\b",
                    r"real and consistent", r"cannot be traced\b",
                    r"bit-level reproducibility", r"\binadmissible\b",
                    r"certified ones", r"core scientific result",
                    r"the decisive test", r"settles each comparison",
                    r"No method removes", r"guarantees that",
                    r"the single most consequential", r"fails Daubert scrutiny"):
        assert not re.search(pattern, body), f"absolute wording is back: {pattern!r}"


def test_ieee_build_derives_cleanly_from_the_source(tex):
    """The ieee access version is generated, never edited. Its front matter
    must meet the venue's stated limits: 150 to 250 abstract words, three to
    ten keywords in alphabetical order, a biography section when named."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_ieee", ROOT / "scripts" / "build_ieee.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, _missing = mod.convert(tex, mod.default_class())
    assert r"\usepackage{lineno}" not in out
    kw = re.search(r"\\begin\{(?:IEEE)?keywords\}\n(.*?)\n\\end", out, re.S).group(1)
    terms = [t.strip() for t in kw.split(",") if t.strip()]
    assert 3 <= len(terms) <= 10, f"{len(terms)} keywords"
    assert terms == sorted(terms, key=str.lower), "ieee access wants keywords alphabetical"
    abstract = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", out, re.S).group(1)
    words = [w for w in re.sub(r"\\[a-zA-Z]+|[${}~\\]", " ", abstract).split()
             if any(c.isalnum() for c in w)]
    assert 150 <= len(words) <= 250, f"abstract is {len(words)} words"
    if re.search(r"^\s*\\anonymousfalse", tex, re.M):
        assert "Anonymous" not in out
        assert "IEEEbiography" in out, "named build carries no biography section"


def test_frontier_figure_has_one_generator():
    """A second renderer of the frontier figure once shipped a stale drawing,
    six methods and a red 'inadmissible' band, under the current caption."""
    for script in (ROOT / "scripts").glob("*.py"):
        if script.name == "render_frontier.py":
            continue
        assert "paper2_F9_method_frontier" not in script.read_text(encoding="utf-8"), \
            f"{script.name} also writes the frontier figure"


def test_referenced_scripts_exist(tex):
    """Every scripts/*.py named in the manuscript must be a real file."""
    # strip the typesetting first: paths carry \allowbreak and escaped
    # underscores, so matching them in raw source is not worth the regex.
    plain = tex.replace(r"\allowbreak", "").replace(r"\_", "_")
    named = set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", plain))
    assert named, "no scripts named in the manuscript; the regex has rotted"
    missing = [n for n in named if not (ROOT / "scripts" / n).exists()]
    assert not missing, f"manuscript names scripts that do not exist: {missing}"
