"""The revised manuscript states numbers through paper/numbers_v2.tex. This test
locks that file to its sources: regenerating it must reproduce the committed
copy byte for byte, and the config-derived values must equal the code
defaults."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("numbers_v2", ROOT / "scripts" / "render_numbers_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_committed_numbers_file_is_current():
    m = _mod()
    committed = (ROOT / "paper" / "numbers_v2.tex").read_text(encoding="utf-8")
    assert committed == m.render(), "paper/numbers_v2.tex is stale: run scripts/render_numbers_v2.py"


def test_macro_names_are_valid_control_words():
    """A LaTeX control word is letters only: \\calibP99Max would parse as
    \\calibP followed by the text 99Max."""
    m = _mod()
    import re
    names = re.findall(r"\\providecommand\{\\([^}]*)\}", m.render())
    assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names), \
        [n for n in names if not re.fullmatch(r"[A-Za-z]+", n)]


def test_typed_claims_about_the_calibration_hold():
    """Sentences the manuscript types around the macros."""
    import csv
    rows = list(csv.DictReader((ROOT / "results/solver_calibration/tolerance.csv").open(encoding="utf-8")))
    assert sum(int(r["n_capped"]) for r in rows) == 0, "paper says no calibration solve reached the cap"
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    assert "No calibration solve reached the iteration cap" in tex


def test_every_macro_the_manuscript_uses_is_generated():
    m = _mod()
    import re
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    generated = set(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", m.render()))
    used = set(re.findall(r"\\((?:csfr|rec|calib|oracle)[A-Z][A-Za-z]*)", tex))
    assert used, "the manuscript uses no number macros; the regex has rotted"
    assert used <= generated, f"manuscript uses undefined number macros: {sorted(used - generated)}"


def test_word_build_inlines_the_number_macros():
    """pandoc resolves \\input against its working directory and drops a file it
    cannot find with only a warning, so every number would vanish from the
    docx. The docx preparation must inline the macro file."""
    spec = importlib.util.spec_from_file_location("docx_build", ROOT / "scripts" / "build_ieee_docx.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    out = d.inline_inputs("pre\n\\input{numbers_v2}\npost $\\csfrTol$\n")
    assert "\\input{numbers_v2}" not in out
    assert "\\providecommand{\\csfrTol}" in out and out.endswith("post $\\csfrTol$\n")
    import inspect
    body = inspect.getsource(d.prepare)
    assert body.index("inline_inputs(") < body.index("resolve_anonymous("), \
        "prepare() must inline the inputs before any other step reads the tex"


def test_sci_formatting():
    m = _mod()
    assert m._sci(1e-6) == "10^{-6}"
    assert m._sci(5e-12) == "5\\times 10^{-12}"
    assert m._sci(1.07e-3, 1) == "1.1\\times 10^{-3}"
    assert m._sci(2.1e-14, 1) == "2.1\\times 10^{-14}"
    assert m._sci(1e-3, 1) == "1.0\\times 10^{-3}"


def test_config_macros_match_code_defaults():
    m = _mod()
    from csfr_solver import SolverConfig
    s = SolverConfig()
    c = m.config_macros()
    assert float(c["csfrLamLone"]) == s.lam_l1
    assert float(c["csfrLamTV"]) == s.lam_tv
    assert float(c["csfrLamFC"]) == s.lam_fc
    assert c["csfrTol"] == m._sci(s.tol)


def test_calibration_macros_follow_the_selection_rule():
    """The calibrated tolerance is the loosest whose p99 error is below 0.01
    grey levels in every calibration cell; the next looser one fails somewhere."""
    m = _mod()
    import csv
    rows = list(csv.DictReader((ROOT / "results/solver_calibration/tolerance.csv").open(encoding="utf-8")))
    from csfr_solver import SolverConfig
    grid = {float(r["tol"]) for r in rows}
    passing = [t for t in grid
               if all(float(r["max_err_p99"]) < 0.01 for r in rows if float(r["tol"]) == t)]
    assert SolverConfig().tol == max(passing), "default tol is not the calibrated one"
    c = m.calibration_macros()
    assert c["calibNextFailCells"] != m.PLACEHOLDER
