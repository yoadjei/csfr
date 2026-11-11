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
    used = set(re.findall(r"\\((?:csfr|rec|calib|oracle|corpus|methods|env|sweep|stats|fallback|bu|shiftmap|dictlearn)[A-Z][A-Za-z]*)", tex))
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


def test_shiftmap_macros_take_the_worst_cell_of_both_corpora(tmp_path):
    m = _mod()
    import csv

    def write(sweep, rows):
        with (tmp_path / f"{sweep}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["family", "level", "n_realisations", "frac_patches_differ",
                                              "cell_mean_range"])
            w.writeheader()
            w.writerows(rows)

    write("sweep_v2_govdocs1", [{"family": "CF1", "level": "L1", "n_realisations": 4,
                                 "frac_patches_differ": 0.10, "cell_mean_range": 0.05}])
    assert set(m.shiftmap_macros(tmp_path).values()) == {m.PLACEHOLDER}      # one corpus is not enough
    write("sweep_v2_dfrws", [{"family": "CF2", "level": "L5", "n_realisations": 4,
                              "frac_patches_differ": 0.42, "cell_mean_range": 0.013}])
    assert m.shiftmap_macros(tmp_path) == {"shiftmapRealisations": "4", "shiftmapDifferMax": "42",
                                           "shiftmapCellRangeMax": "0.05"}


def test_fallback_macros_wait_for_a_complete_primary_sweep(tmp_path):
    """A resumed sweep leaves fallbacks.csv with some cells at fewer seeds and
    others missing; the macros must stay pending, not fail or report a part."""
    m = _mod()
    import csv
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seeds: [0, 1]\nfamilies: [CF1]\nlevels: {L1: 0.05, L5: 0.7}\n")
    fields = ["sweep", "family", "level", "method", "n_seeds", "n", "n_fallback", "frac_fallback"]

    def write(rows):
        with (tmp_path / "fb.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([dict(zip(fields, (m.SWEEPS[0], *r))) for r in rows])

    full = [("CF1", "L1", "dictlearn", 2, 10, 9, 0.9), ("CF1", "L1", "criminisi", 2, 10, 0, 0.0),
            ("CF1", "L5", "dictlearn", 2, 10, 10, 1.0), ("CF1", "L5", "criminisi", 2, 10, 4, 0.4)]
    write(full[:3] + [("CF1", "L5", "criminisi", 1, 5, 2, 0.4)])           # one cell at one seed
    assert set(m.fallback_macros(tmp_path / "fb.csv", cfg).values()) == {m.PLACEHOLDER}
    write(full)
    assert m.fallback_macros(tmp_path / "fb.csv", cfg) == {
        "fallbackDictCFoneMin": "90.0", "fallbackDictCFoneMax": "100.0", "fallbackExemplarCFoneLfive": "40.0"}


def test_dictlearn_cap_macros_take_the_largest_shift_and_refuse_a_mismatch(tmp_path):
    m = _mod()
    import csv
    import pytest

    def write(rows):
        with (tmp_path / f"{m.SWEEPS[0]}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    assert set(m.dictlearn_cap_macros(tmp_path).values()) == {m.PLACEHOLDER}
    row = {"cap_released": 200, "cap_reference": 20000, "n_patches": 200, "matches_sweep": "True"}
    write([{**row, "diff_mean": -0.031, "diff_abs_max": 1.26}, {**row, "diff_mean": 0.012, "diff_abs_max": 0.4}])
    assert m.dictlearn_cap_macros(tmp_path) == {
        "dictlearnCapFactor": "100", "dictlearnCapCells": "2", "dictlearnCapPatches": "200",
        "dictlearnCapMaxCellShift": "0.03", "dictlearnCapMaxPatchShift": "1.3"}
    write([{**row, "diff_mean": 0.0, "diff_abs_max": 0.0, "matches_sweep": "False"}])
    with pytest.raises(ValueError, match="did not reproduce"):
        m.dictlearn_cap_macros(tmp_path)


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
