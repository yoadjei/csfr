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
