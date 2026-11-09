"""Generate paper/numbers_v2.tex: every number the revised manuscript states in
running text, as LaTeX macros, from the code and the result CSVs.

    python scripts/render_numbers_v2.py

The manuscript uses the macros (\\csfrLamLone, ...) instead of typed numbers.
tests/test_numbers_v2.py regenerates the file and fails if the committed copy
is stale, so a number in the paper cannot drift from its source. Result-derived
macros are emitted only when their CSV exists; until then the macro expands to
a visible placeholder.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_objective import HF_CUTOFF  # noqa: E402
from csfr_solver import SolverConfig  # noqa: E402
from sensitivity import RecordConfig  # noqa: E402

OUT = ROOT / "paper" / "numbers_v2.tex"
PLACEHOLDER = r"\textbf{[pending]}"


def _sci(x: float, digits: int = 0) -> str:
    """Math-mode scientific notation: _sci(1e-6) -> 10^{-6};
    _sci(5e-12) -> 5\\times 10^{-12}; _sci(1.07e-3, 1) -> 1.1\\times 10^{-3}."""
    m, e = f"{x:.{digits}e}".split("e")
    e = int(e)
    return f"10^{{{e}}}" if m == "1" else f"{m}\\times 10^{{{e}}}"


def config_macros() -> dict[str, str]:
    s, r = SolverConfig(), RecordConfig()
    return {
        "csfrLamLone": f"{s.lam_l1:g}", "csfrLamTV": f"{s.lam_tv:g}", "csfrLamFC": f"{s.lam_fc:g}",
        "csfrEps": f"{s.eps:g}", "csfrHFcut": f"{HF_CUTOFF:g}",
        "csfrTol": _sci(s.tol), "csfrMaxIter": f"{s.max_iter:,}".replace(",", "\\,"),
        "csfrCheckEvery": f"{s.check_every}",
        "recCGrtol": _sci(r.cg_rtol), "recCGmax": f"{r.cg_max_iter:,}".replace(",", "\\,"),
        "recBoundMargin": _sci(r.bound_margin),
        "recSupport": _sci(r.support_rel), "recTopK": f"{r.top_k}", "recRefTol": _sci(r.ref_tol),
    }


CALIB = ROOT / "results" / "solver_calibration"


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pending(*names: str) -> dict[str, str]:
    """Macros whose source result does not exist yet: a visible placeholder, so
    the manuscript compiles and the gap shows in the PDF."""
    return dict.fromkeys(names, PLACEHOLDER)


def _cell(name: str) -> str:
    return name.replace("_", "~")


def calibration_macros() -> dict[str, str]:
    """From the committed calibration of the stopping rule. These files are part
    of the release, so a missing one is an error, not a pending result."""
    rows = _rows(CALIB / "tolerance.csv")
    tol = SolverConfig().tol
    at = [r for r in rows if float(r["tol"]) == tol]
    nxt = min(float(r["tol"]) for r in rows if float(r["tol"]) > tol)
    fail = [r for r in rows if float(r["tol"]) == nxt and float(r["max_err_p99"]) >= 0.01]
    lb = _rows(CALIB / "lbfgsb.csv")
    init = _rows(CALIB / "init.csv")
    # the rejected projected-step rule, at the threshold it was proposed with
    rej = next(r for r in _rows(CALIB / "projected_step_rule_rejected.csv")
               if r["cell"] == "CF3_L5" and float(r["tol"]) == 0.01)
    ref = next(r for r in rows if r["cell"] == "CF3_L5" and float(r["tol"]) == tol)
    return {
        "calibPctlErrMax": _sci(max(float(r["max_err_p99"]) for r in at), 1),
        "calibPsnrDiffMax": _sci(max(float(r["psnr_abs_diff_max"]) for r in at), 1),
        "calibNcells": str(len(at)), "calibNpatches": str(int(at[0]["n"])),
        "calibCells": ", ".join(_cell(r["cell"]) for r in at),
        "calibCapped": str(sum(int(r["n_capped"]) for r in rows)),
        "calibItersMin": str(min(int(r["iters_median"]) for r in at)),
        "calibItersMax": str(max(int(r["iters_median"]) for r in at)),
        "calibGridLoose": _sci(max(float(r["tol"]) for r in rows)),
        "calibGridTight": _sci(min(float(r["tol"]) for r in rows)),
        "calibNextTol": _sci(nxt),
        "calibNextFailCells": ", ".join(_cell(r["cell"]) for r in fail),
        "calibLbfgsbN": str(int(lb[0]["n"])),
        "calibLbfgsbMaxDiff": _sci(max(float(r["max_abs_diff_max"]) for r in lb), 1),
        "calibInitTol": _sci(float(init[0]["tol"])),
        "calibInitMaxDiff": _sci(max(float(r["max_abs_diff_max"]) for r in init), 1),
        "calibRejStepThresh": _sci(float(rej["tol"])),
        "calibRejStepPsnr": f"{float(rej['psnr_hole']):.2f}",
        "calibRejRefPsnr": f"{float(ref['psnr_hole_ref']):.2f}",
    }


CORPORA = {"Dfrws": "patches", "Gov": "patches_govdocs1_v2", "Kodak": "patches_kodak"}


def corpus_macros() -> dict[str, str]:
    """Size and clustering of the three patch corpora, from the committed
    per-patch source-image labels."""
    out = {}
    for tag, stem in CORPORA.items():
        labels = np.load(ROOT / "data" / f"{stem}_cluster.npy")
        _, counts = np.unique(labels, return_counts=True)
        share = np.sort(counts)[::-1] / len(labels)
        out[f"corpus{tag}N"] = str(len(labels))
        out[f"corpus{tag}Images"] = str(len(counts))
        out[f"corpus{tag}MaxShare"] = f"{100 * share[0]:.1f}"
        out[f"corpus{tag}TopTwoShare"] = f"{100 * share[:2].sum():.0f}"
    return out


SWEEPS = ("sweep_v2_govdocs1", "sweep_v2_dfrws", "sweep_v2_kodak")
FLOOR = "zero_fill"


def method_macros() -> dict[str, str]:
    """Counts of the compared methods, from the primary sweep's configuration:
    every configured baseline except the zero-fill floor, CSFR's TV-only arm
    (also a baseline), and the external method."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "sweep_v2_govdocs1.yaml").read_text(encoding="utf-8"))
    classical = [b for b in cfg["baselines"] if b != FLOOR] + [v for v in cfg["csfr_variants"] if v != "csfr"]
    return {"methodsClassical": str(len(classical)),
            "methodsTotal": str(len(classical) + len(cfg["external"])),
            "methodsAblation": str(len(cfg["ablation_variants"]) + len(cfg["csfr_variants"]) - 1),
            "sweepSeeds": str(len(cfg["seeds"])),
            "sweepAblationSeeds": ", ".join(str(s) for s in cfg["ablation_seeds"])}


def stats_macros() -> dict[str, str]:
    """From the analysis plan the statistics script writes before computing."""
    p = ROOT / "results" / "sweep_v2_govdocs1" / "stats" / "analysis_plan.json"
    if not p.exists():
        return _pending("statsBoot")
    n = int(json.loads(p.read_text(encoding="utf-8"))["bootstrap_resamples"])
    return {"statsBoot": f"{n:,}".replace(",", "\\,")}


def environment_macros() -> dict[str, str]:
    """The software environment of the round-2 sweeps, from their recorded
    run_metadata.json. Every sweep present must agree, because the paper states
    one environment."""
    envs = []
    for s in SWEEPS:
        p = ROOT / "results" / s / "run_metadata.json"
        if p.exists():
            e = json.loads(p.read_text(encoding="utf-8"))["environment"]
            envs.append((s, {k: e[k] for k in ("python", "torch", "opencv", "cpu_count_physical")}))
    if not envs:
        return _pending("envPython", "envTorch", "envOpencv", "envCores")
    first = envs[0][1]
    for s, e in envs[1:]:
        if e != first:
            raise ValueError(f"{s} ran in a different environment: {e} vs {first}")
    return {"envPython": first["python"], "envTorch": first["torch"].replace("+", "{+}"),
            "envOpencv": first["opencv"], "envCores": str(first["cpu_count_physical"])}


def fidelity_macros() -> dict[str, str]:
    """Perturbation sizes the fidelity campaign actually used (E6)."""
    p = ROOT / "results" / "sensitivity_fidelity" / "fidelity.csv"
    if not p.exists():
        return _pending("recDeltaMin", "recDeltaMax")
    deltas = sorted({float(r["delta"]) for r in _rows(p)})
    return {"recDeltaMin": f"{deltas[0]:g}", "recDeltaMax": f"{deltas[-1]:g}"}


def coherence_macros() -> dict[str, str]:
    p = ROOT / "results/oracle_diagnostic/coherence.json"
    if not p.exists():
        return _pending("oracleCoherence", "oracleCoherenceScaled")
    c = json.loads(p.read_text(encoding="utf-8"))
    return {"oracleCoherence": f"{c['mutual_coherence']:.4f}",
            "oracleCoherenceScaled": f"{c['sqrt_N_times_mu']:.2f}"}


def render() -> str:
    macros = {**config_macros(), **calibration_macros(), **corpus_macros(), **method_macros(),
              **environment_macros(), **stats_macros(), **fidelity_macros(), **coherence_macros()}
    lines = ["% GENERATED by scripts/render_numbers_v2.py. Do not edit; re-run the script.",
             "% Every number stated in the running text of the revised manuscript."]
    for k in sorted(macros):
        lines.append(f"\\providecommand{{\\{k}}}{{{macros[k]}}}")
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render()
    OUT.write_text(text, encoding="utf-8")
    pending = text.count(PLACEHOLDER)
    print(f"[numbers] wrote {OUT}; {pending} macro(s) pending a result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
