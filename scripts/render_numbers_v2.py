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


def bu_macros() -> dict[str, str]:
    """The Bounded-Unknown decision rule and outcome thresholds, from the
    evaluation script that applies them (fixed before the data were seen)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bu", ROOT / "scripts" / "run_bu_evaluation.py")
    bu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bu)
    low, high = sorted(bu.THRESHOLDS)
    return {"buPooledMin": f"{bu.RULE['pooled_auroc_min']:.2f}",
            "buWithinMin": f"{bu.RULE['within_cell_auroc_min']:.2f}",
            "buWithinShare": f"{100 * bu.RULE['within_cell_fraction_min']:.0f}",
            "buThreshLow": f"{low:g}", "buThreshHigh": f"{high:g}"}


FALLBACK_METHODS = ("dictlearn", "criminisi")


def fallback_macros(p: Path = ROOT / "results" / "baseline_fallbacks" / "fallbacks.csv",
                    cfg_path: Path = ROOT / "configs" / "sweep_v2_govdocs1.yaml") -> dict[str, str]:
    """Fallback rates on the primary corpus (scripts/baseline_fallbacks.py).
    Pending until the sweep is complete: every configured cell, for both
    methods, at every seed."""
    import yaml
    names = ("fallbackDictCFoneMin", "fallbackDictCFoneMax", "fallbackExemplarCFoneLfive")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    rows = [r for r in _rows(p) if r["sweep"] == SWEEPS[0]] if p.exists() else []
    complete = {(r["family"], r["level"], r["method"]) for r in rows if int(r["n_seeds"]) == len(cfg["seeds"])}
    if not {(f, lv, m) for f in cfg["families"] for lv in cfg["levels"] for m in FALLBACK_METHODS} <= complete:
        return _pending(*names)
    dl = [100 * float(r["frac_fallback"]) for r in rows if r["method"] == "dictlearn" and r["family"] == "CF1"]
    ex = next(r for r in rows if r["method"] == "criminisi" and (r["family"], r["level"]) == ("CF1", "L5"))
    return dict(zip(names, (f"{min(dl):.1f}", f"{max(dl):.1f}", f"{100 * float(ex['frac_fallback']):.1f}")))


SHIFTMAP_REPLICATE_SWEEPS = ("sweep_v2_govdocs1", "sweep_v2_dfrws")


def shiftmap_macros(d: Path = ROOT / "results" / "shiftmap_replicates") -> dict[str, str]:
    """Run-to-run variation of shift-map across fresh interpreters
    (scripts/shiftmap_replicates.py): the realisations per cell, the largest
    share of a cell's patches that differ, and the largest range of a cell mean,
    over every cell of the primary and the forensic corpus. Pending until both
    corpora have been measured."""
    names = ("shiftmapRealisations", "shiftmapDifferMax", "shiftmapCellRangeMax")
    paths = [d / f"{s}.csv" for s in SHIFTMAP_REPLICATE_SWEEPS]
    if not all(p.exists() for p in paths):
        return _pending(*names)
    rows = [r for p in paths for r in _rows(p)]
    realisations = {int(r["n_realisations"]) for r in rows}
    if len(realisations) != 1:
        raise ValueError(f"shift-map replicates differ in realisations per cell: {sorted(realisations)}")
    return dict(zip(names, (str(realisations.pop()),
                            f"{100 * max(float(r['frac_patches_differ']) for r in rows):.0f}",
                            f"{max(float(r['cell_mean_range']) for r in rows):.2f}")))


def dictlearn_cap_macros(d: Path = ROOT / "results" / "dictlearn_inner_cap") -> dict[str, str]:
    """What the inner coordinate-descent cap of the dictionary-learning baseline
    does to its output on the primary corpus (scripts/dictlearn_inner_cap.py):
    the factor of the reference cap, the cells and patches measured, and the
    largest change of a cell mean and of a single patch."""
    names = ("dictlearnCapFactor", "dictlearnCapCells", "dictlearnCapPatches",
             "dictlearnCapMaxCellShift", "dictlearnCapMaxPatchShift")
    p = d / f"{SWEEPS[0]}.csv"
    if not p.exists():
        return _pending(*names)
    rows = _rows(p)
    if not all(r["matches_sweep"] == "True" for r in rows):
        raise ValueError(f"{p}: the released cap did not reproduce the sweep's output")
    factors = {int(r["cap_reference"]) // int(r["cap_released"]) for r in rows}
    patches = {int(r["n_patches"]) for r in rows}
    if len(factors) != 1 or len(patches) != 1:
        raise ValueError(f"{p}: cells differ in reference factor or patches: {factors}, {patches}")
    return dict(zip(names, (str(factors.pop()), str(len(rows)), str(patches.pop()),
                            f"{max(abs(float(r['diff_mean'])) for r in rows):.2f}",
                            f"{max(float(r['diff_abs_max']) for r in rows):.1f}")))


def coherence_macros() -> dict[str, str]:
    p = ROOT / "results/oracle_diagnostic/coherence.json"
    if not p.exists():
        return _pending("oracleCoherence", "oracleCoherenceScaled")
    c = json.loads(p.read_text(encoding="utf-8"))
    return {"oracleCoherence": f"{c['mutual_coherence']:.4f}",
            "oracleCoherenceScaled": f"{c['sqrt_N_times_mu']:.2f}"}


R1_BASELINE_NAMES = {"zero_fill": "zero-fill", "bilinear": "bilinear", "inpaint_telea": "Telea",
                     "inpaint_ns": "Navier-Stokes", "dictlearn": "dictionary learning"}


def r1_hole_whole_macros(d: Path = ROOT / "results" / "r1_hole_vs_whole") -> dict[str, str]:
    """Round-1 paired PSNR differences, CSFR minus each baseline, under
    whole-patch and hole-only scoring (scripts/r1_hole_vs_whole.py; for the
    response to Reviewer 4, comment 4): the comparisons, how many had a patch at
    the cap, the largest change, the largest change not due to the cap, the
    largest departure of any method from exactly reproduced observed pixels, and
    the comparison with the largest change. Raises when a sentence the response
    builds on these would be false: a patch without a hole was dropped, or the
    largest change is not the baseline reaching the cap on patches where CSFR
    does not, each moving the difference by 10 log10(N/|hole|)."""
    names = ("rOneComparisons", "rOneBaselines", "rOneCells", "rOneSeeds", "rOneCappedComparisons",
             "rOneChangeMax", "rOneRestMax", "rOneShiftResidualMax",
             "rOneWorstFamily", "rOneWorstLevel", "rOneWorstBaseline", "rOneWorstCapped", "rOneWorstMove",
             "rOneWorstHolePixels", "rOnePixels", "rOneWorstWhole", "rOneWorstHole")
    cmp_path, res_path = d / "comparisons.csv", d / "shift_residual.csv"
    if not (cmp_path.exists() and res_path.exists()):
        return _pending(*names)
    rows = _rows(cmp_path)
    if any(r["n_whole"] != r["n_hole"] for r in rows):
        raise ValueError(f"{cmp_path}: a comparison dropped patches without a hole")
    w = max(rows, key=lambda r: abs(float(r["change"])))
    holes, n_pix = int(w["hole_pixels_min"]), int(w["n_pixels"])
    shift = 10 * np.log10(n_pix / holes)
    # negative: CSFR's score moves by the shift and the baseline's does not, so
    # the baseline is the method at the cap
    if w["hole_pixels_max"] != w["hole_pixels_min"] or abs(float(w["capped_mean_move"]) + shift) > 1e-3:
        raise ValueError(f"{cmp_path}: the largest change is not the baseline at the cap moving by the shift")
    residual = max(float(r["max_abs_residual_db"]) for r in _rows(res_path))
    return dict(zip(names, (
        str(len(rows)), str(len({r["baseline"] for r in rows})),
        str(len({(r["family"], r["level"]) for r in rows})), str(len({r["seed"] for r in rows})),
        str(sum(float(r["cap_part"]) != 0 for r in rows)),
        f"{max(abs(float(r['change'])) for r in rows):.2f}",
        _sci(max(abs(float(r["rest_part"])) for r in rows)), _sci(residual, 1),
        w["family"], w["level"], R1_BASELINE_NAMES[w["baseline"]], w["n_capped"],
        f"{abs(float(w['capped_mean_move'])):.2f}", str(holes), str(n_pix),
        f"{float(w['whole_diff']):+.2f}", f"{float(w['hole_diff']):+.2f}")))


def render() -> str:
    macros = {**config_macros(), **calibration_macros(), **corpus_macros(), **method_macros(),
              **environment_macros(), **stats_macros(), **fidelity_macros(), **fallback_macros(),
              **bu_macros(), **shiftmap_macros(), **dictlearn_cap_macros(), **coherence_macros(),
              **r1_hole_whole_macros()}
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
