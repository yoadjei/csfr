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


def _tables():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tables_v2", ROOT / "scripts" / "render_tables_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FAMILY_WORDS = {"CF1": "CFone", "CF2": "CFtwo", "CF3": "CFthree", "CF4": "CFfour"}
LEVELS = ("L1", "L2", "L3", "L4", "L5")
COMPARISON_SWEEPS = {"Gov": "sweep_v2_govdocs1", "Dfrws": "sweep_v2_dfrws", "Kodak": "sweep_v2_kodak"}
PROSE_NAMES = {"csfr_tv_only": "the TV-only arm", "bilinear": "linear interpolation", "inpaint_telea": "Telea",
               "inpaint_ns": "Navier--Stokes", "dictlearn": "dictionary learning", "biharmonic": "biharmonic",
               "guleryuz": "block-DCT thresholding", "fsr": "FSR", "shiftmap": "shift-map",
               "criminisi": "exemplar inpainting"}
FAMILY_FIELDS = ("RankMin", "RankMax", "NFirst", "Strongest", "BehindMin", "BehindMax", "NBehindSig",
                 "AheadMax", "NAheadSig", "LamaBehindMax", "LamaAheadMax", "NLamaBehindSig", "NLamaAheadSig",
                 "LedMin", "LedMax", "CapfreeNAhead")
CORPUS_FIELDS = ("NMethods", "NFirst", "GmsdNFirst", "GmsdRankMin", "CapfreeNFirst", "CapfreeNewStrongest",
                 "CapfreeNSignFlip")


def _prose_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def comparison_names(prefix: str) -> list[str]:
    return ([f"q{prefix}{w}{f}" for w in FAMILY_WORDS.values() for f in FAMILY_FIELDS]
            + [f"q{prefix}{f}" for f in CORPUS_FIELDS])


def comparison_macros(prefix: str, sweep: Path) -> dict[str, str]:
    """The comparison of CSFR with the strongest classical method of each cell
    and with LaMa (results/<sweep>/stats/paired_tests.csv, hole PSNR unless
    named; the same reading as Table PAIRED_V2, via render_tables_v2.cell_comparison).
    Per family over the five levels: CSFR's rank range among itself and the
    classical methods; levels where it ranks first; the strongest methods in
    order of first appearance; the smallest and largest deficit where a method
    is ahead (dB, magnitudes) and the levels where its source-image interval
    lies below zero; the largest lead where CSFR is ahead and the levels whose
    interval lies above zero; the same against LaMa; the range of the share of
    source images in which CSFR has the higher mean (percent); the levels at
    which CSFR is ahead of the same strongest method on the cap-free patches. Per corpus: the classical
    method count, the cells CSFR ranks first on hole PSNR and on GMSD, its best
    GMSD rank, and the cells whose strongest method or difference sign changes
    on the cap-free patches. An empty range prints a dash."""
    names = comparison_names(prefix)
    p = sweep / "stats" / "paired_tests.csv"
    if not p.exists():
        return _pending(*names)
    t = _tables()
    rows = _rows(p)
    out: dict[str, str] = {}
    dash = "--"
    fmt = lambda xs, f: f"{f(xs):.2f}" if xs else dash  # noqa: E731
    n_methods = set()
    for fam, word in FAMILY_WORDS.items():
        cs = [t.cell_comparison(rows, fam, lv, "psnr_hole", sweep.name) for lv in LEVELS]
        n_methods |= {c["n"] for c in cs}
        diffs = [float(c["best"]["diff_mean"]) for c in cs]
        lo = [float(c["best"]["ci_cluster_lo"]) for c in cs]
        hi = [float(c["best"]["ci_cluster_hi"]) for c in cs]
        ext = [c["ext"] for c in cs]
        if any(e is None for e in ext):
            raise ValueError(f"{sweep.name} {fam}: LaMa is not scored at every level")
        ld = [float(e["diff_mean"]) for e in ext]
        strongest = list(dict.fromkeys(PROSE_NAMES[c["best"]["method"]] for c in cs))
        vals = {"RankMin": str(min(c["rank"] for c in cs)), "RankMax": str(max(c["rank"] for c in cs)),
                "NFirst": str(sum(c["rank"] == 1 for c in cs)), "Strongest": _prose_list(strongest),
                "BehindMin": fmt([-d for d in diffs if d < 0], min),
                "BehindMax": fmt([-d for d in diffs if d < 0], max),
                "NBehindSig": str(sum(h < 0 for h in hi)),
                "AheadMax": fmt([d for d in diffs if d > 0], max),
                "NAheadSig": str(sum(x > 0 for x in lo)),
                "LedMin": f"{100 * min(float(c['best']['frac_images_csfr_leads']) for c in cs):.0f}",
                "LedMax": f"{100 * max(float(c['best']['frac_images_csfr_leads']) for c in cs):.0f}",
                "CapfreeNAhead": str(sum(float(c["best"]["diff_mean_capfree"]) > 0 for c in cs
                                         if c["best"].get("diff_mean_capfree", "") != "")),
                "LamaBehindMax": fmt([-d for d in ld if d < 0], max),
                "LamaAheadMax": fmt([d for d in ld if d > 0], max),
                "NLamaBehindSig": str(sum(float(e["ci_cluster_hi"]) < 0 for e in ext)),
                "NLamaAheadSig": str(sum(float(e["ci_cluster_lo"]) > 0 for e in ext))}
        out.update({f"q{prefix}{word}{k}": v for k, v in vals.items()})
    if len(n_methods) != 1:
        raise ValueError(f"{sweep.name}: cells compare different numbers of methods: {sorted(n_methods)}")
    cells = [(f, lv) for f in FAMILY_WORDS for lv in LEVELS]
    psnr = [t.cell_comparison(rows, f, lv, "psnr_hole", sweep.name) for f, lv in cells]
    gmsd = [t.cell_comparison(rows, f, lv, "gmsd_hole", sweep.name) for f, lv in cells]
    capfree = [c["best"] for c in psnr if c["best"].get("diff_mean_capfree", "") != ""]
    new_strongest = first_capfree = 0    # on the cap-free patches: a different strongest method; CSFR first
    for (f, lv), c in zip(cells, psnr):
        cf = [r for r in rows if (r["family"], r["level"], r["metric"]) == (f, lv, "psnr_hole")
              and r["method"] != t.EXTERNAL and r.get("diff_mean_capfree", "") != ""]
        if cf:
            new_strongest += min(cf, key=lambda r: float(r["diff_mean_capfree"]))["method"] != c["best"]["method"]
            first_capfree += all(float(r["diff_mean_capfree"]) > 0 for r in cf)
    out.update({f"q{prefix}NMethods": str(n_methods.pop()),
                f"q{prefix}NFirst": str(sum(c["rank"] == 1 for c in psnr)),
                f"q{prefix}GmsdNFirst": str(sum(c["rank"] == 1 for c in gmsd)),
                f"q{prefix}GmsdRankMin": str(min(c["rank"] for c in gmsd)),
                f"q{prefix}CapfreeNFirst": str(first_capfree),
                f"q{prefix}CapfreeNewStrongest": str(new_strongest),
                f"q{prefix}CapfreeNSignFlip": str(sum((float(b["diff_mean"]) > 0) != (float(b["diff_mean_capfree"]) > 0)
                                                     for b in capfree))})
    return out


MECH_NAMES = ("mechContigCappedMin", "mechContigCappedMax", "mechContigCapPartMin", "mechContigCapPartMax",
              "mechContigRestMin", "mechContigRestMax", "mechContigFlatShare", "mechContigBiasMin",
              "mechContigBiasMax", "mechContigWithinShare", "mechContigCsfrPsnrMin", "mechContigCsfrPsnrMax",
              "mechContigTvExactShare", "mechDfrwsContigCappedMax")
CONTIGUOUS = ("CF3", "CF4")


def mechanism_macros(d: Path = ROOT / "results" / "cap_mechanism") -> dict[str, str]:
    """Where the contiguous-loss deficit on the primary corpus comes from
    (scripts/cap_mechanism.py, GovDocs1, every CF3 and CF4 cell): the range of
    patches the strongest method reconstructs to the cap; the range of their
    part of the difference (dB, magnitudes) and of the rest; the share of the
    capped patch-runs whose true hole is flat; per cell, the median signed error
    of CSFR's fill of those flat holes (grey levels, magnitudes); the share of
    flat capped runs CSFR fills within one grey level; the range of the
    per-cell median of CSFR's hole PSNR on the capped runs; and the share of the
    flat capped runs that CSFR's TV-only arm also reproduces exactly. The text says that
    the capped patches account for the whole deficit, that CSFR is ahead on the
    rest, and that its flat fills are darker than the page, in every such cell:
    the macros are refused if any of these fails, and Kodak, said to have no
    capped patch in these families, must have none. For DFRWS 2006: the most
    patches the strongest method caps in one CF3 or CF4 cell."""
    gov, kodak, dfrws = (d / f"sweep_v2_{s}.csv" for s in ("govdocs1", "kodak", "dfrws"))
    if not (gov.exists() and kodak.exists() and dfrws.exists()):
        return _pending(*MECH_NAMES)
    rows = [r for r in _rows(gov) if r["family"] in CONTIGUOUS]
    if len(rows) != len(CONTIGUOUS) * len(LEVELS):
        raise ValueError(f"{gov.name}: {len(rows)} contiguous-loss cells, not {len(CONTIGUOUS) * len(LEVELS)}")
    cap = [float(r["capped_part"]) for r in rows]
    rest = [float(r["rest_part"]) for r in rows]
    if not all(c < 0 < x and c + x < 0 for c, x in zip(cap, rest)):
        raise ValueError("a contiguous-loss cell of GovDocs1 is not a deficit carried by its capped patches")
    bias = [float(r["csfr_flat_mean_err_median"]) for r in rows]
    if not all(b < 0 for b in bias):
        raise ValueError("a cell's flat fills are not darker than the page")
    if any(int(r["n_patches_capped"]) for r in _rows(kodak) if r["family"] in CONTIGUOUS):
        raise ValueError("Kodak has a capped patch under CF3 or CF4")
    runs = sum(int(r["n_runs_capped"]) for r in rows)
    flat = sum(int(r["n_runs_capped_flat"]) for r in rows)
    within = sum(int(r["n_runs_capped_flat_csfr_within_1"]) for r in rows)
    tv = sum(int(r["n_runs_capped_flat_tv_only_at_cap"]) for r in rows)
    capped = [int(r["n_patches_capped"]) for r in rows]
    psnr = [float(r["csfr_psnr_capped_median"]) for r in rows]
    return dict(zip(MECH_NAMES, (
        str(min(capped)), str(max(capped)), f"{-max(cap):.2f}", f"{-min(cap):.2f}",
        f"{min(rest):.2f}", f"{max(rest):.2f}", f"{100 * flat / runs:.0f}",
        f"{-max(bias):.2f}", f"{-min(bias):.2f}", f"{100 * within / flat:.0f}",
        f"{min(psnr):.0f}", f"{max(psnr):.0f}", f"{100 * tv / flat:.0f}",
        str(max(int(r["n_patches_capped"]) for r in _rows(dfrws) if r["family"] in CONTIGUOUS)))))


ABL_NAMES = ("ablNoTvMin", "ablNoTvMax", "ablTvNBehind", "ablTvBehindMax", "ablTvNAhead", "ablTvAheadMax",
             "ablTvBcFlatMin", "ablTvBcFlatMax", "ablTvBcRestMin", "ablTvBcRestMax", "ablTvBcAheadMin",
             "ablTvBcAheadMax", "ablTvBcFlatCapShare", "ablTvScatterBehindLevels", "ablTvScatterAheadLevels",
             "ablTvScatterAheadShareMax", "ablNoFcScatterBehindLevels", "ablNoFcScatterBehindMin",
             "ablNoFcScatterBehindMax", "ablNoFcScatterAheadShareMax", "ablNoFcScatterAheadLevels",
             "ablNoFcScatterAheadMax", "ablNoFcScatterNsLevels", "ablNoFcBcRestMaxAbs", "ablClipMaxAbs",
             "ablClipShare", "ablClipMaxExcess")
SCATTERED = ("CF1",)
BURST_CONTIGUOUS = ("CF2", "CF3", "CF4")


def level_span(levels: list[str]) -> str:
    """Prose for one run of consecutive levels: 'L5', 'L4 and L5', 'L1 to L3'."""
    idx = [LEVELS.index(lv) for lv in levels]
    if not idx or idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"levels {levels} are not one run of consecutive levels")
    if len(idx) == 1:
        return levels[0]
    return f"{levels[0]} and {levels[1]}" if len(idx) == 2 else f"{levels[0]} to {levels[-1]}"


def ablation_macros(mech: Path = ROOT / "results" / "ablation_mechanism" / "sweep_v2_govdocs1.csv",
                    ranges: Path = ROOT / "results" / "sweep_v2_govdocs1" / "stats" / "range_violations.csv"
                    ) -> dict[str, str]:
    """What each term of CSFR contributes on the primary corpus (Section VI-C;
    scripts/ablation_mechanism.py, CSFR minus the arm, hole PSNR, dB):

      NoTv      the DCT-only arm (no TV term): the range of the difference.
      Tv        the TV-only arm (no DCT or high-frequency term): cells behind and
                ahead with the largest gap; under CF2-CF4 (Bc) the part of the
                flat holes (magnitudes), the rest, the share of patches CSFR is
                ahead on, and the share of flat holes the arm reconstructs to
                the cap; under CF1 (Scatter) the levels where the arm is ahead
                and where CSFR is, and the largest share of patches CSFR is
                ahead on where the arm is ahead.
      NoFc      the arm without the high-frequency penalty: under CF1 the levels
                where it is significantly ahead (Holm p < ALPHA), its lead there
                and CSFR's largest patch share, the levels where CSFR is
                significantly ahead and by how much, and the levels where
                neither is; under CF2-CF4 the largest |rest part|.
      Clip      the arm without C3: the largest |difference| in any cell, and
                the range figure of Table ABLATION_V2.

    The text states the following, and the macros are refused if any fails:
    removing TV lowers hole PSNR in every cell with Holm p < ALPHA; against the
    TV-only arm, under CF2-CF4 the flat holes favour the arm while CSFR is ahead
    on the other holes and in the median in every cell; CF1 has no flat hole,
    the arm is ahead (on most patches) at the lowest CF1 levels and CSFR at
    the others; without the penalty, the flat holes favour the arm in every
    CF2-CF4 cell; with C3 no imputed pixel leaves the range."""
    if not (mech.exists() and ranges.exists()):
        return _pending(*ABL_NAMES)
    alpha = _tables().ALPHA
    rows = _rows(mech)

    def arm(method: str, families: tuple[str, ...]) -> list[dict]:
        out = [r for r in rows if r["method"] == method and r["family"] in families]
        if len(out) != len(families) * len(LEVELS):
            raise ValueError(f"{mech.name}: {method} has {len(out)} cells in {families}")
        return out

    def col(rs: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rs]

    every = SCATTERED + BURST_CONTIGUOUS
    no_tv = arm("csfr_dct_only", every)
    if not all(float(r["diff_mean"]) > 0 and float(r["p_holm"]) < alpha for r in no_tv):
        raise ValueError("a cell where removing TV does not lower hole PSNR significantly")

    tv, tv_bc, tv_sc = arm("csfr_tv_only", every), arm("csfr_tv_only", BURST_CONTIGUOUS), arm("csfr_tv_only", SCATTERED)
    if not all(float(r["flat_part"]) < 0 < float(r["rest_part"]) and float(r["diff_median"]) > 0 for r in tv_bc):
        raise ValueError("a CF2-CF4 cell where the flat holes do not favour the TV-only arm, or CSFR is not "
                         "ahead on the other holes and in the median")
    if any(int(r["n_flat"]) for r in tv_sc):
        raise ValueError("a CF1 cell with a flat hole")
    sc_behind = [r for r in tv_sc if float(r["diff_mean"]) < 0]
    sc_ahead = [r for r in tv_sc if float(r["diff_mean"]) > 0]
    if [r["level"] for r in sc_behind + sc_ahead] != list(LEVELS) or not sc_behind or not sc_ahead:
        raise ValueError("under CF1 the TV-only arm is not ahead at the lowest levels and CSFR at the rest")
    if max(col(sc_behind, "frac_patches_csfr_ahead")) >= 0.5:
        raise ValueError("a CF1 level where the TV-only arm is ahead in the mean but not on most patches")
    d_tv = col(tv, "diff_mean")
    flat_cap = sum(int(r["n_flat_arm_at_cap"]) for r in tv_bc) / sum(int(r["n_flat"]) for r in tv_bc)

    fc_bc, fc_sc = arm("csfr_no_fc", BURST_CONTIGUOUS), arm("csfr_no_fc", SCATTERED)
    if not all(float(r["flat_part"]) < 0 for r in fc_bc):
        raise ValueError("a CF2-CF4 cell where the flat holes do not favour the arm without the penalty")
    sig = [r for r in fc_sc if float(r["p_holm"]) < alpha]
    fc_behind = [r for r in sig if float(r["diff_mean"]) < 0]
    fc_ahead = [r for r in sig if float(r["diff_mean"]) > 0]
    fc_ns = [r for r in fc_sc if float(r["p_holm"]) >= alpha]

    clip = arm("csfr_no_clip", every)
    share, worst, kept_max = _tables().range_figures(_rows(ranges))
    if kept_max != 0:
        raise ValueError("an arm that keeps C3 has an imputed pixel outside the range")

    def lv(rs: list[dict]) -> str:
        return level_span([r["level"] for r in rs])

    fmt = "{:.2f}".format
    return dict(zip(ABL_NAMES, (
        fmt(min(col(no_tv, "diff_mean"))), fmt(max(col(no_tv, "diff_mean"))),
        str(sum(d < 0 for d in d_tv)), fmt(-min(d_tv)), str(sum(d > 0 for d in d_tv)), fmt(max(d_tv)),
        fmt(-max(col(tv_bc, "flat_part"))), fmt(-min(col(tv_bc, "flat_part"))),
        fmt(min(col(tv_bc, "rest_part"))), fmt(max(col(tv_bc, "rest_part"))),
        f"{100 * min(col(tv_bc, 'frac_patches_csfr_ahead')):.0f}",
        f"{100 * max(col(tv_bc, 'frac_patches_csfr_ahead')):.0f}", f"{100 * flat_cap:.0f}",
        lv(sc_behind), lv(sc_ahead), f"{100 * max(col(sc_behind, 'frac_patches_csfr_ahead')):.0f}",
        lv(fc_behind), fmt(-max(col(fc_behind, "diff_mean"))), fmt(-min(col(fc_behind, "diff_mean"))),
        f"{100 * max(col(fc_behind, 'frac_patches_csfr_ahead')):.0f}", lv(fc_ahead),
        fmt(max(col(fc_ahead, "diff_mean"))), lv(fc_ns),
        fmt(max(abs(x) for x in col(fc_bc, "rest_part"))),
        fmt(max(abs(x) for x in col(clip, "diff_mean"))), f"{share:.2f}", f"{worst:.1f}")))


def all_comparison_macros(results: Path = ROOT / "results") -> dict[str, str]:
    return {k: v for prefix, s in COMPARISON_SWEEPS.items()
            for k, v in comparison_macros(prefix, results / s).items()}


def render() -> str:
    macros = {**config_macros(), **calibration_macros(), **corpus_macros(), **method_macros(),
              **environment_macros(), **stats_macros(), **fidelity_macros(), **fallback_macros(),
              **bu_macros(), **shiftmap_macros(), **dictlearn_cap_macros(), **coherence_macros(),
              **r1_hole_whole_macros(), **all_comparison_macros(), **mechanism_macros(),
              **ablation_macros()}
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
