"""Generate the round-2 AUTO tables of the manuscript from code and result CSVs.

    python scripts/render_tables_v2.py

Each generator returns the LaTeX body spliced between % BEGIN AUTO <TAG> and
% END AUTO <TAG> (render_stats_tables.splice, which searches the manuscript and
the supplement), or None when its source result does not exist yet; a pending
table is reported and left as it is. tests/test_render_tables_v2.py checks
every generator against its source and that the committed tex is current.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_solver import SolverConfig  # noqa: E402
from masks import SEVERITY, describe_families  # noqa: E402
from quality import CAP  # noqa: E402

_spec = importlib.util.spec_from_file_location("_rst", ROOT / "scripts" / "render_stats_tables.py")
_rst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rst)
splice = _rst.splice


def rule_to_latex(rule: str) -> str:
    """The plain-text generation rule of masks.describe_families, typeset."""
    rule = re.sub(r"round\(([^)]*)\)", lambda m: r"$\mathrm{round}(" + m.group(1).replace("*", "") + ")$", rule)
    rule = re.sub(r"\{(\d+),\.\.\.,(\d+)\}", r"$\\{\1,\\ldots,\2\\}$", rule)
    return re.sub(r"\breaches f\b", "reaches $f$", rule)


def severity() -> str:
    """Corruption families (generation rule) and the nominal erased fraction f
    of each severity level."""
    rows = []
    for fam, text in describe_families().items():
        name, rule = text.split(": ", 1)
        rows.append(f"{fam} & {name} & {rule_to_latex(rule)} \\\\")
    levels = " & ".join(f"{lvl}~{f:.2f}" for lvl, f in SEVERITY.items())
    rows += [r"\midrule",
             r"\multicolumn{3}{@{}l}{Erased fraction $f$: " + levels.replace(" & ", ", ") + r"} \\"]
    return "\n".join(rows)


SWEEP = ROOT / "results" / "sweep_v2_govdocs1"
FAMILIES = ("CF1", "CF2", "CF3", "CF4")
MAIN_LEVELS = ("L1", "L3", "L5")
# row order and display names; zero-fill is the floor and is set apart
METHOD_ORDER = [("csfr", "CSFR"), ("csfr_tv_only", "TV-only"), ("bilinear", "Linear"),
                ("inpaint_telea", "Telea"), ("inpaint_ns", "Navier--Stokes"),
                ("dictlearn", "Dict.\\ learning"), ("biharmonic", "Biharmonic"),
                ("guleryuz", "Block-DCT thr."), ("fsr", "FSR"), ("shiftmap", "Shift-map"),
                ("criminisi", "Exemplar"), ("lama", "LaMa")]
FLOOR = ("zero_fill", "Zero-fill (floor)")


def _summary(sweep: Path) -> dict[tuple[str, str, str], dict] | None:
    """The sweep's summary keyed by (family, level, method), or None until the
    sweep is complete: every configured cell and method (the external one
    included) at every seed. The sweep rewrites summary.csv after each cell, so
    an incomplete one is pending, not an error; the submission gate fails while
    anything is pending."""
    import csv
    import json
    p, meta = sweep / "summary.csv", sweep / "run_metadata.json"
    if not p.exists() or not meta.exists():
        return None
    cfg = json.loads(meta.read_text(encoding="utf-8"))["config"]
    with p.open(encoding="utf-8") as f:
        s = {(r["family"], r["level"], r["method"]): r for r in csv.DictReader(f)}
    methods = cfg.get("csfr_variants", []) + cfg.get("baselines", []) + cfg.get("external", [])
    n = len(cfg["seeds"])
    complete = all(int(s.get((f, lv, m), {}).get("n_seeds", 0)) == n
                   for f in cfg["families"] for lv in cfg["levels"] for m in methods)
    return s if complete else None


FALLBACKS = ROOT / "results" / "baseline_fallbacks" / "fallbacks.csv"
DAGGER = r"$^{\dagger}$"


def _note(text: str, ncols: int) -> str:
    return rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\footnotesize {text}}} \\"


def cap_note(s: dict, cells: list[tuple[str, str]]) -> str:
    """How many patch-runs reach the hole-PSNR cap: the largest count of any
    method in any tabulated cell."""
    rows = [r for (f, lv, _), r in s.items() if (f, lv) in cells]
    worst = max(rows, key=lambda r: int(r["n_at_cap"]))
    k = int(worst["n_at_cap"])
    if k == 0:
        return rf"No patch-run reaches the {CAP:g}\,dB cap."
    name = dict([*METHOD_ORDER, FLOOR])[worst["method"]]
    return (rf"Patch-runs at the {CAP:g}\,dB cap: at most {k} of {worst['n_patches']} per cell "
            rf"({name}, {worst['family']}~{worst['level']}).")


def fallback_cells(sweep: Path, path: Path = FALLBACKS) -> set[tuple[str, str, str]]:
    """(family, level, method) where the method fell back to the observed mean
    in most of its patch-runs (scripts/baseline_fallbacks.py)."""
    import csv
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["family"], r["level"], r["method"]) for r in csv.DictReader(f)
                if r["sweep"] == sweep.name and float(r["frac_fallback"]) > 0.5}


def tabular(colspec: str, header: list[str], body: list[str]) -> str:
    """A complete booktabs tabular: generated tables own their layout, so a
    header cannot drift from its rows."""
    return "\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}"])


def family_level_header(levels: tuple[str, ...], first: str = "Method") -> list[str]:
    """Two header rows: the families over their levels."""
    k = len(levels)
    spans = [(2 + i * k, 1 + (i + 1) * k) for i in range(len(FAMILIES))]
    rules = "".join(rf"\cmidrule({'lr' if i < len(spans) - 1 else 'l'}){{{a}-{b}}}"
                    for i, (a, b) in enumerate(spans))
    return [" & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{f}}}" for f in FAMILIES) + r" \\",
            rules,
            f"{first} & " + " & ".join(lv for _ in FAMILIES for lv in levels) + r" \\"]


def method_by_cell_table(sweep: Path, metric: str, higher_is_better: bool, digits: int,
                         levels: tuple[str, ...] = MAIN_LEVELS, with_cap_note: bool = False,
                         fallbacks: Path = FALLBACKS) -> str | None:
    """Rows are methods, columns are (family, level) cells; the best value of
    each column among the compared methods (not the floor) is bold, and a value
    a method mostly produced by its mean-fill fallback carries a dagger. Methods
    the sweep does not configure have no row. Returns the complete tabular."""
    s = _summary(sweep)
    if s is None:
        return None
    cells = [(f, lv) for f in FAMILIES for lv in levels]
    best = {}
    for c in cells:
        vals = [float(s[(*c, m)][f"{metric}_mean"]) for m, _ in METHOD_ORDER if (*c, m) in s]
        if not vals:
            raise ValueError(f"{sweep.name}: no results for cell {c[0]} {c[1]}")
        best[c] = max(vals) if higher_is_better else min(vals)
    fell_back = fallback_cells(sweep, fallbacks)

    def row(key: str, name: str, mark_best: bool) -> str:
        out = []
        for c in cells:
            txt = f"{float(s[(*c, key)][f'{metric}_mean']):.{digits}f}"
            if mark_best and txt == f"{best[c]:.{digits}f}":
                txt = rf"\textbf{{{txt}}}"
            out.append(txt + DAGGER if (*c, key) in fell_back else txt)
        return f"{name} & " + " & ".join(out) + r" \\"

    lines = [row(k, n, True) for k, n in METHOD_ORDER if any((*c, k) in s for c in cells)]
    lines += [r"\midrule", row(*FLOOR, False)]
    notes = []
    if any((*c, m) in fell_back for c in cells for m, _ in METHOD_ORDER):
        notes.append(DAGGER + r" The method filled the hole with the observed mean in most patch-runs "
                     r"(Section~\ref{sec:experiments:methods}).")
    if with_cap_note:
        notes.append(cap_note(s, cells))
    if notes:
        lines += [r"\midrule", *(_note(t, 1 + len(cells)) for t in notes)]
    return tabular(rf"@{{}}l*{{{len(cells)}}}{{r}}@{{}}", family_level_header(levels), lines)


ALL_LEVELS = ("L1", "L2", "L3", "L4", "L5")
EXTERNAL = "lama"


def signed(x: float, digits: int = 2) -> str:
    """+0.52 / $-$1.21: a typographic minus in text mode."""
    return f"{x:+.{digits}f}".replace("-", "$-$")


def interval(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{signed(lo, digits)}, {signed(hi, digits)}]"


def _csv(path: Path) -> list[dict] | None:
    import csv
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


COMPARISON_SPEC = r"@{}l c l r c r r r c@{}"
COMPARISON_HEADER = [
    r" & & \multicolumn{5}{c}{Strongest classical method} & \multicolumn{2}{c}{LaMa} \\",
    r"\cmidrule(lr){3-7}\cmidrule(l){8-9}",
    r"Cell & Rank & Method & $\Delta$ & 95\% CI & Led (\%) & $\Delta_{\mathrm{cf}}$ & $\Delta$ & 95\% CI \\"]


def comparison_row(rows: list[dict], fam: str, lv: str, metric: str, where: str) -> str:
    """One cell of a sweep's paired tests: CSFR's rank among CSFR and the
    classical methods, the comparison with the strongest classical method
    (difference, interval over source images, share of images CSFR leads,
    cap-free difference), and the comparison with the external method."""
    cell = [r for r in rows if (r["family"], r["level"], r["metric"]) == (fam, lv, metric)]
    classical = [r for r in cell if r["method"] != EXTERNAL]
    if not classical:
        raise ValueError(f"{where}: no {metric} comparisons for {fam} {lv}")
    best = max(classical, key=lambda r: float(r["method_mean"]))
    rank = 1 + sum(float(r["method_mean"]) > float(r["csfr_mean"]) for r in classical)
    ext = next((r for r in cell if r["method"] == EXTERNAL), None)
    cf = best.get("diff_mean_capfree", "")
    out = [f"{fam}~{lv}", f"{rank}/{len(classical) + 1}", dict(METHOD_ORDER)[best["method"]],
           signed(float(best["diff_mean"])),
           interval(float(best["ci_cluster_lo"]), float(best["ci_cluster_hi"])),
           f"{100 * float(best['frac_images_csfr_leads']):.0f}",
           signed(float(cf)) if cf != "" else "--"]
    out += ([signed(float(ext["diff_mean"])), interval(float(ext["ci_cluster_lo"]), float(ext["ci_cluster_hi"]))]
            if ext else ["--", "--"])
    return " & ".join(out) + r" \\"


def comparison_rows(sweep: Path, levels: tuple[str, ...], metric: str = "psnr_hole") -> list[str] | None:
    """comparison_row for every cell of a sweep, families separated by rules."""
    rows = _csv(sweep / "stats" / "paired_tests.csv")
    if rows is None:
        return None
    body = []
    for fam in FAMILIES:
        if body:
            body.append(r"\midrule")
        body += [comparison_row(rows, fam, lv, metric, sweep.name) for lv in levels]
    return body


def paired_v2(sweep: Path = SWEEP) -> str | None:
    """The primary corpus at every severity level."""
    body = comparison_rows(sweep, ALL_LEVELS)
    return None if body is None else tabular(COMPARISON_SPEC, COMPARISON_HEADER, body)


SECONDARY = {"DFRWS 2006": ROOT / "results" / "sweep_v2_dfrws", "Kodak": ROOT / "results" / "sweep_v2_kodak"}


def external_v2(sweeps: dict[str, Path] = SECONDARY) -> str | None:
    """The comparison of PAIRED_V2 on the secondary corpora at L1, L3 and L5,
    one panel per corpus; pending until every corpus has its statistics."""
    panels = {name: comparison_rows(p, MAIN_LEVELS) for name, p in sweeps.items()}
    if any(b is None for b in panels.values()):
        return None
    body = []
    for name, rows in panels.items():
        if body:
            body.append(r"\midrule")
        body += [rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{name}}}}} \\", *rows]
    return tabular(COMPARISON_SPEC, COMPARISON_HEADER, body)


ABLATION_ARMS = [("csfr_tv_only", r"TV only ($\lambda_{\ell_1} = \lambda_{\mathrm{FC}} = 0$)"),
                 ("csfr_dct_only", r"DCT only ($\lambda_{\mathrm{TV}} = 0$)"),
                 ("csfr_no_fc", r"No high-frequency penalty ($\lambda_{\mathrm{FC}} = 0$)"),
                 ("csfr_no_clip", r"No range constraint (C3)")]
NO_RANGE = "csfr_no_clip"
ALPHA = 0.05


def range_note(rows: list[dict]) -> str:
    """The range figure of the ablation: imputed pixels outside the range without
    C3, pooled over cells, and what the arms that keep C3 do."""
    out = [r for r in rows if r["method"] == NO_RANGE]
    kept = [r for r in rows if r["method"] != NO_RANGE]
    n_out = sum(float(r["frac_pixels_out"]) * int(r["n_imputed"]) for r in out)
    share = 100 * n_out / sum(int(r["n_imputed"]) for r in out)
    worst = max(float(r["max_excess"]) for r in out)
    kept_max = max(float(r["frac_pixels_out"]) for r in kept)
    rest = ("with it, none do" if kept_max == 0
            else rf"with it, at most {100 * kept_max:.2g}\% in any cell")
    return (rf"Without C3, {share:.2f}\% of the imputed pixels leave $[0, 255]$, by up to "
            rf"{worst:.1f} grey levels; {rest}.")


def ablation_v2(sweep: Path = SWEEP, metric: str = "psnr_hole") -> str | None:
    """Hole PSNR of CSFR minus each ablation arm per cell at the ablation
    seeds, starred where the Holm-adjusted Wilcoxon p is below ALPHA, with the
    range figure as a note."""
    rows = _csv(sweep / "stats" / "paired_tests_ablation.csv")
    ranges = _csv(sweep / "stats" / "range_violations.csv")
    if rows is None or ranges is None:
        return None
    by = {(r["family"], r["level"], r["method"]): r for r in rows if r["metric"] == metric}
    cells = [(f, lv) for f in FAMILIES for lv in MAIN_LEVELS]
    body = []
    for arm, name in ABLATION_ARMS:
        out = []
        for c in cells:
            r = by.get((*c, arm))
            if r is None:
                raise ValueError(f"{sweep.name}: no ablation comparison for {arm} in {c[0]} {c[1]}")
            star = "$^{*}$" if float(r["p_holm"]) < ALPHA else ""
            out.append(signed(float(r["diff_mean"])) + star)
        body.append(f"{name} & " + " & ".join(out) + r" \\")
    ncols = 1 + len(cells)
    body += [r"\midrule",
             _note(rf"$^{{*}}$ Holm-adjusted Wilcoxon $p < {ALPHA:g}$ across the arms.", ncols),
             _note(range_note(ranges), ncols)]
    return tabular(rf"@{{}}l*{{{len(cells)}}}{{r}}@{{}}", family_level_header(MAIN_LEVELS, "Arm"), body)


SOLVER_DIR = ROOT / "results" / "solver_comparison"


def tex_pow10(x: float) -> str:
    """1e-06 -> $10^{-6}$; 5e-12 -> $5 \\times 10^{-12}$."""
    m, e = f"{x:.0e}".split("e")
    return f"$10^{{{int(e)}}}$" if m == "1" else rf"${m} \times 10^{{{int(e)}}}$"


def solver_arm_name(arm: str) -> str:
    """The display name of a solver-comparison arm."""
    if m := re.fullmatch(r"adam(\d+)_(released|cpu)", arm):
        where = {"released": "released, GPU", "cpu": "CPU"}[m.group(2)]
        return f"Adam, {m.group(1)} iterations ({where})"
    if m := re.fullmatch(r"fista_gap(.+)", arm):
        tol = float(m.group(1))
        return f"Convergent, gap {tex_pow10(tol)}" + (" (default)" if tol == SolverConfig().tol else "")
    raise ValueError(f"unknown solver arm {arm}")


def solver_cmp(d: Path = SOLVER_DIR) -> str | None:
    """Per arm: hole PSNR per family (mean over levels and seeds), time per
    cell, median iterations and the share of patches that met the certificate.
    The CPU re-run of the released budget exists at one seed for its time
    only, so it shows no quality, and a note gives how far its outputs are
    from the released GPU outputs."""
    summ, cells = _csv(d / "quality_vs_compute.csv"), _csv(d / "per_cell.csv")
    if summ is None or cells is None:
        return None
    arms = list(dict.fromkeys(r["arm"] for r in summ))
    by = {(r["arm"], r["family"]): r for r in summ}
    body = []
    for arm in arms:
        rs = [by[(arm, f)] for f in FAMILIES]
        timing_only = arm == "adam600_cpu"
        psnr = ["--"] * len(FAMILIES) if timing_only else [f"{float(r['psnr_hole']):.2f}" for r in rs]
        secs = np.mean([float(r["seconds_per_cell"]) for r in rs])
        iters = np.median([float(r["iterations_median"]) for r in rs])
        conv = rs[0]["converged_frac"]
        conv = f"{100 * np.mean([float(r['converged_frac']) for r in rs]):.1f}" if conv != "" else "--"
        body.append(" & ".join([solver_arm_name(arm), *psnr, f"{secs:.0f}", f"{iters:.0f}", conv]) + r" \\")
    diffs = [float(r["max_abs_diff_vs_released"]) for r in cells if r.get("max_abs_diff_vs_released", "") != ""]
    if diffs:
        body += [r"\midrule",
                 _note(r"The CPU re-run of the released budget differs from the released outputs by at most "
                       rf"{max(diffs):.2g} grey levels; its row reports time only.", 4 + len(FAMILIES))]
    header = [r" & \multicolumn{4}{c}{Hole PSNR (dB)} & & & \\", r"\cmidrule(lr){2-5}",
              "Solver & " + " & ".join(FAMILIES) + r" & s/cell & Iter. & Cert.\ (\%) \\"]
    return tabular(r"@{}l*{4}{r}rrr@{}", header, body)


FIDELITY_DIR = ROOT / "results" / "sensitivity_fidelity"
FIDELITY_DELTA = 1.0


def pct(x: float, sig: int = 3) -> str:
    """A fraction as a percentage with `sig` significant digits."""
    return f"{100 * x:.{sig}g}"


def fidelity(d: Path = FIDELITY_DIR, delta: float = FIDELITY_DELTA) -> str | None:
    """Per cell at perturbation `delta` and the default tolerance: queries,
    median and 90th percentile relative error against the reference re-solve,
    SRF against the reference and the operational re-solve, top-k sign
    agreement, and the time of one record column."""
    rows = _csv(d / "summary.csv")
    if rows is None:
        return None
    tol = SolverConfig().tol
    by = {(r["family"], r["level"]): r for r in rows
          if float(r["delta"]) == delta and float(r["tol"]) == tol}
    body = []
    for fam in FAMILIES:
        if body:
            body.append(r"\midrule")
        for lv in MAIN_LEVELS:
            r = by.get((fam, lv))
            if r is None:
                raise ValueError(f"fidelity: no summary for {fam} {lv} at delta {delta:g}, tol {tol:g}")
            body.append(" & ".join([f"{fam}~{lv}", r["n"], pct(float(r["rel_err_median_ref"])),
                                    pct(float(r["rel_err_p90_ref"])), f"{100 * float(r['srf_ref']):.1f}",
                                    f"{100 * float(r['srf_op']):.1f}",
                                    f"{100 * float(r['sign_agreement_mean_ref']):.1f}",
                                    f"{1000 * float(r['record_seconds_mean']):.0f}"]) + r" \\")
    header = [r" & & \multicolumn{2}{c}{Rel.\ error (\%)} & \multicolumn{2}{c}{SRF (\%)} & & \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
              r"Cell & Queries & Median & P90 & ref & op & Sign (\%) & ms \\"]
    return tabular(r"@{}l r r r r r r r@{}", header, body)


RECORD_KINDS = [("csfr", "implicit_function", "CSFR"),
                ("bilinear", "exact_barycentric", "Linear, exact"),
                ("bilinear", "finite_difference_delta1", r"Linear, FD $\delta = 1$"),
                ("bilinear", "finite_difference_delta4", r"Linear, FD $\delta = 4$"),
                ("inpaint_telea", "finite_difference_delta1", r"Telea, FD $\delta = 1$"),
                ("inpaint_telea", "finite_difference_delta4", r"Telea, FD $\delta = 4$"),
                ("inpaint_ns", "finite_difference_delta1", r"Navier--Stokes, FD $\delta = 1$"),
                ("inpaint_ns", "finite_difference_delta4", r"Navier--Stokes, FD $\delta = 4$")]
RECORD_GROUPS = [("CF1--CF2", ("CF1", "CF2")), ("CF3--CF4", ("CF3", "CF4"))]


def records_cmp(d: Path = FIDELITY_DIR) -> str | None:
    """Per record kind and family group, medians over query pixels: L1 norm,
    support size, |w|-weighted mean distance from the query, and for finite
    differences the relative difference between delta 1 and 4. A kind with no
    non-zero record in a group (linear interpolation outside its convex hull)
    shows dashes."""
    rows = _csv(d / "records_compare.csv")
    if rows is None:
        return None
    body = []
    for method, record, name in RECORD_KINDS:
        out = []
        for _, fams in RECORD_GROUPS:
            rs = [r for r in rows if r["method"] == method and r["record"] == record
                  and r["cell"].split("_")[0] in fams]
            if not rs:
                raise ValueError(f"records: no {method} {record} rows for {fams}")
            nz = [r for r in rs if float(r["l1_norm"]) > 0]
            if not nz:
                out += ["--"] * 4
                continue
            med = lambda k, rr=nz: float(np.median([float(r[k]) for r in rr]))  # noqa: E731
            fd = [float(r["fd_rel_diff_delta1_vs_4"]) for r in nz if r.get("fd_rel_diff_delta1_vs_4", "") != ""]
            out += [f"{med('l1_norm'):.2f}", f"{med('support'):.0f}", f"{med('mean_distance'):.1f}",
                    f"{np.median(fd):.2f}" if fd else "--"]
        body.append(f"{name} & " + " & ".join(out) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{g}}}" for g, _ in RECORD_GROUPS) + r" \\",
              r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
              r"Record & $\ell_1$ & Supp. & Dist. & FD $1/4$ & $\ell_1$ & Supp. & Dist. & FD $1/4$ \\"]
    return tabular(r"@{}l*{8}{r}@{}", header, body)


def quality() -> str | None:
    return method_by_cell_table(SWEEP, "psnr_hole", higher_is_better=True, digits=2, with_cap_note=True)


def gmsd_table() -> str | None:
    return method_by_cell_table(SWEEP, "gmsd_hole", higher_is_better=False, digits=3)


TABLES = {"SEVERITY": severity, "QUALITY": quality, "GMSD": gmsd_table, "PAIRED_V2": paired_v2,
          "EXTERNAL_V2": external_v2, "ABLATION_V2": ablation_v2, "SOLVER_CMP": solver_cmp,
          "FIDELITY": fidelity, "RECORDS_CMP": records_cmp}


def main() -> int:
    for tag, fn in TABLES.items():
        body = fn()
        if body is None:
            print(f"[tables_v2] {tag}: source result not available yet, left as is")
            continue
        splice(tag, body)
        print(f"[tables_v2] {tag}: spliced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
