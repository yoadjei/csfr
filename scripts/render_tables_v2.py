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
import json
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
# what a dagger cites, in the document its table is in (the supplement cannot see the paper's labels)
MAIN_FALLBACK_REF = r"Section~\ref{sec:experiments:methods}"
SUPP_FALLBACK_REF = r"Table~\ref{tab:supp:fallbacks}"


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


def family_level_header(levels: tuple[str, ...], first: str = "Method",
                        families: tuple[str, ...] = FAMILIES) -> list[str]:
    """Two header rows: the families over their levels."""
    k = len(levels)
    spans = [(2 + i * k, 1 + (i + 1) * k) for i in range(len(families))]
    rules = "".join(rf"\cmidrule({'lr' if i < len(spans) - 1 else 'l'}){{{a}-{b}}}"
                    for i, (a, b) in enumerate(spans))
    return [" & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{f}}}" for f in families) + r" \\",
            rules,
            f"{first} & " + " & ".join(lv for _ in families for lv in levels) + r" \\"]


def method_by_cell_table(sweep: Path, metric: str, higher_is_better: bool | None, digits: int,
                         levels: tuple[str, ...] = MAIN_LEVELS, with_cap_note: bool = False,
                         fallbacks: Path = FALLBACKS, families: tuple[str, ...] = FAMILIES,
                         column: str | None = None, fallback_ref: str = MAIN_FALLBACK_REF) -> str | None:
    """Rows are methods, columns are (family, level) cells of the summary
    column `{metric}_mean` (or `column`, for a count such as n_at_cap). The best
    value of each column among the compared methods (not the floor) is bold,
    unless higher_is_better is None (a descriptive statistic), and a value a
    method mostly produced by its mean-fill fallback carries a dagger, whose note
    cites fallback_ref (a label of the document the table is in). Methods
    the sweep does not configure have no row. Returns the complete tabular."""
    s = _summary(sweep)
    if s is None:
        return None
    cells = [(f, lv) for f in families for lv in levels]
    fell_back = fallback_cells(sweep, fallbacks)
    notes = []
    if any((*c, m) in fell_back for c in cells for m, _ in METHOD_ORDER):
        notes.append(DAGGER + r" The method filled the hole with the observed mean in most patch-runs "
                     rf"({fallback_ref}).")
    if with_cap_note:
        notes.append(cap_note(s, cells))
    return cell_table(s, column or f"{metric}_mean", higher_is_better, digits, levels, fell_back, notes,
                      FLOOR, sweep.name, families)


def cell_table(s: dict, column: str, higher_is_better: bool | None, digits: int, levels: tuple[str, ...],
               marked: set, notes: list[str], floor: tuple[str, str], where: str,
               families: tuple[str, ...] = FAMILIES) -> str:
    """The method-by-cell tabular of summary rows s keyed (family, level,
    method), showing `column`: best compared value bold per column (none when
    higher_is_better is None), `marked` cells daggered, the floor row set
    apart, `notes` at the foot."""
    cells = [(f, lv) for f in families for lv in levels]
    best = {}
    for c in cells:
        vals = [float(s[(*c, m)][column]) for m, _ in METHOD_ORDER if (*c, m) in s]
        if not vals:
            raise ValueError(f"{where}: no results for cell {c[0]} {c[1]}")
        best[c] = max(vals) if higher_is_better else min(vals)

    def row(key: str, name: str, mark_best: bool) -> str:
        out = []
        for c in cells:
            txt = f"{float(s[(*c, key)][column]):.{digits}f}"
            if mark_best and txt == f"{best[c]:.{digits}f}":
                txt = rf"\textbf{{{txt}}}"
            out.append(txt + DAGGER if (*c, key) in marked else txt)
        return f"{name} & " + " & ".join(out) + r" \\"

    compared = higher_is_better is not None
    lines = [row(k, n, compared) for k, n in METHOD_ORDER if any((*c, k) in s for c in cells)]
    lines += [r"\midrule", row(*floor, False)]
    if notes:
        lines += [r"\midrule", *(_note(t, 1 + len(cells)) for t in notes)]
    return tabular(rf"@{{}}l*{{{len(cells)}}}{{r}}@{{}}", family_level_header(levels, families=families), lines)


OCR_DIR = ROOT / "results" / "ocr"
OCR_FLOOR = ("corrupted_input", "Corrupted input (floor)")


def ocr(d: Path = OCR_DIR) -> str | None:
    """Character error rate per method and cell (lower is better), with the
    recognition of the corrupted input as the floor."""
    rows = _csv(d / "summary.csv")
    if rows is None:
        return None
    s = {(r["family"], r["level"], r["method"]): r for r in rows}
    return cell_table(s, "cer_mean", False, 3, MAIN_LEVELS, set(), [], OCR_FLOOR, "ocr")


DOWNSTREAM_DIR = ROOT / "results" / "downstream_v2" / "stats"
CLASSIFIERS = [("cnn", "CNN"), ("resnet", "ResNet")]
DOWNSTREAM_ALIAS = {"telea": "inpaint_telea", "ns": "inpaint_ns"}   # harness names -> sweep names


def downstream_v2(d: Path = DOWNSTREAM_DIR) -> str | None:
    """Per method and classifier: accuracy per family (mean over levels), the
    mean ECE of the temperature-scaled confidence over cells, and the cells
    where CSFR is ahead or behind the method (Holm-adjusted McNemar p < ALPHA)."""
    met, st = _csv(d / "metrics.csv"), _csv(d / "stats.csv")
    if met is None or st is None:
        return None
    alias = lambda m: DOWNSTREAM_ALIAS.get(m, m)  # noqa: E731
    present = {alias(r["method"]) for r in met if r["family"] != "clean"}
    methods = [(k, n) for k, n in [*METHOD_ORDER, FLOOR] if k in present]
    body = []
    for k, name in methods:
        out = [name]
        for clf, _ in CLASSIFIERS:
            mr = [r for r in met if r["classifier"] == clf and alias(r["method"]) == k]
            out += [f"{np.mean([float(r['accuracy']) for r in mr if r['family'] == f]):.3f}" for f in FAMILIES]
            out.append(f"{np.mean([float(r['ece_ts']) for r in mr]):.3f}")
            sr = [r for r in st if r["classifier"] == clf and alias(r["method"]) == k]
            ahead = sum(float(r["p_holm"]) < ALPHA and int(r["statistic"]) > 0 for r in sr)
            behind = sum(float(r["p_holm"]) < ALPHA and int(r["statistic"]) < 0 for r in sr)
            out.append(f"{ahead}/{behind}" if sr else "--")
        body.append(" & ".join(out) + r" \\")
    span = len(FAMILIES) + 2                      # accuracy per family, ECE, significance counts
    clean = {r["classifier"]: float(r["accuracy"]) for r in met if r["family"] == "clean"}
    body += [r"\midrule", _note("Clean images: accuracy " + ", ".join(
        f"{n} {clean[c]:.3f}" for c, n in CLASSIFIERS if c in clean) +
        rf". $\pm$: cells where CSFR is ahead / behind (Holm-adjusted McNemar $p < {ALPHA:g}$).",
        1 + len(CLASSIFIERS) * span)]
    header = [" & " + " & ".join(rf"\multicolumn{{{span}}}{{c}}{{{n}}}" for _, n in CLASSIFIERS) + r" \\",
              rf"\cmidrule(lr){{2-{1 + span}}}\cmidrule(l){{{2 + span}-{1 + 2 * span}}}",
              "Method" + (" & " + " & ".join(FAMILIES) + r" & ECE & $\pm$") * len(CLASSIFIERS) + r" \\"]
    return tabular(rf"@{{}}l*{{{2 * span}}}{{r}}@{{}}", header, body)


ALL_LEVELS = ("L1", "L2", "L3", "L4", "L5")
EXTERNAL = "lama"


def signed(x: float, digits: int = 2) -> str:
    """+0.52 / $-$1.21: a typographic minus in text mode."""
    return f"{x:+.{digits}f}".replace("-", "$-$")


def plain(x: float, digits: int = 1) -> str:
    """0.52 / $-$1.21: no plus sign, a typographic minus in text mode."""
    return f"{x:.{digits}f}".replace("-", "$-$")


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


def cell_comparison(rows: list[dict], fam: str, lv: str, metric: str, where: str) -> dict:
    """The paired tests of one cell. best: the strongest classical method, the
    one furthest ahead of CSFR; rank: CSFR's place among itself and the n - 1
    classical methods; ext: the external method's row, None if not scored.
    Read from the signed difference (positive favours CSFR for every metric),
    so it holds whether higher or lower is better."""
    cell = [r for r in rows if (r["family"], r["level"], r["metric"]) == (fam, lv, metric)]
    classical = [r for r in cell if r["method"] != EXTERNAL]
    if not classical:
        raise ValueError(f"{where}: no {metric} comparisons for {fam} {lv}")
    return {"best": min(classical, key=lambda r: float(r["diff_mean"])),
            "rank": 1 + sum(float(r["diff_mean"]) < 0 for r in classical),
            "n": len(classical) + 1,
            "ext": next((r for r in cell if r["method"] == EXTERNAL), None)}


def comparison_row(rows: list[dict], fam: str, lv: str, metric: str, where: str) -> str:
    """One cell of a sweep's paired tests: CSFR's rank among CSFR and the
    classical methods, the comparison with the strongest classical method
    (difference, interval over source images, share of images CSFR leads,
    cap-free difference), and the comparison with the external method."""
    c = cell_comparison(rows, fam, lv, metric, where)
    best, ext = c["best"], c["ext"]
    cf = best.get("diff_mean_capfree", "")
    out = [f"{fam}~{lv}", f"{c['rank']}/{c['n']}", dict(METHOD_ORDER)[best["method"]],
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


# short names: the full-width table has twelve numeric columns; the caption gives the weights each arm zeroes
ABLATION_ARMS = [("csfr_tv_only", "TV only"), ("csfr_dct_only", "DCT only"),
                 ("csfr_no_fc", "No HF penalty"), ("csfr_no_clip", "No C3")]
NO_RANGE = "csfr_no_clip"
ALPHA = 0.05


def range_figures(rows: list[dict]) -> tuple[float, float, float]:
    """The range figure of the ablation (stats/range_violations.csv): the
    percentage of imputed pixels outside [0, 255] without C3, pooled over cells;
    the largest excess in grey levels; and the largest fraction outside in any
    cell of an arm that keeps C3."""
    out = [r for r in rows if r["method"] == NO_RANGE]
    kept = [r for r in rows if r["method"] != NO_RANGE]
    n_out = sum(float(r["frac_pixels_out"]) * int(r["n_imputed"]) for r in out)
    share = 100 * n_out / sum(int(r["n_imputed"]) for r in out)
    worst = max(float(r["max_excess"]) for r in out)
    return share, worst, max(float(r["frac_pixels_out"]) for r in kept)


def range_note(rows: list[dict]) -> str:
    share, worst, kept_max = range_figures(rows)
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
    """Per arm: hole PSNR per family (mean over levels and seeds), CPU time
    per cell, median iterations per patch and the share of patches that met
    the certificate. The released outputs were timed on another device, so
    their row shows no time. The CPU re-run of the released budget exists at
    one seed for its time only, so it shows no quality, and a note gives how
    far its outputs are from the released ones."""
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
        on_cpu = all(r["device"] == "cpu" for r in rs)
        secs = f"{np.mean([float(r['seconds_per_cell']) for r in rs]):.0f}" if on_cpu else "--"
        iters = np.median([float(r["iterations_median"]) for r in rs])
        conv = rs[0]["converged_frac"]
        conv = f"{100 * np.mean([float(r['converged_frac']) for r in rs]):.1f}" if conv != "" else "--"
        body.append(" & ".join([solver_arm_name(arm), *psnr, secs, f"{iters:.0f}", conv]) + r" \\")
    diffs = [float(r["max_abs_diff_vs_released"]) for r in cells if r.get("max_abs_diff_vs_released", "") != ""]
    if diffs:
        seeds = sorted({r["seed"] for r in cells if r.get("max_abs_diff_vs_released", "") != ""})
        agree = ("reproduces the released outputs bit for bit in every cell" if max(diffs) == 0
                 else f"differs from the released outputs by at most {max(diffs):.2g} grey levels")
        body += [r"\midrule",
                 _note(rf"The CPU re-run at mask seed~{', '.join(seeds)} {agree}; its row reports time only.",
                       4 + len(FAMILIES)),
                 _note("The released run was timed on its GPU, so its time is not shown.", 4 + len(FAMILIES))]
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


def record_rows(rows: list[dict], method: str, record: str, fams: tuple[str, ...]) -> list[dict]:
    """The rows of one record kind in one family group of records_compare.csv."""
    rs = [r for r in rows if r["method"] == method and r["record"] == record and r["cell"].split("_")[0] in fams]
    if not rs:
        raise ValueError(f"records: no {method} {record} rows for {fams}")
    return rs


def records_cmp(d: Path = FIDELITY_DIR) -> str | None:
    """Per record kind and family group: over the queries whose record is not
    zero, the medians of its L1 norm, support size and |w|-weighted mean
    distance from the query; and the share of queries whose record is zero (an
    8-bit method whose output did not change, or linear interpolation for an
    observed pixel that is no vertex of a triangle covering the hole). A kind
    with no non-zero record in a group shows dashes for the medians."""
    rows = _csv(d / "records_compare.csv")
    if rows is None:
        return None
    body = []
    for method, record, name in RECORD_KINDS:
        out = []
        for _, fams in RECORD_GROUPS:
            rs = record_rows(rows, method, record, fams)
            nz = [r for r in rs if float(r["l1_norm"]) > 0]
            zero = f"{100 * (1 - len(nz) / len(rs)):.0f}"
            if not nz:
                out += ["--"] * 3 + [zero]
                continue
            med = lambda k, rr=nz: float(np.median([float(r[k]) for r in rr]))  # noqa: E731
            out += [f"{med('l1_norm'):.2f}", f"{med('support'):.0f}", f"{med('mean_distance'):.1f}", zero]
        body.append(f"{name} & " + " & ".join(out) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{g}}}" for g, _ in RECORD_GROUPS) + r" \\",
              r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
              r"Record & $\ell_1$ & Supp. & Dist. & Zero (\%) & $\ell_1$ & Supp. & Dist. & Zero (\%) \\"]
    return tabular(r"@{}l*{8}{r}@{}", header, body)


ORACLE_DIR = ROOT / "results" / "oracle_diagnostic"
ORACLE_CORPORA = [("govdocs1_v2", "GovDocs1"), ("dfrws2006", "DFRWS 2006")]


def sci_compact(x: float) -> str:
    """Plain below 1000, else one-decimal mantissa times a power of ten."""
    if x < 1000:
        return f"{x:.0f}"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def oracle(d: Path = ORACLE_DIR) -> str | None:
    """Per cell and corpus: median oracle hole PSNR, the median and 90th
    percentile condition number of the observed rows of the oracle basis, and
    the share of patches above 10^3. Pending until every cell of both corpora is
    in oracle.csv, which the diagnostic rewrites after each cell."""
    rows = _csv(d / "oracle.csv")
    if rows is None:
        return None
    by = {(r["corpus"], r["family"], r["level"]): r for r in rows}
    if any((c, f, lv) not in by for c, _ in ORACLE_CORPORA for f in FAMILIES for lv in MAIN_LEVELS):
        return None
    body = []
    for fam in FAMILIES:
        if body:
            body.append(r"\midrule")
        for lv in MAIN_LEVELS:
            out = [f"{fam}~{lv}"]
            for corpus, _ in ORACLE_CORPORA:
                r = by[(corpus, fam, lv)]
                out += [plain(float(r["oracle_psnr_hole_median"])), sci_compact(float(r["cond_median"])),
                        sci_compact(float(r["cond_p90"])), f"{100 * float(r['frac_cond_above_1e3']):.0f}"]
            body.append(" & ".join(out) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{n}}}" for _, n in ORACLE_CORPORA) + r" \\",
              r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
              "Cell" + r" & PSNR & $\kappa_{50}$ & $\kappa_{90}$ & $>10^{3}$ (\%)" * len(ORACLE_CORPORA) + r" \\"]
    return tabular(r"@{}l*{8}{r}@{}", header, body)


BU_DIR = ROOT / "results" / "bu_evaluation"
BU_PREDICTORS = [("isolated_missing_frac", "Isolated-missing fraction (Bounded-Unknown)"),
                 ("mean_hole_distance", "Mean distance to an observed pixel"),
                 ("hole_frac", "Hole fraction")]


def bu(d: Path = BU_DIR) -> str | None:
    """Per outcome threshold and predictor: AUROC pooled over cells, per family,
    and the median within cells with outcome variance; then the decision of the
    pre-specified rule per threshold."""
    import json
    rows = _csv(d / "auroc.csv")
    dec_path = d / "decision.json"
    if rows is None or not dec_path.exists():
        return None
    decision = json.loads(dec_path.read_text(encoding="utf-8"))["by_threshold"]
    fmt = lambda a: "--" if a in ("", None) else f"{float(a):.2f}"  # noqa: E731
    body = []
    for thr in sorted({float(r["threshold_db"]) for r in rows}):
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{Outcome: CSFR hole PSNR below {thr:g}\,dB}}}} \\")
        at = [r for r in rows if float(r["threshold_db"]) == thr]
        for key, name in BU_PREDICTORS:
            pr = [r for r in at if r["predictor"] == key]
            pooled = next(r["auroc"] for r in pr if r["scope"] == "pooled")
            fam = [next((r["auroc"] for r in pr if r["scope"] == "family" and r["group"] == f), "")
                   for f in FAMILIES]
            within = [float(r["auroc"]) for r in pr if r["scope"] == "cell" and r["auroc"] != ""]
            body.append(" & ".join([name, fmt(pooled), *map(fmt, fam),
                                    f"{np.median(within):.2f}" if within else "--"]) + r" \\")
    notes = []
    for thr, dd in decision.items():
        verdict = "kept as a contribution" if dd["keep_as_contribution"] else "reported as a diagnostic"
        best = dict(BU_PREDICTORS).get(dd["best_predictor"], "none")
        notes.append(rf"Below {thr.replace('dB', r'\,dB')}: best predictor {best.lower()}; "
                     rf"{100 * dd['fraction_cells_within_auroc_above_min']:.0f}\% of "
                     rf"{dd['cells_with_outcome_variance']} cells above the within-cell minimum; {verdict}.")
    body += [r"\midrule", *(_note(t, 7) for t in notes)]
    header = [r" & & \multicolumn{4}{c}{By family} & \\", r"\cmidrule(lr){3-6}",
              "Predictor & Pooled & " + " & ".join(FAMILIES) + r" & Within (median) \\"]
    return tabular(r"@{}l*{6}{r}@{}", header, body)


JPEG_DIR = ROOT / "results" / "jpeg_corruption"
JPEG_KINDS = [("truncate", "Truncation"), ("overwrite", "Overwritten run"), ("flip", "Flipped byte")]
JPEG_CONDITIONS = [("oracle", "Oracle"), ("decoder", "Decoder"), ("offset", "Offset")]


def starred(diff: float, p: float) -> str:
    return signed(diff) + ("$^{*}$" if p < ALPHA else "")


def jpeg(d: Path = JPEG_DIR) -> str | None:
    """Per corruption and mask: cases, hole PSNR of CSFR, of the best classical
    method and of LaMa, CSFR's paired differences to both (starred where the
    uncorrected Wilcoxon p is below ALPHA), and the mask's coverage of the
    oracle loss and its false-positive share, over the cases with visible damage.
    The run is scored twice, the second time with LaMa, so the table waits until
    every corruption and mask has a LaMa row."""
    summ, mq = _csv(d / "summary.csv"), _csv(d / "masks_quality.csv")
    if summ is None or mq is None:
        return None
    names = dict(METHOD_ORDER)
    by = {(r["kind"], r["condition"], r["method"]): r for r in summ}
    if any((k, c, EXTERNAL) not in by for (k, c, meth) in by if meth == "csfr"):
        return None
    body = []
    for kind, kname in JPEG_KINDS:
        if body:
            body.append(r"\midrule")
        for cond, cname in JPEG_CONDITIONS:
            c = by.get((kind, cond, "csfr"))
            if c is None:
                raise ValueError(f"jpeg: no CSFR summary for {kind} {cond}")
            best = by[(kind, cond, c["vs_best_classical"])]
            lama = by.get((kind, cond, EXTERNAL))
            vis = [r for r in mq if r["kind"] == kind and r["condition"] == cond and r["visible_damage"] == "True"]
            cover = 100 * np.mean([float(r["coverage_of_oracle_loss"]) for r in vis])
            fp = 100 * np.mean([float(r["false_pos_frac"]) for r in vis])
            out = [kname if cond == JPEG_CONDITIONS[0][0] else "", cname, c["n_cases"],
                   f"{float(c['psnr_hole_mean']):.2f}", names[best["method"]], f"{float(best['psnr_hole_mean']):.2f}",
                   f"{float(lama['psnr_hole_mean']):.2f}" if lama else "--",
                   starred(float(c["diff_psnr_hole_vs_best_classical"]), float(c["p_wilcoxon_vs_best_classical"])),
                   (starred(float(c["diff_psnr_hole_vs_lama"]), float(c["p_wilcoxon_vs_lama"]))
                    if c.get("diff_psnr_hole_vs_lama", "") != "" else "--"),
                   f"{cover:.1f}", f"{fp:.2f}"]
            body.append(" & ".join(out) + r" \\")
    body += [r"\midrule", _note(rf"$^{{*}}$ Wilcoxon signed-rank $p < {ALPHA:g}$, paired by case, "
                                r"not corrected for multiplicity.", 11)]
    header = [r" & & & & \multicolumn{2}{c}{Best classical} & & \multicolumn{2}{c}{CSFR $-$} & "
              r"\multicolumn{2}{c}{Mask (\%)} \\",
              r"\cmidrule(lr){5-6}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
              r"Corruption & Mask & Cases & CSFR & Method & PSNR & LaMa & classical & LaMa & Cover & FP \\"]
    return tabular(r"@{}l l r r l r r r r r r@{}", header, body)


MASKERR_DIR = ROOT / "results" / "mask_robustness"
MASKERR_CELLS = ("CF1_L3", "CF2_L3", "CF3_L3", "CF1_L5", "CF2_L5", "CF3_L5")   # as Section VI-H states them


def perturbation_name(p: str) -> str:
    if m := re.fullmatch(r"(erode|dilate)(\d+)", p):
        return f"{m.group(1).capitalize()} {m.group(2)}\\,px"
    if m := re.fullmatch(r"flip(\d+)pct", p):
        return f"Flip {m.group(1)}\\%"
    raise ValueError(f"unknown perturbation {p}")


def maskerr(d: Path = MASKERR_DIR, cells: tuple[str, ...] = MASKERR_CELLS) -> str | None:
    """Per perturbation: each method's mean change in hole PSNR against the true
    mask (perturbed minus exact, paired by patch), averaged over the cells, and
    the realised false-positive and false-negative shares of the wrong mask.
    The run rewrites robustness.csv after every cell and is scored a second time
    with LaMa, so the table waits until every cell has every perturbation, each
    with a LaMa row."""
    rows = _csv(d / "robustness.csv")
    if rows is None:
        return None
    perts = [p for p in dict.fromkeys(r["perturbation"] for r in rows) if p != "exact"]
    have = {(r["cell"], r["perturbation"]) for r in rows if r["method"] == EXTERNAL}
    if any((c, p) not in have for c in cells for p in ("exact", *perts)):
        return None
    present = {r["method"] for r in rows}
    methods = [(k, n) for k, n in METHOD_ORDER if k in present]
    body = []
    for p in perts:
        at = [r for r in rows if r["perturbation"] == p]
        out = [perturbation_name(p)]
        for k, _ in methods:
            v = [float(r["degradation_mean"]) for r in at if r["method"] == k and r.get("degradation_mean", "") != ""]
            out.append(signed(float(np.mean(v))) if v else "--")
        cells = {r["cell"]: r for r in at}.values()
        out += [f"{100 * np.mean([float(r['false_pos_frac']) for r in cells]):.2f}",
                f"{100 * np.mean([float(r['false_neg_frac']) for r in cells]):.2f}"]
        body.append(" & ".join(out) + r" \\")
    header = [r" & \multicolumn{" + str(len(methods)) + r"}{c}{Change in hole PSNR (dB)} & "
              r"\multicolumn{2}{c}{Wrong mask (\%)} \\",
              rf"\cmidrule(lr){{2-{1 + len(methods)}}}\cmidrule(l){{{2 + len(methods)}-{3 + len(methods)}}}",
              "Perturbation & " + " & ".join(n for _, n in methods) + r" & FP & FN \\"]
    return tabular(rf"@{{}}l*{{{len(methods) + 2}}}{{r}}@{{}}", header, body)


CHILD_METHODS = {"shiftmap"}   # run in a fresh interpreter (src/baselines.py)
CHILD_NOTE = ("Shift-map runs in a fresh interpreter per call: its time includes starting it, "
              "and its memory is not in the peak shown.")


def _cell_label(c: str) -> str:
    return c.replace("_", "~")


FULLRES_DIR = ROOT / "results" / "fullres"


def fullres(d: Path = FULLRES_DIR) -> str | None:
    """Per method: hole PSNR per corruption (mean over images and channels),
    the median time per image and the largest rise in resident memory."""
    summ, cost = _csv(d / "summary.csv"), _csv(d / "cost.csv")
    if summ is None or cost is None:
        return None
    corrs = list(dict.fromkeys(r["corruption"] for r in summ))
    by = {(r["corruption"], r["method"]): r for r in summ}
    present = {r["method"] for r in summ}
    methods = [(k, n) for k, n in METHOD_ORDER if k in present]
    body = []
    for k, name in methods:
        q = [f"{float(by[(c, k)]['psnr_hole']):.2f}" if (c, k) in by else "--" for c in corrs]
        cs = [r for r in cost if r["method"] == k]
        secs = f"{np.median([float(r['seconds']) for r in cs]):.1f}" if cs else "--"
        mem = f"{max(float(r['rss_peak_delta_mb']) for r in cs):.0f}" if cs else "--"
        body.append(" & ".join([name, *q, secs, mem]) + r" \\")
    if present & CHILD_METHODS:
        body += [r"\midrule", _note(CHILD_NOTE, 3 + len(corrs))]
    header = [rf" & \multicolumn{{{len(corrs)}}}{{c}}{{Hole PSNR (dB)}} & & \\",
              rf"\cmidrule(lr){{2-{1 + len(corrs)}}}",
              "Method & " + " & ".join(_cell_label(c) for c in corrs) + r" & s/image & MB \\"]
    return tabular(rf"@{{}}l*{{{len(corrs) + 2}}}{{r}}@{{}}", header, body)


BENCH_DIR = ROOT / "results" / "bench_v2"


def bench_v2(d: Path = BENCH_DIR) -> str | None:
    """Per method and bench cell: milliseconds per patch, with the largest rise
    in resident memory; for CSFR also the median iterations and the time of one
    record row."""
    rows = _csv(d / "runtime_memory.csv")
    if rows is None:
        return None
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    by = {(r["cell"], r["method"]): r for r in rows}
    present = {r["method"] for r in rows}
    methods = [(k, n) for k, n in METHOD_ORDER if k in present]
    body = []
    for k, name in methods:
        ms = [f"{float(by[(c, k)]['ms_per_patch']):.1f}" if (c, k) in by else "--" for c in cells]
        mem = max(float(by[(c, k)]["rss_peak_delta_mb"]) for c in cells if (c, k) in by)
        body.append(" & ".join([name, *ms, f"{mem:.0f}"]) + r" \\")
    csfr = [by[(c, "csfr")] for c in cells if (c, "csfr") in by]
    notes = []
    if csfr:
        iters = " / ".join(f"{float(r['iterations_median']):.0f}" for r in csfr)
        rec = " / ".join(f"{float(r['record_ms_per_query']):.1f}" for r in csfr)
        notes.append(rf"CSFR, per cell: median iterations {iters}; one record row {rec}\,ms.")
    if present & CHILD_METHODS:
        notes.append(CHILD_NOTE)
    if notes:
        body += [r"\midrule", *(_note(t, 2 + len(cells)) for t in notes)]
    header = [rf" & \multicolumn{{{len(cells)}}}{{c}}{{ms per patch}} & \\",
              rf"\cmidrule(lr){{2-{1 + len(cells)}}}",
              "Method & " + " & ".join(_cell_label(c) for c in cells) + r" & MB \\"]
    return tabular(rf"@{{}}l*{{{len(cells) + 1}}}{{r}}@{{}}", header, body)


# ------------------------------------------------------------------ supplement

FAMILY_PANELS = (("CF1", "CF2"), ("CF3", "CF4"))
PANEL_SEP = "\n\\par\\medskip\n"


def num_sig(x: float, sig: int = 2) -> str:
    """`sig` significant digits, with a power of ten outside [0.01, 10^sig):
    0.318 -> 0.32; 6.7e-4 -> $6.7 \\times 10^{-4}$."""
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 10 ** sig:
        return f"{x:.{sig}g}"
    m, e = f"{x:.{sig - 1}e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def five_level_panels(sweep: Path, metric: str, higher_is_better: bool | None, digits: int,
                      with_cap_note: bool = False, column: str | None = None,
                      fallbacks: Path = FALLBACKS) -> str | None:
    """A method-by-cell table at every severity level, in two panels (CF1 and
    CF2, then CF3 and CF4) so that each fits the page width. These tables are in
    the supplement, so a dagger cites the supplement's fallback table."""
    parts = [method_by_cell_table(sweep, metric, higher_is_better, digits, ALL_LEVELS, with_cap_note,
                                  fallbacks, families=fams, column=column, fallback_ref=SUPP_FALLBACK_REF)
             for fams in FAMILY_PANELS]
    return None if parts[0] is None else PANEL_SEP.join(parts)


def supp_psnr() -> str | None:
    return five_level_panels(SWEEP, "psnr_hole", True, 2, with_cap_note=True)


def supp_ssim() -> str | None:
    return five_level_panels(SWEEP, "ssim_hole", True, 3)


def supp_hrp() -> str | None:
    """Descriptive only: no value is marked best (Section V-D of the paper)."""
    return five_level_panels(SWEEP, "hrp", None, 2)


def supp_cap() -> str | None:
    """Patch-runs per cell at the hole-PSNR cap, the counts behind the cap-free analysis."""
    return five_level_panels(SWEEP, "n_at_cap", None, 0, column="n_at_cap")


CAP_MECHANISM_DIR = ROOT / "results" / "cap_mechanism"
CAP_MECHANISM_CORPORA = [("sweep_v2_govdocs1", "GovDocs1"), ("sweep_v2_dfrws", "DFRWS 2006"),
                         ("sweep_v2_kodak", "Kodak")]


def supp_cap_mechanism(d: Path = CAP_MECHANISM_DIR) -> str | None:
    """Per corpus and cell (L1, L3, L5): CSFR's difference to the strongest
    classical method split into the patches that method reconstructs to the cap
    and the rest (scripts/cap_mechanism.py), the capped patches, the share of
    the capped patch-runs whose true hole is flat, and the share of those flat
    holes CSFR fills within one grey level. Pending until every corpus is in."""
    panels = {name: _csv(d / f"{s}.csv") for s, name in CAP_MECHANISM_CORPORA}
    if any(rows is None for rows in panels.values()):
        return None
    pct = lambda a, b: f"{100 * a / b:.0f}" if b else "--"  # noqa: E731
    body = []
    for name, rows in panels.items():
        by = {(r["family"], r["level"]): r for r in rows}
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        for fam in FAMILIES:
            for lv in MAIN_LEVELS:
                r = by[(fam, lv)]
                flat = int(r["n_runs_capped_flat"])
                body.append(" & ".join([
                    f"{fam}~{lv}", dict(METHOD_ORDER)[r["strongest"]], signed(float(r["diff_mean"])),
                    signed(float(r["capped_part"])), signed(float(r["rest_part"])),
                    f"{r['n_patches_capped']}/{r['n_patches']}", pct(flat, int(r["n_runs_capped"])),
                    pct(int(r["n_runs_capped_flat_csfr_within_1"]), flat)]) + r" \\")
    header = [r"Cell & Strongest & $\Delta$ & Capped & Rest & Capped patches & Flat (\%) & Within 1 (\%) \\"]
    return tabular(r"@{}l l r r r r r r@{}", header, body)


def p_value(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def rank_side(d: list[float]) -> str:
    """The side the Wilcoxon signed-rank statistic of the differences d favours,
    zeros dropped as scipy's default does: (+) CSFR, (-) the method."""
    from scipy.stats import rankdata
    x = np.asarray(d, dtype=float)
    x = x[x != 0]
    r = rankdata(np.abs(x))
    plus, minus = r[x > 0].sum(), r[x < 0].sum()
    return r"\,(+)" if plus > minus else r"\,($-$)" if minus > plus else ""


def supp_image_level(results: Path = ROOT / "results", metric: str = "psnr_hole") -> str | None:
    """Per corpus and cell (L1, L3, L5), CSFR against the strongest classical
    method of the cell (the one Table PAIRED_V2 names) with the source image as
    the unit: the number of images, the share in which CSFR has the higher mean,
    the mean of the per-image mean differences (stats/per_image.csv) with its
    bootstrap interval over images, and the two-sided Wilcoxon signed-rank p
    over the image means, not adjusted, with the side its statistic favours (a
    rank test can favour CSFR where the mean does not). Pending until every
    corpus has its statistics."""
    panels = {}
    for sweep, name in CAP_MECHANISM_CORPORA:
        paired, per = (_csv(results / sweep / "stats" / f) for f in ("paired_tests.csv", "per_image.csv"))
        if paired is None or per is None:
            return None
        panels[name] = (sweep, paired, per)
    body = []
    for name, (sweep, paired, per) in panels.items():
        means: dict[tuple[str, str, str], list[float]] = {}
        for r in per:
            if r["metric"] == metric:
                means.setdefault((r["family"], r["level"], r["method"]), []).append(float(r["diff_mean"]))
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        for fam in FAMILIES:
            for lv in MAIN_LEVELS:
                b = cell_comparison(paired, fam, lv, metric, sweep)["best"]
                img = means.get((fam, lv, b["method"]), [])
                if len(img) != int(b["n_images"]):
                    raise ValueError(f"{sweep} {fam} {lv}: {len(img)} image means for {b['n_images']} images")
                body.append(" & ".join([
                    f"{fam}~{lv}", dict(METHOD_ORDER)[b["method"]], b["n_images"],
                    f"{100 * float(b['frac_images_csfr_leads']):.0f}", signed(float(np.mean(img))),
                    interval(float(b["ci_image_lo"]), float(b["ci_image_hi"])),
                    p_value(float(b["p_image_wilcoxon"])) + rank_side(img)]) + r" \\")
    header = [r"Cell & Strongest & Images & Led (\%) & $\Delta_{\mathrm{img}}$ & 95\% CI & $p$ (side) \\"]
    return tabular(r"@{}l l r r r c r@{}", header, body)


def supp_bu_calibration(d: Path = BU_DIR) -> str | None:
    """Calibration of the cross-validated logistic model of the three mask
    statistics at each outcome threshold (scripts/run_bu_evaluation.py): per bin
    of predicted probability, the reconstructions in it, the mean predicted
    probability and the observed failure rate; then the model's AUROC and
    expected calibration error (decision.json)."""
    import json
    rows = _csv(d / "calibration.csv")
    if rows is None or not (d / "decision.json").exists():
        return None
    dec = json.loads((d / "decision.json").read_text(encoding="utf-8"))["by_threshold"]
    body = []
    for t in sorted({float(r["threshold_db"]) for r in rows}):
        dd = dec[f"{t:g}dB"]
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{CSFR hole PSNR below {t:g}\,dB}}}} \\")
        for r in (r for r in rows if float(r["threshold_db"]) == t):
            body.append(" & ".join([f"{float(r['lo']):.1f}--{float(r['hi']):.1f}", r["n"],
                                    f"{float(r['mean_predicted']):.3f}", f"{float(r['observed_rate']):.3f}"])
                        + r" \\")
        body.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\footnotesize {dd['cv_folds']}-fold AUROC "
                    rf"{dd['logistic_cv_auroc']:.2f}, ECE {dd['logistic_cv_ece_10bins']:.3f}}} \\")
    header = [r"Predicted & $n$ & Mean predicted & Observed \\"]
    return tabular(r"@{}l r r r@{}", header, body)


FALLBACK_ROWS = [("dictlearn", "Dict.\\ learning"), ("criminisi", "Exemplar")]


def supp_fallbacks(sweep: Path = SWEEP, path: Path = FALLBACKS) -> str | None:
    """Share of patch-runs (%) in which a method filled the hole with the
    observed mean, per cell of the sweep; pending until every cell has every
    seed (a resumed sweep leaves a partial file)."""
    import json
    rows, meta = _csv(path), sweep / "run_metadata.json"
    if rows is None or not meta.exists():
        return None
    n_seeds = len(json.loads(meta.read_text(encoding="utf-8"))["config"]["seeds"])
    by = {(r["family"], r["level"], r["method"]): r for r in rows if r["sweep"] == sweep.name}
    if not all(int(by.get((f, lv, m), {}).get("n_seeds", 0)) == n_seeds
               for f in FAMILIES for lv in ALL_LEVELS for m, _ in FALLBACK_ROWS):
        return None
    parts = []
    for fams in FAMILY_PANELS:
        cells = [(f, lv) for f in fams for lv in ALL_LEVELS]
        body = [f"{name} & " + " & ".join(f"{100 * float(by[(*c, m)]['frac_fallback']):.1f}" for c in cells)
                + r" \\" for m, name in FALLBACK_ROWS]
        parts.append(tabular(rf"@{{}}l*{{{len(cells)}}}{{r}}@{{}}",
                             family_level_header(ALL_LEVELS, families=fams), body))
    return PANEL_SEP.join(parts)


CALIB_DIR = ROOT / "results" / "solver_calibration"


def supp_calibration(d: Path = CALIB_DIR) -> str | None:
    """Stopping-rule calibration per probe cell and gap tolerance: median and
    99th percentile of the per-patch maximum pixel error against the reference
    solve (grey levels), the largest hole-PSNR difference to it (dB), median
    and largest iterations, and patches at the iteration cap."""
    rows = _csv(d / "tolerance.csv")
    if rows is None:
        return None
    default = SolverConfig().tol
    body = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        if body:
            body.append(r"\midrule")
        for r in sorted((r for r in rows if r["cell"] == cell), key=lambda r: -float(r["tol"])):
            tol = float(r["tol"])
            body.append(" & ".join([cell.replace("_", "~"), tex_pow10(tol) + (r"$^{*}$" if tol == default else ""),
                                    num_sig(float(r["max_err_median"])), num_sig(float(r["max_err_p99"])),
                                    num_sig(float(r["psnr_abs_diff_max"])), r["iters_median"], r["iters_max"],
                                    r["n_capped"]]) + r" \\")
    body += [r"\midrule", _note(r"$^{*}$ The default: the loosest tolerance whose 99th percentile is below "
                                r"0.01 grey levels in every cell.", 8)]
    header = [r" & & \multicolumn{2}{c}{Max.\ error} & & \multicolumn{2}{c}{Iterations} & \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
              r"Cell & Gap tol. & Median & P99 & $|\Delta\mathrm{PSNR}|$ & Median & Max & Capped \\"]
    return tabular(r"@{}l c r r r r r r@{}", header, body)


def cell_certificates(d: Path, cell: str, arm: str = "csfr") -> dict | None:
    """Every certificate of one arm in one sweep cell, pooled over mask seeds:
    iterations, converged flags and gaps per patch-solve, and the tolerances
    the seeds were solved at. None if a seed of the cell has no certificate."""
    seeds = sorted(p for p in (d / cell).iterdir() if p.is_dir() and p.name.startswith("seed"))
    certs = [s / f"certificate_{arm}.json" for s in seeds]
    if not certs or not all(c.exists() for c in certs):
        return None
    loaded = [json.loads(c.read_text(encoding="utf-8")) for c in certs]
    out = {k: np.concatenate([c[k] for c in loaded]) for k in ("iterations", "converged", "gap")}
    out["tols"] = {float(c["config"]["tol"]) for c in loaded}
    out["seeds"] = tuple(s.name for s in seeds)
    return out


def supp_convergence(results: Path = ROOT / "results") -> str | None:
    """Per cell and corpus, over every patch and mask seed of the CSFR sweep:
    the median and largest iteration count and the median Frank-Wolfe gap at
    release, read from the certificates the sweep writes. A note gives the
    tolerance and how many patch-solves met the certificate."""
    cells = [f"{f}_{lv}" for f in FAMILIES for lv in ALL_LEVELS]
    data = {}
    for sweep, name in CAP_MECHANISM_CORPORA:
        for cell in cells:
            c = cell_certificates(results / sweep, cell) if (results / sweep / cell).is_dir() else None
            if c is None:
                return None
            data[name, cell] = c
        if len({data[name, cell]["seeds"] for cell in cells}) != 1:
            return None                                  # a cell is missing a seed the others have
    tols = set().union(*(c["tols"] for c in data.values()))
    if len(tols) != 1:
        raise ValueError(f"the sweeps were solved at different tolerances: {sorted(tols)}")
    tol = tols.pop()
    names = [n for _, n in CAP_MECHANISM_CORPORA]
    body = []
    for cell in cells:
        if body and cell.endswith("_L1"):
            body.append(r"\midrule")
        out = []
        for n in names:
            c = data[n, cell]
            out += [f"{np.median(c['iterations']):.0f}", f"{c['iterations'].max():.0f}",
                    num_sig(float(np.median(c["gap"])))]
        body.append(" & ".join([cell.replace("_", "~"), *out]) + r" \\")
    met = {n: sum(int(data[n, cell]["converged"].sum()) for cell in cells) for n in names}
    total = {n: sum(len(data[n, cell]["converged"]) for cell in cells) for n in names}
    per_cell = {n: {len(data[n, cell]["converged"]) for cell in cells} for n in names}
    ncols = 1 + 3 * len(names)
    if all(met[n] == total[n] for n in names):
        cert = rf"Every patch-solve met the certificate, a gap of at most {tex_pow10(tol)}."
    else:
        cert = "Patch-solves that met the certificate: " + "; ".join(f"{n} {met[n]} of {total[n]}" for n in names) + "."
    sizes = ", ".join(f"{n} {'/'.join(str(k) for k in sorted(per_cell[n]))}" for n in names)
    body += [r"\midrule", _note(f"Gap: the median gap at release. {cert}", ncols),
             _note(f"Patch-solves per cell (patches times mask seeds): {sizes}.", ncols)]
    header = [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n in names) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(names))),
              "Cell" + r" & It.\ med. & It.\ max & Gap" * len(names) + r" \\"]
    return tabular(rf"@{{}}l*{{{3 * len(names)}}}{{r}}@{{}}", header, body)


SHIFTMAP_DIR = ROOT / "results" / "shiftmap_replicates"
SHIFTMAP_CORPORA = [("sweep_v2_govdocs1", "GovDocs1"), ("sweep_v2_dfrws", "DFRWS 2006")]


def supp_shiftmap(d: Path = SHIFTMAP_DIR) -> str | None:
    """Run-to-run variation of shift-map per cell at mask seed 0, per corpus:
    the share of patches whose output differs between realisations, the SD and
    range of the cell's mean hole PSNR across realisations, and the largest
    range of one patch's hole PSNR (dB)."""
    data = {s: _csv(d / f"{s}.csv") for s, _ in SHIFTMAP_CORPORA}
    if any(v is None for v in data.values()):
        return None
    by = {(s, r["family"], r["level"]): r for s, rows in data.items() for r in rows}
    body = []
    for fam in FAMILIES:
        if body:
            body.append(r"\midrule")
        for lv in ALL_LEVELS:
            out = [f"{fam}~{lv}"]
            for s, _ in SHIFTMAP_CORPORA:
                r = by.get((s, fam, lv))
                if r is None:
                    raise ValueError(f"shift-map replicates: no row for {s} {fam} {lv}")
                out += [f"{100 * float(r['frac_patches_differ']):.0f}", f"{float(r['cell_mean_sd']):.3f}",
                        f"{float(r['cell_mean_range']):.3f}", f"{float(r['patch_range_max']):.2f}"]
            body.append(" & ".join(out) + r" \\")
    k = sorted({int(r["n_realisations"]) for rows in data.values() for r in rows})
    body += [r"\midrule", _note(f"{'/'.join(map(str, k))} realisations per cell: the sweep's output and "
                                f"{'/'.join(str(n - 1) for n in k)} runs in fresh interpreters.", 9)]
    header = [" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{n}}}" for _, n in SHIFTMAP_CORPORA) + r" \\",
              r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
              "Cell" + r" & Differ (\%) & SD & Range & Patch" * len(SHIFTMAP_CORPORA) + r" \\"]
    return tabular(r"@{}l*{8}{r}@{}", header, body)


DICTCAP_DIR = ROOT / "results" / "dictlearn_inner_cap"


def supp_dictcap(d: Path = DICTCAP_DIR, sweep: Path = SWEEP) -> str | None:
    """The inner coordinate-descent cap of dictionary learning per cell: the
    patches whose dictionary was learned, mean hole PSNR at the released and
    the reference cap, and their difference (mean and largest per patch)."""
    rows = _csv(d / f"{sweep.name}.csv")
    if rows is None:
        return None
    body = []
    for fam in dict.fromkeys(r["family"] for r in rows):
        if body:
            body.append(r"\midrule")
        for r in (r for r in rows if r["family"] == fam):
            body.append(" & ".join([f"{fam}~{r['level']}", f"{r['n_learned']}/{r['n_patches']}",
                                    f"{float(r['psnr_released']):.2f}", f"{float(r['psnr_reference']):.2f}",
                                    signed(float(r["diff_mean"]), 3), f"{float(r['diff_abs_max']):.2f}"]) + r" \\")
    released, reference = rows[0]["cap_released"], rows[0]["cap_reference"]
    header = [rf" & & \multicolumn{{2}}{{c}}{{Hole PSNR (dB)}} & \multicolumn{{2}}{{c}}{{Difference (dB)}} \\",
              r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
              rf"Cell & Learned & cap {released} & cap {reference} & Mean & Max $|\cdot|$ \\"]
    return tabular(r"@{}l r r r r r@{}", header, body)


def supp_fidelity(d: Path = FIDELITY_DIR, delta: float = FIDELITY_DELTA) -> str | None:
    """Panel A: per cell and perturbation at the default tolerance, the median
    relative error against the reference re-solve and SRF against the reference
    and the operational re-solve. Panel B: the same at perturbation `delta`
    per gap tolerance on the tolerance-study cells."""
    rows, tol_rows = _csv(d / "summary.csv"), _csv(d / "tol_study_summary.csv")
    if rows is None or tol_rows is None:
        return None
    default = SolverConfig().tol
    at_default = [r for r in rows if float(r["tol"]) == default]
    deltas = sorted({float(r["delta"]) for r in at_default})
    cells = list(dict.fromkeys((r["family"], r["level"]) for r in at_default))
    by = {(r["family"], r["level"], float(r["delta"])): r for r in at_default}

    def triple(r: dict) -> list[str]:
        return [pct(float(r["rel_err_median_ref"])), f"{100 * float(r['srf_ref']):.1f}",
                f"{100 * float(r['srf_op']):.1f}"]

    body_a = [" & ".join([f"{f}~{lv}", *(v for dl in deltas for v in triple(by[(f, lv, dl)]))]) + r" \\"
              for f, lv in cells]
    spans = "".join(rf"\cmidrule({'lr' if i < len(deltas) - 1 else 'l'}){{{2 + 3 * i}-{4 + 3 * i}}}"
                    for i in range(len(deltas)))
    header_a = [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\delta = {dl:g}$}}" for dl in deltas) + r" \\",
                spans, "Cell" + r" & Med.\ (\%) & SRF ref & SRF op" * len(deltas) + r" \\"]
    panel_a = tabular(rf"@{{}}l*{{{3 * len(deltas)}}}{{r}}@{{}}", header_a, body_a)
    body_b = [" & ".join([f"{r['family']}~{r['level']}", tex_pow10(float(r["tol"])), *triple(r),
                          f"{100 * float(r['converged_frac_op']):.0f}"]) + r" \\"
              for r in sorted((r for r in tol_rows if float(r["delta"]) == delta),
                              key=lambda r: (r["family"], r["level"], -float(r["tol"])))]
    header_b = [rf"Cell & Gap tol. & Med.\ (\%) & SRF ref & SRF op & Converged (\%) \\"]
    panel_b = tabular(r"@{}l c r r r r@{}", header_b, body_b)
    return PANEL_SEP.join([panel_a, panel_b])


def quality() -> str | None:
    return method_by_cell_table(SWEEP, "psnr_hole", higher_is_better=True, digits=2, with_cap_note=True)


def gmsd_table() -> str | None:
    return method_by_cell_table(SWEEP, "gmsd_hole", higher_is_better=False, digits=3)


TABLES = {"SEVERITY": severity, "QUALITY": quality, "GMSD": gmsd_table, "PAIRED_V2": paired_v2,
          "EXTERNAL_V2": external_v2, "ABLATION_V2": ablation_v2, "SOLVER_CMP": solver_cmp,
          "FIDELITY": fidelity, "RECORDS_CMP": records_cmp, "ORACLE": oracle, "BU": bu,
          "JPEG": jpeg, "MASKERR": maskerr, "FULLRES": fullres, "BENCH_V2": bench_v2,
          "DOWNSTREAM_V2": downstream_v2, "OCR": ocr,
          # supplement
          "SUPP_PSNR": supp_psnr, "SUPP_SSIM": supp_ssim, "SUPP_HRP": supp_hrp, "SUPP_CAP": supp_cap,
          "SUPP_FALLBACKS": supp_fallbacks, "SUPP_CALIBRATION": supp_calibration,
          "SUPP_SHIFTMAP": supp_shiftmap, "SUPP_DICTCAP": supp_dictcap, "SUPP_FIDELITY": supp_fidelity,
          "SUPP_CAP_MECHANISM": supp_cap_mechanism, "SUPP_IMAGE_LEVEL": supp_image_level, "SUPP_BU_CALIBRATION": supp_bu_calibration,
          "SUPP_CONVERGENCE": supp_convergence}


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
