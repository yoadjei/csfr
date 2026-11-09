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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
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
    import csv
    p = sweep / "summary.csv"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return {(r["family"], r["level"], r["method"]): r for r in csv.DictReader(f)}


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
    a method mostly produced by its mean-fill fallback carries a dagger. A cell
    a method did not run in is a dash. Returns the complete tabular."""
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
            r = s.get((*c, key))
            if r is None:
                out.append("--")
                continue
            v = float(r[f"{metric}_mean"])
            txt = f"{v:.{digits}f}"
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


def paired_v2(sweep: Path = SWEEP, metric: str = "psnr_hole") -> str | None:
    """Per cell: CSFR's rank among CSFR and the classical methods, the
    comparison with the strongest classical method (difference, interval over
    source images, fraction of images CSFR leads, cap-free difference), and the
    comparison with the external method."""
    rows = _csv(sweep / "stats" / "paired_tests.csv")
    if rows is None:
        return None
    names = dict(METHOD_ORDER)
    body = []
    for fam in FAMILIES:
        if body:
            body.append(r"\midrule")
        for lv in ALL_LEVELS:
            cell = [r for r in rows if (r["family"], r["level"], r["metric"]) == (fam, lv, metric)]
            classical = [r for r in cell if r["method"] != EXTERNAL]
            if not classical:
                raise ValueError(f"{sweep.name}: no {metric} comparisons for {fam} {lv}")
            best = max(classical, key=lambda r: float(r["method_mean"]))
            rank = 1 + sum(float(r["method_mean"]) > float(r["csfr_mean"]) for r in classical)
            ext = next((r for r in cell if r["method"] == EXTERNAL), None)
            cf = best.get("diff_mean_capfree", "")
            out = [f"{fam}~{lv}", f"{rank}/{len(classical) + 1}", names[best["method"]],
                   signed(float(best["diff_mean"])),
                   interval(float(best["ci_cluster_lo"]), float(best["ci_cluster_hi"])),
                   f"{100 * float(best['frac_images_csfr_leads']):.0f}",
                   signed(float(cf)) if cf != "" else "--"]
            out += ([signed(float(ext["diff_mean"])),
                     interval(float(ext["ci_cluster_lo"]), float(ext["ci_cluster_hi"]))]
                    if ext else ["--", "--"])
            body.append(" & ".join(out) + r" \\")
    header = [r" & & \multicolumn{5}{c}{Strongest classical method} & \multicolumn{2}{c}{LaMa} \\",
              r"\cmidrule(lr){3-7}\cmidrule(l){8-9}",
              r"Cell & Rank & Method & $\Delta$ & 95\% CI & Led (\%) & $\Delta_{\mathrm{cf}}$ & $\Delta$ & 95\% CI \\"]
    return tabular(r"@{}l c l r c r r r c@{}", header, body)


def quality() -> str | None:
    return method_by_cell_table(SWEEP, "psnr_hole", higher_is_better=True, digits=2, with_cap_note=True)


def gmsd_table() -> str | None:
    return method_by_cell_table(SWEEP, "gmsd_hole", higher_is_better=False, digits=3)


TABLES = {"SEVERITY": severity, "QUALITY": quality, "GMSD": gmsd_table, "PAIRED_V2": paired_v2}


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
