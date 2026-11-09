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


def cap_note(s: dict, cells: list[tuple[str, str]], ncols: int) -> str:
    """Table row reporting how many patch-runs reach the hole-PSNR cap: the
    largest count of any method in any tabulated cell."""
    rows = [r for (f, lv, _), r in s.items() if (f, lv) in cells]
    worst = max(rows, key=lambda r: int(r["n_at_cap"]))
    k = int(worst["n_at_cap"])
    if k == 0:
        text = rf"No patch-run reaches the {CAP:g}\,dB cap."
    else:
        name = dict([*METHOD_ORDER, FLOOR])[worst["method"]]
        text = (rf"Patch-runs at the {CAP:g}\,dB cap: at most {k} of {worst['n_patches']} per cell "
                rf"({name}, {worst['family']}~{worst['level']}).")
    return rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\footnotesize {text}}} \\"


def method_by_cell_table(sweep: Path, metric: str, higher_is_better: bool, digits: int,
                         levels: tuple[str, ...] = MAIN_LEVELS, with_cap_note: bool = False) -> str | None:
    """Rows are methods, columns are (family, level) cells; the best value of
    each column among the compared methods (not the floor) is bold. A cell a
    method did not run in is a dash."""
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

    def row(key: str, name: str, mark_best: bool) -> str:
        out = []
        for c in cells:
            r = s.get((*c, key))
            if r is None:
                out.append("--")
                continue
            v = float(r[f"{metric}_mean"])
            txt = f"{v:.{digits}f}"
            out.append(rf"\textbf{{{txt}}}" if mark_best and txt == f"{best[c]:.{digits}f}" else txt)
        return f"{name} & " + " & ".join(out) + r" \\"

    lines = [row(k, n, True) for k, n in METHOD_ORDER if any((*c, k) in s for c in cells)]
    lines += [r"\midrule", row(*FLOOR, False)]
    if with_cap_note:
        lines += [r"\midrule", cap_note(s, cells, 1 + len(cells))]
    return "\n".join(lines)


def quality() -> str | None:
    return method_by_cell_table(SWEEP, "psnr_hole", higher_is_better=True, digits=2, with_cap_note=True)


def gmsd_table() -> str | None:
    return method_by_cell_table(SWEEP, "gmsd_hole", higher_is_better=False, digits=3)


TABLES = {"SEVERITY": severity, "QUALITY": quality, "GMSD": gmsd_table}


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
