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


TABLES = {"SEVERITY": severity}


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
