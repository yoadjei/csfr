"""Emit the LaTeX rows for the paired-inference, runtime, and external-validation
tables from the released CSVs, so every number in the manuscript is generated
rather than transcribed.

Writes into results/tables_tex/ and splices the marked blocks in the .tex.

Usage:
    python scripts/render_stats_tables.py
"""
from __future__ import annotations

import csv
import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
OUT = ROOT / "results" / "tables_tex"

STATS = ROOT / "results" / "paired_stats" / "paired_tests_seedavg.csv"
BENCH = ROOT / "results" / "bench" / "runtime_memory.csv"
EXT = ROOT / "results" / "csfr_sweep_govdocs1" / "summary.csv"
PRIMARY = ROOT / "results" / "csfr_sweep_2d_v2" / "summary.csv"

LABEL = {"zero_fill": "Zero-fill", "bilinear": "Linear", "inpaint_telea": "Telea",
         "inpaint_ns": "NS", "dictlearn": "Dict.", "csfr": "CSFR"}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def splice(tag: str, body: str) -> None:
    """Replace the content between % BEGIN AUTO <tag> / % END AUTO <tag>."""
    t = TEX.read_text(encoding="utf-8")
    pat = re.compile(rf"(% BEGIN AUTO {tag}\n).*?(% END AUTO {tag})", re.S)
    if not pat.search(t):
        print(f"[tables] marker {tag} absent; wrote file only")
        return
    TEX.write_text(pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), t),
                   encoding="utf-8")
    print(f"[tables] spliced {tag}")


def emit(name: str, tag: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body + "\n", encoding="utf-8")
    splice(tag, body)


# ------------------------------------------------------- paired inference ---
def stars(p: float) -> str:
    return "" if p >= 0.05 else (r"$^{*}$" if p >= 1e-3 else r"$^{**}$")


def paired_block() -> str:
    """CSFR vs the strongest baseline in each cell, seed-averaged per patch."""
    rows = [r for r in read(STATS) if r["metric"] == "psnr"]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["family"], r["level"])].append(r)
    out = []
    for fam in ["CF1", "CF2", "CF3", "CF4"]:
        for lvl in ["L1", "L2", "L3", "L4", "L5"]:
            cand = by[(fam, lvl)]
            r = max(cand, key=lambda r: float(r["base_mean"]))
            d = float(r["mean_diff"])
            cl_lo, cl_hi = float(r["diff_ci_lo_cluster"]), float(r["diff_ci_hi_cluster"])
            # A cluster interval spanning zero means the sign is not established
            # once source-image dependence is respected.
            robust = "yes" if (cl_lo > 0) == (cl_hi > 0) else "no"
            out.append(
                f"{fam} & {lvl} & {LABEL[r['baseline']]} & "
                f"{d:+.2f}{stars(float(r['p_wilcoxon_holm']))} & "
                f"[{float(r['diff_ci_lo']):+.2f}, {float(r['diff_ci_hi']):+.2f}] & "
                f"[{cl_lo:+.2f}, {cl_hi:+.2f}] & "
                f"{float(r['cohens_dz']):+.2f} & {robust} \\\\")
        if fam != "CF4":
            out.append(r"\midrule")
    return "\n".join(out)


# ------------------------------------------------------ runtime and memory ---
def bench_block() -> str:
    if not BENCH.exists():
        return "% benchmark not yet run"
    rows = read(BENCH)
    order = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn", "csfr"]
    by = {r["method"]: r for r in rows}
    out = []
    for m in order:
        if m not in by:
            continue
        r = by[m]
        out.append(
            f"{r['label']} & {float(r['ms_per_patch']):.2f} & "
            f"{float(r['mean_s']):.3f} & {float(r['sd_s']):.3f} & "
            f"{float(r['median_s']):.3f} & {float(r['peak_alloc_mb']):.1f} \\\\")
    return "\n".join(out)


# -------------------------------------------------------- external corpus ---
def external_block() -> str:
    if not EXT.exists():
        return "% external sweep not yet run"
    def load(path):
        d = collections.defaultdict(dict)
        for r in read(path):
            d[(r["family"], r["level"])][r["method"]] = r
        return d
    ext, pri = load(EXT), load(PRIMARY)
    meth = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn", "csfr"]
    out = []
    for fam in ["CF1", "CF2", "CF3", "CF4"]:
        for lvl in ["L1", "L3", "L5"]:
            cells = ext[(fam, lvl)]
            if not cells:
                continue
            ps = {m: float(cells[m]["psnr_db_mean"]) for m in meth if m in cells}
            best = max(ps, key=ps.get)
            # rank of CSFR on this corpus, and the primary-corpus gap for contrast.

            rank = sorted(ps, key=ps.get, reverse=True).index("csfr") + 1
            gap_ext = ps["csfr"] - max(v for m, v in ps.items() if m != "csfr")
            pri_ps = {m: float(pri[(fam, lvl)][m]["psnr_db_mean"])
                      for m in meth if m in pri[(fam, lvl)]}
            gap_pri = pri_ps["csfr"] - max(v for m, v in pri_ps.items() if m != "csfr")
            cells_fmt = " & ".join(
                (r"\textbf{" + f"{ps[m]:.2f}" + "}") if m == best else f"{ps[m]:.2f}"
                for m in meth)
            out.append(f"{fam} & {lvl} & {cells_fmt} & {rank} & {gap_ext:+.2f} & {gap_pri:+.2f} \\\\")
        if fam != "CF4":
            out.append(r"\midrule")
    return "\n".join(out)


def budget_block() -> str:
    path = ROOT / "results" / "budget_sensitivity" / "budget.csv"
    if not path.exists():
        return "% budget sensitivity not yet run"
    rows = read(path)
    by = collections.defaultdict(dict)
    gap = {}
    for r in rows:
        by[r["cell"]][int(r["budget"])] = float(r["psnr_db"])
        if int(r["budget"]) == max(int(x["budget"]) for x in rows if x["cell"] == r["cell"]):
            gap[r["cell"]] = float(r["gap_to_best_baseline"])
    order = ["CF1_L5", "CF2_L3", "CF2_L5", "CF3_L5", "CF4_L5"]
    budgets = sorted({int(r["budget"]) for r in rows})
    out = []
    for cell in order:
        if cell not in by:
            continue
        vals = " & ".join(f"{by[cell][b]:.2f}" for b in budgets)
        out.append(f"{cell.replace('_', '~')} & {vals} & {gap[cell]:+.2f} \\\\")
    return "\n".join(out)


def main() -> int:
    emit("paired_rows.tex", "PAIRED", paired_block())
    emit("bench_rows.tex", "BENCH", bench_block())
    emit("external_rows.tex", "EXTERNAL", external_block())
    emit("budget_rows.tex", "BUDGET_SENS", budget_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
