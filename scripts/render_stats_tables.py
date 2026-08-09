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
SUPP = ROOT / "paper" / "paper2_supplementary.tex"
OUT = ROOT / "results" / "tables_tex"

STATS = ROOT / "results" / "paired_stats" / "paired_tests_seedavg.csv"
BENCH = ROOT / "results" / "bench" / "runtime_memory.csv"
EXT = ROOT / "results" / "csfr_sweep_govdocs1" / "summary.csv"
PRIMARY = ROOT / "results" / "csfr_sweep_2d_v2" / "summary.csv"
DOWNSTREAM_METRICS = ROOT / "results" / "downstream" / "metrics.csv"
DOWNSTREAM_STATS = ROOT / "results" / "downstream" / "stats.csv"

LABEL = {"zero_fill": "Zero-fill", "bilinear": "Linear", "inpaint_telea": "Telea",
         "inpaint_ns": "NS", "dictlearn": "Dict.", "csfr": "CSFR",
         # downstream harness names for the opencv methods, plus the
         # external-prior inpainter. LaMa carries no provenance, so its
         # traceability score is 0 by construction rather than by measurement.
         "telea": "Telea", "ns": "NS", "lama": "LaMa (ext.)"}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def splice(tag: str, body: str) -> None:
    """Replace the content between % BEGIN AUTO <tag> / % END AUTO <tag>.

    Searches the main manuscript and the supplementary document, because the
    defensive tables live in the supplement. A tag must appear in exactly one
    of them: a tag found in neither is reported, since a silent miss would let
    a released table drift away from the CSV it claims to come from.
    """
    targets = [p for p in (TEX, SUPP) if p.exists()]
    pat = re.compile(rf"(% BEGIN AUTO {tag}\n).*?(% END AUTO {tag})", re.S)
    hit = False
    for path in targets:
        t = path.read_text(encoding="utf-8")
        if not pat.search(t):
            continue
        path.write_text(pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), t),
                        encoding="utf-8")
        print(f"[tables] spliced {tag} -> {path.name}")
        hit = True
    if not hit:
        print(f"[tables] marker {tag} absent from both documents; wrote file only")


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


# -------------------------------------------------------- downstream utility ---
def downstream_block() -> str:
    """CSFR confident-error vs strongest baseline, per cell.

    Reports CSFR and baseline confident-error (joint and conditional), with
    sample sizes, signal-covering subset sizes, and bootstrap CI significance
    marker. Bootstrap CI computed on signal-covering subset (centre_erased_frac
    >= 0.25); CI is degenerate when signal subset is empty, complete, or near-
    complete. See metrics.csv n_signal column for subset size per cell.
    """
    if not DOWNSTREAM_METRICS.exists():
        return "% downstream metrics not yet run"

    metrics_rows = read(DOWNSTREAM_METRICS)
    metrics_by = collections.defaultdict(dict)
    for r in metrics_rows:
        key = (r["family"], r["level"], r["method"])
        metrics_by[key] = r

    # Load stats to get bootstrap CI info
    stats_rows = read(DOWNSTREAM_STATS) if DOWNSTREAM_STATS.exists() else []
    stats_by = {}
    for r in stats_rows:
        key = (r["family"], r["level"], r["method_b"])
        stats_by[key] = r

    out = []
    for fam in ["CF1", "CF2", "CF3", "CF4"]:
        for lvl in ["L1", "L2", "L3", "L4", "L5"]:
            # Get CSFR row
            csfr_key = (fam, lvl, "csfr")
            if csfr_key not in metrics_by:
                continue
            csfr_row = metrics_by[csfr_key]

            # strongest baseline in this cell, by accuracy. lama is included:
            # it is the external-prior inpainter and it wins several severe
            # cells outright, so excluding it would silently compare CSFR
            # against a weaker opponent than the one that was actually run.
            baseline_accs = {m: float(metrics_by[(fam, lvl, m)]["accuracy"])
                           for m in ["zero_fill", "bilinear", "telea", "ns",
                                     "dictlearn", "lama"]
                           if (fam, lvl, m) in metrics_by}
            if not baseline_accs:
                continue
            best_baseline = max(baseline_accs, key=baseline_accs.get)
            baseline_row = metrics_by[(fam, lvl, best_baseline)]

            # Extract metrics
            csfr_acc = float(csfr_row["accuracy"])
            csfr_recovery = float(csfr_row["recovery"])
            csfr_conf_joint = float(csfr_row["conf_err_joint"])
            csfr_conf_cond = float(csfr_row["conf_err_cond"])
            csfr_n_conf = int(float(csfr_row["n_confident"]))
            csfr_n_signal = int(float(csfr_row["signal_n"]))

            baseline_acc = float(baseline_row["accuracy"])
            baseline_conf_joint = float(baseline_row["conf_err_joint"])
            baseline_n_conf = int(float(baseline_row["n_confident"]))

            # Check if bootstrap CI excludes zero (indicates significance)
            stats_key = (fam, lvl, best_baseline)
            ci_marker = ""
            if stats_key in stats_by:
                stat_row = stats_by[stats_key]
                ci_lo_str = stat_row.get("ci_lo_conferr_diff_signal", "")
                ci_hi_str = stat_row.get("ci_hi_conferr_diff_signal", "")
                if ci_lo_str and ci_hi_str:
                    try:
                        ci_lo = float(ci_lo_str)
                        ci_hi = float(ci_hi_str)
                        # CI excludes zero if both same sign
                        if (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0):
                            ci_marker = r"$^{*}$"
                    except (ValueError, TypeError):
                        pass

            # Format: Family Level Baseline CSFR_acc recovery CSFR_conf_joint
            # CSFR_n_conf Baseline_conf_joint Baseline_n_conf n_signal marker
            out.append(
                f"{fam} & {lvl} & {LABEL.get(best_baseline, best_baseline)} & "
                f"{csfr_acc:.3f} & {csfr_recovery:+.3f} & "
                f"{csfr_conf_joint:.3f} ({csfr_n_conf:d}) & "
                f"{baseline_conf_joint:.3f} ({baseline_n_conf:d}) & "
                f"n_sig={csfr_n_signal}{ci_marker} \\\\")

        if fam != "CF4":
            out.append(r"\midrule")

    return "\n".join(out)


def downstream_abstain_block() -> str:
    """CSFR abstention and signal-covering subset: Bounded-Unknown firing rate
    and isolated-missing-fraction distribution. Reports n_signal for each cell;
    note that signal-covering subset (centre_erased_frac >= 0.25) is degenerate
    for CF1 (all-or-none), CF3/CF4 (nearly complete by geometry). Only CF2
    shows meaningful variation. Bounded-Unknown boolean is nearly a (family,
    level) indicator; continuous isolated_missing_frac is the informative
    statistic.
    """
    if not DOWNSTREAM_METRICS.exists():
        return "% downstream metrics not yet run"
    rows = read(DOWNSTREAM_METRICS)
    csfr_rows = [r for r in rows if r["method"] == "csfr"]
    if not csfr_rows:
        return "% no CSFR rows"

    out = []
    for r in csfr_rows:
        fam, lvl = r["family"], r["level"]
        bu_rate = float(r["bu_rate"]) if r.get("bu_rate") and r["bu_rate"] != "" else 0.0
        imf_mean = float(r["imf_mean"]) if r.get("imf_mean") and r["imf_mean"] != "" else 0.0
        imf_sd = float(r["imf_sd"]) if r.get("imf_sd") and r["imf_sd"] != "" else 0.0
        imf_median = float(r["imf_median"]) if r.get("imf_median") and r["imf_median"] != "" else 0.0
        n_signal = int(float(r["signal_n"])) if r.get("signal_n") else 0

        out.append(
            f"{fam} & {lvl} & {bu_rate:.3f} & "
            f"{imf_mean:.3f} & {imf_median:.3f} & {imf_sd:.3f} & {n_signal:d} \\\\")

    return "\n".join(out)


def main() -> int:
    emit("paired_rows.tex", "PAIRED", paired_block())
    emit("bench_rows.tex", "BENCH", bench_block())
    emit("external_rows.tex", "EXTERNAL", external_block())
    emit("budget_rows.tex", "BUDGET_SENS", budget_block())
    emit("downstream_rows.tex", "DOWNSTREAM", downstream_block())
    emit("downstream_abstain_rows.tex", "DOWNSTREAM_ABSTAIN", downstream_abstain_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
