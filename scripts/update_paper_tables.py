"""Splice v2 sweep numbers into the manuscript: PSNR/SSIM table, HRP table,
CVR table, and the ablation table (inserted as its own subsection if absent).
Idempotent: replaces content between markers each run."""
from __future__ import annotations

import csv
import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
SUMMARY = ROOT / "results" / "csfr_sweep_2d_v2" / "summary.csv"

METH = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn", "csfr"]
LABEL = {"zero_fill": "Zero-fill", "bilinear": "Linear", "inpaint_telea": "Telea",
         "inpaint_ns": "NS", "dictlearn": "Dict.", "csfr": "CSFR"}
ABL = ["csfr_c1", "csfr_c1c2", "csfr_c1c3", "csfr_c1c4", "csfr"]
ABL_LABEL = {"csfr_c1": "C1 only", "csfr_c1c2": "C1+C2 (TV)", "csfr_c1c3": "C1+C3 (bounds)",
             "csfr_c1c4": "C1+C4 (freq.)", "csfr": "Full CSFR"}


def load():
    rows = collections.defaultdict(dict)
    with SUMMARY.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["family"], r["level"])][r["method"]] = {
                k: float(v) for k, v in r.items()
                if k not in ("family", "level", "method")}
    return rows


def bold(v, is_best, fmt):
    s = fmt.format(v)
    return "\\textbf{" + s + "}" if is_best else s


def psnr_ssim_block(rows):
    out = []
    for fam in ["CF1", "CF2", "CF3", "CF4"]:
        for lvl in ["L1", "L3", "L5"]:
            c = rows[(fam, lvl)]
            ps = {m: c[m]["psnr_db_mean"] for m in METH}
            ss = {m: c[m]["ssim_mean"] for m in METH}
            bp, bs = max(ps, key=ps.get), max(ss, key=ss.get)
            out.append(
                f"{fam} & {lvl} & "
                + " & ".join(bold(ps[m], m == bp, "{:.2f}") for m in METH) + " & "
                + " & ".join(bold(ss[m], m == bs, "{:.3f}") for m in METH) + r" \\")
        if fam != "CF4":
            out.append(r"\midrule")
    return "\n".join(out)


def fam_mean(rows, method, key):
    per = collections.defaultdict(list)
    for (fam, lvl), c in rows.items():
        per[fam].append(c[method][key])
    return {f: sum(v) / len(v) for f, v in per.items()}


def hrp_block(rows):
    fams = ["CF1", "CF2", "CF3", "CF4"]
    vals = {m: fam_mean(rows, m, "hrp_mean") for m in METH}
    best = {f: min(METH, key=lambda m: vals[m][f]) for f in fams}
    out = []
    for m in METH:
        out.append(LABEL[m] + " & " + " & ".join(
            bold(vals[m][f], best[f] == m, "{:.2f}") for f in fams) + r" \\")
    return "\n".join(out)


def cvr_block(rows):
    out = []
    for m in METH:
        cs = []
        for i in range(1, 5):
            v = [rows[k][m][f"cvr_c{i}"] for k in rows]
            cs.append(100 * sum(v) / len(v))
        out.append(LABEL[m] + " & " + " & ".join(f"{c:.2f}" for c in cs) + r" \\")
    return "\n".join(out)


def ablation_block(rows):
    out = []
    for m in ABL:
        psnr = sum(rows[k][m]["psnr_db_mean"] for k in rows) / len(rows)
        ssim = sum(rows[k][m]["ssim_mean"] for k in rows) / len(rows)
        hrp = sum(rows[k][m]["hrp_mean"] for k in rows) / len(rows)
        c2 = 100 * sum(rows[k][m]["cvr_c2"] for k in rows) / len(rows)
        c3 = 100 * sum(rows[k][m]["cvr_c3"] for k in rows) / len(rows)
        c4 = 100 * sum(rows[k][m]["cvr_c4"] for k in rows) / len(rows)
        out.append(f"{ABL_LABEL[m]} & {psnr:.2f} & {ssim:.3f} & {hrp:.2f} & "
                   f"{c2:.2f} & {c3:.2f} & {c4:.2f} \\\\")
    return "\n".join(out)


def splice(tex: str, marker: str, content: str) -> str:
    begin, end = f"% BEGIN AUTO {marker}", f"% END AUTO {marker}"
    block = f"{begin}\n{content}\n{end}"
    if begin in tex:
        return re.sub(re.escape(begin) + r".*?" + re.escape(end),
                      lambda _m: block, tex, flags=re.S)
    raise SystemExit(f"marker {marker} not found in tex — add markers first")


def main():
    rows = load()
    tex = TEX.read_text(encoding="utf-8")
    tex = splice(tex, "PSNR-SSIM", psnr_ssim_block(rows))
    tex = splice(tex, "HRP", hrp_block(rows))
    tex = splice(tex, "CVR", cvr_block(rows))
    tex = splice(tex, "ABLATION", ablation_block(rows))
    TEX.write_text(tex, encoding="utf-8")
    print("tables spliced into", TEX.name)


if __name__ == "__main__":
    main()
