"""Render the measured method-frontier figure (F9): PSNR vs forensic-validity
composite for the six same-harness methods, from the v2 summary CSV."""
from __future__ import annotations

import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results/csfr_sweep_2d_v2/summary.csv"
OUT = ROOT / "paper/figures/paper2_F9_method_frontier.png"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42})

METH = {"zero_fill": ("Zero-fill", "#7f7f7f", "o"),
        "bilinear": ("Bilinear", "#4477AA", "s"),
        "inpaint_telea": ("Telea", "#2e7d32", "^"),
        "inpaint_ns": ("Navier-Stokes", "#e65100", "v"),
        "dictlearn": ("Dict. learning", "#6a1b9a", "x"),
        "csfr": ("CSFR", "#003366", "D")}
# per-method label offsets (points) to avoid collisions of near-coincident points
LABEL_OFFSET = {"inpaint_telea": (-14, -22), "inpaint_ns": (8, 10),
                "dictlearn": (8, 8), "csfr": (10, 4),
                "bilinear": (8, -16), "zero_fill": (8, 8)}
TS = {"csfr": 1.0}  # provenance map released only by CSFR; others release none

rows = collections.defaultdict(lambda: collections.defaultdict(list))
with SUMMARY.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["method"] in METH:
            for k in ("psnr_db_mean", "hrp_mean", "cvr_c1", "cvr_c2", "cvr_c3", "cvr_c4"):
                rows[r["method"]][k].append(float(r[k]))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
hrp_all = [sum(v["hrp_mean"]) / len(v["hrp_mean"]) for v in rows.values()]
hrp_max = max(hrp_all)
for m, (label, color, marker) in METH.items():
    v = rows[m]
    psnr = sum(v["psnr_db_mean"]) / len(v["psnr_db_mean"])
    cvr = sum(sum(v[f"cvr_c{i}"]) / len(v[f"cvr_c{i}"]) for i in range(1, 5)) / 4
    hrp = sum(v["hrp_mean"]) / len(v["hrp_mean"])
    validity = ((1 - cvr) + (1 - hrp / hrp_max) + TS.get(m, 0.0)) / 3
    kw = {} if marker == "x" else {"edgecolor": "black", "linewidth": 0.6}
    ax.scatter(psnr, validity, s=110, color=color, marker=marker, zorder=3, **kw)
    ax.annotate(label, (psnr, validity), xytext=LABEL_OFFSET.get(m, (6, 5)),
                textcoords="offset points", fontsize=10, color=color,
                fontweight="bold")

ax.axhspan(0, 1 / 3, color="#d32f2f", alpha=0.08)
ax.text(14.5, 0.05, "TS = 0 region (DL methods, cited): inadmissible\n"
        "regardless of PSNR", fontsize=8.5, color="#b71c1c", style="italic")
ax.set_xlabel("Mean PSNR over 20-cell sweep (dB)")
ax.set_ylabel("Forensic validity composite")
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT.name)
