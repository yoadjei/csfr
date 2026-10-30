"""Render the exploratory comparison figure (F9): mean PSNR against the
study-defined auditability composite for the seven evaluated methods, from the
v2 summary CSV. This is the only generator of paper2_F9_method_frontier.png;
scripts/render_figures.py no longer draws its own copy."""
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

# okabe-ito colour-blind-safe palette. the previous scheme put red (#d62728),
# green (#2e7d32) and orange (#e65100) on three adjacent points, which is the
# textbook deuteranopia failure: those three are the ones a red-green colour
# blind reader cannot separate, and they were the three clustered markers.
# marker shape carries the same information independently of hue, so the figure
# also reads in greyscale, and the assigned colours differ in lightness as well
# as hue for the same reason.
METH = {"zero_fill": ("Zero-fill", "#999999", "o"),
        "bilinear": ("Bilinear", "#56B4E9", "s"),
        "inpaint_telea": ("Telea", "#009E73", "^"),
        "inpaint_ns": ("Navier-Stokes", "#E69F00", "v"),
        "dictlearn": ("Dict. learning", "#CC79A7", "x"),
        "lama": ("LaMa (external-prior)", "#D55E00", "*"),
        "csfr": ("CSFR", "#003366", "D")}
# per-method label offsets (points) to avoid collisions of near-coincident points
# telea, navier-stokes and lama sit within 0.8 db and 0.011 validity of each
# other, so their labels are fanned out with leader lines rather than placed
# adjacent to the markers, where they overlapped each other and the points.
# dictlearn sits just below the 2/3 ceiling, so its label goes below the marker:
# placed above, it straddles the line and reads as though the method were inside
# the region only CSFR reaches.
LABEL_OFFSET = {"inpaint_telea": (-56, -30), "inpaint_ns": (-16, 42),
                "dictlearn": (8, -20), "lama": (-30, -40), "csfr": (12, 6),
                "bilinear": (10, -18), "zero_fill": (10, 8)}
# methods whose label needs a leader line back to its marker
LEADERED = {"inpaint_telea", "inpaint_ns", "lama"}
# audit record coverage by method. only the csfr implementation was instrumented
# to emit a sensitivity record; the composite needs a number where the paper
# reports "not available", so the others enter as 0. that is a statement about
# the evaluated software, not about what the algorithms could be made to log.
ARC = {"csfr": 1.0}

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
    validity = ((1 - cvr) + (1 - hrp / hrp_max) + ARC.get(m, 0.0)) / 3
    kw = {} if marker == "x" else {"edgecolor": "black", "linewidth": 0.6}
    ax.scatter(psnr, validity, s=110, color=color, marker=marker, zorder=3, **kw)
    arrow = (dict(arrowstyle="-", color=color, linewidth=0.7,
                  shrinkA=0, shrinkB=4) if m in LEADERED else None)
    ax.annotate(label, (psnr, validity), xytext=LABEL_OFFSET.get(m, (6, 5)),
                textcoords="offset points", fontsize=10, color=color,
                fontweight="bold", arrowprops=arrow)

# the composite is ((1 - cvr) + (1 - hrp/hrp_max) + ARC)/3, so a method entered
# at ARC = 0 cannot exceed 2/3 however good its reconstruction is. that ceiling
# is a property of the composite, and the region above it is reachable only by
# an implementation that released an audit record.
#
# an earlier version shaded the bottom third and called it "the TS = 0 region,
# inadmissible for forensic use". that was wrong three ways: six of the seven
# methods sit at 0 on this axis and five of them plot above the band; the only
# point in the band was zero-fill, which is not an external-prior method; and
# asserting inadmissibility contradicts this paper's own position that
# admissibility is a determination for a court.
ARC_CEILING = 2 / 3
ax.axhspan(ARC_CEILING, 1.0, color="#003366", alpha=0.06)
ax.axhline(ARC_CEILING, color="#003366", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(14.3, 0.94,
        "Above 2/3 only with a released audit record: at ARC = 0 the composite\n"
        "cannot exceed 2/3, whatever the reconstruction quality",
        fontsize=8.5, color="#003366", style="italic")
ax.set_xlabel("Mean PSNR over 20-cell sweep (dB)")
ax.set_ylabel("Auditability composite (exploratory, study-defined)")
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT.name)
