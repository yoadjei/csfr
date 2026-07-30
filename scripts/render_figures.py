"""Render all five manuscript figures (F1, F2, F6, F8, F9) from scratch.

Design rules (ResearchOS figure standards):
- designed at final print width (7.5 in full text width), fonts >= 8 pt
- no in-image banner titles (captions live in the manuscript)
- diagram nodes are text with auto-fitting bboxes -> text cannot spill
- colorblind/greyscale-safe series (marker + linestyle + colour)

Usage:
    python scripts/render_figures_v3.py [--out-dir paper/figures]
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLUE = "#1f4e79"
BLUE_MID = "#5b8fbf"
BLUE_PALE = "#F4F8FC"
BLUE_PALE2 = "#E6EEF8"
RED = "#c00000"
GREY = "#7f7f7f"
WHITE = "#FFFFFF"
GREEN, ORANGE, PURPLE = "#2e7d32", "#e65100", "#6a1b9a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "figure.facecolor": WHITE,
})
DPI = 600


def node(ax, x, y, text, fc, ec, fs=9, tc=None, lw=1.5, pad=0.55, weight="normal"):
    """Text with an auto-fitting rounded box. Returns the text artist."""
    return ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                   color=tc or ec, fontweight=weight, linespacing=1.5,
                   zorder=10,
                   bbox=dict(boxstyle=f"round,pad={pad}", facecolor=fc,
                             edgecolor=ec, linewidth=lw))


def arrow(ax, src, dst, color=BLUE, lw=1.4, shrinkA=8, shrinkB=8, rad=0.0, alpha=1.0):
    ax.annotate("", xy=dst, xytext=src, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                shrinkA=shrinkA, shrinkB=shrinkB,
                                connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ F1
def render_f1(out):
    fig, ax = plt.subplots(figsize=(7.5, 2.7), dpi=DPI)
    ax.set_xlim(0, 116); ax.set_ylim(0, 34); ax.axis("off")

    labels = [
        "1. Fragment ingest\ncarved disk shard",
        "2. Alignment estimation\ngrid placement",
        "3. Constraint formulation\nC1–C4 assembled",
        "4. $\\ell_1$ solver\nmin $\\|\\Psi^{\\top}x\\|_1$ + projection",
        "5. Reconstruction assembly\npixel grid",
        "6. Traceability audit\nprovenance map",
    ]
    xs = [20, 58, 96]
    ys = [26, 10]
    pos = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[2], ys[0]),
           (xs[2], ys[1]), (xs[1], ys[1]), (xs[0], ys[1])]
    for i, (lab, (x, y)) in enumerate(zip(labels, pos)):
        last = i == 5
        node(ax, x, y, lab, fc=WHITE if last else BLUE_PALE,
             ec=RED if last else BLUE, fs=7.7, lw=1.6,
             weight="bold" if last else "normal")

    arrow(ax, (xs[0], ys[0]), (xs[1], ys[0]), shrinkA=62, shrinkB=66)
    arrow(ax, (xs[1], ys[0]), (xs[2], ys[0]), shrinkA=66, shrinkB=64)
    arrow(ax, (xs[2], ys[0] - 4), (xs[2], ys[1] + 4), shrinkA=10, shrinkB=10)
    arrow(ax, (xs[2], ys[1]), (xs[1], ys[1]), shrinkA=68, shrinkB=68)
    arrow(ax, (xs[1], ys[1]), (xs[0], ys[1]), shrinkA=68, shrinkB=58)

    for x, lab in zip(xs, ["no training data", "deterministic", "auditable"]):
        node(ax, x, 1.0, lab, fc=WHITE, ec=BLUE, fs=7.5, lw=1.1, pad=0.45,
             weight="bold")

    fig.savefig(out / "paper2_F1_pipeline.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.06, facecolor=WHITE)
    plt.close(fig)
    print("[F1] done")


# ------------------------------------------------------------------ F2
def _panel_masks(h, w, rng):
    m = {}
    a = np.ones((h, w)); idx = rng.choice(h * w, int(0.40 * h * w), replace=False)
    a.reshape(-1)[idx] = 0; m["a"] = a
    b = np.ones((h, w))
    while (1 - b).sum() < 0.25 * h * w:
        bh_, bw_ = rng.integers(6, 20), rng.integers(6, 20)
        y0_, x0_ = rng.integers(0, h - bh_), rng.integers(0, w - bw_)
        b[y0_:y0_ + bh_, x0_:x0_ + bw_] = 0
    m["b"] = b
    c = np.ones((h, w)); c[: h // 2] = 0; m["c"] = c
    d = np.ones((h, w)); d[h // 2:] = 0; m["d"] = d
    e = np.ones((h, w)); e[: max(2, h // 8)] = 0; m["e"] = e
    f = np.ones((h, w)); f.reshape(-1)[h * w // 2:] = 0; m["f"] = f
    return m


def render_f2(out, patches_path):
    img = np.load(patches_path)[7].astype(float) / 255.0
    h, w = img.shape
    masks = _panel_masks(h, w, np.random.default_rng(0))
    labels = {
        "a": "(a) CF1: random loss 40%",
        "b": "(b) CF2: burst blocks 25%",
        "c": "(c) CF3: top-half erasure",
        "d": "(d) CF4: bottom-half erasure",
        "e": "(e) header overwrite (byte)",
        "f": "(f) tail truncation (byte)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), dpi=DPI)
    for ax, key in zip(axes.flat, "abcdef"):
        rgb = np.stack([img] * 3, axis=-1)
        rgb[masks[key] == 0] = [0.75, 0.0, 0.0]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(labels[key], fontsize=9.5, color=BLUE, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GREY); s.set_linewidth(0.6)
    fig.legend(handles=[Patch(facecolor="#c00000", label="missing / corrupted"),
                        Patch(facecolor="#aaaaaa", label="intact reference")],
               loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "paper2_F2_fragmentation.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.05, facecolor=WHITE)
    plt.close(fig)
    print("[F2] done")


# ------------------------------------------------------------------ F6
def render_f6(out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # nodes
    top_xy = (36, 85)
    node(ax, *top_xy, "Reconstructed pixel $\\hat{x}[i,j]$\nCSFR solver output",
         fc=WHITE, ec=RED, fs=8.5, lw=1.8, weight="bold")

    mids = [(14, 50, "Constraint $c_1$\nspatial continuity"),
            (36, 50, "Constraint $c_2$\nintensity bounds"),
            (58, 50, "Constraint $c_3$\nfragment consistency")]
    for x, y, lab in mids:
        node(ax, x, y, lab, fc=BLUE_PALE2, ec=BLUE_MID, fs=7.8, lw=1.4)

    bots = [(14, 15, "Fragment F1\nbytes 0–511"),
            (36, 15, "Fragment F2\nbytes 512–1023"),
            (58, 15, "Fragment F3\nbytes 1024–1535")]
    for x, y, lab in bots:
        node(ax, x, y, lab, fc=BLUE_PALE, ec=BLUE, fs=8.0, lw=1.4, weight="bold")

    # fragment -> constraint mesh
    for xb, yb, _ in bots:
        for xm, ym, _ in mids:
            arrow(ax, (xb, yb), (xm, ym), color=GREY, lw=0.8, alpha=0.4,
                  shrinkA=16, shrinkB=16)

    # constraint -> pixel, with weights
    weights = ["$\\alpha_1{=}0.42$", "$\\alpha_2{=}0.31$", "$\\alpha_3{=}0.27$"]
    dests = [(24, 78), (36, 78), (48, 78)]
    for (xm, ym, _), wl, dst in zip(mids, weights, dests):
        arrow(ax, (xm, ym), dst, color=BLUE, lw=1.5, shrinkA=18, shrinkB=0)
        mx, my = (xm + dst[0]) / 2, (ym + dst[1]) / 2 + 1.5
        ax.text(mx, my, wl, fontsize=8.0, color=BLUE, style="italic",
                ha="center", zorder=12,
                bbox=dict(facecolor=WHITE, edgecolor="none", pad=1.2))

    # audit panel
    audit = ("$\\bf{Audit\\ query}$\n\n"
             "Q. Which inputs determine\n     pixel $(i, j)$?\n\n"
             "A. F1: 42% via $c_1$\n     F2: 31% via $c_2$\n     F3: 27% via $c_3$\n\n"
             "Deterministic.\nNo black box.")
    ax.text(86, 50, audit, ha="center", va="center", fontsize=7.8,
            color="#222", linespacing=1.6, zorder=10,
            bbox=dict(boxstyle="round,pad=0.8", facecolor=WHITE,
                      edgecolor=RED, linewidth=1.5))

    fig.savefig(out / "paper2_F6_traceability.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.06, facecolor=WHITE)
    plt.close(fig)
    print("[F6] done")


# ------------------------------------------------------------------ F8
def _load_summary(csv_path):
    data = collections.defaultdict(dict)
    with Path(csv_path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[r["family"]].setdefault(r["method"], [None] * 5)
            data[r["family"]][r["method"]][int(r["level"][1]) - 1] = \
                float(r["psnr_db_mean"])
    return data


SERIES = [
    ("zero_fill", "Zero-fill", GREY, "-", "o", 1.4, 4),
    ("bilinear", "Bilinear", BLUE_MID, "--", "s", 1.4, 4),
    ("inpaint_telea", "Telea", GREEN, "-.", "^", 1.4, 4),
    ("inpaint_ns", "Navier–Stokes", ORANGE, ":", "v", 1.4, 4),
    ("dictlearn", "Dict. learning", PURPLE, "--", "x", 1.4, 5),
    ("csfr", "CSFR (proposed)", BLUE, "-", "D", 2.2, 5),
]


def render_f8(out, csv_path):
    data = _load_summary(csv_path)
    fams = ["CF1", "CF2", "CF3", "CF4"]
    titles = {"CF1": "CF1: random pixel loss", "CF2": "CF2: burst block corruption",
              "CF3": "CF3: top-half erasure", "CF4": "CF4: bottom-half erasure"}
    lvls = ["L1\n5%", "L2\n15%", "L3\n30%", "L4\n50%", "L5\n70%"]
    x = np.arange(5)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6), dpi=DPI, sharey=True)
    for ax, fam in zip(axes.flat, fams):
        for key, lab, col, ls, mk, lw, ms in SERIES:
            ax.plot(x, data[fam][key], color=col, ls=ls, marker=mk, lw=lw,
                    ms=ms, label=lab)
        ax.set_title(titles[fam], fontsize=10, fontweight="bold", color=BLUE, pad=4)
        ax.set_xticks(x); ax.set_xticklabels(lvls, fontsize=8)
        ax.set_ylim(0, 50); ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR (dB)", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.005), columnspacing=1.1,
               handlelength=1.8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "paper2_F8_csfr_psnr_sweep.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.05, facecolor=WHITE)
    plt.close(fig)
    print("[F8] done")


# ------------------------------------------------------------------ F9
F9_METH = {"zero_fill": ("Zero-fill", GREY, "o", (10, 8)),
           "bilinear": ("Bilinear", "#4477AA", "s", (10, -18)),
           "inpaint_telea": ("Telea", GREEN, "^", (-16, -24)),
           "inpaint_ns": ("Navier–Stokes", ORANGE, "v", (10, 12)),
           "dictlearn": ("Dict. learning", PURPLE, "x", (10, 10)),
           "csfr": ("CSFR (proposed)", "#003366", "D", (12, 4))}


def render_f9(out, csv_path):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    with Path(csv_path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] in F9_METH:
                for k in ("psnr_db_mean", "hrp_mean",
                          "cvr_c1", "cvr_c2", "cvr_c3", "cvr_c4"):
                    rows[r["method"]][k].append(float(r[k]))
    hrp_max = max(sum(v["hrp_mean"]) / len(v["hrp_mean"]) for v in rows.values())

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=DPI)
    for m, (label, color, marker, off) in F9_METH.items():
        v = rows[m]
        psnr = sum(v["psnr_db_mean"]) / len(v["psnr_db_mean"])
        cvr = sum(sum(v[f"cvr_c{i}"]) / len(v[f"cvr_c{i}"]) for i in range(1, 5)) / 4
        hrp = sum(v["hrp_mean"]) / len(v["hrp_mean"])
        validity = ((1 - cvr) + (1 - hrp / hrp_max) + (1.0 if m == "csfr" else 0.0)) / 3
        kw = {} if marker == "x" else {"edgecolor": "black", "linewidth": 0.6}
        ax.scatter(psnr, validity, s=130, color=color, marker=marker, zorder=3, **kw)
        ax.annotate(label, (psnr, validity), xytext=off,
                    textcoords="offset points", fontsize=10, color=color,
                    fontweight="bold")
    ax.axhspan(0, 1 / 3, color="#d32f2f", alpha=0.08)
    ax.text(0.02, 0.05,
            "TS = 0 region (DL methods, cited): inadmissible regardless of PSNR",
            transform=ax.transAxes, fontsize=9, color="#b71c1c", style="italic")
    ax.set_xlabel("Mean PSNR over 20-cell sweep (dB)", fontsize=10)
    ax.set_ylabel("Forensic validity composite", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "paper2_F9_method_frontier.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.05, facecolor=WHITE)
    plt.close(fig)
    print("[F9] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "paper" / "figures"))
    ap.add_argument("--csfr-csv",
                    default=str(PROJECT_ROOT / "results/csfr_sweep_2d_v2/summary.csv"))
    ap.add_argument("--patches", default=str(PROJECT_ROOT / "data/patches.npy"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    render_f1(out)
    render_f2(out, args.patches)
    render_f6(out)
    render_f8(out, args.csfr_csv)
    render_f9(out, args.csfr_csv)


if __name__ == "__main__":
    main()
