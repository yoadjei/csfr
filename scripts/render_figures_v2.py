"""Generate the round-2 figures of the manuscript as vector PDFs.

    python scripts/render_figures_v2.py

Each figure is a pair of functions: a data function that reads saved results
(and recomputes what a figure needs from them) and returns None while those
results do not exist, and a draw function that writes paper/figures/<name>.pdf.
The numbers a figure shows are written next to it in
results/figures_v2/<name>.json. Text stays vector with embedded TrueType fonts;
image panels are embedded at their native resolution, one cell per pixel. The
PDFs carry no creation date, so an unchanged figure is byte-identical.

Figures so far:
  record_walkthrough  the sensitivity record of one CSFR reconstruction: three
                      imputed pixels' rows, how their weight falls off with
                      distance, and the perturb-and-re-solve check of one column.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_objective import HI  # noqa: E402
from csfr_solver import SolverConfig, solve  # noqa: E402
from sensitivity import (RecordConfig, fidelity_patch, free_set,  # noqa: E402
                         sensitivity_column, sensitivity_row)

_spec = importlib.util.spec_from_file_location("_fid", ROOT / "scripts" / "run_sensitivity_fidelity.py")
_fid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fid)
imputed_queries, observed_queries = _fid.imputed_queries, _fid.observed_queries

FIG_DIR = ROOT / "paper" / "figures"
DATA_DIR = ROOT / "results" / "figures_v2"

TEXT_WIDTH_IN = 7.16                 # IEEE two-column text width
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, embedded
         "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"}
# categorical slots 1-3 of the dataviz reference palette (blue, orange, aqua):
# every pair passes the colour-vision checks on a light surface; aqua is below
# 3:1 contrast, so each series also has its own marker and a legend entry
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK_MUTED = "#52514e"
# diverging: blue (negative) and red (positive) arms, neutral grey at zero
SIGNED = LinearSegmentedColormap.from_list(
    "signed", ["#184f95", "#5598e7", "#f0efec", "#e36f6e", "#a8302f"])
HOLE = "#f0efec"                     # the neutral grey: hole pixels in image panels


def _save(fig, name: str, fig_dir: Path) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.pdf"
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def sci(x: float, digits: int = 1) -> str:
    """1.7e-11 -> 1.7 x 10^-11 in mathtext; 0 -> 0."""
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ------------------------------------------------------------ record walkthrough

WALK_CELL, WALK_SEED, WALK_PATCH = "CF3_L5", 0, 0     # the patch whose manifest E6 writes
WALK_DELTAS = (0.25, 1.0, 4.0)


def walkthrough_data(sweep: Path = ROOT / "results" / "sweep_v2_govdocs1",
                     patches_path: Path = ROOT / "data" / "patches_govdocs1_v2.npy",
                     cell: str = WALK_CELL, seed: int = WALK_SEED, patch: int = WALK_PATCH,
                     deltas: tuple[float, ...] = WALK_DELTAS) -> dict | None:
    """Recompute the sensitivity record of one saved CSFR reconstruction.

    The patch is re-solved with the configuration the sweep recorded and checked
    against the saved output. Rows: the imputed pixels the fidelity experiment
    queries in its manifest (distance 1 and 3 from the observed region, and the
    deepest). Column: the observed pixel at distance 1 from the hole, predicted
    and measured by perturb-and-re-solve against the float64 reference solve, at
    each delta; the relative errors are checked against
    sensitivity.fidelity_patch, so the figure shows what the experiment measures.
    """
    d = sweep / cell / f"seed{seed}"
    if not all((d / f).exists() for f in ("y.npy", "mask.npy", "x_hat_csfr.npy", "metrics.json")):
        return None
    y = np.load(d / "y.npy")[patch:patch + 1]
    M = np.load(d / "mask.npy")[patch:patch + 1]
    truth = np.load(patches_path)[patch].astype(np.float32)
    if not np.array_equal(truth * M[0], y[0]):
        raise RuntimeError(f"{d}: patch {patch} of {patches_path.name} is not the observed patch")
    cfg = SolverConfig(**json.loads((d / "metrics.json").read_text(encoding="utf-8"))["methods"]["csfr"]["solver"])
    X, cert = solve(torch.from_numpy(y), torch.from_numpy(M), cfg)
    dev = float(np.abs(X[0].numpy() - np.load(d / "x_hat_csfr.npy")[patch]).max())
    if not bool(cert.converged[0]) or dev > 1e-3:
        raise RuntimeError(f"{d}: re-solved patch {patch} differs from the sweep by {dev}")

    Mb = torch.from_numpy(M[0] == 1)
    yi = torch.from_numpy(y[0]).to(torch.float64)
    x = X[0].to(torch.float64)
    rc = RecordConfig()
    hole_dist = _fid.distance_transform_cdt(M[0] == 1, metric="chessboard")  # observed: distance to the hole
    rows = []
    for q in imputed_queries(M[0] == 1):
        w, info = sensitivity_row(x, yi, Mb, q, cfg, rc)
        rows.append({"pixel": list(q), "row": w.numpy(), "cg_iters": int(info["cg_iters"]),
                     "cg_residual": float(info["cg_residual"]), "support": int(info["support"]),
                     "affine_defect": info["affine_defect"]})

    dist, k = observed_queries(M[0] == 1)[0]
    free = free_set(x, Mb, rc.bound_margin)
    pred, _ = sensitivity_column(x, yi, Mb, k, cfg, rc)
    signs = [-1.0 if float(yi[k]) + dl > HI else 1.0 for dl in deltas]
    Y = yi.unsqueeze(0).repeat(len(deltas) + 1, 1, 1).clone()
    for j, (dl, s) in enumerate(zip(deltas, signs), start=1):
        Y[j][k] += s * dl
    Ms = Mb.to(torch.float64).unsqueeze(0).repeat(len(deltas) + 1, 1, 1)
    Xr, cr = solve(Y, Ms, replace(cfg, dtype="float64", tol=rc.ref_tol, max_iter=rc.ref_max_iter))
    official = fidelity_patch(x, yi, Mb, [k], cfg, rc, deltas=deltas, operational=False)
    column = []
    for j, (dl, s, off) in enumerate(zip(deltas, signs, official), start=1):
        actual = ((Xr[j] - Xr[0]) / (s * dl))[free]
        rel = float((actual - pred[free]).norm() / actual.norm())
        if abs(rel - off["rel_l2_error_ref"]) > 1e-9 * max(1.0, rel):
            raise RuntimeError(f"delta {dl}: relative error {rel} differs from fidelity_patch's "
                               f"{off['rel_l2_error_ref']}")
        column.append({"delta": dl, "actual": actual.numpy(), "rel_l2_error_ref": rel,
                       "sign_agreement_topk": off["sign_agreement_topk"],
                       "ref_converged": bool(cr.converged[0] and cr.converged[j])})
    return {"sweep": sweep.name, "cell": cell, "seed": seed, "patch": patch,
            "truth": truth, "observed": M[0] == 1, "reconstruction": x.numpy(),
            "hole_distance": hole_dist, "iterations": int(cert.iterations[0]),
            "solver": asdict(cfg), "record": asdict(rc),
            "rows": rows, "column_pixel": list(k), "column_distance": int(dist),
            "predicted": pred[free].numpy(), "n_free": int(free.sum()), "column": column}


def walkthrough_numbers(data: dict) -> dict:
    """The numbers the figure shows (and its JSON records): no arrays."""
    return {"sweep": data["sweep"], "cell": data["cell"], "seed": data["seed"], "patch": data["patch"],
            "solver_iterations": data["iterations"], "solver": data["solver"], "record": data["record"],
            "n_observed": int(data["observed"].sum()), "n_free": data["n_free"],
            "rows": [{k: v for k, v in r.items() if k != "row"} | {"max_abs_weight": float(np.abs(r["row"]).max())}
                     for r in data["rows"]],
            "column_pixel": data["column_pixel"], "column_distance": data["column_distance"],
            "column": [{k: v for k, v in c.items() if k != "actual"} for c in data["column"]]}


def _image(ax, img: np.ndarray, title: str, hole: np.ndarray | None = None) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if hole is not None:
        ax.imshow(np.where(hole, 1.0, np.nan), cmap=ListedColormap([HOLE]), vmin=0, vmax=1,
                  interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label positions near ys, in the same order, at least gap apart and inside
    [lo, hi] (as far as the span allows)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [0.0] * len(ys)
    prev = -np.inf
    for i in order:
        out[i] = max(float(ys[i]), prev + gap)
        prev = out[i]
    over = max(out) - hi
    if over > 0:
        out = [v - over for v in out]
    return [max(v, lo) for v in out]


def _callouts(ax, points: list[tuple[tuple[int, int], str]], shape: tuple[int, int]) -> None:
    """Mark pixels with a cross and label them outside the image on the nearer
    side, on leader lines, so pixels a row apart stay readable."""
    h, w = shape
    for side in ("left", "right"):
        pts = [p for p in points if (p[0][1] < w / 2) == (side == "left")]
        if not pts:
            continue
        ys = spread([r for (r, _), _ in pts], gap=0.11 * h, lo=0, hi=h - 1)
        x_text = -0.06 * w if side == "left" else w - 1 + 0.06 * w
        for ((r, c), label), yl in zip(pts, ys):
            ax.plot(c, r, marker="x", ms=3, mew=0.8, color="#0b0b0b", clip_on=False)
            ax.annotate(label, xy=(c, r), xytext=(x_text, yl), textcoords="data",
                        ha="right" if side == "left" else "left", va="center", fontsize=7,
                        annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#0b0b0b",
                                    "shrinkA": 1, "shrinkB": 2})


def draw_walkthrough(data: dict, fig_dir: Path = FIG_DIR, name: str = "record_walkthrough") -> Path:
    """Two rows of four panels. Top: the source patch, the observed patch with
    the queried pixels, CSFR's reconstruction, and predicted against measured
    column entries. Bottom: the three rows on a symmetric scale each, and the
    row weight per observed row against its distance from the hole."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH_IN, 4.15), layout="constrained")
        hole = ~data["observed"]
        _image(axes[0, 0], data["truth"], "(a) Source patch")
        _image(axes[0, 1], data["truth"] * data["observed"], "(b) Observed (grey: hole)", hole)
        _callouts(axes[0, 1], [(tuple(r["pixel"]), str(i)) for i, r in enumerate(data["rows"], start=1)]
                  + [(tuple(data["column_pixel"]), "k")], hole.shape)
        _image(axes[0, 2], data["reconstruction"], "(c) CSFR reconstruction")

        ax = axes[0, 3]
        pred = data["predicted"]
        for i, c in reversed(list(enumerate(data["column"]))):       # largest delta drawn first, beneath
            ax.scatter(pred, c["actual"], s=6, marker=MARKERS[i], facecolors="none",
                       edgecolors=SERIES[i], linewidths=0.5,
                       label=rf"$\delta={c['delta']:g}$: error {100 * c['rel_l2_error_ref']:.2g}%")
        lim = [min(pred.min(), *(c["actual"].min() for c in data["column"])),
               max(pred.max(), *(c["actual"].max() for c in data["column"]))]
        ax.plot(lim, lim, color=INK_MUTED, lw=0.6, zorder=0)
        ax.set_xlabel(r"predicted $\partial\hat x_i/\partial y_k$")
        ax.set_ylabel("measured (re-solve)")
        ax.set_title("(d) Column of pixel k")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False, handletextpad=0.2,
                  borderaxespad=0.2, markerscale=1.5)

        for i, (r, ax) in enumerate(zip(data["rows"], axes[1, :3]), start=1):
            w = np.where(data["observed"], r["row"], np.nan)
            vmax = float(np.nanmax(np.abs(w)))
            im = ax.imshow(w, cmap=SIGNED, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.imshow(np.where(hole, 1.0, np.nan), cmap=ListedColormap([HOLE]), vmin=0, vmax=1,
                      interpolation="nearest")
            _callouts(ax, [(tuple(r["pixel"]), str(i))], hole.shape)
            ax.set_xticks([])
            ax.set_yticks([])
            pr, pc = r["pixel"]
            ax.set_title(f"({'efg'[i - 1]}) Row of pixel {i} ({pr}, {pc})")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            defect = r["affine_defect"]
            cb.set_label(f"CG {r['cg_iters']} it., residual {sci(r['cg_residual'])}\n"
                         + (rf"$1-\sum w$ = {sci(defect)}" if defect is not None else "a hole pixel at a bound")
                         + f", support {r['support']}", fontsize=7, labelpad=2)

        ax = axes[1, 3]
        dist = data["hole_distance"]
        depth = np.arange(1, int(dist[data["observed"]].max()) + 1)
        for i, r in enumerate(data["rows"]):
            mass = [float(np.abs(r["row"][(dist == d) & data["observed"]]).sum()) for d in depth]
            ax.semilogy(depth, mass, marker=MARKERS[i], ms=3, lw=1.0, color=SERIES[i], label=f"pixel {i + 1}")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * (hi / lo) ** 0.35)          # headroom for the legend row
        ax.set_xlabel("distance from the hole (pixels)")
        ax.set_ylabel(r"$\sum |w|$ at that distance")
        ax.set_title("(h) Reach of the rows")
        ax.legend(loc="upper center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8,
                  handlelength=1.5, borderaxespad=0.1)
        for a in (axes[0, 3], axes[1, 3]):
            a.spines[["top", "right"]].set_visible(False)
            a.grid(True, color="#e4e3df", lw=0.4)
            a.set_axisbelow(True)
        return _save(fig, name, fig_dir)


def record_walkthrough(fig_dir: Path = FIG_DIR, data_dir: Path = DATA_DIR, **kw) -> Path | None:
    data = walkthrough_data(**kw)
    if data is None:
        return None
    _write_json(data_dir / "record_walkthrough.json", walkthrough_numbers(data))
    return draw_walkthrough(data, fig_dir)


FIGURES = {"record_walkthrough": record_walkthrough}


def main(argv: list[str] | None = None) -> int:
    names = argv if argv else list(FIGURES)
    for n in names:
        out = FIGURES[n]()
        print(f"[figures_v2] {n}: {'pending' if out is None else out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
