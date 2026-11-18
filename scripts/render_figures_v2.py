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

import csv
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, SymLogNorm  # noqa: E402
from scipy.ndimage import label  # noqa: E402

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


# ------------------------------------------------------------ record comparison

CMP_CELL, CMP_SEED, CMP_PATCH = "CF2_L3", 0, 0      # the first patch of the cell, as in the walkthrough
CMP_N_PATCHES = 10                                    # run_sensitivity_fidelity --patches (default)
# (method, record, panel title): the records of one observed pixel, as records_compare.csv names them
CMP_RECORDS = (
    ("csfr", "implicit_function", "CSFR"),
    ("bilinear", "exact_barycentric", "Linear interp., exact"),
    ("inpaint_telea", "finite_difference_delta1", r"Telea, $\delta=1$"),
    ("inpaint_telea", "finite_difference_delta4", r"Telea, $\delta=4$"),
    ("inpaint_ns", "finite_difference_delta1", r"Navier-Stokes, $\delta=1$"),
    ("inpaint_ns", "finite_difference_delta4", r"Navier-Stokes, $\delta=4$"),
)
CMP_STATS = ("l1_norm", "max_abs", "support", "mean_distance")
CMP_LINTHRESH = 1e-3            # the maps are linear below this fraction of their largest entry


def comparison_columns(y: np.ndarray, M: np.ndarray, patch: int, k: tuple[int, int]) -> dict:
    """The columns of observed pixel k of one patch, as records_unit in
    scripts/run_sensitivity_fidelity.py computes them: the batch is solved
    together with the default configuration, then CSFR's implicit-function
    column, the exact barycentric column of linear interpolation, and
    finite-difference columns of Telea and Navier-Stokes at delta 1 and 4.
    records_unit computes every query of a cell at once, and its file cannot
    change without invalidating the saved units, so these lines repeat it;
    comparison_data checks the result against the saved row. Keyed by
    (method, record)."""
    cfg, rc = SolverConfig(), RecordConfig()
    X, _ = solve(torch.from_numpy(y), torch.from_numpy(M), cfg)
    Mb = M[patch] == 1
    col, _ = sensitivity_column(X[patch], torch.from_numpy(y[patch]), torch.from_numpy(Mb), k, cfg, rc)
    exact = np.zeros(Mb.shape)
    for p, entries in _fid.bilinear_weights(M[patch]).items():
        for q, wt in entries:
            if q == k:
                exact[p] = wt
    out = {("csfr", "implicit_function"): col.numpy(), ("bilinear", "exact_barycentric"): exact}
    y1, M1 = y[patch:patch + 1].astype(np.float32), M[patch:patch + 1]
    for name in ("inpaint_telea", "inpaint_ns"):
        fn = _fid.BASELINES[name]
        for dl in (1.0, 4.0):
            out[name, f"finite_difference_delta{dl:g}"] = _fid.finite_difference_column(
                lambda a, b, fn=fn: fn(a, b), y1, M1, k, dl)
    return out


def comparison_data(sweep: Path = ROOT / "results" / "sweep_v2_govdocs1",
                    saved: Path = ROOT / "results" / "sensitivity_fidelity" / "records_compare.csv",
                    cell: str = CMP_CELL, seed: int = CMP_SEED, patch: int = CMP_PATCH,
                    n_patches: int = CMP_N_PATCHES) -> dict | None:
    """The records of the observed pixel at distance 1 from the hole of one patch,
    recomputed and checked against the row the record comparison saved for it, so
    the figure shows the records Table RECORDS_CMP summarises."""
    d = sweep / cell / f"seed{seed}"
    if not saved.exists() or not all((d / f).exists() for f in ("y.npy", "mask.npy")):
        return None
    y, M, _, _ = _fid._cell_inputs(sweep, cell, seed, n_patches)
    Mb = M[patch] == 1
    dist, k = observed_queries(Mb)[0]
    if dist != 1:
        raise RuntimeError(f"{d}: patch {patch} has no observed pixel at distance 1 from the hole")
    cols = comparison_columns(y, M, patch, k)
    rows = {(r["method"], r["record"]): r for r in _fid_rows(saved)
            if r["cell"] == cell and int(r["patch"]) == patch and int(r["query_distance"]) == dist}
    records = []
    for method, record, title in CMP_RECORDS:
        stats = _fid._record_stats(cols[method, record], k, ~Mb)
        row = rows.get((method, record))
        if row is None:
            raise RuntimeError(f"{saved.name}: no {method} {record} row for {cell} patch {patch}")
        for key in CMP_STATS:
            a, b = float(stats[key]), float(row[key])
            if not (np.isclose(a, b, rtol=1e-6, atol=1e-9) or (np.isnan(a) and np.isnan(b))):
                raise RuntimeError(f"{method} {record}: {key} {a} differs from the saved {b}")
        records.append({"method": method, "record": record, "title": title, "column": cols[method, record],
                        **{key: float(stats[key]) for key in CMP_STATS},
                        "components_reached": components_reached(cols[method, record], ~Mb)})
    return {"sweep": sweep.name, "cell": cell, "seed": seed, "patch": patch, "query_pixel": list(k),
            "query_distance": int(dist), "observed": Mb, "y": y[patch], "records": records,
            "n_hole_components": int(label(~Mb, structure=np.ones((3, 3)))[1])}


def components_reached(column: np.ndarray, hole: np.ndarray) -> int:
    """How many connected parts of the hole (8-connected) hold an entry of the
    column above the support threshold of the record comparison (1e-6 of the
    largest |entry|)."""
    a = np.abs(np.where(hole, column, 0.0))
    if a.max() == 0:
        return 0
    parts, _ = label(hole, structure=np.ones((3, 3)))
    return len(np.unique(parts[a > 1e-6 * a.max()]))


def _fid_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def comparison_numbers(data: dict) -> dict:
    return {k: data[k] for k in ("sweep", "cell", "seed", "patch", "query_pixel", "query_distance",
                                 "n_hole_components")} | {
        "n_hole": int((~data["observed"]).sum()),
        "records": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()
                     if k != "column"} for r in data["records"]]}


def reach(column: np.ndarray, k: tuple[int, int], hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sum |column| over the hole pixels at each Chebyshev distance from k."""
    rr, cc = np.nonzero(hole)
    d = np.maximum(np.abs(rr - k[0]), np.abs(cc - k[1]))
    a = np.abs(column[rr, cc])
    ds = np.arange(1, int(d.max()) + 1)
    return ds, np.array([a[d == x].sum() for x in ds])


def draw_comparison(data: dict, fig_dir: Path = FIG_DIR, name: str = "record_comparison") -> Path:
    """Two rows of four panels. Top: the observed patch with the queried pixel k,
    the CSFR column and the exact column of linear interpolation, and how the
    weight of those two columns falls off with distance from k. Bottom: the
    finite-difference columns of Telea and Navier-Stokes at delta 1 and 4. Each
    column is on its own symmetric logarithmic scale over the hole, linear below
    CMP_LINTHRESH of its largest entry, so that entries a hundred times smaller
    than the largest stay visible; observed pixels are white."""
    hole = ~data["observed"]
    k = tuple(data["query_pixel"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH_IN, 4.15), layout="constrained")
        _image(axes[0, 0], data["y"], "(a) Observed (grey: hole)", hole)
        _callouts(axes[0, 0], [(k, "k")], hole.shape)
        maps = [axes[0, 1], axes[0, 2], *axes[1, :]]
        for i, (r, ax) in enumerate(zip(data["records"], maps)):
            c = np.where(hole, r["column"], np.nan)
            vmax = float(np.nanmax(np.abs(c))) or 1.0
            norm = SymLogNorm(linthresh=CMP_LINTHRESH * vmax, linscale=0.5, vmin=-vmax, vmax=vmax, base=10)
            im = ax.imshow(c, cmap=SIGNED, norm=norm, interpolation="nearest")
            ax.contour(hole.astype(float), levels=[0.5], colors=INK_MUTED, linewidths=0.4)
            _callouts(ax, [(k, "k")], hole.shape)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({'bcefgh'[i]}) {r['title']}")
            if r["l1_norm"] == 0:
                ax.text(0.5, 0.5, "zero: output\nunchanged", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color=INK_MUTED,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.02)
            cb.set_ticks([-vmax, -1e-2 * vmax, 0.0, 1e-2 * vmax, vmax],
                         labels=["−1", r"$-10^{-2}$", "0", r"$10^{-2}$", "1"])
            cb.ax.tick_params(labelsize=6)
            scale = f"c / {r['max_abs']:.2g}" if r["max_abs"] > 0 else "c = 0"
            cb.set_label(rf"{scale};  $\ell_1$ {r['l1_norm']:.3g}, support {r['support']:.0f}",
                         fontsize=7, labelpad=2)

        ax = axes[0, 3]
        for i, r in enumerate(data["records"][:2]):
            ds, mass = reach(r["column"], k, hole)
            keep = mass > 0
            ax.semilogy(ds[keep], mass[keep], marker=MARKERS[i], ms=3, lw=1.0, color=SERIES[i],
                        label=r["title"])
        ax.set_xlabel("distance from k (pixels)")
        ax.set_ylabel(r"$\sum |c|$ at that distance")
        ax.set_title("(d) Reach of the columns")
        ax.legend(loc="lower right", frameon=False, handletextpad=0.3, handlelength=1.5, borderaxespad=0.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e4e3df", lw=0.4)
        ax.set_axisbelow(True)
        return _save(fig, name, fig_dir)


def record_comparison(fig_dir: Path = FIG_DIR, data_dir: Path = DATA_DIR, **kw) -> Path | None:
    data = comparison_data(**kw)
    if data is None:
        return None
    _write_json(data_dir / "record_comparison.json", comparison_numbers(data))
    return draw_comparison(data, fig_dir)


FIGURES = {"record_walkthrough": record_walkthrough, "record_comparison": record_comparison}


def main(argv: list[str] | None = None) -> int:
    names = argv if argv else list(FIGURES)
    for n in names:
        out = FIGURES[n]()
        print(f"[figures_v2] {n}: {'pending' if out is None else out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
