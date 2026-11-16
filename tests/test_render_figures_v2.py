"""Round-2 figures: the record walkthrough recomputes what it shows from a saved
reconstruction, refuses inputs that are not that reconstruction, and writes a
vector PDF with embedded fonts that is the same bytes on every run."""
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from csfr_solver import SolverConfig, solve  # noqa: E402
from masks import make_mask  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("figures_v2", ROOT / "scripts" / "render_figures_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spread_keeps_order_and_gap_inside_the_span():
    m = _mod()
    assert m.spread([5, 6, 7], gap=3, lo=0, hi=20) == [5, 8, 11]
    assert m.spread([7, 5], gap=3, lo=0, hi=20) == [8, 5]          # input order kept
    assert m.spread([18, 19], gap=3, lo=0, hi=20) == [17, 20]       # pushed back inside
    assert m.spread([0], gap=3, lo=0, hi=20) == [0]


def test_sci():
    m = _mod()
    assert m.sci(1.678e-11) == r"$1.7\times10^{-11}$"
    assert m.sci(-1.4e-10) == r"$-1.4\times10^{-10}$"
    assert m.sci(0.0) == "0"


def _sweep(tmp_path, cell="CF3_L3", n=2, size=16):
    """A one-cell sweep directory as run_sweep_v2 leaves it, for n small patches."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    patches = np.stack([np.clip(60 + 6 * xx + 3 * yy + rng.normal(0, 8, (size, size)), 0, 255)
                        for _ in range(n)]).astype(np.float32)
    M = make_mask(patches.shape, "CF3", 0.5, np.random.default_rng(0))
    y = patches * M
    cfg = SolverConfig(tol=1e-6, max_iter=20000, init="zero", dtype="float64")
    X, cert = solve(torch.from_numpy(y), torch.from_numpy(M), cfg)
    assert bool(cert.converged.all())
    d = tmp_path / "sweep" / cell / "seed0"
    d.mkdir(parents=True)
    np.save(d / "y.npy", y)
    np.save(d / "mask.npy", M)
    np.save(d / "x_hat_csfr.npy", X.numpy().astype(np.float32))
    (d / "metrics.json").write_text(json.dumps({"methods": {"csfr": {"solver": asdict(cfg)}}}))
    np.save(tmp_path / "patches.npy", patches)
    return {"sweep": tmp_path / "sweep", "patches_path": tmp_path / "patches.npy", "cell": cell,
            "seed": 0, "patch": 0}, M


def test_walkthrough_is_pending_without_its_cell(tmp_path):
    m = _mod()
    assert m.record_walkthrough(fig_dir=tmp_path / "fig", data_dir=tmp_path / "data",
                                sweep=tmp_path, patches_path=tmp_path / "none.npy") is None
    assert not (tmp_path / "fig").exists()


def test_walkthrough_records_the_queried_rows_and_a_validated_column(tmp_path):
    m = _mod()
    kw, M = _sweep(tmp_path)
    pdf = m.record_walkthrough(fig_dir=tmp_path / "fig", data_dir=tmp_path / "data", **kw)
    doc = pdf.read_bytes()
    assert b"/FontFile2" in doc and b"/Type3" not in doc            # embedded TrueType, no bitmap fonts
    numbers = json.loads((tmp_path / "data" / "record_walkthrough.json").read_text())
    observed = M[0] == 1
    assert [tuple(r["pixel"]) for r in numbers["rows"]] == m.imputed_queries(observed)
    assert tuple(numbers["column_pixel"]) == m.observed_queries(observed)[0][1]
    assert numbers["n_observed"] == int(observed.sum())
    for r in numbers["rows"]:
        assert abs(r["affine_defect"]) < 1e-8                       # exact rows sum to one
        assert r["cg_residual"] <= asdict(m.RecordConfig())["cg_rtol"]
    assert [c["delta"] for c in numbers["column"]] == list(m.WALK_DELTAS)
    assert all(c["ref_converged"] and c["rel_l2_error_ref"] < 0.1 for c in numbers["column"])

    again = m.record_walkthrough(fig_dir=tmp_path / "fig2", data_dir=tmp_path / "data2", **kw)
    assert again.read_bytes() == doc                                # no dates, same bytes


def test_walkthrough_refuses_an_output_it_cannot_reproduce(tmp_path):
    m = _mod()
    kw, _ = _sweep(tmp_path)
    x_path = kw["sweep"] / kw["cell"] / "seed0" / "x_hat_csfr.npy"
    x = np.load(x_path)
    x[0, -1, -1] += 1.0
    np.save(x_path, x)
    with pytest.raises(RuntimeError, match="differs from the sweep"):
        m.walkthrough_data(**kw)


def test_walkthrough_refuses_patches_that_are_not_the_observed_ones(tmp_path):
    m = _mod()
    kw, _ = _sweep(tmp_path)
    p = np.load(kw["patches_path"])
    np.save(kw["patches_path"], p[::-1].copy())
    with pytest.raises(RuntimeError, match="not the observed patch"):
        m.walkthrough_data(**kw)


def test_every_included_figure_is_generated_and_current(tmp_path):
    r"""Each \includegraphics of the manuscript and the supplement names a PDF
    that a generator of render_figures_v2 writes, and the committed file is the
    generator's current output (checked where the inputs are present)."""
    import re
    m = _mod()
    paper = ROOT / "paper"
    included = sorted({g for p in ("paper2_reconstruction.tex", "paper2_supplementary.tex")
                       for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                                           (paper / p).read_text(encoding="utf-8"))})
    assert included, "no figures included"
    for g in included:
        name = Path(g).stem
        assert g == f"figures/{name}.pdf" and name in m.FIGURES, f"{g} has no generator in render_figures_v2"
        fresh = m.FIGURES[name](fig_dir=tmp_path, data_dir=tmp_path / "data")
        if fresh is None:
            continue                                                 # inputs absent on this machine
        assert fresh.read_bytes() == (paper / g).read_bytes(), f"{g} is stale: run scripts/render_figures_v2.py"
