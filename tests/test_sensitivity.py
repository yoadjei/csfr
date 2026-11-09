"""Sensitivity record at the converged solution (round 2): rows and columns are
two views of one Jacobian, observed pixels have the identity row, weights live
on observed pixels only, the record matches finite differences of the solution
map, and the manifest carries every parameter that affects a number."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from csfr_solver import SolverConfig, solve  # noqa: E402
from sensitivity import (RecordConfig, sensitivity_row, sensitivity_column,  # noqa: E402
                         fidelity_columns, fidelity_patch, free_set, write_manifest)


def _solved(n=12, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    ref = (40 + 150 * (xx > n // 2) + 2.0 * yy).astype(np.float32)[None]
    M = np.ones_like(ref)
    M[0].reshape(-1)[rng.choice(n * n, n * n // 3, replace=False)] = 0
    y = torch.from_numpy(ref * M)
    Mt = torch.from_numpy(M)
    cfg = SolverConfig(tol=5e-12, max_iter=100000)
    x, cert = solve(y, Mt, cfg)
    return x[0], y[0].double(), Mt[0].bool(), cfg, cert


def _pick(Mb):
    hole = np.argwhere(~Mb.numpy())
    obs = np.argwhere(Mb.numpy())
    return tuple(map(int, hole[len(hole) // 2])), tuple(map(int, obs[len(obs) // 3]))


def test_row_and_column_agree():
    x, y, Mb, cfg, _ = _solved()
    i, k = _pick(Mb)
    row, ri = sensitivity_row(x, y, Mb, i, cfg, RecordConfig())
    col, ci = sensitivity_column(x, y, Mb, k, cfg, RecordConfig())
    assert ri["kind"] == "imputed" and ci["cg_iters"] > 0
    assert abs(float(row[k]) - float(col[i])) <= 1e-6 * (1 + abs(float(col[i])))


def test_observed_pixel_has_identity_row():
    x, y, Mb, cfg, _ = _solved()
    _, k = _pick(Mb)
    row, info = sensitivity_row(x, y, Mb, k, cfg, RecordConfig())
    assert info["kind"] == "observed"
    assert float(row[k]) == 1.0 and float(row.abs().sum()) == 1.0


def test_weights_live_on_observed_pixels_only():
    x, y, Mb, cfg, _ = _solved()
    i, _ = _pick(Mb)
    row, info = sensitivity_row(x, y, Mb, i, cfg, RecordConfig())
    assert float(row[~Mb].abs().sum()) == 0.0
    assert info["support"] > 0


def test_rows_sum_to_one_when_no_hole_pixel_is_at_a_bound():
    """J is invariant under X -> X + c (DC excluded, TV uses differences), so the
    Hessian annihilates constants and, with no hole pixel at a bound, every row
    of -H_FF^-1 H_F,Omega sums to one. Any damping of H_FF breaks this, which is
    why the record is undamped by default."""
    x, y, Mb, cfg, _ = _solved()
    i, _ = _pick(Mb)
    row, info = sensitivity_row(x, y, Mb, i, cfg, RecordConfig())
    assert info["n_hole_at_bound"] == 0
    assert abs(float(row.sum()) - 1.0) < 1e-8
    assert abs(info["affine_defect"]) < 1e-8
    damped, dinfo = sensitivity_row(x, y, Mb, i, cfg, RecordConfig(damping_rel=1e-4))
    assert abs(float(damped.sum()) - 1.0) > 1e-6, "damping no longer biases the row; revisit the test"
    assert dinfo["affine_defect"] == 1.0 - float(damped.sum())


def test_affine_identity_is_not_claimed_when_a_bound_is_active():
    x, y, Mb, cfg, _ = _solved()
    i, _ = _pick(Mb)
    x = x.clone()
    hole = np.argwhere(~Mb.numpy())
    other = next(tuple(map(int, p)) for p in hole if tuple(map(int, p)) != i)
    x[other] = 0.0  # a hole pixel at the lower bound
    _, info = sensitivity_row(x, y, Mb, i, cfg, RecordConfig())
    assert info["n_hole_at_bound"] == 1 and info["affine_defect"] is None


def test_record_is_undamped_by_default():
    assert RecordConfig().damping_rel == 0.0


def test_column_is_zero_off_the_free_set():
    x, y, Mb, cfg, _ = _solved()
    _, k = _pick(Mb)
    col, _ = sensitivity_column(x, y, Mb, k, cfg, RecordConfig())
    free = free_set(x, Mb, RecordConfig().bound_margin)
    assert float(col[~free].abs().sum()) == 0.0


def test_column_is_the_derivative_of_the_solution_map():
    """First-order behaviour: the finite-difference error vanishes with the
    perturbation and grows linearly in it (the Taylor remainder), which is what
    distinguishes a correct derivative from an approximately right vector."""
    x, y, Mb, cfg, _ = _solved()
    _, k = _pick(Mb)
    res = fidelity_columns(x, y, Mb, k, cfg, RecordConfig(), deltas=(1e-3, 1e-2, 1e-1),
                           operational=False)
    e = [r["rel_l2_error_ref"] for r in res]
    assert all(r["ref_converged"] for r in res)
    assert e[0] < 1e-3 and e[1] < 1e-2
    assert 5.0 < e[2] / e[1] < 20.0
    assert all(r["sign_agreement_topk"] == 1.0 for r in res)


def test_batched_patch_fidelity_equals_per_query_fidelity():
    """fidelity_patch solves every query and delta of a patch in one batch; the
    rows must equal the per-query computation."""
    x, y, Mb, cfg, _ = _solved()
    obs = [tuple(map(int, p)) for p in np.argwhere(Mb.numpy())[[3, 40]]]
    rc = RecordConfig()
    batched = fidelity_patch(x, y, Mb, obs, cfg, rc, deltas=(0.25, 1.0), operational=True)
    assert len(batched) == 4
    for q, k in enumerate(obs):
        single = fidelity_columns(x, y, Mb, k, cfg, rc, deltas=(0.25, 1.0), operational=True)
        rows = [r for r in batched if r["query"] == q]
        assert [r["query_pixel"] for r in rows] == [list(k), list(k)]
        for a, b in zip(rows, single):
            for key in ["rel_l2_error_ref", "rel_l2_error_op", "sign_agreement_topk", "pred_norm"]:
                assert abs(a[key] - b[key]) <= 1e-6 * (1 + abs(b[key])), key


def test_manifest_is_json_and_carries_parameters(tmp_path):
    x, y, Mb, cfg, cert = _solved()
    i, _ = _pick(Mb)
    m = write_manifest(tmp_path / "m.json", x_hat=x, y=y, Mb=Mb, solver_cfg=cfg,
                       certificate=cert, queries=[i], patch_id="toy", rc=RecordConfig())
    assert (tmp_path / "m.json").exists()
    assert m["record_parameters"]["damping_rel"] == 0.0
    assert m["queries"][0]["status"] == "imputed"
    assert m["queries"][0]["n_hole_at_bound"] == 0
    assert abs(m["queries"][0]["affine_defect"]) < 1e-8
    assert m["solver_configuration"]["tol"] == 5e-12
    assert "objective" in m and m["objective"]["dc_excluded_from_l1"] is True
