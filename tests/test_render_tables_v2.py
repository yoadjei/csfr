"""Round-2 AUTO tables: each generator reproduces its source exactly, and the
committed manuscript carries the current output of every generator."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"


def _mod():
    spec = importlib.util.spec_from_file_location("tables_v2", ROOT / "scripts" / "render_tables_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block(tag: str) -> str:
    t = TEX.read_text(encoding="utf-8")
    m = re.search(rf"% BEGIN AUTO {tag}\n(.*?)\n% END AUTO {tag}", t, re.S)
    assert m, f"AUTO {tag} markers missing from the manuscript"
    return m.group(1)


def test_rule_to_latex():
    m = _mod()
    assert m.rule_to_latex("exactly round(f*H*W) pixels erased") == r"exactly $\mathrm{round}(fHW)$ pixels erased"
    assert m.rule_to_latex("uniform on {4,...,31} and") == r"uniform on $\{4,\ldots,31\}$ and"
    assert m.rule_to_latex("until the erased fraction reaches f (the") == "until the erased fraction reaches $f$ (the"


def test_severity_table_follows_the_mask_code():
    m = _mod()
    from masks import BURST_MIN_SIDE, SEVERITY
    body = m.severity()
    rows = [r for r in body.splitlines() if re.match(r"CF\d &", r)]
    assert [r.split(" & ")[0] for r in rows] == ["CF1", "CF2", "CF3", "CF4"]
    assert rf"$\{{{BURST_MIN_SIDE},\ldots,31\}}$" in body
    for lvl, f in SEVERITY.items():
        assert f"{lvl}~{f:.2f}" in body


def _body(tab: str) -> list[str]:
    """The rows of a generated family-by-level tabular, without its frame and header."""
    lines = tab.splitlines()
    assert lines[0].startswith(r"\begin{tabular}") and lines[1] == r"\toprule"
    assert lines[5] == r"\midrule" and lines[-2:] == [r"\bottomrule", r"\end{tabular}"]
    return lines[6:-2]


def test_family_level_header_spans_each_family_over_its_levels():
    m = _mod()
    h = m.family_level_header(("L1", "L3", "L5"))
    assert h[0] == r" & \multicolumn{3}{c}{CF1} & \multicolumn{3}{c}{CF2} & \multicolumn{3}{c}{CF3} & \multicolumn{3}{c}{CF4} \\"
    assert h[1] == r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}"
    assert h[2].count(" & ") == 12 and h[2].startswith("Method & L1 & L3 & L5 & L1")


def _fixture_summary(tmp_path, values, n_seeds=3):
    """summary.csv with the columns run_sweep_v2.write_summary writes that the
    generator reads, and a run_metadata.json whose configuration is exactly the
    (family, level, method) keys of values -> psnr_hole_mean."""
    import csv
    import json
    rows = [{"family": f, "level": lv, "method": m, "psnr_hole_mean": v, "gmsd_hole_mean": 1.0 / v,
             "n_seeds": n_seeds, "n_patches": 900,
             "n_at_cap": int(m == "guleryuz" and (f, lv) == ("CF2", "L3")) * 7}
            for (f, lv, m), v in values.items()]
    with (tmp_path / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = {"seeds": [0, 1, 2], "families": sorted({f for f, _, _ in values}),
           "levels": {lv: 0.0 for lv in sorted({lv for _, lv, _ in values})},
           "csfr_variants": [], "external": [], "baselines": sorted({m for _, _, m in values})}
    (tmp_path / "run_metadata.json").write_text(json.dumps({"config": cfg}))


def test_method_by_cell_table_orders_bolds_and_dashes(tmp_path):
    m = _mod()
    cells = [(f, lv) for f in m.FAMILIES for lv in m.MAIN_LEVELS]
    values = {}
    for i, c in enumerate(cells):
        values[(*c, "csfr")] = 30.0 + i
        values[(*c, "guleryuz")] = 31.0 + i if c[0] == "CF1" else 29.0 + i
        values[(*c, "zero_fill")] = 99.0          # the floor never takes the bold
        values[(*c, "lama")] = 20.0
    _fixture_summary(tmp_path, values)
    none = tmp_path / "no_fallbacks.csv"
    lines = _body(m.method_by_cell_table(tmp_path, "psnr_hole", True, 2, fallbacks=none))
    # unconfigured methods (Telea, ...) have no row; the floor is set apart
    assert [ln.split(" & ")[0] for ln in lines] == ["CSFR", "Block-DCT thr.", "LaMa", r"\midrule",
                                                   "Zero-fill (floor)"]
    csfr, gul, lama = (ln.split(" & ") for ln in lines[:3])
    assert gul[1] == r"\textbf{31.00}" and csfr[1] == "30.00"        # CF1 L1: Guleryuz best
    assert csfr[4] == r"\textbf{33.00}" and gul[4] == "32.00"         # CF2 L1: CSFR best
    assert lama[-1] == r"20.00 \\"
    assert r"\textbf" not in lines[-1]
    lower = m.method_by_cell_table(tmp_path, "gmsd_hole", False, 3, fallbacks=none)
    assert r"\textbf{0.032}" in _body(lower)[1]                 # 1/31, lower is better
    noted = _body(m.method_by_cell_table(tmp_path, "psnr_hole", True, 2, with_cap_note=True,
                                         fallbacks=none))
    assert noted[:-2] == lines
    assert noted[-1].startswith(r"\multicolumn{13}{@{}l}")
    assert "at most 7 of 900 per cell (Block-DCT thr., CF2~L3)" in noted[-1]


def test_majority_fallback_cells_carry_a_dagger(tmp_path):
    import csv
    m = _mod()
    cells = [(f, lv) for f in m.FAMILIES for lv in m.MAIN_LEVELS]
    _fixture_summary(tmp_path, {(*c, meth): 30.0 for c in cells for meth in ("csfr", "dictlearn", "zero_fill")})
    fb = tmp_path / "fallbacks.csv"
    with fb.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sweep", "family", "level", "method", "frac_fallback"])
        w.writeheader()
        w.writerow({"sweep": tmp_path.name, "family": "CF1", "level": "L1", "method": "dictlearn",
                    "frac_fallback": 0.99})
        w.writerow({"sweep": tmp_path.name, "family": "CF1", "level": "L3", "method": "dictlearn",
                    "frac_fallback": 0.5})                    # not a majority
        w.writerow({"sweep": "another_sweep", "family": "CF2", "level": "L1", "method": "dictlearn",
                    "frac_fallback": 1.0})                    # another corpus
    lines = _body(m.method_by_cell_table(tmp_path, "psnr_hole", True, 2, fallbacks=fb))
    dl = lines[1].split(" & ")
    assert dl[0] == r"Dict.\ learning"
    assert dl[1] == r"\textbf{30.00}" + m.DAGGER and m.DAGGER not in "".join(dl[2:])
    assert lines[-1].startswith(r"\multicolumn{13}{@{}l}{\footnotesize " + m.DAGGER)


def test_an_incomplete_sweep_is_pending(tmp_path):
    import json
    m = _mod()
    cells = [(f, lv) for f in m.FAMILIES for lv in m.MAIN_LEVELS]
    values = {(*c, meth): 30.0 for c in cells for meth in ("csfr", "zero_fill")}
    _fixture_summary(tmp_path, values, n_seeds=1)                      # one of three seeds so far
    assert m.method_by_cell_table(tmp_path, "psnr_hole", True, 2) is None
    _fixture_summary(tmp_path, values)
    assert m.method_by_cell_table(tmp_path, "psnr_hole", True, 2) is not None
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    meta["config"]["external"] = ["lama"]                              # not scored yet
    (tmp_path / "run_metadata.json").write_text(json.dumps(meta))
    assert m.method_by_cell_table(tmp_path, "psnr_hole", True, 2) is None


def _paired_fixture(tmp_path, drop_lama_in=None, drop_cell=None):
    import csv
    rows = []
    for f in ("CF1", "CF2", "CF3", "CF4"):
        for lv in ("L1", "L2", "L3", "L4", "L5"):
            if (f, lv) == drop_cell:
                continue
            for meth, mm in (("guleryuz", 32.0), ("fsr", 30.5), ("lama", 31.0)):
                if meth == "lama" and (f, lv) == drop_lama_in:
                    continue
                r = {"family": f, "level": lv, "metric": "psnr_hole", "method": meth,
                     "csfr_mean": 31.0, "method_mean": mm, "diff_mean": 31.0 - mm,
                     "ci_cluster_lo": 31.0 - mm - 0.5, "ci_cluster_hi": 31.0 - mm + 0.25,
                     "frac_images_csfr_leads": 0.25}
                if meth != "lama":
                    r["diff_mean_capfree"] = 31.0 - mm + 0.1
                rows.append(r)
            rows.append({"family": f, "level": lv, "metric": "gmsd_hole", "method": "fsr",
                         "csfr_mean": 0.1, "method_mean": 0.9, "diff_mean": 0.8,
                         "ci_cluster_lo": 0.7, "ci_cluster_hi": 0.9, "frac_images_csfr_leads": 1.0})
    (tmp_path / "stats").mkdir()
    with (tmp_path / "stats" / "paired_tests.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def test_paired_v2_strongest_classical_rank_and_external(tmp_path):
    m = _mod()
    _paired_fixture(tmp_path, drop_lama_in=("CF2", "L3"))
    lines = m.paired_v2(tmp_path).splitlines()
    rows = [ln for ln in lines if ln.startswith("CF")]
    assert len(rows) == 20 and lines.count(r"\midrule") == 1 + 3     # header rule + between families
    cf1 = rows[0].split(" & ")
    # CSFR (31.0) is behind Guleryuz (32.0), ahead of FSR; LaMa is not ranked
    assert cf1[:7] == ["CF1~L1", "2/3", "Block-DCT thr.", "$-$1.00", "[$-$1.50, $-$0.75]", "25", "$-$0.90"]
    assert cf1[7:] == ["+0.00", r"[$-$0.50, +0.25] \\"]
    cf2_l3 = next(r for r in rows if r.startswith("CF2~L3")).split(" & ")
    assert cf2_l3[7:] == ["--", r"-- \\"]


def test_paired_v2_missing_cell_is_an_error(tmp_path):
    import pytest
    m = _mod()
    _paired_fixture(tmp_path, drop_cell=("CF3", "L4"))
    with pytest.raises(ValueError, match="CF3 L4"):
        m.paired_v2(tmp_path)


def test_external_v2_has_one_panel_per_corpus_and_waits_for_both(tmp_path):
    m = _mod()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _paired_fixture(a)
    assert m.external_v2({"First": a, "Second": b}) is None            # b has no statistics yet
    _paired_fixture(b, drop_lama_in=("CF1", "L3"))
    lines = m.external_v2({"First": a, "Second": b}).splitlines()
    panels = [ln for ln in lines if ln.startswith(r"\multicolumn{9}")]
    assert panels == [r"\multicolumn{9}{@{}l}{\emph{First}} \\", r"\multicolumn{9}{@{}l}{\emph{Second}} \\"]
    rows = [ln for ln in lines if ln.startswith("CF")]
    assert len(rows) == 2 * 12 and all(r.split(" & ")[0][-2:] in ("L1", "L3", "L5") for r in rows)
    second_cf1_l3 = [r for r in rows if r.startswith("CF1~L3")][1].split(" & ")
    assert second_cf1_l3[7:] == ["--", r"-- \\"]


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def test_ablation_v2_stars_holm_significance_and_states_the_range(tmp_path):
    m = _mod()
    arms = [a for a, _ in m.ABLATION_ARMS]
    rows = [{"family": f, "level": lv, "metric": "psnr_hole", "method": a,
             "diff_mean": 0.5 if a != "csfr_no_fc" else -0.02, "p_holm": 0.001 if a != "csfr_no_fc" else 0.4}
            for f in m.FAMILIES for lv in m.MAIN_LEVELS for a in arms]
    _write_csv(tmp_path / "stats" / "paired_tests_ablation.csv", rows)
    assert m.ablation_v2(tmp_path) is None                         # range figure not computed yet
    _write_csv(tmp_path / "stats" / "range_violations.csv",
               [{"method": "csfr_no_clip", "n_imputed": 1000, "frac_pixels_out": 0.02, "max_excess": 12.34},
                {"method": "csfr_no_clip", "n_imputed": 3000, "frac_pixels_out": 0.0, "max_excess": 0.0},
                {"method": "csfr", "n_imputed": 1000, "frac_pixels_out": 0.0, "max_excess": 0.0}])
    lines = _body(m.ablation_v2(tmp_path))
    assert [ln.split(" & ")[0] for ln in lines[:4]] == [n for _, n in m.ABLATION_ARMS]
    assert lines[0].split(" & ")[1] == "+0.50$^{*}$"
    assert lines[2].split(" & ")[1] == "$-$0.02"                     # not significant: no star
    assert r"Without C3, 0.50\% of the imputed pixels leave $[0, 255]$, by up to 12.3 grey levels; " \
           r"with it, none do." in lines[-1]


def test_solver_arm_names():
    m = _mod()
    assert m.solver_arm_name("adam600_released") == "Adam, 600 iterations (released, GPU)"
    assert m.solver_arm_name("adam2400_cpu") == "Adam, 2400 iterations (CPU)"
    assert m.solver_arm_name("fista_gap1e-06") == "Convergent, gap $10^{-6}$ (default)"
    assert m.solver_arm_name("fista_gap1e-10") == "Convergent, gap $10^{-10}$"
    assert m.tex_pow10(5e-12) == r"$5 \times 10^{-12}$"


def test_solver_cmp_reports_time_only_for_the_cpu_rerun(tmp_path):
    m = _mod()
    arms = ["adam600_released", "adam600_cpu", "adam2400_cpu", "fista_gap1e-06"]
    summ = [{"family": f, "arm": a, "psnr_hole": 30.0 + i, "seconds_per_cell": 10.0 * (i + 1),
             "iterations_median": 600 if a.startswith("adam600") else (2400 if "2400" in a else 750),
             "converged_frac": 0.99 if a.startswith("fista") else ""}
            for i, a in enumerate(arms) for f in m.FAMILIES]
    _write_csv(tmp_path / "quality_vs_compute.csv", summ)
    _write_csv(tmp_path / "per_cell.csv", [{"arm": "adam600_cpu", "max_abs_diff_vs_released": 0.0021},
                                           {"arm": "fista_gap1e-06", "max_abs_diff_vs_released": ""}])
    lines = _body(m.solver_cmp(tmp_path))
    assert lines[0] == r"Adam, 600 iterations (released, GPU) & 30.00 & 30.00 & 30.00 & 30.00 & 10 & 600 & -- \\"
    assert lines[1] == r"Adam, 600 iterations (CPU) & -- & -- & -- & -- & 20 & 600 & -- \\"
    assert lines[3].endswith(r"& 33.00 & 40 & 750 & 99.0 \\")
    assert "at most 0.0021 grey levels" in lines[-1]


def test_fidelity_reads_the_default_tolerance_at_one_grey_level(tmp_path):
    m = _mod()
    from csfr_solver import SolverConfig
    tol = SolverConfig().tol
    rows = []
    for f in m.FAMILIES:
        for lv in m.MAIN_LEVELS:
            for delta, t in ((1.0, tol), (4.0, tol), (1.0, 1e-10)):
                rows.append({"family": f, "level": lv, "delta": delta, "tol": t, "n": 30,
                             "rel_err_median_ref": 0.00213 if (delta, t) == (1.0, tol) else 0.9,
                             "rel_err_p90_ref": 0.0451, "srf_ref": 0.9667, "srf_op": 0.9333,
                             "sign_agreement_mean_ref": 0.998, "record_seconds_mean": 0.0123})
    _write_csv(tmp_path / "summary.csv", rows)
    lines = _body(m.fidelity(tmp_path))
    assert lines[0] == r"CF1~L1 & 30 & 0.213 & 4.51 & 96.7 & 93.3 & 99.8 & 12 \\"
    assert sum(ln.startswith("CF") for ln in lines) == 12


def test_records_cmp_dashes_an_empty_record_and_pools_family_groups(tmp_path):
    m = _mod()
    rows = []
    for cell in ("CF1_L3", "CF2_L3", "CF3_L3", "CF4_L3"):
        contiguous = cell[:3] in ("CF3", "CF4")
        for method, record, _ in m.RECORD_KINDS:
            empty = method == "bilinear" and record == "exact_barycentric" and contiguous
            r = {"cell": cell, "method": method, "record": record, "l1_norm": 0.0 if empty else 1.5,
                 "support": 0 if empty else 40, "mean_distance": "" if empty else 2.25}
            if record.startswith("finite"):
                r["fd_rel_diff_delta1_vs_4"] = 0.75 if method != "bilinear" else 0.0
            rows.append(r)
    _write_csv(tmp_path / "records_compare.csv", rows)
    lines = _body(m.records_cmp(tmp_path))
    assert lines[0] == r"CSFR & 1.50 & 40 & 2.2 & -- & 1.50 & 40 & 2.2 & -- \\"
    assert lines[1] == r"Linear, exact & 1.50 & 40 & 2.2 & -- & -- & -- & -- & -- \\"
    assert lines[2].endswith(r"& 0.00 \\") and lines[4].endswith(r"& 0.75 \\")


def test_sci_compact():
    m = _mod()
    assert m.sci_compact(812.4) == "812"
    assert m.sci_compact(1.234e16) == r"$1.2{\times}10^{16}$"


def test_oracle_has_a_column_group_per_corpus(tmp_path):
    m = _mod()
    rows = [{"corpus": c, "family": f, "level": lv, "oracle_psnr_hole_median": 41.26, "cond_median": 12.0,
             "cond_p90": 3.4e5, "frac_cond_above_1e3": 0.25}
            for c, _ in m.ORACLE_CORPORA for f in m.FAMILIES for lv in m.MAIN_LEVELS]
    _write_csv(tmp_path / "oracle.csv", rows)
    lines = _body(m.oracle(tmp_path))
    assert lines[0] == (r"CF1~L1 & 41.3 & 12 & $3.4{\times}10^{5}$ & 25 & 41.3 & 12 & $3.4{\times}10^{5}$ & 25 \\")


def test_bu_reports_auroc_per_scope_and_the_rule_decision(tmp_path):
    import json
    m = _mod()
    rows = []
    for thr in (20.0, 25.0):
        for key, _ in m.BU_PREDICTORS:
            a = 0.9 if key == "mean_hole_distance" else 0.6
            rows.append({"threshold_db": thr, "scope": "pooled", "group": "all", "predictor": key, "auroc": a})
            rows += [{"threshold_db": thr, "scope": "family", "group": f, "predictor": key, "auroc": a}
                     for f in m.FAMILIES]
            rows += [{"threshold_db": thr, "scope": "cell", "group": f"s:CF1_L{i}", "predictor": key,
                      "auroc": v} for i, v in enumerate((0.5, 0.7, ""))]          # one cell without variance
    _write_csv(tmp_path / "auroc.csv", rows)
    dec = {"by_threshold": {f"{t:g}dB": {"best_predictor": "mean_hole_distance", "pooled_auroc": 0.9,
                                         "cells_with_outcome_variance": 2,
                                         "fraction_cells_within_auroc_above_min": 0.5,
                                         "keep_as_contribution": t == 25.0} for t in (20.0, 25.0)}}
    (tmp_path / "decision.json").write_text(json.dumps(dec))
    lines = _body(m.bu(tmp_path))
    assert lines[0].startswith(r"\multicolumn{7}{@{}l}{\emph{Outcome: CSFR hole PSNR below 20\,dB}}")
    assert lines[2] == r"Mean distance to an observed pixel & 0.90 & 0.90 & 0.90 & 0.90 & 0.90 & 0.60 \\"
    assert "Below 20\\,dB: best predictor mean distance to an observed pixel; 50\\% of 2 cells" in lines[-2]
    assert lines[-2].rstrip(" \\").endswith("reported as a diagnostic.}")
    assert lines[-1].rstrip(" \\").endswith("kept as a contribution.}")


def test_committed_tables_are_current():
    m = _mod()
    for tag, fn in m.TABLES.items():
        body = fn()
        if body is not None:
            assert _block(tag) == body, f"AUTO {tag} is stale: run scripts/render_tables_v2.py"
