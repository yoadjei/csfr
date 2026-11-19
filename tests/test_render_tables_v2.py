"""Round-2 AUTO tables: each generator reproduces its source exactly, and the
committed manuscript carries the current output of every generator."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
SUPP = ROOT / "paper" / "paper2_supplementary.tex"


def _mod():
    spec = importlib.util.spec_from_file_location("tables_v2", ROOT / "scripts" / "render_tables_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block(tag: str) -> str:
    """The AUTO block of `tag` in the manuscript or the supplement (exactly one)."""
    hits = [m.group(1) for p in (TEX, SUPP)
            if (m := re.search(rf"% BEGIN AUTO {tag}\n(.*?)\n% END AUTO {tag}", p.read_text(encoding="utf-8"), re.S))]
    assert len(hits) == 1, f"AUTO {tag} markers appear in {len(hits)} documents, not one"
    return hits[0]


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


def _solver_summary(m, tmp_path, diff):
    arms = ["adam600_released", "adam600_cpu", "adam2400_cpu", "fista_gap1e-06"]
    summ = [{"family": f, "arm": a, "psnr_hole": 30.0 + i, "seconds_per_cell": 10.0 * (i + 1),
             "iterations_median": 600 if a.startswith("adam600") else (2400 if "2400" in a else 750),
             "converged_frac": 0.99 if a.startswith("fista") else "",
             "device": "cuda" if a == "adam600_released" else "cpu"}
            for i, a in enumerate(arms) for f in m.FAMILIES]
    _write_csv(tmp_path / "quality_vs_compute.csv", summ)
    _write_csv(tmp_path / "per_cell.csv", [{"arm": "adam600_cpu", "seed": 0, "max_abs_diff_vs_released": diff},
                                           {"arm": "fista_gap1e-06", "seed": 0, "max_abs_diff_vs_released": ""}])
    return _body(m.solver_cmp(tmp_path))


def test_solver_cmp_shows_cpu_time_only_and_quality_only_where_measured(tmp_path):
    m = _mod()
    lines = _solver_summary(m, tmp_path, 0.0021)
    assert lines[0] == r"Adam, 600 iterations (released, GPU) & 30.00 & 30.00 & 30.00 & 30.00 & -- & 600 & -- \\"
    assert lines[1] == r"Adam, 600 iterations (CPU) & -- & -- & -- & -- & 20 & 600 & -- \\"
    assert lines[3].endswith(r"& 33.00 & 40 & 750 & 99.0 \\")
    assert "at mask seed~0 differs from the released outputs by at most 0.0021 grey levels" in lines[-2]
    assert "timed on its GPU" in lines[-1]


def test_solver_cmp_states_a_bit_for_bit_reproduction(tmp_path):
    lines = _solver_summary(_mod(), tmp_path, 0.0)
    assert "reproduces the released outputs bit for bit in every cell" in lines[-2]
    assert "at most" not in lines[-2]


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


def test_records_cmp_counts_zero_records_and_takes_medians_over_the_rest(tmp_path):
    m = _mod()
    rows = []
    for cell in ("CF1_L3", "CF2_L3", "CF3_L3", "CF4_L3"):
        contiguous = cell[:3] in ("CF3", "CF4")
        for method, record, _ in m.RECORD_KINDS:
            empty = ((method == "bilinear" and record == "exact_barycentric" and contiguous)
                     or (method == "inpaint_telea" and record == "finite_difference_delta1" and cell == "CF1_L3"))
            rows.append({"cell": cell, "method": method, "record": record, "l1_norm": 0.0 if empty else 1.5,
                         "support": 0 if empty else 40, "mean_distance": "" if empty else 2.25})
    _write_csv(tmp_path / "records_compare.csv", rows)
    lines = _body(m.records_cmp(tmp_path))
    assert lines[0] == r"CSFR & 1.50 & 40 & 2.2 & 0 & 1.50 & 40 & 2.2 & 0 \\"
    assert lines[1] == r"Linear, exact & 1.50 & 40 & 2.2 & 0 & -- & -- & -- & 100 \\"
    assert lines[4] == r"Telea, FD $\delta = 1$ & 1.50 & 40 & 2.2 & 50 & 1.50 & 40 & 2.2 & 0 \\"


def test_supp_image_level_reads_the_strongest_method_per_image_and_waits_for_every_corpus(tmp_path):
    m = _mod()
    for i, (sweep, _) in enumerate(m.CAP_MECHANISM_CORPORA):
        if i == 2:
            assert m.supp_image_level(tmp_path) is None                  # a corpus without stats
        d = tmp_path / sweep / "stats"
        d.mkdir(parents=True)
        paired, per = [], []
        for f in m.FAMILIES:
            for lv in ("L1", "L2", "L3", "L4", "L5"):
                for method, diff, p in (("guleryuz", -1.0, 0.25), ("bilinear", 5.0, 1e-5)):
                    paired.append({"family": f, "level": lv, "metric": "psnr_hole", "method": method,
                                   "diff_mean": diff, "n_images": 2, "frac_images_csfr_leads": 0.5,
                                   "ci_image_lo": diff - 0.5, "ci_image_hi": diff + 0.5, "p_image_wilcoxon": p})
                    per += [{"family": f, "level": lv, "metric": "psnr_hole", "method": method, "image": k,
                             "n_patches": 3, "diff_mean": diff + dv} for k, dv in ((1, 0.5), (2, -0.25))]
        _write_csv(d / "paired_tests.csv", paired)
        _write_csv(d / "per_image.csv", per)
    tab = m.supp_image_level(tmp_path).splitlines()
    assert tab[3] == r"\midrule" and tab[-2:] == [r"\bottomrule", r"\end{tabular}"]   # one header line
    lines = tab[4:-2]
    assert lines[0] == r"\multicolumn{7}{@{}l}{\emph{GovDocs1}} \\"
    # mean of the image means -0.5 and -1.25
    assert lines[1] == r"CF1~L1 & Block-DCT thr. & 2 & 50 & $-$0.88 & [$-$1.50, $-$0.50] & 0.250\,($-$) \\"
    assert m.rank_side([-10.0, 1.0, 2.0, 3.0]) == r"\,(+)"     # mean -1, but ranks + 1+2+3 against - 4
    assert m.rank_side([-3.0, 1.0, 0.0]) == r"\,($-$)" and m.rank_side([1.0, -1.0]) == ""
    assert sum(ln.startswith("CF") for ln in lines) == 3 * 12


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
    _write_csv(tmp_path / "oracle.csv", rows[:-1])                     # a run still in progress
    assert m.oracle(tmp_path) is None


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


def test_jpeg_rows_per_corruption_and_mask(tmp_path):
    m = _mod()
    summ, mq = [], []
    for kind, _ in m.JPEG_KINDS:
        for cond, _ in m.JPEG_CONDITIONS:
            summ += [{"kind": kind, "condition": cond, "method": "csfr", "n_cases": 20, "psnr_hole_mean": 25.0,
                      "vs_best_classical": "fsr", "diff_psnr_hole_vs_best_classical": -0.4,
                      "p_wilcoxon_vs_best_classical": 0.01, "diff_psnr_hole_vs_lama": 0.3,
                      "p_wilcoxon_vs_lama": 0.2},
                     {"kind": kind, "condition": cond, "method": "fsr", "n_cases": 20, "psnr_hole_mean": 25.4},
                     {"kind": kind, "condition": cond, "method": "lama", "n_cases": 20, "psnr_hole_mean": 24.7}]
            mq += [{"kind": kind, "condition": cond, "visible_damage": v, "coverage_of_oracle_loss": c,
                    "false_pos_frac": f} for v, c, f in (("True", 0.9, 0.01), ("True", 0.7, 0.03),
                                                         ("False", 0.0, 0.5))]
    _write_csv(tmp_path / "summary.csv", [r for r in summ if r["method"] != "lama"])
    _write_csv(tmp_path / "masks_quality.csv", mq)
    assert m.jpeg(tmp_path) is None                               # LaMa not scored yet
    _write_csv(tmp_path / "summary.csv", summ)
    lines = _body(m.jpeg(tmp_path))
    assert lines[0] == (r"Truncation & Oracle & 20 & 25.00 & FSR & 25.40 & 24.70 & $-$0.40$^{*}$ & +0.30 "
                        r"& 80.0 & 2.00 \\")                      # invisible cases excluded from the mask means
    assert lines[1].startswith(" & Decoder & ")
    assert sum(1 for ln in lines if " & " in ln and not ln.startswith(r"\multicolumn")) == 9


def _maskerr_fixture(d, cells=("CF1_L3", "CF3_L5"), methods=("csfr", "bilinear", "lama"), skip=()):
    """robustness.csv and geometry.csv for `cells`: each method loses 1, 2 or 3 dB
    (by its position) times the cell's factor under erode1 and flip5pct;
    linear interpolation returns the floor with the true mask under CF3."""
    rob, geo = [], []
    for ci, cell in enumerate(cells):
        floor = 5.0 + ci
        for p in ("exact", "erode1", "flip5pct"):
            geo.append({"cell": cell, "perturbation": p, "n": 4, "zero_fill_psnr_hole": floor,
                        "fp_share_of_hole": 0.0 if p != "flip5pct" else 0.05 * (ci + 1),
                        "fn_share_of_observed": {"exact": 0.0, "erode1": 0.2, "flip5pct": 0.05}[p] * (ci + 1)})
            for mi, k in enumerate(methods):
                if (cell, p, k) in skip:
                    continue
                exact = floor if (k == "bilinear" and cell.startswith("CF3")) else 20.0 + mi
                deg = 0.0 if p == "exact" else -(mi + 1) * (ci + 1)
                rob.append({"cell": cell, "perturbation": p, "method": k, "psnr_hole_mean": exact + deg,
                            "degradation_mean": deg, "false_pos_frac": 0.0, "false_neg_frac": 0.0})
    _write_csv(d / "robustness.csv", rob)
    _write_csv(d / "geometry.csv", geo)


def test_maskerr_gives_each_family_the_mean_over_its_levels(tmp_path):
    m = _mod()
    _maskerr_fixture(tmp_path, cells=("CF2_L3", "CF2_L5", "CF3_L5"))
    lines = _body(m.maskerr(tmp_path, cells=("CF2_L3", "CF2_L5", "CF3_L5")))
    assert lines[0] == r"\multicolumn{6}{@{}l}{\textit{CF2, mean over L3 and L5}} \\"
    assert lines[1] == r"Erode 1\,px & $-$1.50 & $-$3.00 & $-$4.50 & 0.00 & 30.00 \\"
    assert lines[2] == r"Flip 5\% & $-$1.50 & $-$3.00 & $-$4.50 & 7.50 & 7.50 \\"
    assert lines[4] == r"\multicolumn{6}{@{}l}{\textit{CF3, mean over L3 and L5}} \\"
    assert lines[5] == r"Erode 1\,px & $-$3.00 & -- & $-$9.00 & 0.00 & 60.00 \\"   # linear: at the floor
    assert lines[-1].startswith(r"\multicolumn{6}{@{}l}{\footnotesize --: Linear returns the zero-fill floor")


def test_supp_maskerr_gives_every_cell(tmp_path):
    m = _mod()
    _maskerr_fixture(tmp_path)
    lines = _body(m.supp_maskerr(tmp_path, cells=("CF3_L5", "CF1_L3")))
    labels = [ln for ln in lines if ln.startswith(r"\multicolumn{6}{@{}l}{\textit")]
    assert labels == [r"\multicolumn{6}{@{}l}{\textit{CF1 L3}} \\", r"\multicolumn{6}{@{}l}{\textit{CF3 L5}} \\"]
    assert r"Erode 1\,px & $-$1.00 & $-$2.00 & $-$3.00 & 0.00 & 20.00 \\" in lines


def test_maskerr_waits_for_every_cell_perturbation_lama_row_and_the_geometry(tmp_path):
    m = _mod()
    cells = ("CF1_L3", "CF3_L5")
    _maskerr_fixture(tmp_path, cells=cells, skip={("CF3_L5", "flip5pct", k) for k in ("csfr", "bilinear", "lama")})
    assert m.maskerr(tmp_path, cells=cells) is None                       # last cell still running
    _maskerr_fixture(tmp_path, cells=cells, skip={("CF3_L5", "flip5pct", "lama")})
    assert m.maskerr(tmp_path, cells=cells) is None                       # LaMa not scored yet
    _maskerr_fixture(tmp_path, cells=cells)
    (tmp_path / "geometry.csv").unlink()
    assert m.maskerr(tmp_path, cells=cells) is None                       # geometry not computed
    _maskerr_fixture(tmp_path, cells=cells)
    assert m.maskerr(tmp_path, cells=cells) is not None and m.supp_maskerr(tmp_path, cells=cells) is not None


def test_main_splices_only_the_tags_named(monkeypatch):
    m = _mod()
    spliced = []
    monkeypatch.setattr(m, "TABLES", {"A": lambda: "a", "B": lambda: "b", "C": lambda: None})
    monkeypatch.setattr(m, "splice", lambda tag, body: spliced.append((tag, body)))
    assert m.main(["B"]) == 0 and spliced == [("B", "b")]
    assert m.main([]) == 0 and spliced[1:] == [("A", "a"), ("B", "b")]
    with pytest.raises(SystemExit):
        m.main(["MASKERRR"])


def test_fullres_and_bench_note_the_isolated_method(tmp_path):
    m = _mod()
    corrs = ("CF1_L3", "CF2_L3", "CF3_L3")
    _write_csv(tmp_path / "summary.csv", [{"corruption": c, "method": k, "n_images": 3, "psnr_hole": 30.0}
                                          for c in corrs for k in ("csfr", "shiftmap", "lama")])
    _write_csv(tmp_path / "cost.csv", [{"corruption": c, "image": i, "method": k, "seconds": s, "rss_peak_delta_mb": mb}
                                       for c in corrs for i in range(3) for k, s, mb in
                                       (("csfr", 10.0 + i, 50.0 + i), ("shiftmap", 2.0, 5.0))])
    lines = _body(m.fullres(tmp_path))
    assert lines[0] == r"CSFR & 30.00 & 30.00 & 30.00 & 11.0 & 52 \\"
    assert lines[2] == r"LaMa & 30.00 & 30.00 & 30.00 & -- & -- \\"     # cost recorded in its own environment
    assert m.CHILD_NOTE in lines[-1]
    bench = [{"cell": c, "method": k, "ms_per_patch": ms, "rss_peak_delta_mb": 12.0,
              **({"iterations_median": 800.0, "record_ms_per_query": 4.25} if k == "csfr" else {})}
             for c in ("CF1_L3", "CF2_L5", "CF3_L5") for k, ms in (("csfr", 40.0), ("bilinear", 1.25))]
    _write_csv(tmp_path / "runtime_memory.csv", bench)
    (tmp_path / "environment.json").write_text("{}")
    lines = _body(m.bench_v2(tmp_path))
    assert lines[0] == r"CSFR & 40.0 & 40.0 & 40.0 & 12 \\"
    assert "median iterations 800 / 800 / 800; one record row 4.2 / 4.2 / 4.2" in lines[-1]
    assert m.CHILD_NOTE not in "".join(lines)                             # no shift-map in this bench


def test_ocr_bolds_the_lowest_error_and_sets_the_input_apart(tmp_path):
    m = _mod()
    rows = [{"family": f, "level": lv, "method": k, "cer_mean": v}
            for f in m.FAMILIES for lv in m.MAIN_LEVELS
            for k, v in (("csfr", 0.12), ("guleryuz", 0.10), ("lama", 0.11), ("corrupted_input", 0.05))]
    _write_csv(tmp_path / "summary.csv", rows)
    lines = _body(m.ocr(tmp_path))
    assert lines[1].split(" & ")[1] == r"\textbf{0.100}"            # lowest among compared methods
    assert lines[-1].startswith("Corrupted input (floor) & 0.050") and r"\textbf" not in lines[-1]


def test_downstream_v2_accuracy_ece_and_significance_counts(tmp_path):
    m = _mod()
    met, st = [], []
    for clf, _ in m.CLASSIFIERS:
        met.append({"classifier": clf, "family": "clean", "level": "", "method": "clean", "accuracy": 0.95,
                    "ece_ts": 0.01})
        for f in m.FAMILIES:
            for lv in m.ALL_LEVELS:
                for k, acc in (("csfr", {"L1": 0.9, "L5": 0.5}.get(lv, 0.7)), ("telea", 0.5), ("lama", 0.6)):
                    met.append({"classifier": clf, "family": f, "level": lv, "method": k, "accuracy": acc,
                                "ece_ts": 0.05})
                st.append({"classifier": clf, "family": f, "level": lv, "method": "telea",
                           "statistic": 30 if f != "CF4" else -3, "p_holm": 0.001 if f != "CF4" else 0.6})
    _write_csv(tmp_path / "metrics.csv", met)
    _write_csv(tmp_path / "stats.csv", st)
    lines = _body(m.downstream_v2(tmp_path))
    assert lines[0] == (r"CSFR & 0.700 & 0.700 & 0.700 & 0.700 & 0.050 & -- & 0.700 & 0.700 & 0.700 & 0.700 "
                        r"& 0.050 & -- \\")
    assert lines[1].startswith("Telea & 0.500") and lines[1].endswith(r"& 0.050 & 15/0 \\")
    assert "CNN 0.950, ResNet 0.950" in lines[-1]


def _write_rows(path, rows):
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def _five_level_sweep(tmp_path, n_seeds=3):
    import json
    m = _mod()
    rows = [{"family": f, "level": lv, "method": meth, "n_seeds": n_seeds, "n_patches": 600,
             "psnr_hole_mean": v, "ssim_hole_mean": v / 100, "hrp_mean": v / 10,
             "n_at_cap": 3 if meth == "guleryuz" else 0}
            for f in m.FAMILIES for lv in m.ALL_LEVELS
            for meth, v in (("csfr", 30.0), ("guleryuz", 31.0), ("zero_fill", 8.0))]
    _write_rows(tmp_path / "summary.csv", rows)
    cfg = {"seeds": [0, 1, 2], "families": list(m.FAMILIES), "levels": {lv: 0.0 for lv in m.ALL_LEVELS},
           "csfr_variants": ["csfr"], "baselines": ["guleryuz", "zero_fill"], "external": []}
    (tmp_path / "run_metadata.json").write_text(json.dumps({"config": cfg}))


def test_num_sig():
    m = _mod()
    assert (m.num_sig(0.318), m.num_sig(0.0), m.num_sig(42.0)) == ("0.32", "0", "42")
    assert m.num_sig(6.7e-4) == r"$6.7 \times 10^{-4}$" and m.num_sig(1234.0) == r"$1.2 \times 10^{3}$"


def test_five_level_panels_split_the_families_and_leave_descriptive_values_unbolded(tmp_path):
    m = _mod()
    _five_level_sweep(tmp_path)
    none = tmp_path / "no_fallbacks.csv"
    a, b = m.five_level_panels(tmp_path, "psnr_hole", True, 2, fallbacks=none).split(m.PANEL_SEP)
    assert a.startswith(r"\begin{tabular}{@{}l*{10}{r}@{}}") and "{CF2}" in a and "CF3" not in a
    assert "{CF3}" in b and "{CF4}" in b and "CF1" not in b
    assert _body(a)[1].split(" & ")[1] == r"\textbf{31.00}"            # Guleryuz best everywhere
    hrp = m.five_level_panels(tmp_path, "hrp", None, 2, fallbacks=none)
    assert r"\textbf" not in hrp and _body(hrp.split(m.PANEL_SEP)[0])[0].startswith("CSFR & 3.00 &")
    cap = m.five_level_panels(tmp_path, "n_at_cap", None, 0, column="n_at_cap", fallbacks=none)
    assert _body(cap.split(m.PANEL_SEP)[0])[1] == "Block-DCT thr. & " + " & ".join(["3"] * 10) + r" \\"
    _five_level_sweep(tmp_path, n_seeds=2)
    assert m.five_level_panels(tmp_path, "psnr_hole", True, 2, fallbacks=none) is None


def test_supp_fallbacks_wait_for_every_seed_and_print_percent(tmp_path):
    m = _mod()
    _five_level_sweep(tmp_path)
    rows = [{"sweep": tmp_path.name, "family": f, "level": lv, "method": meth, "n_seeds": 3,
             "frac_fallback": 0.995 if meth == "dictlearn" and f == "CF1" else 0.0}
            for f in m.FAMILIES for lv in m.ALL_LEVELS for meth in ("dictlearn", "criminisi")]
    _write_rows(tmp_path / "fb.csv", rows[:-1] + [{**rows[-1], "n_seeds": 2}])
    assert m.supp_fallbacks(tmp_path, tmp_path / "fb.csv") is None
    _write_rows(tmp_path / "fb.csv", rows)
    a, _ = m.supp_fallbacks(tmp_path, tmp_path / "fb.csv").split(m.PANEL_SEP)
    assert _body(a)[0] == r"Dict.\ learning & " + " & ".join(["99.5"] * 5 + ["0.0"] * 5) + r" \\"


def test_supp_calibration_lists_tolerances_loosest_first_and_marks_the_default(tmp_path):
    m = _mod()
    from csfr_solver import SolverConfig
    base = {"cell": "CF1_L3", "max_err_median": 0.0067, "max_err_p99": 0.035, "psnr_abs_diff_max": 0.0037,
            "iters_median": 190, "iters_max": 360, "n_capped": 0}
    _write_rows(tmp_path / "tolerance.csv", [{**base, "tol": SolverConfig().tol}, {**base, "tol": 1e-4}])
    rows = _body(m.supp_calibration(tmp_path))
    assert rows[0] == r"CF1~L3 & $10^{-4}$ & $6.7 \times 10^{-3}$ & 0.035 & $3.7 \times 10^{-3}$ & 190 & 360 & 0 \\"
    assert rows[1].split(" & ")[1] == m.tex_pow10(SolverConfig().tol) + r"$^{*}$"


def test_supp_shiftmap_needs_both_corpora(tmp_path):
    m = _mod()
    rows = [{"family": f, "level": lv, "n_realisations": 4, "frac_patches_differ": 0.25, "cell_mean_sd": 0.0123,
             "cell_mean_range": 0.031, "patch_range_max": 1.234} for f in m.FAMILIES for lv in m.ALL_LEVELS]
    _write_rows(tmp_path / "sweep_v2_govdocs1.csv", rows)
    assert m.supp_shiftmap(tmp_path) is None
    _write_rows(tmp_path / "sweep_v2_dfrws.csv", rows)
    body = _body(m.supp_shiftmap(tmp_path))
    assert body[0] == r"CF1~L1 & 25 & 0.012 & 0.031 & 1.23 & 25 & 0.012 & 0.031 & 1.23 \\"
    assert "4 realisations per cell: the sweep's output and 3 runs" in body[-1]


def test_supp_dictcap_signs_the_difference(tmp_path):
    m = _mod()
    rows = [{"family": "CF2", "level": "L3", "n_patches": 200, "n_learned": 180, "cap_released": 200,
             "cap_reference": 20000, "psnr_released": 18.21, "psnr_reference": 18.26, "diff_mean": -0.049,
             "diff_abs_max": 1.15, "matches_sweep": True}]
    _write_rows(tmp_path / "sweep_x.csv", rows)
    tab = m.supp_dictcap(tmp_path, tmp_path / "sweep_x")
    assert _body(tab) == [r"CF2~L3 & 180/200 & 18.21 & 18.26 & $-$0.049 & 1.15 \\"]
    assert "cap 200 & cap 20000" in tab


def test_supp_fidelity_has_a_delta_panel_and_a_tolerance_panel(tmp_path):
    m = _mod()
    from csfr_solver import SolverConfig
    tol = SolverConfig().tol
    r = {"family": "CF1", "level": "L3", "rel_err_median_ref": 0.0123, "srf_ref": 0.98, "srf_op": 0.95,
         "converged_frac_op": 1.0}
    _write_rows(tmp_path / "summary.csv", [{**r, "delta": dl, "tol": tol} for dl in (0.25, 1.0, 4.0)])
    _write_rows(tmp_path / "tol_study_summary.csv", [{**r, "delta": 1.0, "tol": t} for t in (1e-8, 1e-4)]
                + [{**r, "delta": 4.0, "tol": 1e-4}])
    a, b = m.supp_fidelity(tmp_path).split(m.PANEL_SEP)
    assert r"\multicolumn{3}{c}{$\delta = 0.25$}" in a
    assert a.splitlines()[-3] == r"CF1~L3 & 1.23 & 98.0 & 95.0 & 1.23 & 98.0 & 95.0 & 1.23 & 98.0 & 95.0 \\"
    rows_b = [ln for ln in b.splitlines() if ln.startswith("CF1")]
    assert [ln.split(" & ")[1] for ln in rows_b] == ["$10^{-4}$", "$10^{-8}$"]     # delta 1 only, loosest first


def test_committed_tables_are_current():
    m = _mod()
    for tag, fn in m.TABLES.items():
        body = fn()
        if body is not None:
            assert _block(tag) == body, f"AUTO {tag} is stale: run scripts/render_tables_v2.py"


def test_supp_cap_mechanism_has_a_panel_per_corpus_and_waits_for_all_three(tmp_path):
    m = _mod()
    row = {"strongest": "criminisi", "diff_mean": -1.5, "capped_part": -2.25, "rest_part": 0.75,
           "n_patches_capped": 14, "n_patches": 600, "n_runs_capped": 40, "n_runs_capped_flat": 10,
           "n_runs_capped_csfr_within_1": 3, "n_runs_capped_flat_csfr_within_1": 3}
    names = [s for s, _ in m.CAP_MECHANISM_CORPORA]
    for s in names[:2]:
        _write_csv(tmp_path / f"{s}.csv", [{"family": f, "level": lv, **row}
                                           for f in m.FAMILIES for lv in m.ALL_LEVELS])
    assert m.supp_cap_mechanism(tmp_path) is None                       # Kodak not in yet
    _write_csv(tmp_path / f"{names[2]}.csv", [{"family": f, "level": lv, **row, "n_runs_capped": 0,
                                                "n_runs_capped_flat": 0, "n_patches_capped": 0}
                                               for f in m.FAMILIES for lv in m.ALL_LEVELS])
    lines = m.supp_cap_mechanism(tmp_path).splitlines()
    assert [ln for ln in lines if ln.startswith(r"\multicolumn{8}")] == [
        rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{n}}}}} \\" for _, n in m.CAP_MECHANISM_CORPORA]
    rows = [ln for ln in lines if ln.startswith("CF")]
    assert len(rows) == 3 * 12
    assert rows[0] == r"CF1~L1 & Exemplar & $-$1.50 & $-$2.25 & +0.75 & 14/600 & 25 & 30 \\"
    assert rows[-1].endswith(r"& 0/600 & -- & -- \\")                   # no capped run: shares undefined


def test_supp_bu_calibration_lists_the_bins_and_the_model_scores(tmp_path):
    import json
    m = _mod()
    assert m.supp_bu_calibration(tmp_path) is None
    _write_csv(tmp_path / "calibration.csv", [
        {"threshold_db": t, "bin": b, "lo": 0.1 * b, "hi": 0.1 * (b + 1), "n": 100 * b,
         "mean_predicted": 0.1 * b + 0.05, "observed_rate": 0.1 * b + 0.04}
        for t in (20.0, 25.0) for b in (1, 2)])
    assert m.supp_bu_calibration(tmp_path) is None                      # waits for the decision
    dec = {f"{t:g}dB": {"logistic_cv_auroc": 0.738,
                        "logistic_cv_ece_10bins": 0.0125, "cv_folds": 10} for t in (20.0, 25.0)}
    (tmp_path / "decision.json").write_text(json.dumps({"by_threshold": dec}), encoding="utf-8")
    lines = m.supp_bu_calibration(tmp_path).splitlines()
    assert sum(ln.startswith(r"\multicolumn{4}{@{}l}{\emph{CSFR hole PSNR below") for ln in lines) == 2
    assert "0.1--0.2 & 100 & 0.150 & 0.140 \\\\" in lines
    assert lines.count(r"\multicolumn{4}{@{}l}{\footnotesize 10-fold AUROC 0.74, ECE 0.013} \\") == 2


def test_plain_prints_a_typographic_minus_and_no_plus():
    m = _mod()
    assert (m.plain(-0.31), m.plain(24.94), m.plain(-166.36, 2)) == ("$-$0.3", "24.9", "$-$166.36")


def _sweep_certs(root, tol=1e-6, missing=None, kodak_short=True):
    import json
    m = _mod()
    for sweep, _ in m.CAP_MECHANISM_CORPORA:
        for f in m.FAMILIES:
            for lv in m.ALL_LEVELS:
                for seed in (0, 1):
                    if (sweep, f"{f}_{lv}", seed) == missing:
                        continue
                    d = root / sweep / f"{f}_{lv}" / f"seed{seed}"
                    d.mkdir(parents=True)
                    big = 100 * (seed + 1) if (f, lv) == ("CF1", "L1") else 50
                    last_ok = not (kodak_short and sweep == "sweep_v2_kodak" and seed == 1)
                    (d / "certificate_csfr.json").write_text(json.dumps(
                        {"iterations": [10, 20, big], "converged": [True, True, last_ok],
                         "gap": [1e-7, 3e-7, 9e-7], "config": {"tol": tol}}), encoding="utf-8")
    return root


def test_supp_convergence_reports_iterations_and_gaps_per_cell(tmp_path):
    m = _mod()
    lines = _body(m.supp_convergence(_sweep_certs(tmp_path)))
    assert lines[0] == (r"CF1~L1 & 20 & 200 & $3.0 \times 10^{-7}$ & 20 & 200 & $3.0 \times 10^{-7}$ "
                        r"& 20 & 200 & $3.0 \times 10^{-7}$ \\")
    assert "GovDocs1 120 of 120; DFRWS 2006 120 of 120; Kodak 100 of 120." in lines[-2]
    assert "Every patch-solve" not in lines[-2]
    assert "Patch-solves per cell (patches times mask seeds): GovDocs1 6, DFRWS 2006 6, Kodak 6." in lines[-1]


def test_supp_convergence_says_every_patch_solve_only_when_all_met(tmp_path):
    lines = _body(_mod().supp_convergence(_sweep_certs(tmp_path, kodak_short=False)))
    assert r"Every patch-solve met the certificate, a gap of at most $10^{-6}$." in lines[-2]


def test_supp_convergence_waits_for_every_seed_and_refuses_mixed_tolerances(tmp_path):
    import json
    m = _mod()
    assert m.supp_convergence(_sweep_certs(tmp_path / "a", missing=("sweep_v2_dfrws", "CF3_L2", 1))) is None
    root = _sweep_certs(tmp_path / "b")
    p = root / "sweep_v2_dfrws" / "CF3_L2" / "seed1" / "certificate_csfr.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    c["config"]["tol"] = 1e-4
    p.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError):
        m.supp_convergence(root)


def test_generators_of_running_stages_wait_for_complete_results(tmp_path):
    """FULLRES, BENCH_V2, OCR and DOWNSTREAM_V2 give None while their source is
    partial: a cell or method missing, or the LaMa pass not yet scored (L28)."""
    m = _mod()
    corrs = m.FULLRES_CORRUPTIONS
    summ = [{"corruption": c, "method": k, "n_images": 3, "psnr_hole": 30.0} for c in corrs for k in ("csfr", "lama")]
    cost = [{"corruption": c, "image": 0, "method": "csfr", "seconds": 1.0, "rss_peak_delta_mb": 1.0} for c in corrs]
    d = tmp_path / "fullres"
    d.mkdir()
    _write_csv(d / "cost.csv", cost)
    _write_csv(d / "summary.csv", [r for r in summ if r["method"] != "lama"])
    assert m.fullres(d) is None                                         # before the LaMa pass
    _write_csv(d / "summary.csv", summ[:-1])
    assert m.fullres(d) is None                                         # a corruption without LaMa
    _write_csv(d / "summary.csv", [{**r, "n_images": 2 if r["method"] == "lama" else 3} for r in summ])
    assert m.fullres(d) is None                                         # LaMa on fewer images
    _write_csv(d / "cost.csv", cost[:-1])
    _write_csv(d / "summary.csv", summ)
    assert m.fullres(d) is None                                         # a cost row missing
    _write_csv(d / "cost.csv", cost)
    assert m.fullres(d) is not None

    d = tmp_path / "bench"
    d.mkdir()
    rows = [{"cell": c, "method": k, "ms_per_patch": 1.0, "rss_peak_delta_mb": 1.0,
             **({"iterations_median": 800.0, "record_ms_per_query": 4.0} if k == "csfr" else {})}
            for c in m.BENCH_CELLS for k in ("csfr", "bilinear")]
    _write_csv(d / "runtime_memory.csv", rows[:-1])
    (d / "environment.json").write_text("{}")
    assert m.bench_v2(d) is None                                        # last method of the last cell running
    _write_csv(d / "runtime_memory.csv", rows)
    (d / "environment.json").unlink()
    assert m.bench_v2(d) is None                                        # the bench has not finished
    (d / "environment.json").write_text("{}")
    assert m.bench_v2(d) is not None

    d = tmp_path / "ocr"
    d.mkdir()
    rows = [{"family": f, "level": lv, "method": k, "cer_mean": 0.1}
            for f in m.FAMILIES for lv in m.MAIN_LEVELS for k in ("csfr", "corrupted_input")]
    _write_csv(d / "summary.csv", rows)
    assert m.ocr(d) is None                                             # LaMa not recognised yet
    lama = [{**r, "method": "lama"} for r in rows if r["method"] == "csfr"]
    _write_csv(d / "summary.csv", rows + lama[:-1])
    assert m.ocr(d) is None
    _write_csv(d / "summary.csv", rows + lama)
    assert m.ocr(d) is not None

    d = tmp_path / "downstream"
    d.mkdir()
    met = [{"classifier": clf, "family": f, "level": lv, "method": k, "accuracy": 0.5, "ece_ts": 0.05}
           for clf, _ in m.CLASSIFIERS for f in m.FAMILIES for lv in m.ALL_LEVELS for k in ("csfr", "lama")]
    clean = [{"classifier": clf, "family": "clean", "level": "", "method": "clean", "accuracy": 0.9, "ece_ts": 0.01}
             for clf, _ in m.CLASSIFIERS]
    _write_csv(d / "stats.csv", [{"classifier": "cnn", "family": "CF1", "level": "L1", "method": "lama",
                                  "statistic": 1, "p_holm": 0.5}])
    _write_csv(d / "metrics.csv", clean + [r for r in met if r["classifier"] == "cnn"])
    assert m.downstream_v2(d) is None                                   # the second classifier not run
    _write_csv(d / "metrics.csv", clean + [r for r in met if r["method"] != "lama"])
    assert m.downstream_v2(d) is None                                   # LaMa not scored
    _write_csv(d / "metrics.csv", clean[:1] + met)
    assert m.downstream_v2(d) is None                                   # a classifier without its clean row
    _write_csv(d / "metrics.csv", clean + met)
    assert m.downstream_v2(d) is not None
