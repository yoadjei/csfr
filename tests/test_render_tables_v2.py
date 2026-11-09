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


def _fixture_summary(tmp_path, values):
    """summary.csv with the columns run_sweep_v2.write_summary writes that the
    generator reads; values maps (family, level, method) -> psnr_hole_mean."""
    import csv
    rows = [{"family": f, "level": lv, "method": m, "psnr_hole_mean": v, "gmsd_hole_mean": 1.0 / v,
             "n_patches": 900, "n_at_cap": int(m == "guleryuz" and (f, lv) == ("CF2", "L3")) * 7}
            for (f, lv, m), v in values.items()]
    with (tmp_path / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_method_by_cell_table_orders_bolds_and_dashes(tmp_path):
    m = _mod()
    cells = [(f, lv) for f in m.FAMILIES for lv in m.MAIN_LEVELS]
    values = {}
    for i, c in enumerate(cells):
        values[(*c, "csfr")] = 30.0 + i
        values[(*c, "guleryuz")] = 31.0 + i if c[0] == "CF1" else 29.0 + i
        values[(*c, "zero_fill")] = 99.0          # the floor never takes the bold
        if c != ("CF4", "L5"):
            values[(*c, "lama")] = 20.0
    _fixture_summary(tmp_path, values)
    none = tmp_path / "no_fallbacks.csv"
    lines = _body(m.method_by_cell_table(tmp_path, "psnr_hole", True, 2, fallbacks=none))
    assert [ln.split(" & ")[0] for ln in lines] == ["CSFR", "Block-DCT thr.", "LaMa", r"\midrule",
                                                   "Zero-fill (floor)"]
    csfr, gul, lama = (ln.split(" & ") for ln in lines[:3])
    assert gul[1] == r"\textbf{31.00}" and csfr[1] == "30.00"        # CF1 L1: Guleryuz best
    assert csfr[4] == r"\textbf{33.00}" and gul[4] == "32.00"         # CF2 L1: CSFR best
    assert lama[-1] == r"-- \\"                                      # did not run in CF4 L5
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


def test_missing_cell_is_an_error(tmp_path):
    m = _mod()
    _fixture_summary(tmp_path, {("CF1", "L1", "csfr"): 30.0})
    import pytest
    with pytest.raises(ValueError, match="no results for cell CF1 L3"):
        m.method_by_cell_table(tmp_path, "psnr_hole", True, 2)


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


def test_committed_tables_are_current():
    m = _mod()
    for tag, fn in m.TABLES.items():
        body = fn()
        if body is not None:
            assert _block(tag) == body, f"AUTO {tag} is stale: run scripts/render_tables_v2.py"
