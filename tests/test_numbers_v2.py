"""The revised manuscript states numbers through paper/numbers_v2.tex. This test
locks that file to its sources: regenerating it must reproduce the committed
copy byte for byte, and the config-derived values must equal the code
defaults."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("numbers_v2", ROOT / "scripts" / "render_numbers_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_committed_numbers_file_is_current():
    m = _mod()
    committed = (ROOT / "paper" / "numbers_v2.tex").read_text(encoding="utf-8")
    assert committed == m.render(), "paper/numbers_v2.tex is stale: run scripts/render_numbers_v2.py"


def test_macro_names_are_valid_control_words():
    """A LaTeX control word is letters only: \\calibP99Max would parse as
    \\calibP followed by the text 99Max."""
    m = _mod()
    import re
    names = re.findall(r"\\providecommand\{\\([^}]*)\}", m.render())
    assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names), \
        [n for n in names if not re.fullmatch(r"[A-Za-z]+", n)]


def test_typed_claims_about_the_calibration_hold():
    """Sentences the manuscript types around the macros."""
    import csv
    rows = list(csv.DictReader((ROOT / "results/solver_calibration/tolerance.csv").open(encoding="utf-8")))
    assert sum(int(r["n_capped"]) for r in rows) == 0, "paper says no calibration solve reached the cap"
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    assert "No calibration solve reached the iteration cap" in tex


def test_typed_claims_about_the_primary_comparison_hold():
    """The words the quality section types around the comparison macros
    ("every", "none", "the other way") hold for the committed values, and the
    sentences that carry them are still in the manuscript."""
    import re
    v = dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{(.*)\}$",
                        (ROOT / "paper" / "numbers_v2.tex").read_text(encoding="utf-8"), re.M))
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    claims = {
        r"The strongest method at every level is \qGovCFoneStrongest":
            " and " not in v["qGovCFoneStrongest"] and v["qGovCFoneNFirst"] == "0",
        "Block thresholding stays ahead at every level on the patches that no method reconstructs to the cap":
            v["qGovCFoneStrongest"] == "block-DCT thresholding" and v["qGovCFoneCapfreeNAhead"] == "0",
        "CSFR is ahead of LaMa at every CF1 level": v["qGovCFoneLamaBehindMax"] == "--",
        "LaMa is ahead of CSFR at every CF2 level": v["qGovCFtwoLamaAheadMax"] == "--",
        "Per source image the comparison goes the other way":
            v["qGovCFthreeNFirst"] == v["qGovCFfourNFirst"] == "0"
            and min(int(v["qGovCFthreeLedMin"]), int(v["qGovCFfourLedMin"])) > 50,
        "CSFR is ahead of the strongest method at every CF3 and CF4 level":
            v["qGovCFthreeCapfreeNAhead"] == v["qGovCFfourCapfreeNAhead"] == "5",
        "and below zero at none": v["qGovCFthreeNLamaBehindSig"] == v["qGovCFfourNLamaBehindSig"] == "0",
        "CSFR is not the best method on hole GMSD in any cell": v["qGovGmsdNFirst"] == "0",
    }
    for sentence, holds in claims.items():
        assert sentence in tex, f"sentence gone from the manuscript (update this test): {sentence}"
        assert holds, f"the manuscript says '{sentence}', which the generated values contradict"


def test_every_macro_the_manuscript_uses_is_generated():
    m = _mod()
    import re
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    generated = set(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", m.render()))
    used = set(re.findall(r"\\((?:csfr|rec|calib|oracle|corpus|methods|env|sweep|stats|fallback|bu|shiftmap|dictlearn|mech|q(?:Gov|Dfrws|Kodak))[A-Z][A-Za-z]*)", tex))
    assert used, "the manuscript uses no number macros; the regex has rotted"
    assert used <= generated, f"manuscript uses undefined number macros: {sorted(used - generated)}"


def test_word_build_inlines_the_number_macros():
    """pandoc resolves \\input against its working directory and drops a file it
    cannot find with only a warning, so every number would vanish from the
    docx. The docx preparation must inline the macro file."""
    spec = importlib.util.spec_from_file_location("docx_build", ROOT / "scripts" / "build_ieee_docx.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    out = d.inline_inputs("pre\n\\input{numbers_v2}\npost $\\csfrTol$\n")
    assert "\\input{numbers_v2}" not in out
    assert "\\providecommand{\\csfrTol}" in out and out.endswith("post $\\csfrTol$\n")
    import inspect
    body = inspect.getsource(d.prepare)
    assert body.index("inline_inputs(") < body.index("resolve_anonymous("), \
        "prepare() must inline the inputs before any other step reads the tex"


def test_shiftmap_macros_take_the_worst_cell_of_both_corpora(tmp_path):
    m = _mod()
    import csv

    def write(sweep, rows):
        with (tmp_path / f"{sweep}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["family", "level", "n_realisations", "frac_patches_differ",
                                              "cell_mean_range"])
            w.writeheader()
            w.writerows(rows)

    write("sweep_v2_govdocs1", [{"family": "CF1", "level": "L1", "n_realisations": 4,
                                 "frac_patches_differ": 0.10, "cell_mean_range": 0.05}])
    assert set(m.shiftmap_macros(tmp_path).values()) == {m.PLACEHOLDER}      # one corpus is not enough
    write("sweep_v2_dfrws", [{"family": "CF2", "level": "L5", "n_realisations": 4,
                              "frac_patches_differ": 0.42, "cell_mean_range": 0.013}])
    assert m.shiftmap_macros(tmp_path) == {"shiftmapRealisations": "4", "shiftmapDifferMax": "42",
                                           "shiftmapCellRangeMax": "0.05"}


def test_fallback_macros_wait_for_a_complete_primary_sweep(tmp_path):
    """A resumed sweep leaves fallbacks.csv with some cells at fewer seeds and
    others missing; the macros must stay pending, not fail or report a part."""
    m = _mod()
    import csv
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seeds: [0, 1]\nfamilies: [CF1]\nlevels: {L1: 0.05, L5: 0.7}\n")
    fields = ["sweep", "family", "level", "method", "n_seeds", "n", "n_fallback", "frac_fallback"]

    def write(rows):
        with (tmp_path / "fb.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([dict(zip(fields, (m.SWEEPS[0], *r))) for r in rows])

    full = [("CF1", "L1", "dictlearn", 2, 10, 9, 0.9), ("CF1", "L1", "criminisi", 2, 10, 0, 0.0),
            ("CF1", "L5", "dictlearn", 2, 10, 10, 1.0), ("CF1", "L5", "criminisi", 2, 10, 4, 0.4)]
    write(full[:3] + [("CF1", "L5", "criminisi", 1, 5, 2, 0.4)])           # one cell at one seed
    assert set(m.fallback_macros(tmp_path / "fb.csv", cfg).values()) == {m.PLACEHOLDER}
    write(full)
    assert m.fallback_macros(tmp_path / "fb.csv", cfg) == {
        "fallbackDictCFoneMin": "90.0", "fallbackDictCFoneMax": "100.0", "fallbackExemplarCFoneLfive": "40.0"}


def test_dictlearn_cap_macros_take_the_largest_shift_and_refuse_a_mismatch(tmp_path):
    m = _mod()
    import csv
    import pytest

    def write(rows):
        with (tmp_path / f"{m.SWEEPS[0]}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    assert set(m.dictlearn_cap_macros(tmp_path).values()) == {m.PLACEHOLDER}
    row = {"cap_released": 200, "cap_reference": 20000, "n_patches": 200, "matches_sweep": "True"}
    write([{**row, "diff_mean": -0.031, "diff_abs_max": 1.26}, {**row, "diff_mean": 0.012, "diff_abs_max": 0.4}])
    assert m.dictlearn_cap_macros(tmp_path) == {
        "dictlearnCapFactor": "100", "dictlearnCapCells": "2", "dictlearnCapPatches": "200",
        "dictlearnCapMaxCellShift": "0.03", "dictlearnCapMaxPatchShift": "1.3"}
    write([{**row, "diff_mean": 0.0, "diff_abs_max": 0.0, "matches_sweep": "False"}])
    with pytest.raises(ValueError, match="did not reproduce"):
        m.dictlearn_cap_macros(tmp_path)


def test_sci_formatting():
    m = _mod()
    assert m._sci(1e-6) == "10^{-6}"
    assert m._sci(5e-12) == "5\\times 10^{-12}"
    assert m._sci(1.07e-3, 1) == "1.1\\times 10^{-3}"
    assert m._sci(2.1e-14, 1) == "2.1\\times 10^{-14}"
    assert m._sci(1e-3, 1) == "1.0\\times 10^{-3}"


def test_config_macros_match_code_defaults():
    m = _mod()
    from csfr_solver import SolverConfig
    s = SolverConfig()
    c = m.config_macros()
    assert float(c["csfrLamLone"]) == s.lam_l1
    assert float(c["csfrLamTV"]) == s.lam_tv
    assert float(c["csfrLamFC"]) == s.lam_fc
    assert c["csfrTol"] == m._sci(s.tol)


def test_calibration_macros_follow_the_selection_rule():
    """The calibrated tolerance is the loosest whose p99 error is below 0.01
    grey levels in every calibration cell; the next looser one fails somewhere."""
    m = _mod()
    import csv
    rows = list(csv.DictReader((ROOT / "results/solver_calibration/tolerance.csv").open(encoding="utf-8")))
    from csfr_solver import SolverConfig
    grid = {float(r["tol"]) for r in rows}
    passing = [t for t in grid
               if all(float(r["max_err_p99"]) < 0.01 for r in rows if float(r["tol"]) == t)]
    assert SolverConfig().tol == max(passing), "default tol is not the calibrated one"
    c = m.calibration_macros()
    assert c["calibNextFailCells"] != m.PLACEHOLDER


def _r1_write(d, rows, residual=1.2e-6):
    import csv
    d.mkdir(parents=True, exist_ok=True)
    fields = ["family", "level", "seed", "baseline", "hole_pixels_min", "hole_pixels_max", "n_pixels", "n_whole",
              "n_hole", "n_capped", "whole_diff", "hole_diff", "change", "cap_part", "rest_part", "dropped_part",
              "capped_mean_move"]
    with (d / "comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, 0) for k in fields} for r in rows)
    with (d / "shift_residual.csv").open("w", newline="", encoding="utf-8") as f:
        f.write(f"method,n_patches_checked,max_abs_residual_db\ncsfr,10,{residual}\n")


def _r1_row(**kw):
    import math
    move = -10 * math.log10(4096 / 192)
    base = dict(family="CF4", level="L1", seed=0, baseline="inpaint_telea", hole_pixels_min=192, hole_pixels_max=192,
                n_pixels=4096, n_whole=300, n_hole=300, n_capped=3, whole_diff=0.106, hole_diff=0.106 + 3 * move / 300,
                change=3 * move / 300, cap_part=3 * move / 300, rest_part=4e-8, capped_mean_move=move)
    return {**base, **kw}


def test_r1_hole_whole_macros(tmp_path):
    """The response to R4.4: pending without the CSVs; the largest change and
    its cell; a comparison without capped patches does not count as capped."""
    m = _mod()
    assert set(m.r1_hole_whole_macros(tmp_path / "none").values()) == {m.PLACEHOLDER}
    _r1_write(tmp_path, [_r1_row(), _r1_row(family="CF1", baseline="bilinear", n_capped=0, change=1e-9,
                                            cap_part=0.0, rest_part=1e-9, capped_mean_move="nan")])
    c = m.r1_hole_whole_macros(tmp_path)
    assert (c["rOneComparisons"], c["rOneCappedComparisons"], c["rOneChangeMax"]) == ("2", "1", "0.13")
    assert (c["rOneBaselines"], c["rOneCells"], c["rOneSeeds"]) == ("2", "2", "1")
    assert (c["rOneWorstFamily"], c["rOneWorstLevel"], c["rOneWorstBaseline"]) == ("CF4", "L1", "Telea")
    assert (c["rOneWorstMove"], c["rOneWorstWhole"], c["rOneWorstHole"]) == ("13.29", "+0.11", "-0.03")
    assert c["rOneRestMax"] == r"4\times 10^{-8}" and c["rOneShiftResidualMax"] == r"1.2\times 10^{-6}"


# a dropped patch; a move that is not the shift; a hole that varies; CSFR, not the baseline, at the cap
@pytest.mark.parametrize("bad", [dict(n_hole=299), dict(capped_mean_move=-5.0), dict(hole_pixels_max=200),
                                 dict(capped_mean_move=10 * __import__("math").log10(4096 / 192))])
def test_r1_hole_whole_macros_refuse_a_false_explanation(tmp_path, bad):
    m = _mod()
    _r1_write(tmp_path, [_r1_row(**bad)])
    with pytest.raises(ValueError):
        m.r1_hole_whole_macros(tmp_path)


def _paired(d, cells, missing_lama=None):
    """stats/paired_tests.csv for cells {(fam, lv): {method: (diff, lo, hi, diff_capfree)}}
    plus one GMSD row per cell; CSFR's mean is 30 dB, a method's 30 - diff."""
    import csv
    rows = []
    for (f, lv), methods in cells.items():
        for meth, (diff, lo, hi, cf) in methods.items():
            if meth == "lama" and (f, lv) == missing_lama:
                continue
            rows.append({"family": f, "level": lv, "metric": "psnr_hole", "method": meth,
                         "csfr_mean": 30.0, "method_mean": 30.0 - diff, "diff_mean": diff,
                         "ci_cluster_lo": lo, "ci_cluster_hi": hi,
                         "frac_images_csfr_leads": 0.1 + 0.1 * int(lv[1]),          # 20 % at L1 .. 60 % at L5
                         "diff_mean_capfree": "" if meth == "lama" else cf})
        rows.append({"family": f, "level": lv, "metric": "gmsd_hole", "method": "fsr",
                     "csfr_mean": 0.2, "method_mean": 0.1, "diff_mean": -0.1,
                     "ci_cluster_lo": -0.2, "ci_cluster_hi": -0.05, "frac_images_csfr_leads": 0.0,
                     "diff_mean_capfree": ""})
    (d / "stats").mkdir(parents=True)
    with (d / "stats" / "paired_tests.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _comparison_cells():
    """CF1: block-DCT ahead at every level by 1.0 + 0.5 i, significant at four;
    CF2: FSR ahead at L1-L3, CSFR ahead at L4-L5 (significant at L5);
    CF3, CF4: CSFR first everywhere, by 0.2. LaMa: behind CSFR on CF1, ahead
    elsewhere. One cell (CF2 L4) whose cap-free patches change the strongest
    method and the sign."""
    cells = {}
    for i, lv in enumerate(("L1", "L2", "L3", "L4", "L5")):
        g = 1.0 + 0.5 * i
        cells[("CF1", lv)] = {"guleryuz": (-g, -g - 0.5, -0.1 if i else 0.3, -g),
                              "fsr": (2.0, 1.0, 3.0, 2.0), "lama": (3.0, 1.0, 5.0, "")}
        f2 = (-0.4, -0.6, -0.2, -0.4) if i < 3 else (0.3, 0.1 if i == 4 else -0.1, 0.5, 0.3)
        cf2_fsr_capfree = -0.2 if lv == "L4" else f2[3]
        cells[("CF2", lv)] = {"fsr": (*f2[:3], cf2_fsr_capfree),
                              "biharmonic": (0.5 if lv != "L4" else 0.4, 0.2, 0.8, -0.3 if lv == "L4" else 0.5),
                              "lama": (-1.0, -1.5, -0.5, "")}
        for f in ("CF3", "CF4"):
            cells[(f, lv)] = {"inpaint_ns": (0.2, -0.1, 0.4, 0.2), "biharmonic": (0.8, 0.5, 1.1, 0.8),
                             "lama": (-0.6, -1.2, 0.1, "")}
    return cells


def test_comparison_macros_read_rank_gap_and_intervals_per_family(tmp_path):
    m = _mod()
    assert set(m.comparison_macros("Gov", tmp_path).values()) == {m.PLACEHOLDER}
    _paired(tmp_path, _comparison_cells())
    q = m.comparison_macros("Gov", tmp_path)
    assert set(q) == set(m.comparison_names("Gov"))
    assert (q["qGovCFoneRankMin"], q["qGovCFoneRankMax"], q["qGovCFoneNFirst"]) == ("2", "2", "0")
    assert q["qGovCFoneStrongest"] == "block-DCT thresholding"
    assert (q["qGovCFoneBehindMin"], q["qGovCFoneBehindMax"], q["qGovCFoneNBehindSig"]) == ("1.00", "3.00", "4")
    assert (q["qGovCFoneAheadMax"], q["qGovCFoneNAheadSig"]) == ("--", "0")
    assert (q["qGovCFoneLamaAheadMax"], q["qGovCFoneLamaBehindMax"], q["qGovCFoneNLamaAheadSig"]) == ("3.00", "--", "5")
    assert (q["qGovCFtwoRankMin"], q["qGovCFtwoRankMax"], q["qGovCFtwoNFirst"]) == ("1", "2", "2")
    assert q["qGovCFtwoStrongest"] == "FSR"                          # best by the signed difference
    assert (q["qGovCFtwoBehindMin"], q["qGovCFtwoBehindMax"], q["qGovCFtwoNBehindSig"]) == ("0.40", "0.40", "3")
    assert (q["qGovCFtwoAheadMax"], q["qGovCFtwoNAheadSig"]) == ("0.30", "1")
    assert (q["qGovCFtwoLamaBehindMax"], q["qGovCFtwoNLamaBehindSig"]) == ("1.00", "5")
    assert (q["qGovCFthreeNFirst"], q["qGovCFthreeStrongest"], q["qGovCFthreeNLamaBehindSig"]) == ("5", "Navier--Stokes", "0")
    assert q["qGovNMethods"] == "3" and q["qGovNFirst"] == "12"      # CF2 L4-L5, CF3, CF4
    assert (q["qGovGmsdNFirst"], q["qGovGmsdRankMin"]) == ("0", "2")
    assert (q["qGovCapfreeNewStrongest"], q["qGovCapfreeNSignFlip"]) == ("1", "1")
    assert q["qGovCapfreeNFirst"] == "11"                           # CF2 L4 is no longer first
    assert (q["qGovCFoneLedMin"], q["qGovCFoneLedMax"]) == ("20", "60")
    assert (q["qGovCFoneCapfreeNAhead"], q["qGovCFtwoCapfreeNAhead"], q["qGovCFthreeCapfreeNAhead"]) == ("0", "1", "5")


def _mech_row(fam, lv, **kw):
    r = {"sweep": "s", "family": fam, "level": lv, "strongest": "criminisi", "n_patches": 600, "n_seeds": 3,
         "diff_mean": -1.0, "capped_part": -2.0, "rest_part": 1.0, "n_patches_capped": 20, "n_runs_capped": 60,
         "n_runs_capped_flat": 30, "n_runs_capped_csfr_within_1": 10, "n_runs_capped_flat_csfr_within_1": 10,
         "n_runs_capped_flat_tv_only_at_cap": 27,
         "csfr_psnr_capped_median": 55.0, "csfr_flat_mean_err_median": -0.5, "csfr_flat_max_abs_err_median": 1.0}
    r.update(kw)
    return r


def _mech(d, gov_overrides=None, kodak_capped=0):
    gov = [_mech_row(f, lv) for f in ("CF1", "CF2", "CF3", "CF4") for lv in ("L1", "L2", "L3", "L4", "L5")]
    cf3_l1 = {"n_patches_capped": 5, "capped_part": -4.5, "rest_part": 0.25,
              "csfr_flat_mean_err_median": -0.1, "csfr_psnr_capped_median": 48.0, **(gov_overrides or {})}
    gov[10] = _mech_row("CF3", "L1", **cf3_l1)
    kodak = [_mech_row(f, lv, n_patches_capped=kodak_capped if f == "CF4" else 0)
             for f in ("CF1", "CF2", "CF3", "CF4") for lv in ("L1", "L2", "L3", "L4", "L5")]
    _write_rows(d / "sweep_v2_govdocs1.csv", gov)
    _write_rows(d / "sweep_v2_kodak.csv", kodak)


def _write_rows(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_mechanism_macros_read_the_contiguous_cells_of_the_primary_corpus(tmp_path):
    m = _mod()
    assert set(m.mechanism_macros(tmp_path).values()) == {m.PLACEHOLDER}
    _mech(tmp_path)
    q = m.mechanism_macros(tmp_path)
    assert (q["mechContigCappedMin"], q["mechContigCappedMax"]) == ("5", "20")
    assert (q["mechContigCapPartMin"], q["mechContigCapPartMax"]) == ("2.00", "4.50")
    assert (q["mechContigRestMin"], q["mechContigRestMax"]) == ("0.25", "1.00")
    assert (q["mechContigBiasMin"], q["mechContigBiasMax"]) == ("0.10", "0.50")
    assert (q["mechContigFlatShare"], q["mechContigWithinShare"]) == ("50", "33")
    assert (q["mechContigCsfrPsnrMin"], q["mechContigCsfrPsnrMax"]) == ("48", "55")
    assert q["mechContigTvExactShare"] == "90"


@pytest.mark.parametrize("bad", [{"rest_part": -0.1}, {"capped_part": 0.5, "rest_part": 0.1},
                                 {"capped_part": -0.2, "rest_part": 0.3}, {"csfr_flat_mean_err_median": 0.1}])
def test_mechanism_macros_refuse_a_cell_that_contradicts_the_text(tmp_path, bad):
    m = _mod()
    _mech(tmp_path, gov_overrides=bad)
    with pytest.raises(ValueError):
        m.mechanism_macros(tmp_path)


def test_mechanism_macros_refuse_a_capped_kodak_patch(tmp_path):
    m = _mod()
    _mech(tmp_path, kodak_capped=1)
    with pytest.raises(ValueError, match="Kodak"):
        m.mechanism_macros(tmp_path)


def test_comparison_macros_need_lama_at_every_level(tmp_path):
    m = _mod()
    _paired(tmp_path, _comparison_cells(), missing_lama=("CF3", "L2"))
    with pytest.raises(ValueError, match="LaMa is not scored"):
        m.comparison_macros("Gov", tmp_path)
