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
    SECONDARY = ("Dfrws", "Kodak")
    PENDING = _mod().PLACEHOLDER
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
        # Secondary corpora (Section VI-G).
        "On both corpora the strongest method at every CF1 level is block-DCT thresholding":
            all(v[f"q{c}CFoneStrongest"] == "block-DCT thresholding" and v[f"q{c}CFoneNFirst"] == "0" for c in SECONDARY),
        "with intervals that exclude zero at every level, and CSFR is ahead of LaMa at every CF1 level":
            all(v[f"q{c}CFoneNBehindSig"] == "5" and v[f"q{c}CFoneLamaBehindMax"] == "--" for c in SECONDARY),
        "Under contiguous loss CSFR is closer to the strongest method than on the primary corpus at every level":
            max(float(v[f"q{c}{f}BehindMax"]) for c in SECONDARY for f in ("CFthree", "CFfour"))
            < min(float(v[f"qGov{f}BehindMin"]) for f in ("CFthree", "CFfour")),
        "Across these 20 cells the interval excludes zero against CSFR in one cell and in its favour in three, all three on Kodak":
            sum(int(v[f"q{c}{f}NBehindSig"]) for c in SECONDARY for f in ("CFthree", "CFfour")) == 1
            and sum(int(v[f"qKodak{f}NAheadSig"]) for f in ("CFthree", "CFfour")) == 3
            and sum(int(v[f"qDfrws{f}NAheadSig"]) for f in ("CFthree", "CFfour")) == 0,
        "The differences on DFRWS 2006 are small enough that even these few patches matter":
            any(v[f"qDfrws{f}CapfreeNAhead"] != v[f"qDfrws{f}NFirst"] for f in ("CFthree", "CFfour")),
        r"Under burst loss CSFR is first at \qDfrwsCFtwoNFirst\ of the five levels on each corpus":
            v["qDfrwsCFtwoNFirst"] == v["qKodakCFtwoNFirst"],
        "On DFRWS 2006 LaMa is ahead of CSFR at every CF2 level":
            v["qDfrwsCFtwoLamaAheadMax"] == "--" and v["qDfrwsCFtwoNLamaBehindSig"] == "5",
        "It is not the best method on hole GMSD in any cell of either corpus":
            all(v[f"q{c}GmsdNFirst"] == "0" for c in SECONDARY),
        # Constraint ablation (Section VI-C). ablation_macros refuses data that break its
        # "every"/"no"/"none" sentences, so for those the sentence must still be there and
        # the macros must be generated.
        "Without it (DCT only), hole PSNR is lower in every cell": v["ablNoTvMin"] != PENDING,
        "they cost more often than they gain":
            int(v["ablTvNBehind"]) > int(v["ablTvNAhead"]) and int(v["ablTvNBehind"]) + int(v["ablTvNAhead"]) == 20,
        "in every CF2 to CF4 cell those holes move the mean difference in its favour": v["ablTvBcFlatMin"] != PENDING,
        "On the other holes CSFR is ahead in every such cell": v["ablTvBcRestMin"] != PENDING,
        "the median difference favours CSFR in every such cell": v["ablTvBcAheadMin"] != PENDING,
        "Scattered loss leaves no flat hole": v["ablTvScatterBehindLevels"] != PENDING,
        "Under burst and contiguous loss the flat holes favour the arm without the penalty in every cell":
            v["ablNoFcBcRestMaxAbs"] != PENDING,
        "with C3 none do": v["ablClipShare"] != PENDING,
        # Record fidelity (Section VI-D). record_fidelity_macros refuses a sign disagreement,
        # a failure away from the hole and an operational SRF that differs from the reference.
        "the record agrees in sign with the measured response for every query": v["recNQueries"] != PENDING,
        "The operational re-solve gives the SRF of the reference in every cell": v["recSrfPooled"] != PENDING,
        "Every failure is next to the hole": v["recFarN"] != PENDING,
        "Both ratios are four for the second-order remainder of an exact derivative":
            1 / float(v["recDeltaMin"]) == 4 == float(v["recDeltaMax"]),
        "The departure is therefore almost entirely the curvature of the solution map":
            all(3.5 < float(v[k]) < 4.5 for k in ("recRatioLow", "recRatioHigh")),
        "SRF falls as the perturbation grows":
            float(v["recSrfSmallDelta"]) > float(v["recSrfPooled"]) > float(v["recSrfLargeDelta"]),
        # Comparison records (Section VI-E). record_comparison_macros refuses a zero CSFR
        # column and a non-zero exact linear record under CF3-CF4.
        "For every query its column is non-zero": v["recCsfrSupportMinShare"] != PENDING,
        "The records of the comparison methods are concentrated near the query":
            float(v["recCmpReachMax"]) < float(v["recCsfrReachMin"]),
        "and for every CF3 and CF4 query": v["recLinZeroScatter"] != PENDING,
        "the value it takes when the one-grey-level record is zero": v["recPdeFdDiffMed"] == "1.00",
        # Discussion, convexity (sec:discussion:convexity).
        "a non-convex iteration, is ahead of CSFR at every level of all three corpora":
            all(v[f"q{c}CFoneStrongest"] == "block-DCT thresholding" and v[f"q{c}CFoneNFirst"] == "0"
                for c in ("Gov", *SECONDARY)),
        "on each corpus the deficit shrinks at every step from L1 to L5":
            all(v[f"q{c}CFoneLevelTrend"] == "rises" for c in ("Gov", *SECONDARY)),
        "Both arms gain most over CSFR at L1 and less at every step to L5, as does block-DCT thresholding":
            v["ablTvScatterTrend"] == v["ablNoFcScatterTrend"] == "rises"
            and all(v[f"q{c}CFoneLevelTrend"] == "rises" for c in ("Gov", *SECONDARY)),
        "the TV-only arm, which is also convex, is ahead at CF1": v["ablTvScatterBehindLevels"] != PENDING,
        # Discussion, where CSFR is best suited (sec:discussion:scope; R1.5, R3.4).
        "Under scattered loss it is not: block-DCT thresholding is ahead at every level of every corpus":
            all(v[f"q{c}CFoneStrongest"] == "block-DCT thresholding" and v[f"q{c}CFoneNFirst"] == "0"
                for c in ("Gov", *SECONDARY)),
        "Under burst loss LaMa is ahead of CSFR at every level on the primary corpus and on DFRWS 2006":
            v["qGovCFtwoLamaAheadMax"] == v["qDfrwsCFtwoLamaAheadMax"] == "--",
        r"CSFR is first at some levels and elsewhere at most \qDfrwsCFthreeBehindMax":
            sum(int(v[f"q{c}{f}NFirst"]) for c in SECONDARY for f in ("CFthree", "CFfour")) > 0
            and float(v["qDfrwsCFthreeBehindMax"]) == max(float(v[f"q{c}{f}BehindMax"])
                                                         for c in SECONDARY for f in ("CFthree", "CFfour")),
        "CSFR is ahead of the strongest method at every contiguous-loss level on the patches that no method":
            v["qGovCFthreeCapfreeNAhead"] == v["qGovCFfourCapfreeNAhead"] == "5",
        "it has the higher mean in most source images":
            min(int(v["qGovCFthreeLedMin"]), int(v["qGovCFfourLedMin"])) > 50,
        "which CSFR fills slightly dark": v["mechContigBiasMin"] != PENDING,       # mechanism_macros refuses otherwise
        "under light scattered loss on most patches as well":
            v["ablTvScatterAheadShareMax"] != PENDING and int(v["ablTvScatterAheadShareMax"]) < 50,
        "while CSFR is ahead of it on the other holes and in the median under burst and contiguous loss":
            v["ablTvBcRestMin"] != PENDING,
        # What the observed pixels can determine (sec:results:oracle; R2.1b, R2.1c).
        # oracle_macros refuses data that break its "every"/"no" sentences.
        "no CF1 patch exceeds $10^{3}$": v["oracleScatterCondMax"] != PENDING,
        "Under burst loss the median condition number rises at every level": v["oracleBurstCondLastMin"] != PENDING,
        "In every CF3 and CF4 cell from that level, on both corpora, the median condition number exceeds $10^{4}$":
            v["oracleContigIllLevel"] != PENDING,
        "which uses a DCT sparsity model without CSFR's penalties, is the strongest method at every level":
            all(v[f"q{c}CFoneStrongest"] == "block-DCT thresholding" and v[f"q{c}CFoneNFirst"] == "0"
                for c in ("Gov", "Dfrws")),
        "removing TV lowers hole PSNR in every cell": v["ablNoTvMin"] != PENDING,
        "CSFR is ahead of the strongest method at every contiguous-loss level of the primary corpus":
            v["qGovCFthreeCapfreeNAhead"] == v["qGovCFfourCapfreeNAhead"] == "5",
        # The Bounded-Unknown diagnostic (sec:results:bu; R1.9, R2.m3). bu_result_macros
        # refuses a met rule, a cell at the within-cell minimum, a contiguous-loss cell
        # whose masks separate the patches, and a model that adds discrimination.
        "The rule is met at neither threshold": v["buNCells"] != PENDING,
        "no statistic reaches an AUROC of": v["buWithinMax"] != PENDING,
        "so its within-cell AUROC is exactly one half": v["buWithinMax"] != PENDING,
        "combining and calibrating the statistics adds no discrimination": v["buLogitAurocLow"] != PENDING,
        "it does not meet the rule": v["buNCells"] != PENDING,
        "does not identify the reconstructions that fail": v["buNCells"] != PENDING,
        # The solver comparison (sec:results:solver, sec:discussion; R1.1, R1.8, R2.3, R2.10).
        # solver_macros refuses a re-run that is not bit-identical, an uncertified patch, a
        # structured family where the longer run does not gain, is not passed by the
        # convergent solver or recovers at most half of the gain, an Adam trace whose PSNR
        # does not rise at every point, and an Adam loss not still falling at the budget.
        "Under scattered loss the arms agree": float(v["solverScatterDiffMax"]) < 0.1,
        "dB ahead of that longer run": v["solverFistaOverLongMin"] != PENDING,
        "reproduced them bit for bit in every cell": v["solverRetimeSeed"] != PENDING,
        "The convergent solver met the certificate on every patch at every tolerance": v["solverNPatches"] != PENDING,
        "so the choice of tolerance does not move the reported quality": float(v["solverTolDiffMax"]) < 0.01,
        "while its hole PSNR rose at every traced point": v["solverTraceLossMinIter"] != PENDING,
        "so its own loss does not say when to stop":
            float(v["solverTraceLossMinPsnr"]) < float(v["solverTraceFinalPsnr"]) - 1.0,
        "A four times larger budget closed most of that gap but not all of it": v["solverGainMin"] != PENDING,
        "Adam's own loss was still falling when the released budget stopped it": v["solverGainMin"] != PENDING,
        # Convergence of the sweeps (sec:results:quality, sec:discussion:scope; R2.3d).
        # convergence_macros refuses an uncertified CSFR patch-solve, a capped one outside
        # the primary TV-only arm or short of the cap, and a TV-only arm no slower than CSFR.
        "Every CSFR solve met the certificate": v["convNSolves"] != PENDING,
        "all of them in the TV-only arm on the primary corpus": v["convArmNCapped"] != PENDING,
        "They are kept and flagged as not converged": int(v["convArmNCapped"]) > 0,
        "It converges more slowly": int(v["convTvItersMax"]) > int(v["convGovItersMax"]),
        # Errors in the mask (sec:results:maskerr; R2.5b). maskerr_macros refuses a false
        # positive that does not carry 0, a dilation that leaves a truly observed pixel next
        # to the hole, a CF2/CF3 method that loses less from dilation than from erosion,
        # block-DCT thresholding not losing least under dilation, and an observed pixel
        # left by a two-pixel erosion of CF1 L5.
        "false negatives cost little and false positives cost much":
            float(v["maskErodeAnyMax"]) < float(v["maskDilateCsfrMin"]),
        "every method loses more from it than from the two-pixel erosion": v["maskDilateGulMin"] != PENDING,
        "every observed pixel next to the remaining hole is a false positive": v["maskDilateGulMin"] != PENDING,
        "Block-DCT thresholding loses least under every dilation of every CF2 and CF3 cell":
            v["maskDilateGulMin"] != PENDING,
        "and a two-pixel erosion leaves none in any": v["maskCFoneErodeNoneLfive"] != PENDING,
        "so the most accurate method loses most": float(v["maskCFoneDilateOverFloorMax"]) < 0.5,
    }
    for sentence, holds in claims.items():
        assert sentence in tex, f"sentence gone from the manuscript (update this test): {sentence}"
        assert holds, f"the manuscript says '{sentence}', which the generated values contradict"


def test_every_macro_the_manuscript_uses_is_generated():
    m = _mod()
    import re
    tex = (ROOT / "paper" / "paper2_reconstruction.tex").read_text(encoding="utf-8")
    generated = set(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", m.render()))
    used = set(re.findall(r"\\((?:csfr|rec|calib|oracle|corpus|methods|env|sweep|stats|fallback|bu|shiftmap|dictlearn|mech|abl|solver|conv|mask|q(?:Gov|Dfrws|Kodak))[A-Z][A-Za-z]*)", tex))
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
    assert (q["qGovCFoneLevelTrend"], q["qGovCFtwoLevelTrend"]) == ("falls", "mixed")   # -1.0 .. -3.0; ties
    assert (m.level_trend([-7.6, -5.1, -2.3]), m.level_trend([1.0, 1.0])) == ("rises", "mixed")
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
    dfrws = [_mech_row(f, lv, n_patches_capped={"CF1": 9, "CF3": 2, "CF4": 3}.get(f, 0))
             for f in ("CF1", "CF2", "CF3", "CF4") for lv in ("L1", "L2", "L3", "L4", "L5")]
    _write_rows(d / "sweep_v2_govdocs1.csv", gov)
    _write_rows(d / "sweep_v2_kodak.csv", kodak)
    _write_rows(d / "sweep_v2_dfrws.csv", dfrws)


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
    assert q["mechDfrwsContigCappedMax"] == "3"                          # CF1's 9 is not contiguous loss
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


ABL_LEVELS = ("L1", "L2", "L3", "L4", "L5")


def _abl_row(arm, fam, lv, diff, p=0.01, **kw):
    r = {"sweep": "s", "method": arm, "family": fam, "level": lv, "n_patches": 600, "n_seeds": 1,
         "diff_mean": diff, "diff_median": diff, "frac_patches_csfr_ahead": 0.9 if diff > 0 else 0.1,
         "p_holm": p, "n_flat": 0, "flat_part": 0.0, "rest_part": diff, "n_flat_arm_at_cap": 0}
    r.update(kw)
    return r


def _abl(tmp_path, overrides=None, kept_out=0.0):
    """An ablation_mechanism CSV and a range_violations CSV that say what the text says."""
    rows = {}
    for i, lv in enumerate(ABL_LEVELS):
        cf1_tv = (-1.0, -0.5, -0.2, 0.2, 0.7)[i]
        rows["csfr_tv_only", "CF1", lv] = _abl_row("csfr_tv_only", "CF1", lv, cf1_tv,
                                                  frac_patches_csfr_ahead=(0.2, 0.25, 0.3, 0.6, 0.9)[i])
        cf1_fc = (-2.4, -2.0, -1.3, -0.3, 0.35)[i]
        rows["csfr_no_fc", "CF1", lv] = _abl_row("csfr_no_fc", "CF1", lv, cf1_fc, p=0.2 if lv == "L4" else 0.01,
                                                frac_patches_csfr_ahead=(0.1, 0.1, 0.14, 0.46, 0.85)[i])
        for fam in ("CF2", "CF3", "CF4"):
            rows["csfr_tv_only", fam, lv] = _abl_row("csfr_tv_only", fam, lv, -1.0, diff_median=0.2,
                                                    frac_patches_csfr_ahead=0.7, n_flat=10, flat_part=-1.3,
                                                    rest_part=0.3, n_flat_arm_at_cap=8)
            rows["csfr_no_fc", fam, lv] = _abl_row("csfr_no_fc", fam, lv, -0.2, n_flat=10, flat_part=-0.25,
                                                  rest_part=0.05)
        for fam in ("CF1", "CF2", "CF3", "CF4"):
            rows["csfr_dct_only", fam, lv] = _abl_row("csfr_dct_only", fam, lv, 0.5)
            rows["csfr_no_clip", fam, lv] = _abl_row("csfr_no_clip", fam, lv, 0.01)
    rows["csfr_tv_only", "CF3", "L1"].update(diff_mean=-3.0, flat_part=-3.5, rest_part=0.5,
                                             frac_patches_csfr_ahead=0.6)
    rows["csfr_tv_only", "CF2", "L5"].update(diff_mean=0.2, flat_part=-0.1, rest_part=0.3,
                                             frac_patches_csfr_ahead=0.86)
    rows["csfr_no_fc", "CF3", "L1"].update(diff_mean=-1.67, flat_part=-1.16, rest_part=-0.51)
    rows["csfr_dct_only", "CF3", "L1"]["diff_mean"] = 7.0
    rows["csfr_no_clip", "CF3", "L2"]["diff_mean"] = -0.09
    for key, kw in (overrides or {}).items():
        rows[key].update(kw)
    mech = tmp_path / "ablation_mechanism.csv"
    _write_rows(mech, list(rows.values()))
    ranges = tmp_path / "range_violations.csv"
    _write_rows(ranges, [
        {"family": "CF1", "level": "L1", "seed": 0, "method": "csfr_no_clip", "n_patches": 600, "n_imputed": 1000,
         "frac_pixels_out": 0.02, "frac_patches_out": 0.3, "max_excess": 50.0},
        {"family": "CF1", "level": "L2", "seed": 0, "method": "csfr_no_clip", "n_patches": 600, "n_imputed": 1000,
         "frac_pixels_out": 0.04, "frac_patches_out": 0.3, "max_excess": 158.4},
        {"family": "CF1", "level": "L1", "seed": 0, "method": "csfr", "n_patches": 600, "n_imputed": 1000,
         "frac_pixels_out": kept_out, "frac_patches_out": 0.0, "max_excess": 0.0}])
    return mech, ranges


def test_ablation_macros_say_what_each_term_contributes(tmp_path):
    m = _mod()
    assert set(m.ablation_macros(tmp_path / "none.csv", tmp_path / "none.csv").values()) == {m.PLACEHOLDER}
    q = m.ablation_macros(*_abl(tmp_path))
    assert (q["ablNoTvMin"], q["ablNoTvMax"]) == ("0.50", "7.00")
    assert (q["ablTvNBehind"], q["ablTvBehindMax"], q["ablTvNAhead"], q["ablTvAheadMax"]) == ("17", "3.00", "3", "0.70")
    assert (q["ablTvBcFlatMin"], q["ablTvBcFlatMax"]) == ("0.10", "3.50")
    assert (q["ablTvBcRestMin"], q["ablTvBcRestMax"]) == ("0.30", "0.50")
    assert (q["ablTvBcAheadMin"], q["ablTvBcAheadMax"], q["ablTvBcFlatCapShare"]) == ("60", "86", "80")
    assert (q["ablTvScatterBehindLevels"], q["ablTvScatterAheadLevels"]) == ("L1 to L3", "L4 and L5")
    assert q["ablTvScatterAheadShareMax"] == "30"
    assert (q["ablNoFcScatterBehindLevels"], q["ablNoFcScatterBehindMin"], q["ablNoFcScatterBehindMax"]) == (
        "L1 to L3", "1.30", "2.40")
    assert (q["ablNoFcScatterAheadShareMax"], q["ablNoFcScatterAheadLevels"], q["ablNoFcScatterAheadMax"]) == (
        "14", "L5", "0.35")
    assert (q["ablNoFcScatterNsLevels"], q["ablNoFcBcRestMaxAbs"]) == ("L4", "0.51")
    assert (q["ablClipMaxAbs"], q["ablClipShare"], q["ablClipMaxExcess"]) == ("0.09", "3.00", "158.4")
    assert (q["ablTvScatterTrend"], q["ablNoFcScatterTrend"]) == ("rises", "rises")
    q = m.ablation_macros(*_abl(tmp_path, overrides={("csfr_no_fc", "CF1", "L2"): {"diff_mean": -2.5}}))
    assert q["ablNoFcScatterTrend"] == "mixed"


@pytest.mark.parametrize("key,bad", [
    (("csfr_dct_only", "CF2", "L3"), {"diff_mean": -0.1}),          # "lowers hole PSNR in every cell"
    (("csfr_dct_only", "CF2", "L3"), {"p_holm": 0.2}),              # "with p < 0.05 in every cell"
    (("csfr_tv_only", "CF4", "L2"), {"rest_part": -0.1}),           # "on the other holes CSFR is ahead in every cell"
    (("csfr_tv_only", "CF4", "L2"), {"diff_median": -0.1}),         # "the median favours CSFR in every cell"
    (("csfr_tv_only", "CF4", "L2"), {"flat_part": 0.1}),            # "the flat holes favour the arm"
    (("csfr_tv_only", "CF1", "L2"), {"n_flat": 1}),                 # "scattered loss leaves no flat hole"
    (("csfr_tv_only", "CF1", "L2"), {"diff_mean": 0.1}),            # arm ahead at the lowest levels, CSFR above
    (("csfr_tv_only", "CF1", "L1"), {"frac_patches_csfr_ahead": 0.6}),  # "on most patches"
    (("csfr_no_fc", "CF2", "L4"), {"flat_part": 0.1}),              # flat holes favour the arm without the penalty
])
def test_ablation_macros_refuse_data_that_contradict_the_text(tmp_path, key, bad):
    m = _mod()
    with pytest.raises(ValueError):
        m.ablation_macros(*_abl(tmp_path, overrides={key: bad}))


def test_ablation_macros_refuse_a_range_violation_with_c3(tmp_path):
    m = _mod()
    with pytest.raises(ValueError, match="C3"):
        m.ablation_macros(*_abl(tmp_path, kept_out=0.001))


FID_QUERIES = [  # cell, distance, e(0.25), e(1), e(4)
    ("CF1_L1", 1, 0.05, 0.20, 0.80), ("CF1_L1", 3, 0.0025, 0.01, 0.04),
    ("CF3_L1", 8, 0.002, 0.008, 0.032), ("CF3_L1", 1, 0.01, 0.04, 0.20)]


def _fid(tmp_path, sign=1.0, far_e1=None, op_shift=0.0):
    rows = []
    for q, (cell, dist, *errs) in enumerate(FID_QUERIES):
        for delta, e in zip((0.25, 1.0, 4.0), errs):
            if far_e1 is not None and dist > 1 and delta == 1.0:
                e = far_e1
            rows.append({"cell": cell, "seed": 0, "patch": q, "query_pixel": "0,0", "query_distance": dist,
                         "delta": delta, "tol": 1e-06, "rel_l2_error_ref": e,
                         "sign_agreement_topk": sign if delta == 1.0 else 1.0})
    _write_rows(tmp_path / "fidelity.csv", rows)
    summary = [{"family": f, "level": lv, "delta": 1.0, "tol": 1e-06,
                "srf_ref": 0.5 if (f, lv) == ("CF1", "L1") else 1.0,
                "srf_op": (0.5 if (f, lv) == ("CF1", "L1") else 1.0) + (op_shift if f == "CF2" else 0.0),
                "rel_err_median_ref": 0.05 if (f, lv) == ("CF1", "L1") else 0.02}
               for f in ("CF1", "CF2", "CF3", "CF4") for lv in ABL_LEVELS]
    _write_rows(tmp_path / "summary.csv", summary)
    return tmp_path / "fidelity.csv", tmp_path / "summary.csv"


def test_record_fidelity_macros_say_where_the_record_is_least_accurate(tmp_path):
    m = _mod()
    assert set(m.record_fidelity_macros(tmp_path / "x.csv", tmp_path / "y.csv").values()) == {m.PLACEHOLDER}
    q = m.record_fidelity_macros(*_fid(tmp_path))
    assert (q["recNQueries"], q["recSrfPooled"], q["recSrfSmallDelta"], q["recSrfLargeDelta"]) == (
        "4", "75.0", "100.0", "50.0")
    assert (q["recSrfCellMin"], q["recSrfCellMax"], q["recErrMedMin"], q["recErrMedMax"]) == (
        "50.0", "100.0", "2", "5")
    assert (q["recNearN"], q["recNearFail"], q["recNearSrf"], q["recNearErrMax"]) == ("2", "1", "50.0", "20")
    assert (q["recFarN"], q["recFarErrMax"]) == ("2", "1")
    assert (q["recRatioLow"], q["recRatioHigh"]) == ("4.00", "4.00")      # e(4)/e(1): 4, 4, 4, 5
    assert (q["recRatioLowPTen"], q["recRatioLowPNinety"]) == ("4.00", "4.00")
    assert (q["recRatioHighPTen"], q["recRatioHighPNinety"]) == ("4.00", "4.70")


@pytest.mark.parametrize("kw,match", [({"sign": 0.9}, "sign"), ({"far_e1": 0.2}, "farther"),
                                      ({"op_shift": 0.1}, "operational")])
def test_record_fidelity_macros_refuse_data_that_contradict_the_text(tmp_path, kw, match):
    m = _mod()
    with pytest.raises(ValueError, match=match):
        m.record_fidelity_macros(*_fid(tmp_path, **kw))


def _records(tmp_path, csfr_support=(100, 99, 50, 50), lin_contig=0.0):
    """Two queries in CF1_L1 and two in CF3_L1; hole of 100 free pixels in CF1, 50 in CF3."""
    queries = [("CF1_L1", "0,0", 100), ("CF1_L1", "0,1", 100), ("CF3_L1", "0,0", 50), ("CF3_L1", "0,1", 50)]
    fd = {  # (method, query index) -> (l1 at delta 1, l1 at delta 4, ||FD1 - FD4|| / ||FD4||)
        ("inpaint_telea", 0): (0.0, 2.0, 1.0), ("inpaint_telea", 1): (1.0, 2.0, 3.0),
        ("inpaint_telea", 2): (0.0, 0.0, float("nan")), ("inpaint_telea", 3): (0.0, 1.0, 1.0),
        ("inpaint_ns", 0): (0.0, 0.0, float("nan")), ("inpaint_ns", 1): (2.0, 3.0, 1.5),
        ("inpaint_ns", 2): (0.0, 1.0, 1.0), ("inpaint_ns", 3): (0.0, 0.0, float("nan"))}
    rows, fid = [], []
    for i, (cell, px, n_free) in enumerate(queries):
        base = {"cell": cell, "patch": 0, "query_pixel": px, "query_distance": 1}
        rows.append({**base, "method": "csfr", "record": "implicit_function", "l1_norm": 2.0,
                     "support": csfr_support[i], "mean_distance": 30.0 + i, "fd_rel_diff_delta1_vs_4": "",
                     "fd_vs_exact_rel_l2": ""})
        lin = 1.0 if i == 0 else (lin_contig if cell.startswith("CF3") else 0.0)
        rows.append({**base, "method": "bilinear", "record": "exact_barycentric", "l1_norm": lin,
                     "support": 2 if lin else 0, "mean_distance": 1.0, "fd_rel_diff_delta1_vs_4": "",
                     "fd_vs_exact_rel_l2": ""})
        for dl in (1, 4):
            rows.append({**base, "method": "bilinear", "record": f"finite_difference_delta{dl}", "l1_norm": lin,
                         "support": 2 if lin else 0, "mean_distance": 1.0,
                         "fd_rel_diff_delta1_vs_4": 0.0 if lin else "nan",
                         "fd_vs_exact_rel_l2": (2e-5 if i == 0 else 0.0) if dl == 4 else ""})
        for m in ("inpaint_telea", "inpaint_ns"):
            one, four, diff = fd[m, i]
            for dl, l1 in ((1, one), (4, four)):
                rows.append({**base, "method": m, "record": f"finite_difference_delta{dl}", "l1_norm": l1,
                             "support": 9 if m == "inpaint_ns" else 4, "mean_distance": 6.0 if l1 else "",
                             "fd_rel_diff_delta1_vs_4": diff, "fd_vs_exact_rel_l2": ""})
        fid.append({"cell": cell, "seed": 0, "patch": 0, "query_pixel": px, "delta": 1.0, "tol": 1e-06,
                    "n_free": n_free})
    _write_rows(tmp_path / "records_compare.csv", rows)
    _write_rows(tmp_path / "fidelity.csv", fid)
    return tmp_path / "records_compare.csv", tmp_path / "fidelity.csv"


def test_record_comparison_macros_set_the_global_record_against_local_ones(tmp_path):
    m = _mod()
    assert set(m.record_comparison_macros(tmp_path / "x.csv", tmp_path / "y.csv").values()) == {m.PLACEHOLDER}
    q = m.record_comparison_macros(*_records(tmp_path))
    assert q["recCsfrSupportMinShare"] == "99.0"
    assert (q["recCsfrReachMin"], q["recCsfrReachMax"]) == ("30.5", "32.5")      # group medians
    assert (q["recCmpSupportMax"], q["recCmpReachMax"]) == ("9", "6.0")
    assert (q["recLinZeroScatter"], q["recLinFdMaxDiff"]) == ("50", r"2.0\times 10^{-5}")
    assert (q["recPdeZeroOne"], q["recPdeZeroFour"]) == ("75", "38")          # 6 and 3 of 8
    assert q["recPdeFdDiffMed"] == "1.00"                                     # 1, 3, 1, 1.5, 1
    assert (q["recPdeFdDiffBothMin"], q["recPdeFdDiffBothMax"]) == ("1.50", "3.00")


@pytest.mark.parametrize("kw,match", [({"csfr_support": (100, 0, 50, 50)}, "zero"),
                                      ({"lin_contig": 1.0}, "CF3")])
def test_record_comparison_macros_refuse_data_that_contradict_the_text(tmp_path, kw, match):
    m = _mod()
    with pytest.raises(ValueError, match=match):
        m.record_comparison_macros(*_records(tmp_path, **kw))


def test_comparison_macros_need_lama_at_every_level(tmp_path):
    m = _mod()
    _paired(tmp_path, _comparison_cells(), missing_lama=("CF3", "L2"))
    with pytest.raises(ValueError, match="LaMa is not scored"):
        m.comparison_macros("Gov", tmp_path)


# ---------------------------------------------------------------- oracle (Section VI-F)

ORACLE_COND = {  # family -> median condition number per level (both corpora)
    "CF1": (1.1, 1.2, 1.4, 1.7, 2.3), "CF2": (2.8, 4.1, 8.4, 36.9, 814.0),
    "CF3": (2.5, 31.0, 1.2e4, 4.9e10, 1e16), "CF4": (2.5, 31.0, 1.2e4, 4.9e10, 1e16)}
ORACLE_PSNR = {"CF1": (25.6, 25.5, 25.4, 25.2, 24.9), "CF2": (24.4, 23.6, 21.8, 16.9, -0.3),
               "CF3": (23.0, 16.2, -16.7, -109.4, -166.4), "CF4": (22.7, 15.9, -15.8, -108.2, -165.3)}


def _oracle(tmp_path, overrides=None):
    rows = []
    for corpus, scale, err in (("govdocs1_v2", 1.0, 0.093), ("dfrws2006", 2.0, 0.103)):
        for fam in ("CF1", "CF2", "CF3", "CF4"):
            for i, lv in enumerate(ABL_LEVELS):
                cond = ORACLE_COND[fam][i] * (scale if fam == "CF2" else 1.0)
                r = {"corpus": corpus, "family": fam, "level": lv, "frac": 0.05, "n_patches": 10,
                     "seeds": "0 1 2", "kappa": 205, "oracle_psnr_hole_median": ORACLE_PSNR[fam][i],
                     "cond_median": cond, "cond_p90": 2 * cond, "frac_cond_above_1e3": 0.5 if cond > 1e3 else 0.0,
                     "frac_rank_deficient": 0.86 if (fam in ("CF3", "CF4") and lv == "L5") else 0.0,
                     "smin_median": 1 / cond, "sparse_approx_rel_err_median": err}
                r.update((overrides or {}).get((corpus, fam, lv), {}))
                rows.append(r)
    _write_rows(tmp_path / "oracle.csv", rows)
    return tmp_path / "oracle.csv"


def test_oracle_macros_say_where_the_observations_determine_the_model(tmp_path):
    m = _mod()
    assert set(m.oracle_macros(tmp_path / "none.csv").values()) == {m.PLACEHOLDER}
    q = m.oracle_macros(_oracle(tmp_path))
    assert (q["oracleKappa"], q["oracleSparseErrMin"], q["oracleSparseErrMax"]) == ("205", "9.3", "10.3")
    assert (q["oracleScatterCondMax"], q["oracleScatterPsnrMin"], q["oracleScatterPsnrMax"]) == (
        "2.30", "24.9", "25.6")
    assert (q["oracleBurstCondFirstMin"], q["oracleBurstCondFirstMax"]) == ("2.8", "5.6")
    assert (q["oracleBurstCondLastMin"], q["oracleBurstCondLastMax"]) == ("814", "1628")
    assert (q["oracleBurstAboveLastMin"], q["oracleBurstAboveLastMax"]) == ("0", "50")
    assert (q["oracleContigIllLevel"], q["oracleContigRankMin"]) == ("L3", "86")


@pytest.mark.parametrize("key,bad", [
    (("dfrws2006", "CF1", "L4"), {"frac_cond_above_1e3": 0.01}),          # "no CF1 patch above 10^3"
    (("govdocs1_v2", "CF2", "L3"), {"cond_median": 50.0}),               # "rises at every level"
    (("govdocs1_v2", "CF3", "L4"), {"cond_median": 900.0}),              # ill from L3 on, at every level above
    (("dfrws2006", "CF4", "L3"), {"oracle_psnr_hole_median": 1.0}),      # the fit fails from the same level
])
def test_oracle_macros_refuse_data_that_contradict_the_text(tmp_path, key, bad):
    with pytest.raises(ValueError):
        _mod().oracle_macros(_oracle(tmp_path, overrides={key: bad}))


# ------------------------------------------------------- Bounded-Unknown (Section VI-K)

def _bu(tmp_path, keep=False, cell_auc=0.55, contig_auc=0.5, logit_gain=0.001):
    import json
    rows = []
    for t, n_fail in ((20.0, 480), (25.0, 680)):
        for p, a in (("isolated_missing_frac", 0.737), ("mean_hole_distance", 0.741), ("hole_frac", 0.62)):
            rows.append({"threshold_db": t, "scope": "pooled", "group": "all", "predictor": p, "n": 1000,
                         "n_fail": n_fail, "auroc": a - (0.02 if t == 25.0 else 0.0)})
            for cell in ("sweep_v2_govdocs1:CF1_L1", "sweep_v2_govdocs1:CF3_L2"):
                rows.append({"threshold_db": t, "scope": "cell", "group": cell, "predictor": p, "n": 100,
                             "n_fail": 50, "auroc": contig_auc if "CF3" in cell else cell_auc})
    _write_rows(tmp_path / "auroc.csv", rows)
    dec = {f"{t:g}dB": {"best_predictor": "mean_hole_distance", "pooled_auroc": 0.74,
                        "cells_with_outcome_variance": 2, "fraction_cells_within_auroc_above_min": 0.0,
                        "keep_as_contribution": keep, "logistic_cv_auroc": 0.74,
                        "logistic_cv_ece_10bins": 0.012, "cv_folds": 10} for t in (20.0, 25.0)}
    dec["25dB"]["logistic_cv_auroc"] = 0.72 + logit_gain
    (tmp_path / "decision.json").write_text(json.dumps({"by_threshold": dec}), encoding="utf-8")
    return tmp_path


def test_bu_result_macros_report_the_rule_as_applied(tmp_path):
    m = _mod()
    assert set(m.bu_result_macros(tmp_path / "none").values()) == {m.PLACEHOLDER}
    q = m.bu_result_macros(_bu(tmp_path))
    assert (q["buNCells"], q["buFailLow"], q["buFailHigh"]) == ("2", "48", "68")
    assert (q["buBuPooledLow"], q["buBuPooledHigh"], q["buDistPooledLow"], q["buDistPooledHigh"]) == (
        "0.74", "0.72", "0.74", "0.72")
    assert (q["buFracPooledLow"], q["buFracPooledHigh"], q["buWithinMax"]) == ("0.62", "0.60", "0.55")
    assert (q["buEceMax"], q["buLogitAurocLow"], q["buLogitAurocHigh"]) == ("0.012", "0.74", "0.72")


@pytest.mark.parametrize("kw", [{"keep": True},           # "the rule is met at neither threshold"
                                {"cell_auc": 0.7},        # "in no cell"
                                {"contig_auc": 0.52},     # "exactly one half in every contiguous-loss cell"
                                {"logit_gain": 0.03}])    # "adds no discrimination"
def test_bu_result_macros_refuse_data_that_contradict_the_text(tmp_path, kw):
    with pytest.raises(ValueError):
        _mod().bu_result_macros(_bu(tmp_path, **kw))


# ------------------------------------------------------- solver comparison (Section VI-B)

SOLVER_PSNR = {"adam600_released": (30.00, 19.70, 14.50, 14.60), "adam600_cpu": (30.00, 19.70, 14.50, 14.60),
               "adam2400_cpu": (30.00, 22.70, 18.50, 18.70), "fista_gap1e-04": (30.05, 22.80, 20.20, 20.60),
               "fista_gap1e-06": (30.05, 22.80, 20.20, 20.60), "fista_gap1e-10": (30.05, 22.80, 20.20, 20.60)}
SOLVER_ITERS = {"fista_gap1e-06": (260, 820, 990, 1000), "adam2400_cpu": (2400,) * 4}
SOLVER_SECS = {"fista_gap1e-06": (7.1, 20.3, 24.4, 27.7), "adam2400_cpu": (29.5, 30.4, 30.6, 33.5)}


def _solver(tmp_path, psnr=None, converged=300, rerun_diff=0.0, adam_psnr=(10.0, 12.0, 14.0, 15.0),
            adam_loss=(2.0, 1.5, 1.0, 2.2)):
    fams = ("CF1", "CF2", "CF3", "CF4")
    table = {**SOLVER_PSNR, **(psnr or {})}
    summ, cells = [], []
    for arm, vals in table.items():
        for i, f in enumerate(fams):
            summ.append({"family": f, "arm": arm, "psnr_hole": vals[i],
                         "seconds_per_cell": SOLVER_SECS.get(arm, (8.0,) * 4)[i],
                         "iterations_median": SOLVER_ITERS.get(arm, (600,) * 4)[i],
                         "device": "cuda" if arm == "adam600_released" else "cpu"})
            for seed in (0, 1):
                if arm == "adam600_cpu" and seed == 1:
                    continue
                tight = 20.0 + i + seed
                cells.append({"arm": arm, "cell": f"{f}_L1", "seed": seed, "n": 300,
                              "psnr_hole": tight + (0.0012 if arm == "fista_gap1e-04" and seed == 1 else 0.0),
                              "n_converged": converged if arm.startswith("fista") else "",
                              "max_abs_diff_vs_released": rerun_diff if arm == "adam600_cpu" else ""})
    _write_rows(tmp_path / "quality_vs_compute.csv", summ)
    _write_rows(tmp_path / "per_cell.csv", cells)
    iters = (200, 600, 800, 1600)
    trace = [{"arm": "adam", "iter": k, "psnr_hole": p, "objective": loss, "gap_max": ""}
             for k, p, loss in zip(iters, adam_psnr, adam_loss)]
    trace += [{"arm": "fista", "iter": k, "psnr_hole": p, "objective": 1.0, "gap_max": g}
              for k, p, g in zip(iters, (15.0, 20.005, 20.0, 20.0), (9.13, 0.5, 0.01, 2e-12))]
    _write_rows(tmp_path / "trace_CF2_L5.csv", trace)
    return tmp_path


def test_solver_macros_say_what_the_fixed_budget_cost(tmp_path):
    m = _mod()
    assert set(m.solver_macros(tmp_path).values()) == {m.PLACEHOLDER}
    q = m.solver_macros(_solver(tmp_path))
    assert (q["solverNPatches"], q["solverRetimeSeed"], q["solverScatterDiffMax"]) == ("300", "0", "0.05")
    assert (q["solverAdamLongGainMin"], q["solverAdamLongGainMax"]) == ("3.00", "4.10")
    assert (q["solverFistaOverLongMin"], q["solverFistaOverLongMax"]) == ("0.10", "1.90")
    assert (q["solverGainMin"], q["solverGainMax"]) == ("3.10", "6.00")
    assert (q["solverTolLoose"], q["solverTolTight"], q["solverTolDiffMax"]) == ("10^{-4}", "10^{-10}", "0.001")
    assert (q["solverItersMin"], q["solverItersMax"], q["solverSecMin"], q["solverSecMax"]) == (
        "260", "1000", "7", "28")
    assert (q["solverAdamSecMin"], q["solverAdamSecMax"]) == ("30", "34")
    assert (q["solverTraceLossMinIter"], q["solverTraceLossMinPsnr"], q["solverTraceLossRatio"]) == (
        "800", "14.0", "2.2")
    assert (q["solverTraceLastIter"], q["solverTraceFinalPsnr"]) == ("1600", "20.0")
    assert (q["solverSettleTol"], q["solverTraceSettledIter"]) == ("0.01", "600")
    assert (q["solverTraceGapFirst"], q["solverTraceGapLast"]) == ("9.1", r"2.0\times 10^{-12}")


@pytest.mark.parametrize("kw", [
    {"rerun_diff": 1e-6},                                                 # "reproduced ... bit for bit"
    {"converged": 299},                                                   # "certified every patch"
    {"psnr": {"adam2400_cpu": (30.0, 19.6, 18.5, 18.7)}},                 # the longer run gains under CF2-CF4
    {"psnr": {"adam2400_cpu": (30.0, 22.9, 18.5, 18.7)}},                 # the convergent solver is ahead
    {"psnr": {"adam2400_cpu": (30.0, 22.7, 16.0, 18.7)}},                 # "closed most of that gap"
    {"adam_psnr": (10.0, 12.0, 11.0, 15.0)},                              # Adam's PSNR "rose at every point"
    {"adam_loss": (2.0, 1.0, 1.5, 2.2)},                                  # "still falling when the budget stopped it"
])
def test_solver_macros_refuse_data_that_contradict_the_text(tmp_path, kw):
    with pytest.raises(ValueError):
        _mod().solver_macros(_solver(tmp_path, **kw))


# ------------------------------------------------------- convergence of the sweeps (Section VI-A)

def _conv(tmp_path, csfr_short=False, capped_arm="csfr_tv_only", capped_sweep="sweep_v2_govdocs1",
          cap_iter=20000, tv_iters=4000.0, n_capped=2):
    import json
    m = _mod()
    for sweep in m.SWEEPS:
        arms = ["csfr", "csfr_tv_only"] + (["csfr_no_fc"] if sweep == m.PRIMARY_SWEEP else [])
        rows = []
        for arm in arms:
            for (fam, lv), it in ((("CF1", "L3"), 300.0), (("CF2", "L5"), 1200.0)):
                short = 0
                if arm == "csfr" and csfr_short and sweep == m.PRIMARY_SWEEP and fam == "CF1":
                    short = 1
                if (arm, sweep, fam) == (capped_arm, capped_sweep, "CF1"):
                    short = n_capped
                rows.append({"corpus": sweep, "method": arm, "kind": "csfr", "family": fam, "level": lv,
                             "n_patches": 600, "n_converged": 600 - short,
                             "iters_median": tv_iters if arm == "csfr_tv_only" else it})
            if (arm, sweep) == (capped_arm, capped_sweep) and n_capped:
                d = tmp_path / sweep / "CF1_L3" / "seed0"
                d.mkdir(parents=True)
                gaps = [4.8e-6, 2.0e-6][:n_capped] + [5e-7] * (600 - n_capped)
                (d / f"certificate_{arm}.json").write_text(json.dumps(
                    {"iterations": [cap_iter] * n_capped + [400] * (600 - n_capped),
                     "converged": [False] * n_capped + [True] * (600 - n_capped), "gap": gaps}), encoding="utf-8")
        rows.append({"corpus": sweep, "method": "guleryuz", "kind": "classical", "family": "CF1", "level": "L3",
                     "n_patches": 600, "n_converged": "", "iters_median": ""})
        _write_rows(tmp_path / sweep / "summary.csv", rows)
    return tmp_path


def test_convergence_macros_count_every_patch_solve(tmp_path):
    m = _mod()
    assert set(m.convergence_macros(tmp_path).values()) == {m.PLACEHOLDER}
    q = m.convergence_macros(_conv(tmp_path))
    assert (q["convNSolves"], q["convItersMin"], q["convItersMax"], q["convGovItersMax"]) == (
        r"3\,600", "300", "1200", "1200")
    assert (q["convArmNSolves"], q["convArmNCapped"], q["convTvNSolves"], q["convTvItersMax"]) == (
        r"4\,800", "2", r"1\,200", "4000")
    assert (q["convArmCappedGapMax"], q["convArmCappedGapRatio"]) == (r"4.8\times 10^{-6}", "4.8")


@pytest.mark.parametrize("kw", [
    {"csfr_short": True},                          # "Every CSFR solve met the certificate"
    {"capped_arm": "csfr_no_fc"},                  # "all of them in the TV-only arm"
    {"capped_sweep": "sweep_v2_dfrws"},            # "on the primary corpus"
    {"cap_iter": 19990},                           # "reached the cap"
    {"tv_iters": 1000.0},                          # "It converges more slowly"
    {"n_capped": 0},                               # the text reports capped patch-solves
])
def test_convergence_macros_refuse_data_that_contradict_the_text(tmp_path, kw):
    with pytest.raises(ValueError):
        _mod().convergence_macros(_conv(tmp_path, **kw))


def _maskerr(tmp_path, fp_value=0.0, border=1.0, gul_least=True, dilate_cheaper=False, cf1_left=0):
    """A complete mask-error result in which every sentence of Section VI-H holds:
    CF2/CF3 cells lose 0.5 or 1 dB from erosion and fall to just above the floor
    under dilation, block-DCT thresholding least; CF1 dilation returns the floor."""
    m = _mod()
    t = m._tables()
    methods = ("csfr", "bilinear", "guleryuz", "lama")
    rob, geo = [], []
    for cell in t.MASKERR_CELLS:
        fam, lv = cell.split("_")
        floor = 5.0
        exact = {"csfr": 20.0, "bilinear": floor if fam == "CF3" else 19.0, "guleryuz": 21.0, "lama": 19.5}
        for p in ("exact", "erode1", "erode2", "dilate1", "dilate2", "flip1pct", "flip5pct"):
            for k in methods:
                if fam == "CF1":
                    psnr = {"exact": exact[k], "erode1": exact[k] - 8, "erode2": exact[k] - 12,
                            "flip1pct": exact[k] - {"csfr": 4.0 if lv == "L5" else 4.9, "bilinear": 3.0,
                                                    "guleryuz": 9.6, "lama": 1.4}[k]}.get(p, floor + 0.05)
                else:
                    over = {"csfr": 0.1, "bilinear": 0.0, "guleryuz": 6.0 if gul_least else 0.0, "lama": 1.0}[k]
                    erode2 = exact[k] - (14.0 if dilate_cheaper and k == "lama" else 1.0)
                    psnr = {"exact": exact[k], "erode1": exact[k] - 0.5, "erode2": erode2,
                            "flip1pct": exact[k] - 8.0, "flip5pct": exact[k] - 12.0}.get(p, floor + over)
                    if fam == "CF3" and k == "bilinear":
                        psnr = floor
                rob.append({"cell": cell, "perturbation": p, "method": k, "psnr_hole_mean": psnr,
                            "degradation_mean": psnr - exact[k]})
            fp = {"dilate1": 0.95 if fam == "CF1" else (0.05 if lv == "L3" else 0.10), "dilate2": 0.99,
                  "flip1pct": 0.01, "flip5pct": 0.05}.get(p, 0.0)
            fn = {"erode1": 0.937 if fam == "CF1" else 0.1, "erode2": 0.999 if fam == "CF1" else 0.2,
                  "flip1pct": 0.01, "flip5pct": 0.05}.get(p, 0.0)
            none = {("CF1_L5", "erode1"): 463, ("CF1_L5", "erode2"): 600 - cf1_left}.get((cell, p), 0)
            geo.append({"cell": cell, "perturbation": p, "n": 600, "hole_share": 0.3, "fp_share_of_hole": fp,
                        "fn_share_of_observed": fn, "n_none_observed": none,
                        "n_hole_all_observed": 478 if (cell, p) == ("CF1_L3", "dilate1") else 0,
                        "border_fp_share": border if p.startswith("dilate") else 0.0,
                        "fp_value_max": fp_value if fp else "nan", "zero_fill_psnr_hole": floor})
    _write_rows(tmp_path / "robustness.csv", rob)
    _write_rows(tmp_path / "geometry.csv", geo)
    return tmp_path


def test_maskerr_macros_say_what_a_wrong_mask_costs(tmp_path):
    m = _mod()
    assert set(m.maskerr_macros(tmp_path).values()) == {m.PLACEHOLDER}
    q = m.maskerr_macros(_maskerr(tmp_path))
    assert (q["maskErodeFnMax"], q["maskErodeCsfrMax"], q["maskErodeAnyMax"]) == ("20", "1.0", "1.0")
    assert (q["maskDilateFpMin"], q["maskDilateFpMax"]) == ("5.0", "10")
    assert (q["maskDilateCsfrMin"], q["maskDilateCsfrMax"], q["maskDilateCsfrOverFloorMax"]) == ("14.9", "14.9", "0.10")
    assert (q["maskDilateGulMin"], q["maskDilateGulMax"]) == ("10.0", "10.0")
    assert (q["maskFlipCsfrMin"], q["maskFlipCsfrMax"]) == ("8.0", "8.0")
    assert (q["maskCFoneDilateAllObs"], q["maskCFoneDilateFpLfive"], q["maskCFoneDilateOverFloorMax"]) == (
        "478", "95.0", "0.05")
    assert (q["maskCFoneErodeFnLthree"], q["maskCFoneErodeNoneLfive"]) == ("93.7", "463")
    assert (q["maskCFoneFlipCsfrMin"], q["maskCFoneFlipCsfrMax"]) == ("4.0", "4.9")
    assert (q["maskCFoneFlipOtherMin"], q["maskCFoneFlipOtherMax"]) == ("1.4", "9.6")


@pytest.mark.parametrize("kw", [
    {"fp_value": 128.0},         # "a false positive, which carries the value 0"
    {"border": 0.98},            # "every observed pixel next to the remaining hole is a false positive"
    {"gul_least": False},        # "Block-DCT thresholding loses least under every dilation"
    {"dilate_cheaper": True},    # "every method loses more from it than from the two-pixel erosion"
    {"cf1_left": 1},             # "and a two-pixel erosion leaves none in any"
])
def test_maskerr_macros_refuse_data_that_contradict_the_text(tmp_path, kw):
    with pytest.raises(ValueError):
        _mod().maskerr_macros(_maskerr(tmp_path, **kw))
