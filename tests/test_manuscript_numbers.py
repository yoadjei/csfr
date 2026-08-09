"""Every hand-written number in the manuscript must match its source CSV.

Tables are AUTO-generated and cannot drift. Prose numbers are typed by hand and
have drifted before: the external prior's largest margin was reported as 8.4 dB
when the sweep says 9.08, the CF1 L1 standard deviations were wrong by 0.5, and
the downstream ceiling quoted the full test set while accuracy recovery is
computed against the evaluation subset. These tests recompute each claim from
the released artefacts.

A test fails if the claim is absent from the manuscript as well as if it is
wrong. That is deliberate: rewording a sentence should force the number in it
to be re-verified rather than silently dropping it from the check.
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper2_reconstruction.tex"
RESULTS = ROOT / "results"

CLASSICAL = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn"]
ALL_METHODS = CLASSICAL + ["lama", "csfr"]


@pytest.fixture(scope="module")
def tex() -> str:
    return TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sweep() -> dict:
    d: dict = collections.defaultdict(dict)
    with (RESULTS / "csfr_sweep_2d_v2" / "summary.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["family"], r["level"])][r["method"]] = r
    return d


def psnr(sweep, fam, lvl, method) -> float:
    return float(sweep[(fam, lvl)][method]["psnr_db_mean"])


def claim(tex: str, pattern: str) -> re.Match:
    """Locate a numeric claim. Absence is a failure, not a skip."""
    m = re.search(pattern, tex)
    assert m is not None, (
        f"claim not found in manuscript: {pattern!r}. If the sentence was "
        f"reworded, update this test and re-verify the number against the CSV."
    )
    return m


def test_cf1_lead_range(tex, sweep):
    """CSFR's lead on random pixel loss, against the best of every other method."""
    leads = []
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        best_other = max(psnr(sweep, "CF1", lvl, m)
                         for m in ALL_METHODS if m != "csfr")
        leads.append(psnr(sweep, "CF1", lvl, "csfr") - best_other)
    m = claim(tex, r"\$1\.0\$ to \$1\.4\$\\,dB")
    assert m
    assert 0.95 <= min(leads) <= 1.05, f"min lead is {min(leads):.2f}"
    assert 1.35 <= max(leads) <= 1.45, f"max lead is {max(leads):.2f}"


def test_lama_wins_thirteen_of_twenty(tex, sweep):
    wins = sum(1 for k in sweep
               if max(ALL_METHODS, key=lambda m: psnr(sweep, k[0], k[1], m)) == "lama")
    claim(tex, r"13 of 20 cells")
    assert wins == 13, f"external prior wins {wins} cells, manuscript says 13"


def test_lama_largest_margin_over_csfr(tex, sweep):
    margin = max(psnr(sweep, f, l, "lama") - psnr(sweep, f, l, "csfr") for f, l in sweep)
    claim(tex, r"by up to \$9\.1\$\\,dB")
    assert 9.0 <= margin <= 9.15, f"largest margin is {margin:.2f} dB"


def test_cf2_margin_endpoints(tex, sweep):
    lo = psnr(sweep, "CF2", "L1", "lama") - psnr(sweep, "CF2", "L1", "csfr")
    hi = psnr(sweep, "CF2", "L5", "lama") - psnr(sweep, "CF2", "L5", "csfr")
    claim(tex, r"\$1\.8\$\\,dB at L1 widening to \$7\.0\$\\,dB at L5")
    assert abs(lo - 1.81) < 0.05, f"CF2 L1 margin is {lo:.2f}"
    assert abs(hi - 7.01) < 0.05, f"CF2 L5 margin is {hi:.2f}"


def test_cf1_l1_means_and_sds(tex, sweep):
    cell = sweep[("CF1", "L1")]
    claim(tex, r"CSFR \$44\.68 \\pm 9\.05\$\\,dB against Navier--Stokes \$43\.49 \\pm 8\.33\$\\,dB")
    assert abs(float(cell["csfr"]["psnr_db_mean"]) - 44.68) < 0.01
    assert abs(float(cell["csfr"]["psnr_db_std"]) - 9.05) < 0.01
    assert abs(float(cell["inpaint_ns"]["psnr_db_mean"]) - 43.49) < 0.01
    assert abs(float(cell["inpaint_ns"]["psnr_db_std"]) - 8.33) < 0.01


def test_deficits_against_classical_baselines(tex, sweep):
    """The paired inference covers the classical baselines only, so these two
    ranges must be computed without the external prior."""
    half = [psnr(sweep, f, l, "csfr") - max(psnr(sweep, f, l, m) for m in CLASSICAL)
            for f in ["CF3", "CF4"] for l in ["L2", "L3", "L4", "L5"]]
    burst = [psnr(sweep, "CF2", l, "csfr") - max(psnr(sweep, "CF2", l, m) for m in CLASSICAL)
             for l in ["L3", "L4", "L5"]]
    claim(tex, r"\$5\.1\$--\$8\.4\$\\,dB on CF3/CF4")
    claim(tex, r"\$1\.7\$--\$5\.2\$\\,dB on CF2")
    assert abs(min(-v for v in half) - 5.13) < 0.05
    assert abs(max(-v for v in half) - 8.37) < 0.05
    assert abs(min(-v for v in burst) - 1.65) < 0.05
    assert abs(max(-v for v in burst) - 5.24) < 0.05


def test_downstream_ceilings(tex):
    env = json.loads((RESULTS / "downstream" / "environment.json").read_text(encoding="utf-8"))
    clean = list(csv.DictReader((RESULTS / "downstream" / "clean.csv").open(encoding="utf-8")))
    subset = sum(int(r["correct"]) for r in clean) / len(clean)
    claim(tex, r"\$0\.8783\$ on the full SVHN test set")
    claim(tex, r"\$0\.8690\$ on the 1000-digit evaluation subset")
    assert abs(env["clean_accuracy"] - subset) < 1e-6
    assert abs(subset - 0.8690) < 0.0005, f"evaluation-subset ceiling is {subset:.4f}"


def test_confident_prediction_counts(tex):
    met = {}
    with (RESULTS / "downstream" / "metrics.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            met[(r["family"], r["level"], r["method"])] = r
    claim(tex, r"16 of 1000 CSFR outputs against 135 of LaMa")
    assert int(float(met[("CF3", "L5", "csfr")]["n_confident"])) == 16
    assert int(float(met[("CF3", "L5", "lama")]["n_confident"])) == 135
    claim(tex, r"\$0\.250\$ and \$0\.489\$")
    assert abs(float(met[("CF3", "L5", "csfr")]["conf_err_cond"]) - 0.250) < 0.002
    assert abs(float(met[("CF3", "L5", "lama")]["conf_err_cond"]) - 0.489) < 0.002


def test_budget_sensitivity_claims(tex, sweep):
    by: dict = collections.defaultdict(dict)
    with (RESULTS / "budget_sensitivity" / "budget.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[r["cell"]][int(r["budget"])] = float(r["psnr_db"])
    bilinear = psnr(sweep, "CF2", "L5", "bilinear")
    claim(tex, r"trails bilinear by \$4\.6\$\\,dB at 600 iterations")
    claim(tex, r"leads it by \$1\.2\$\\,dB at 2400")
    assert abs((bilinear - by["CF2_L5"][600]) - 4.6) < 0.1
    assert abs((by["CF2_L5"][2400] - bilinear) - 1.2) < 0.1
