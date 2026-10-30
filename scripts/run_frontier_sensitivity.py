"""Sensitivity of the exploratory auditability composite to its free parameters
(pre-submission review §4.8: "provide sensitivity analysis for their thresholds
and weights").

The composite combines three study-defined indicators with equal weight:
V = w1(1-CVR) + w2(1-HRP_norm) + w3 ARC,  w = (1/3,1/3,1/3), where ARC is
audit record coverage. Two choices are arbitrary and are stressed here:

  * the weights w1,w2,w3 -- swept over the probability simplex;
  * the HRP normaliser -- HRP has no natural upper bound, so it is rescaled;
    we vary the normalising constant.

The reported quantity is CSFR's rank on the composite. If CSFR's standing
depends on the exact weights, the ranking is not a finding; if it is stable
across the simplex, the ranking is robust to the composite's free parameters.
Either way the composite describes the evaluated software artefacts, not a
forensic-validity scale.

Output: results/sensitivity/frontier_sensitivity.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# the external-prior inpainter belongs in this sweep. the ranking under test
# includes every method the paper measured, and lama is the only generative
# method run under the harness, so a sweep without it tests the ranking against
# no generative method at all. its hrp and all four cvr columns are in the same
# summary csv.
METHODS = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn",
           "lama", "csfr"]
# only the csfr implementation emitted an audit record; the composite needs a
# number where the paper reports "not available", so the others enter as 0.
ARC = {m: (1.0 if m == "csfr" else 0.0) for m in METHODS}


def load(summary: Path) -> dict:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    with summary.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] in METHODS:
                for k in ("psnr_db_mean", "hrp_mean", "cvr_c1", "cvr_c2", "cvr_c3", "cvr_c4"):
                    agg[r["method"]][k].append(float(r[k]))
    out = {}
    for m, v in agg.items():
        cvr = np.mean([np.mean(v[f"cvr_c{i}"]) for i in range(1, 5)])
        out[m] = {"psnr": float(np.mean(v["psnr_db_mean"])),
                  "hrp": float(np.mean(v["hrp_mean"])), "cvr": float(cvr)}
    return out


def validity(stats: dict, w, hrp_norm: float) -> dict:
    w1, w2, w3 = w
    return {m: w1 * (1 - s["cvr"]) + w2 * (1 - s["hrp"] / hrp_norm) + w3 * ARC[m]
            for m, s in stats.items()}


def rank_of(scores: dict, method: str) -> int:
    return sorted(scores, key=scores.get, reverse=True).index(method) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/csfr_sweep_2d_v2/summary.csv")
    ap.add_argument("--out", default="results/sensitivity/frontier_sensitivity.json")
    ap.add_argument("--grid", type=int, default=21, help="Simplex resolution")
    args = ap.parse_args(argv)

    stats = load(ROOT / args.summary)
    hrp_max = max(s["hrp"] for s in stats.values())

    # sweep the weight simplex on a grid.

    ranks = []
    g = args.grid
    for i, j in itertools.product(range(g + 1), repeat=2):
        if i + j > g:
            continue
        w = (i / g, j / g, (g - i - j) / g)
        sc = validity(stats, w, hrp_max)
        ranks.append(rank_of(sc, "csfr"))
    ranks = np.array(ranks)

    # vary the HRP normaliser at equal weights.

    hrp_variants = {}
    for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
        sc = validity(stats, (1 / 3, 1 / 3, 1 / 3), hrp_max * scale)
        hrp_variants[f"{scale}x"] = rank_of(sc, "csfr")

    # the point the paper actually reports.

    base = validity(stats, (1 / 3, 1 / 3, 1 / 3), hrp_max)

    result = {
        "weights_simplex": {
            "n_weightings": int(ranks.size),
            "csfr_rank_min": int(ranks.min()),
            "csfr_rank_max": int(ranks.max()),
            "fraction_csfr_rank1": float((ranks == 1).mean()),
            "note": "CSFR rank on the exploratory auditability composite over the full weight simplex"},
        "hrp_normaliser": hrp_variants,
        "equal_weight_point": {
            "csfr_rank": rank_of(base, "csfr"),
            "scores": {m: round(v, 4) for m, v in base.items()}},
        "interpretation": (
            "ARC is the only component on which CSFR is uniquely maximal, "
            "because only its implementation emitted an audit record in this "
            "study. Wherever w3>0 that component lifts CSFR unless another "
            "method's CVR/HRP advantage outweighs it; the rank range records "
            "where that happens. The composite describes the evaluated software "
            "artefacts and is not a forensic-validity scale."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["weights_simplex"] | {"hrp": hrp_variants}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
