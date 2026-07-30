"""Sensitivity of the forensic-validity frontier to its free parameters
(pre-submission review §4.8: "provide sensitivity analysis for their thresholds
and weights").

The validity composite combines three study-defined indicators with equal
weight:  V = w1(1-CVR) + w2(1-HRP_norm) + w3 TS,  w = (1/3,1/3,1/3). Two choices
are arbitrary and are stressed here:

  * the weights w1,w2,w3 -- swept over the probability simplex;
  * the HRP normaliser -- HRP has no natural upper bound, so it is rescaled;
    we vary the normalising constant.

The reported quantity is CSFR's rank on the validity axis. If the qualitative
frontier claim ("DL off the frontier; CSFR the only certified point") depends on
the exact weights, it is not a real finding; if CSFR's standing is stable across
the simplex, it is.

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
METHODS = ["zero_fill", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn", "csfr"]
TS = {m: (1.0 if m == "csfr" else 0.0) for m in METHODS}


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
    return {m: w1 * (1 - s["cvr"]) + w2 * (1 - s["hrp"] / hrp_norm) + w3 * TS[m]
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
            "note": "CSFR rank on the validity axis over the full weight simplex"},
        "hrp_normaliser": hrp_variants,
        "equal_weight_point": {
            "csfr_rank": rank_of(base, "csfr"),
            "scores": {m: round(v, 4) for m, v in base.items()}},
        "interpretation": (
            "TS is the only axis on which CSFR is uniquely maximal, so wherever "
            "w3>0 CSFR cannot be beaten on validity by a TS=0 method unless that "
            "method's CVR/HRP advantage outweighs the TS gap. The rank range "
            "records exactly where that happens."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["weights_simplex"] | {"hrp": hrp_variants}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
