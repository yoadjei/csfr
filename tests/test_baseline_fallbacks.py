"""The fallback rule of scripts/baseline_fallbacks.py is the one the dictionary
learning baseline applies: a patch it predicts to fall back gets the observed
mean throughout the hole, and one it predicts not to does not."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _mod():
    spec = importlib.util.spec_from_file_location("fallbacks", ROOT / "scripts" / "baseline_fallbacks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_intact_tiles_counts_the_scan_positions():
    m = _mod()
    full = np.ones((64, 64), np.uint8)
    assert m.intact_tiles(full) == 29 * 29          # positions 0, 2, ..., 56 on each axis
    one_hole = full.copy()
    one_hole[0, 0] = 0
    assert m.intact_tiles(one_hole) == 29 * 29 - 1  # only the window at (0, 0) contains it


def test_rule_matches_the_baseline_on_both_sides():
    m = _mod()
    from baselines import baseline_dictlearn
    from masks import make_mask
    y = np.load(ROOT / "data" / "patches.npy")[:2].astype(np.float32)
    rng = np.random.default_rng(0)
    cases = [make_mask((1, 64, 64), "CF1", 0.05, rng)[0], make_mask((1, 64, 64), "CF3", 0.05, rng)[0]]
    seen = set()
    for M in cases:
        out = baseline_dictlearn((y[0] * M)[None], M[None], rng_seed=0)[0]
        mean = float(y[0][M == 1].mean())
        hole = out[M == 0]
        falls_back = m.dictlearn_falls_back(M)
        assert bool(np.all(hole == np.float32(mean))) == falls_back
        seen.add(falls_back)
    assert seen == {True, False}                   # both branches exercised
