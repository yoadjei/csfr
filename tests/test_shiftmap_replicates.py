"""The replicate statistics of scripts/shiftmap_replicates.py."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("reps", ROOT / "scripts" / "shiftmap_replicates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spread_counts_differing_patches_and_the_spread_of_cell_means():
    m = _mod()
    ref = np.full((3, 4, 4), 100.0)
    M = np.ones((3, 4, 4))
    M[:, :2] = 0
    a = ref.copy()
    a[M == 0] += 1.0                                # every patch at 1 grey level error
    b = a.copy()
    b[1][M[1] == 0] += 1.0                          # patch 1 differs in the second run
    s = m.spread(ref, [a, a.copy(), b], M)
    assert s["n_realisations"] == 3 and s["frac_patches_differ"] == 1 / 3
    p1 = 10 * np.log10(255 ** 2 / 1.0)
    p2 = 10 * np.log10(255 ** 2 / 4.0)
    assert abs(s["patch_range_max"] - (p1 - p2)) < 1e-9
    assert abs(s["cell_mean_range"] - (p1 - p2) / 3) < 1e-9
    assert s["cell_mean_sd"] > 0
