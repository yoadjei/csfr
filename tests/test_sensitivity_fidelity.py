"""Query-pixel selection of the fidelity campaign: deterministic, at the stated
Chebyshev distances, on the right side of the mask."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("fid", ROOT / "scripts" / "run_sensitivity_fidelity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block_mask():
    Mb = np.ones((32, 32), bool)
    Mb[10:20, 4:14] = False
    return Mb


def test_observed_queries_are_observed_at_the_stated_distances():
    m = _mod()
    Mb = _block_mask()
    qs = m.observed_queries(Mb)
    dist = distance_transform_cdt(Mb, metric="chessboard")
    assert [d for d, _ in qs] == [1, 3, 8]
    for d, (r, c) in qs:
        assert Mb[r, c]
        assert dist[r, c] == d if d < 8 else dist[r, c] >= 8
    # first in raster order
    d1 = np.argwhere(Mb & (dist == 1))[0]
    assert qs[0][1] == (int(d1[0]), int(d1[1]))


def test_missing_distance_is_skipped():
    m = _mod()
    Mb = np.ones((8, 8), bool)
    Mb[::2, ::2] = False          # scattered loss: no observed pixel is 3 away from the hole
    assert [d for d, _ in m.observed_queries(Mb)] == [1]


def test_imputed_queries_are_hole_pixels_including_the_deepest():
    m = _mod()
    Mb = _block_mask()
    qs = m.imputed_queries(Mb)
    dist = distance_transform_cdt(~Mb, metric="chessboard")
    assert all(not Mb[p] for p in qs)
    assert [int(dist[p]) for p in qs] == [1, 3, int(dist[~Mb].max())]
    assert m.imputed_queries(np.ones((4, 4), bool)) == []
