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


def test_an_existing_measurement_is_kept(tmp_path):
    """Shift-map is not deterministic, so measuring again gives new numbers; a
    resumed campaign must keep the measurement the paper quotes."""
    m = _mod()
    sweep = tmp_path / "sweep_v2_dfrws"
    sweep.mkdir()
    (tmp_path / "out").mkdir()
    kept = tmp_path / "out" / "sweep_v2_dfrws.csv"
    kept.write_text("family,level\nCF1,L1\n")
    assert m.main(["--sweep-dir", str(sweep), "--out-dir", str(tmp_path / "out")]) == 0
    assert kept.read_text() == "family,level\nCF1,L1\n"      # no run_metadata.json was needed either
