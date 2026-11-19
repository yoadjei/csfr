"""The graphical abstract: its numbers come from the generated macros, a pending
number is refused, and the image meets the venue's size limits. Synthetic patch,
no solve."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("graphical_abstract", ROOT / "scripts" / "render_graphical_abstract.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


VALUES = {"csfrTol": "10^{-6}", "recSrfPooled": "90.9", "recNQueries": "529"}


def test_text_lines_carry_the_generated_numbers():
    m = _mod()
    lines = m.text_lines(VALUES, 0.10)
    assert ("unique minimiser, Frank–Wolfe gap ≤ 10", False, "−6") in lines
    assert ("90.9% of 529 queries within 10%", False, "") in lines
    assert m.power_of_ten("10^{-12}") == "−12"
    with pytest.raises(ValueError):
        m.power_of_ten("5\\times 10^{-6}")


@pytest.mark.parametrize("bad", [{"recSrfPooled": r"\textbf{[pending]}"}, {"recNQueries": None}])
def test_a_pending_or_missing_number_is_refused(bad):
    m = _mod()
    v = {**VALUES, **bad}
    v = {k: x for k, x in v.items() if x is not None}
    with pytest.raises(ValueError):
        m.text_lines(v, 0.10)


def test_the_committed_macros_give_every_number():
    m = _mod()
    lines = m.text_lines(m.macro_values(), 0.10)
    assert all("pending" not in t for t, _, _ in lines)


def test_image_size_and_file_limit(tmp_path):
    m = _mod()
    rng = np.random.default_rng(0)
    observed = np.ones((64, 64), bool)
    observed[:45] = False                                      # top rows lost, as in the walkthrough patch
    truth = rng.uniform(0, 255, (64, 64))
    data = {"observed": observed, "truth": truth, "reconstruction": truth}
    q = m.query_pixel(observed)
    assert q == (42, 32)
    row = rng.normal(size=(64, 64))
    cmap = LinearSegmentedColormap.from_list("s", ["#184f95", "#f0efec", "#a8302f"])
    img = m.draw(data, row, q, m.text_lines(VALUES, 0.10), cmap)
    assert img.size == (m.W, m.H) == (660, 295)
    out = tmp_path / "ga.jpg"
    m.save_jpeg(img, out)
    assert out.stat().st_size < 45 * 1024
