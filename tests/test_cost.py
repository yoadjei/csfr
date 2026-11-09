"""Cost measurement: wall-clock repeats, the peak resident set size seen by a
sampling thread, and the tracemalloc peak, from one helper used by the bench
and the full-resolution experiment."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cost import PeakRSS, measure  # noqa: E402


def test_peak_rss_sees_a_transient_allocation():
    with PeakRSS(interval=0.01) as p:
        a = np.ones(64 * 2 ** 20 // 8)          # 64 MB, touched
        time.sleep(0.15)
        del a
    assert p.peak_mb - p.start_mb > 40


def test_measure_reports_repeats_and_memory():
    calls = []
    r = measure(lambda: calls.append(np.zeros(1000).sum()), repeats=3, warmup=1)
    assert len(calls) == 1 + 3 + 1                # warm-up, timed repeats, one memory pass
    for k in ["mean_s", "sd_s", "median_s", "min_s", "max_s", "rss_peak_mb", "rss_start_mb",
              "tracemalloc_peak_mb", "repeats"]:
        assert k in r
    assert r["repeats"] == 3 and r["min_s"] <= r["median_s"] <= r["max_s"]
