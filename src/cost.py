"""Runtime and memory measurement shared by the bench (E15) and the
full-resolution experiment (E11).

Three quantities, measured in separate passes so that profiling never inflates
the timings:
  * wall-clock over timed repeats after a warm-up call (time.perf_counter);
  * peak resident set size during one further call, from a thread sampling
    psutil every `interval` seconds (captures native allocations in OpenCV,
    BLAS and torch, at the sampling resolution);
  * the tracemalloc peak of Python-level allocations during that same call.
"""
from __future__ import annotations

import threading
import time
import tracemalloc
from typing import Callable

import numpy as np


class PeakRSS:
    """Context manager: start_mb and peak_mb of the process resident set size."""

    def __init__(self, interval: float = 0.05):
        import psutil
        self._proc = psutil.Process()
        self.interval = interval
        self.start_mb = self.peak_mb = 0.0
        self._stop = threading.Event()

    def _rss(self) -> float:
        return self._proc.memory_info().rss / 2 ** 20

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRSS":
        self.start_mb = self.peak_mb = self._rss()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self.peak_mb = max(self.peak_mb, self._rss())


def measure(fn: Callable[[], object], repeats: int = 5, warmup: int = 1,
            interval: float = 0.05) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    with PeakRSS(interval) as p:
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {"repeats": repeats, "mean_s": float(t.mean()),
            "sd_s": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            "median_s": float(np.median(t)), "min_s": float(t.min()), "max_s": float(t.max()),
            "rss_start_mb": p.start_mb, "rss_peak_mb": p.peak_mb,
            "rss_peak_delta_mb": p.peak_mb - p.start_mb,
            "tracemalloc_peak_mb": peak / 2 ** 20, "rss_sampling_interval_s": interval}
