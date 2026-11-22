"""Round-2 downstream statistics: accuracy, ECE of the temperature-scaled
confidence, McNemar paired by (patch, seed) with the sign convention of the
round-1 script, and Holm across the methods of a cell."""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["patch_id", "seed", "correct", "confidence", "confidence_ts"]


def _mod():
    spec = importlib.util.spec_from_file_location("ds2", ROOT / "scripts" / "run_downstream_stats_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path, correct, conf=0.9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, c in enumerate(correct):
            w.writerow({"patch_id": i % 10, "seed": i // 10, "correct": int(c), "confidence": conf,
                        "confidence_ts": 1.0})


def test_accuracy_ece_and_mcnemar(tmp_path):
    m = _mod()
    d = tmp_path / "cnn"
    _write(d / "clean.csv", [1] * 20)
    csfr = [1] * 15 + [0] * 5                     # 20 (patch, seed) pairs
    other = [1] * 10 + [0] * 10
    _write(d / "CF1_L3" / "csfr.csv", csfr)
    _write(d / "CF1_L3" / "bilinear.csv", other)
    _write(d / "CF1_L3" / "telea.csv", csfr)
    assert m.main(["--root", str(tmp_path)]) == 0
    met = {(r["family"], r["method"]): r for r in csv.DictReader((tmp_path / "stats" / "metrics.csv").open())}
    assert float(met[("CF1", "csfr")]["accuracy"]) == 0.75
    # temperature-scaled confidence 1.0 everywhere: ECE = 1 - accuracy (the round-1 helper dropped 1.0)
    assert abs(float(met[("CF1", "csfr")]["ece_ts"]) - 0.25) < 1e-12
    assert float(met[("clean", "clean")]["accuracy"]) == 1.0
    st = {r["method"]: r for r in csv.DictReader((tmp_path / "stats" / "stats.csv").open())}
    assert int(st["bilinear"]["b_csfr_only"]) == 5 and int(st["bilinear"]["c_other_only"]) == 0
    assert int(st["bilinear"]["statistic"]) == 5                  # positive: CSFR ahead
    assert float(st["telea"]["p_mcnemar"]) == 1.0                  # identical correctness
    assert float(st["bilinear"]["p_holm"]) >= float(st["bilinear"]["p_mcnemar"])


def test_risk_coverage_enters_tied_confidences_together():
    m = _mod()
    conf = [0.9, 0.9, 0.8, 0.5]
    correct = [1, 0, 1, 1]
    cov, acc = m.risk_coverage(conf, correct)
    assert list(cov) == [0.5, 0.75, 1.0]           # the two 0.9s enter together
    assert list(acc) == [0.5, 2 / 3, 0.75]
    # reordering the tied inputs changes nothing
    assert [list(a) for a in m.risk_coverage([0.9, 0.9, 0.8, 0.5], [0, 1, 1, 1])] == [list(cov), list(acc)]


def test_aurc_is_the_mean_selective_risk_without_ties():
    import numpy as np
    m = _mod()
    rng = np.random.default_rng(0)
    conf = rng.random(50)
    correct = (rng.random(50) < conf).astype(int)
    order = np.argsort(-conf)
    risks = [1 - correct[order[:k]].mean() for k in range(1, 51)]
    assert abs(m.aurc(conf, correct) - float(np.mean(risks))) < 1e-12


def test_selective_accuracy_at_the_first_threshold_that_reaches_the_coverage():
    m = _mod()
    conf, correct = [0.9, 0.9, 0.8, 0.5], [1, 0, 1, 1]
    assert m.selective_accuracy(conf, correct, 0.25) == (0.5, 0.5)     # the tie lifts it to 0.5
    assert m.selective_accuracy(conf, correct, 0.6) == (0.75, 2 / 3)
    assert m.selective_accuracy(conf, correct, 1.0) == (1.0, 0.75)


def test_coverage_curves_per_cell_and_pooled_by_level(tmp_path):
    m = _mod()
    d = tmp_path / "cnn"
    _write(d / "clean.csv", [1] * 20)
    for fam in ("CF1", "CF2"):
        _write(d / f"{fam}_L3" / "csfr.csv", [1] * 15 + [0] * 5)
        _write(d / f"{fam}_L3" / "bilinear.csv", [1] * 10 + [0] * 10)
    assert m.main(["--root", str(tmp_path)]) == 0
    met = {(r["family"], r["method"]): r for r in csv.DictReader((tmp_path / "stats" / "metrics.csv").open())}
    # confidence_ts is 1.0 everywhere: one threshold, risk 1 - accuracy over all coverage
    assert abs(float(met[("CF1", "csfr")]["aurc_ts"]) - 0.25) < 1e-12
    cur = list(csv.DictReader((tmp_path / "stats" / "coverage.csv").open()))
    assert {r["family"] for r in cur} == {"clean", "CF1", "CF2", "all"}
    pooled = [r for r in cur if r["family"] == "all" and r["method"] == "bilinear"]
    assert len(pooled) == len(m.COVERAGE_GRID) and {r["level"] for r in pooled} == {"L3"}
    assert all(float(r["coverage"]) == 1.0 and float(r["accuracy"]) == 0.5 for r in pooled)
