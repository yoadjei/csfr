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
