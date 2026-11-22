"""scripts/train_resnet_classifier.py trains and evaluates on the device it is
given (CPU by default, as on the workstation) and records it with the weights."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("train_resnet", ROOT / "scripts" / "train_resnet_classifier.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("argv, expected", [([], "cpu"), (["--device", "cuda"], "cuda")])
def test_the_device_reaches_training_evaluation_and_the_record(tmp_path, monkeypatch, argv, expected):
    m = _mod()
    x, y = torch.zeros(4, 1, 32, 32), torch.zeros(4, dtype=torch.long)
    seen, saved = {}, {}
    monkeypatch.setattr(m, "load_digit_dataset", lambda prefer, split, download=True: (x, y, "svhn"))

    def train(*args, device=None, **kwargs):
        seen["train"] = str(device)
        return "model", {}

    def evaluate(model, images, labels, device=None):
        seen["eval"] = str(device)
        return 0.97

    monkeypatch.setattr(m, "train_classifier", train)
    monkeypatch.setattr(m, "clean_top1_accuracy", evaluate)
    monkeypatch.setattr(m, "save_classifier", lambda model, out, meta: saved.update(meta))
    monkeypatch.setattr(m, "sha256_file", lambda path: "0")
    out, ckpt = tmp_path / "c.pt", tmp_path / "t.pt"
    assert m.main([*argv, "--out", str(out), "--checkpoint", str(ckpt)]) == 0
    assert seen == {"train": expected, "eval": expected}
    assert saved["device"] == expected
