"""Classifier training saves its state after every epoch and resumes from the last
saved epoch, ending with the weights an uninterrupted run ends with."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import downstream  # noqa: E402
from downstream import train_classifier  # noqa: E402

KW = dict(epochs=3, batch_size=16, lr=1e-3, seed=0, arch="DigitResNet", weight_decay=5e-4, cosine=True)


def _data():
    g = torch.Generator().manual_seed(1)
    return torch.rand(40, 1, 32, 32, generator=g), torch.randint(0, 10, (40,), generator=g)


def _same(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_a_resumed_training_ends_where_an_uninterrupted_one_ends(tmp_path, monkeypatch):
    x, y = _data()
    whole, info_whole = train_classifier(x, y, **KW)
    ckpt = tmp_path / "train.ckpt"
    real = downstream._accuracy
    calls = []

    def killed_in_the_second_epoch(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(downstream, "_accuracy", killed_in_the_second_epoch)
    with pytest.raises(RuntimeError):
        train_classifier(x, y, checkpoint=ckpt, **KW)
    monkeypatch.setattr(downstream, "_accuracy", real)
    resumed, info = train_classifier(x, y, checkpoint=ckpt, **KW)
    assert _same(whole, resumed)
    assert info["history"] == info_whole["history"]
    assert info["resumed_after_epoch"] == 1 and "resumed_after_epoch" not in info_whole


def test_saving_checkpoints_does_not_change_the_result(tmp_path):
    x, y = _data()
    plain, _ = train_classifier(x, y, **KW)
    saved, _ = train_classifier(x, y, checkpoint=tmp_path / "train.ckpt", **KW)
    assert _same(plain, saved)


def test_a_checkpoint_from_other_settings_or_data_is_not_used(tmp_path):
    x, y = _data()
    ckpt = tmp_path / "train.ckpt"
    whole, _ = train_classifier(x, y, **KW)
    train_classifier(x, y, checkpoint=ckpt, **{**KW, "lr": 1e-2})
    model, info = train_classifier(x, y, checkpoint=ckpt, **KW)
    assert _same(model, whole) and "resumed_after_epoch" not in info
    train_classifier(x, (y + 1) % 10, checkpoint=ckpt, **KW)
    model, info = train_classifier(x, y, checkpoint=ckpt, **KW)
    assert _same(model, whole) and "resumed_after_epoch" not in info
