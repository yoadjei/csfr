"""Tests for src/downstream.py: digit data load + frozen DigitCNN."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
MNIST_CACHE = ROOT / "data" / "mnist"

from downstream import (  # noqa: E402
    DigitCNN,
    clean_top1_accuracy,
    freeze,
    load_digit_dataset,
    load_mnist,
    predict,
    set_seed,
    train_classifier,
)


def _synthetic_digits(n: int = 64, size: int = 32, seed: int = 0):
    """Tiny synthetic set: bright blob whose centre encodes the label."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    labels = rng.integers(0, 10, size=n)
    for i, lab in enumerate(labels):
        # place a bright square whose row band depends on the digit
        r0 = (lab % 5) * (size // 5)
        c0 = (lab // 5) * (size // 2)
        images[i, r0 : r0 + 6, c0 : c0 + 6] = 1.0
        images[i] += rng.normal(0, 0.02, size=(size, size)).astype(np.float32)
    x = torch.from_numpy(images).unsqueeze(1).clamp(0, 1)
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


def test_classifier_forward_shape():
    model = DigitCNN()
    x = torch.randn(4, 1, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_predict_returns_labels_and_confidence():
    model = freeze(DigitCNN())
    x = torch.rand(5, 1, 32, 32)
    labels, conf = predict(model, x)
    assert labels.shape == (5,)
    assert conf.shape == (5,)
    assert labels.dtype == torch.int64
    assert torch.all((conf >= 0) & (conf <= 1))


def test_deterministic_seed_reproduces_weights():
    x, y = _synthetic_digits(n=80, seed=1)

    m1, _ = train_classifier(x, y, epochs=2, batch_size=16, seed=42, val_fraction=0.2)
    m2, _ = train_classifier(x, y, epochs=2, batch_size=16, seed=42, val_fraction=0.2)

    for (n1, p1), (n2, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"mismatch in {n1}"


def test_load_mnist_path_and_shape():
    """Load path works for MNIST.

    Reads the repo cache rather than a temp dir, so the suite does not re-fetch
    11 mb on every run. `scripts/verify.sh` runs these tests, and a gate that
    depends on an external download fails for reasons that have nothing to do
    with the code, which is what happened here.
    """
    try:
        images, labels = load_mnist(MNIST_CACHE, split="test", download=True, size=32)
    except URLError as exc:
        pytest.skip(f"mnist absent from cache and unreachable: {exc}")
    assert images.ndim == 4
    assert images.shape[1:] == (1, 32, 32)
    assert images.dtype == torch.float32
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert labels.shape[0] == images.shape[0]
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 9


def test_clean_accuracy_sane_after_short_train():
    """After a few epochs on MNIST, clean top-1 should clear 80%."""
    try:
        train_x, train_y = load_mnist(MNIST_CACHE, split="train", download=True)
        test_x, test_y = load_mnist(MNIST_CACHE, split="test", download=True)
    except URLError as exc:
        pytest.skip(f"mnist absent from cache and unreachable: {exc}")
    # keep the smoke train short: 8k samples, 2 epochs
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(train_x.shape[0], generator=g)[:8000]
    model, info = train_classifier(
        train_x[idx], train_y[idx], epochs=2, batch_size=128, seed=0
    )
    acc = clean_top1_accuracy(model, test_x, test_y)
    print(f"MNIST clean top-1 = {acc:.4f} (val={info['val_acc']:.4f})")
    assert acc > 0.80, f"clean accuracy too low: {acc:.4f}"


def test_load_digit_dataset_svhn_or_fallback():
    """Preferred SVHN load path; accepts MNIST fallback if SVHN blocked."""
    images, labels, corpus = load_digit_dataset(prefer="svhn", split="test")
    assert corpus in ("svhn", "mnist")
    assert images.shape[1] == 1
    assert images.shape[2] == images.shape[3]
    assert labels.shape[0] == images.shape[0]
    # SVHN zero-as-10 mapping: labels must be 0..9
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 9
