"""Downstream digit classifier for the CSFR utility experiment.

Trains a small CNN on clean grayscale digits (SVHN preferred, MNIST fallback),
freezes it, and exposes helpers for loading data, predicting labels with
softmax confidence, and reporting clean-set top-1 accuracy. No post-hoc
calibration here; native calibration is measured later.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SVHN_DIR = ROOT / "data" / "svhn"
DEFAULT_MNIST_DIR = ROOT / "data" / "mnist"
DEFAULT_CKPT = ROOT / "results" / "downstream" / "classifier.pt"
DEFAULT_SEED = 0

SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
}

# original IDX gzip files (no torchvision dependency)
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

PathLike = Union[str, Path]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix RNG state for reproducible training and evaluation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    """Fetch url to dest; resume via HTTP Range when a .part file exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resuming {url} from byte {existing} -> {dest}")
    else:
        print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, mode) as out:
        # if server ignored Range and returned 200, rewrite from scratch
        if existing > 0 and getattr(resp, "status", 200) == 200:
            out.close()
            with open(tmp, "wb") as out2:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out2.write(chunk)
        else:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    tmp.replace(dest)
    return dest


def _rgb_to_gray(x: np.ndarray) -> np.ndarray:
    """Luminance grayscale; x is (H, W, 3) or (N, H, W, 3) uint8/float."""
    # ITU-R BT.601 coefficients
    w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    return (x.astype(np.float32) * w).sum(axis=-1)


def _to_nchw_float01(gray: np.ndarray) -> torch.Tensor:
    """(N, H, W) -> float tensor (N, 1, H, W) in [0, 1]."""
    if gray.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {gray.shape}")
    t = torch.from_numpy(gray.astype(np.float32))
    if t.max() > 1.5:
        t = t / 255.0
    return t.unsqueeze(1).contiguous()


# ── SVHN ──────────────────────────────────────────────────────────────
def download_svhn(root: PathLike = DEFAULT_SVHN_DIR) -> Path:
    """Fetch train/test .mat files into root. Returns the root path."""
    root = Path(root)
    for split, url in SVHN_URLS.items():
        _download(url, root / f"{split}_32x32.mat")
    return root


def load_svhn(
    root: PathLike = DEFAULT_SVHN_DIR,
    split: str = "train",
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load SVHN as grayscale float tensors suitable for CSFR.

    Returns
    -------
    images : Tensor[N, 1, 32, 32] float in [0, 1]
    labels : Tensor[N] long in {0..9}
        classic SVHN encodes digit zero as label 10; mapped to 0 here.
    """
    root = Path(root)
    path = root / f"{split}_32x32.mat"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        download_svhn(root)
    mat = loadmat(str(path))
    # X is (32, 32, 3, N); y is (N, 1)
    x = np.transpose(mat["X"], (3, 0, 1, 2))  # N,H,W,3
    y = mat["y"].astype(np.int64).reshape(-1)
    y = np.where(y == 10, 0, y)
    gray = _rgb_to_gray(x)
    return _to_nchw_float01(gray), torch.from_numpy(y)


# ── MNIST fallback ────────────────────────────────────────────────────
def _parse_idx_images(raw: bytes) -> np.ndarray:
    import gzip
    import struct

    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"bad image magic {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def _parse_idx_labels(raw: bytes) -> np.ndarray:
    import gzip
    import struct

    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    magic, n = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"bad label magic {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=8).copy()


def download_mnist(root: PathLike = DEFAULT_MNIST_DIR) -> Path:
    root = Path(root)
    for name, url in MNIST_URLS.items():
        _download(url, root / Path(url).name)
    return root


def load_mnist(
    root: PathLike = DEFAULT_MNIST_DIR,
    split: str = "train",
    download: bool = True,
    size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST, resize to size×size, return grayscale float tensors."""
    root = Path(root)
    prefix = "train" if split == "train" else "t10k"
    img_path = root / f"{prefix}-images-idx3-ubyte.gz"
    lab_path = root / f"{prefix}-labels-idx1-ubyte.gz"
    if not img_path.exists() or not lab_path.exists():
        if not download:
            raise FileNotFoundError(img_path)
        download_mnist(root)
    images = _parse_idx_images(img_path.read_bytes())
    labels = _parse_idx_labels(lab_path.read_bytes()).astype(np.int64)
    # nearest-neighbour upsample 28→32 so the CNN sees a square digit field
    if size != 28:
        images = _resize_nn(images, size)
    return _to_nchw_float01(images), torch.from_numpy(labels)


def _resize_nn(images: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of (N, H, W) uint8 to (N, size, size)."""
    n, h, w = images.shape
    ys = (np.arange(size) * h / size).astype(np.int64)
    xs = (np.arange(size) * w / size).astype(np.int64)
    return images[:, ys][:, :, xs]


def load_digit_dataset(
    prefer: str = "svhn",
    root: Optional[PathLike] = None,
    split: str = "train",
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Load preferred corpus; fall back to MNIST if SVHN cannot be fetched.

    Returns (images, labels, corpus_name) where corpus_name is 'svhn' or 'mnist'.
    """
    prefer = prefer.lower()
    if prefer not in ("svhn", "mnist"):
        raise ValueError(f"prefer must be 'svhn' or 'mnist', got {prefer!r}")

    if prefer == "mnist":
        r = Path(root) if root is not None else DEFAULT_MNIST_DIR
        imgs, labs = load_mnist(r, split=split, download=download)
        return imgs, labs, "mnist"

    r = Path(root) if root is not None else DEFAULT_SVHN_DIR
    try:
        imgs, labs = load_svhn(r, split=split, download=download)
        return imgs, labs, "svhn"
    except Exception as exc:  # noqa: BLE001  intentional fallback
        print(f"SVHN unavailable ({exc}); falling back to MNIST")
        r = DEFAULT_MNIST_DIR
        imgs, labs = load_mnist(r, split=split, download=download)
        return imgs, labs, "mnist"


# ── CNN ───────────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    """Small 3-block conv net for 32×32 grayscale digits (10 classes)."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16×16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8×8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def freeze(model: nn.Module) -> nn.Module:
    """Disable gradients and put the model in eval mode."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_classifier(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = DEFAULT_SEED,
    device: Optional[torch.device] = None,
    val_fraction: float = 0.1,
) -> Tuple[DigitCNN, dict]:
    """Train DigitCNN on clean digits. Returns (frozen model, info dict)."""
    set_seed(seed)
    device = device or torch.device("cpu")
    n = images.shape[0]
    n_val = max(1, int(n * val_fraction))
    # deterministic shuffle via seeded generator
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(images[train_idx], labels[train_idx])
    val_ds = TensorDataset(images[val_idx], labels[val_idx])
    # reseeds so DataLoader shuffle is also deterministic across runs
    set_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_acc = _accuracy(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"loss={history['train_loss'][-1]:.4f}  val_acc={val_acc:.4f}"
        )

    freeze(model)
    info = {
        "seed": seed,
        "epochs": epochs,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "history": history,
        "val_acc": history["val_acc"][-1],
    }
    return model.cpu(), info


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu()
            correct += int((pred == yb).sum())
            total += yb.numel()
    return correct / max(total, 1)


def save_classifier(
    model: nn.Module,
    path: PathLike = DEFAULT_CKPT,
    meta: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta or {},
        "arch": "DigitCNN",
    }
    torch.save(payload, path)
    return path


def load_classifier(
    path: PathLike = DEFAULT_CKPT,
    device: Optional[torch.device] = None,
    freeze_weights: bool = True,
) -> DigitCNN:
    """Load a saved DigitCNN; frozen by default."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DigitCNN()
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    if freeze_weights:
        freeze(model)
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    images: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict digit labels and softmax-max confidence.

    Parameters
    ----------
    images : Tensor[N, 1, H, W] or [1, H, W] or [H, W]

    Returns
    -------
    labels : LongTensor[N]
    confidence : FloatTensor[N]  (max softmax probability)
    """
    device = device or next(model.parameters()).device
    model.eval()
    if images.ndim == 2:  # H, W
        images = images.unsqueeze(0).unsqueeze(0)
    elif images.ndim == 3:  # N, H, W (channel-less batch)
        images = images.unsqueeze(1)
    elif images.ndim != 4:
        raise ValueError(f"expected 2D/3D/4D images, got shape {tuple(images.shape)}")

    labels_out = []
    conf_out = []
    for start in range(0, images.shape[0], batch_size):
        xb = images[start : start + batch_size].to(device)
        logits = model(xb)
        prob = F.softmax(logits, dim=1)
        conf, lab = prob.max(dim=1)
        labels_out.append(lab.cpu())
        conf_out.append(conf.cpu())
    return torch.cat(labels_out), torch.cat(conf_out)


@torch.no_grad()
def clean_top1_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> float:
    """Top-1 accuracy of the (frozen) classifier on a clean labelled set."""
    pred, _ = predict(model, images, device=device, batch_size=batch_size)
    return float((pred == labels.cpu()).float().mean())


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_and_save(
    *,
    prefer: str = "svhn",
    epochs: int = 3,
    seed: int = DEFAULT_SEED,
    ckpt_path: PathLike = DEFAULT_CKPT,
    max_train: Optional[int] = None,
) -> dict:
    """End-to-end: load data, train, evaluate clean test accuracy, save ckpt."""
    set_seed(seed)
    train_x, train_y, corpus = load_digit_dataset(prefer=prefer, split="train")
    test_x, test_y, _ = load_digit_dataset(prefer=corpus, split="test", download=False)

    if max_train is not None and max_train < train_x.shape[0]:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(train_x.shape[0], generator=g)[:max_train]
        train_x, train_y = train_x[idx], train_y[idx]

    model, info = train_classifier(
        train_x, train_y, epochs=epochs, seed=seed
    )
    acc = clean_top1_accuracy(model, test_x, test_y)
    meta = {
        "corpus": corpus,
        "seed": seed,
        "epochs": epochs,
        "clean_test_top1": acc,
        "n_train": info["n_train"],
        "n_val": info["n_val"],
        "val_acc": info["val_acc"],
        "input_shape": list(train_x.shape[1:]),
    }
    path = save_classifier(model, ckpt_path, meta=meta)
    meta["ckpt_path"] = str(path)
    meta["ckpt_sha256"] = sha256_file(path)
    meta["ckpt_bytes"] = path.stat().st_size
    print(f"clean test top-1 = {acc:.4f}  ({corpus})")
    print(f"saved {path} ({meta['ckpt_bytes']} bytes)")
    return meta


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train frozen digit CNN for CSFR downstream")
    p.add_argument("--prefer", choices=["svhn", "mnist"], default="svhn")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--max-train", type=int, default=None,
                   help="optional cap on training set size for smoke runs")
    args = p.parse_args()
    train_and_save(
        prefer=args.prefer,
        epochs=args.epochs,
        seed=args.seed,
        ckpt_path=args.ckpt,
        max_train=args.max_train,
    )
