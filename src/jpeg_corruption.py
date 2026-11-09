"""Byte-level JPEG corruption and the missing-pixel masks a carver can build (E9).

The corruption acts on the entropy-coded scan of a baseline JPEG, as damage on
a storage medium would: truncation (the file ends early), overwrite (a run of
foreign bytes inside the scan) and a single flipped byte. A standard decoder in
tolerant mode turns the damaged file into an image: Pillow with
LOAD_TRUNCATED_IMAGES fills an unreadable tail with mid-grey 128, and a corrupted
run desynchronises the Huffman decoding, so the blocks after it carry garbage.

Three masks of the lost pixels (1 = observed, 0 = missing), from most to least
informed:
  oracle   blocks whose decode differs from the clean decode (needs the original;
           an upper bound on what any detector can achieve)
  decoder  blocks the decoder filled with uniform mid-grey (what the decoder
           itself reveals; misses desynchronised garbage)
  offset   every MCU row from the first corrupted byte onward, mapped from the
           byte offset through the scan length, with a safety margin (what a
           carver knowing where the damage starts can build)
"""
from __future__ import annotations

import io

import numpy as np

KINDS = ("truncate", "overwrite", "flip")
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def _segments(b: bytes):
    """(marker, offset, length) of the header segments up to and including SOS."""
    if b[:2] != SOI:
        raise ValueError("not a JPEG (no SOI)")
    i = 2
    while i + 4 <= len(b):
        if b[i] != 0xFF:
            raise ValueError(f"marker expected at offset {i}")
        m = b[i + 1]
        if m == 0xFF:                      # fill byte
            i += 1
            continue
        if m in (0x01,) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(b[i + 2:i + 4], "big")
        yield m, i, seg
        if m == 0xDA:
            return
        i += 2 + seg
    raise ValueError("no SOS segment")


def locate_scan(b: bytes) -> tuple[int, int]:
    """(first byte of the entropy-coded scan, offset of the final EOI marker)."""
    start = None
    for m, i, seg in _segments(b):
        if m == 0xDA:
            start = i + 2 + seg
    end = b.rfind(EOI)
    if start is None or end <= start:
        raise ValueError("scan not found")
    return start, end


def mcu_height(b: bytes) -> int:
    """Pixel height of one MCU row: 8 times the largest vertical sampling factor."""
    for m, i, seg in _segments(b):
        if m in (0xC0, 0xC1, 0xC2):
            n = b[i + 9]
            vmax = max(b[i + 10 + 3 * c + 1] & 0x0F for c in range(n))
            return 8 * vmax
    raise ValueError("no SOF segment")


def corrupt(b: bytes, kind: str, rng: np.random.Generator,
            scan: tuple[int, int]) -> tuple[bytes, int]:
    """Damage the scan; returns (bytes, offset of the first corrupted scan byte)."""
    s0, s1 = scan
    n = s1 - s0
    if kind == "truncate":
        cut = s0 + int(rng.uniform(0.3, 0.9) * n)
        return b[:cut] + EOI, cut
    if kind == "overwrite":
        L = int(rng.integers(512, 4097))
        L = min(L, max(1, n // 2))
        start = s0 + int(rng.integers(0, n - L + 1))
        out = bytearray(b)
        out[start:start + L] = rng.integers(0, 256, L, dtype=np.uint8).tobytes()
        return bytes(out), start
    if kind == "flip":
        off = s0 + int(rng.integers(0, n))
        out = bytearray(b)
        out[off] ^= 0xFF
        return bytes(out), off
    raise ValueError(f"unknown corruption kind {kind}")


def decode_tolerant(b: bytes, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, str]:
    """Luma of a possibly damaged JPEG as float32, and the decoder that produced it.

    Pillow in truncated-image mode first; OpenCV (libjpeg-turbo) if Pillow raises;
    a uniform mid-grey image of `shape` if both fail (then nothing was decoded).
    """
    from PIL import Image, ImageFile
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(b)) as im:
            return np.asarray(im.convert("L"), dtype=np.float32), "pil"
    except Exception:
        pass
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float32), "cv2"
    except Exception:
        pass
    if shape is None:
        raise ValueError("undecodable and no shape given")
    return np.full(shape, 128.0, np.float32), "grey"


def _block_stat(img: np.ndarray, block: int, fn) -> np.ndarray:
    """fn over every block x block tile (edge tiles may be smaller), broadcast back."""
    h, w = img.shape
    out = np.empty((h, w), dtype=bool)
    for r in range(0, h, block):
        for c in range(0, w, block):
            out[r:r + block, c:c + block] = fn(img[r:r + block, c:c + block], r, c)
    return out


def oracle_mask(clean: np.ndarray, decoded: np.ndarray, block: int = 8,
                thresh: float = 4.0) -> np.ndarray:
    diff = np.abs(clean.astype(np.float64) - decoded.astype(np.float64))
    lost = _block_stat(diff, block, lambda t, r, c: t.mean() > thresh)
    return (~lost).astype(np.float32)


def decoder_mask(decoded: np.ndarray, block: int = 8) -> np.ndarray:
    d = decoded.astype(np.float64)
    lost = _block_stat(d, block, lambda t, r, c: t.std() < 0.5 and 120.0 <= t.mean() <= 136.0)
    return (~lost).astype(np.float32)


def offset_mask(shape: tuple[int, int], scan_start: int, scan_end: int, corrupt_offset: int,
                mcu_h: int, margin_rows: int = 2) -> np.ndarray:
    """Missing from the MCU row the corrupted byte maps to (minus a margin) downward."""
    h, w = shape
    n_rows = -(-h // mcu_h)
    frac = (corrupt_offset - scan_start) / max(scan_end - scan_start, 1)
    first = max(0, int(np.floor(frac * n_rows)) - margin_rows)
    M = np.ones((h, w), np.float32)
    M[first * mcu_h:, :] = 0.0
    return M


def mask_errors(M: np.ndarray, M_oracle: np.ndarray) -> dict:
    """False positives: pixels marked observed that the oracle marks lost; false
    negatives: pixels marked lost that the oracle marks observed (fractions of
    all pixels), and the fraction of the oracle loss the mask covers."""
    lost_o = M_oracle == 0
    lost = M == 0
    return {"false_pos_frac": float((~lost & lost_o).mean()),
            "false_neg_frac": float((lost & ~lost_o).mean()),
            "coverage_of_oracle_loss": float((lost & lost_o).sum() / max(lost_o.sum(), 1)),
            "missing_frac": float(lost.mean())}
