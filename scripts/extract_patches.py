"""Carve images from a forensic corpus and dice them into evaluation patches.

Referenced by ``data/README.md``. Reproduces the committed DFRWS 2006 patch set
and builds the external-validation corpora on the same protocol.

Sampling protocol (fixed, seeded, and recorded in the emitted manifest):
  1. Carve JPEG byte ranges by SOI/EOI signature (disk-image sources) or read
     image entries directly (zip / directory sources).
  2. Keep only images that decode cleanly and whose shorter side is >= MIN_SIDE.
  3. Dice each image on a **non-overlapping** grid (stride >= patch size), so no
     two patches in the corpus share a pixel.
  4. Drop near-constant patches (per-patch std < MIN_STD): PSNR is degenerate on
     flat content and would dominate the aggregate.
  5. Cap the patches contributed by any single source image at MAX_PER_IMAGE, to
     limit how far one image can dominate the corpus.
  6. Sample to the target count without replacement under the given seed.

The manifest records, for every retained patch, the source image it came from
(``source_image`` index). Patches from one image are statistically dependent, so
downstream inference must treat the source image as the cluster unit; see
``scripts/run_paired_stats.py``.

Usage:
    # external validation corpus (independent of DFRWS 2006)
    python scripts/extract_patches.py --source-type zip \
        --input "B:/rsencarver/Datasets/GovDocs1/zipfiles" \
        --name govdocs1 --n-target 300 --out data/patches_govdocs1.npy

    # naturally corrupted, full-resolution, colour case studies
    python scripts/extract_patches.py --source-type diskimage \
        --input "B:/rsencarver/Datasets/NickMikus/11-carve-fat/11-carve-fat.dd" \
        --name mikus_fat --keep-full-images data/full_mikus_fat --colour
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".bmp")


# ------------------------------------------------------------------ carve ---
def carve_jpegs(buf: bytes, min_bytes: int = 2_000,
                max_bytes: int = 8_000_000) -> list[bytes]:
    """Signature-carve JPEG byte ranges from a raw byte stream.

    Each SOI marker is paired with the *first following* EOI rather than
    consuming EOIs in lockstep: 0xFFD8 also occurs inside entropy-coded data and
    in thumbnails, and a lockstep pairing lets those false headers swallow the
    terminators belonging to real images. Candidates are deduplicated by content
    so an image recovered through several false headers is carved once.
    """
    b = np.frombuffer(buf, dtype=np.uint8)
    marker = (b[:-1] == 0xFF)
    soi = np.where(marker & (b[1:] == 0xD8))[0]
    eoi = np.where(marker & (b[1:] == 0xD9))[0]
    out: list[bytes] = []
    seen: set[bytes] = set()
    for s in soi:
        k = int(np.searchsorted(eoi, s))
        if k >= len(eoi):
            break
        e = int(eoi[k]) + 2
        if not (min_bytes <= e - s <= max_bytes):
            continue
        blob = buf[s:e]
        digest = hashlib.sha256(blob).digest()
        if digest not in seen:
            seen.add(digest)
            out.append(blob)
    return out


def iter_source_blobs(source_type: str, input_path: Path):
    """Yield ``(identifier, raw_bytes)`` for every candidate image in a source."""
    if source_type == "diskimage":
        for i, blob in enumerate(carve_jpegs(input_path.read_bytes())):
            yield f"{input_path.name}#carved{i:04d}", blob
    elif source_type == "zip":
        archives = (sorted(input_path.glob("*.zip")) if input_path.is_dir()
                    else [input_path])
        for arc in archives:
            with zipfile.ZipFile(arc) as z:
                for name in z.namelist():
                    if name.lower().endswith(IMAGE_SUFFIXES):
                        yield f"{arc.name}:{name}", z.read(name)
    elif source_type == "dir":
        for p in sorted(input_path.rglob("*")):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                yield str(p.relative_to(input_path)), p.read_bytes()
    else:
        raise ValueError(source_type)


def decode(raw: bytes, min_side: int, colour: bool) -> np.ndarray | None:
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    if min(img.size) < min_side:
        return None
    return np.asarray(img.convert("RGB" if colour else "L"), dtype=np.uint8)


# ------------------------------------------------------------------- dice ---
def dice(im: np.ndarray, patch: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut a non-overlapping patch grid. Returns (patches, top-left coords)."""
    h, w = im.shape[:2]
    ys = np.arange(0, h - patch + 1, stride)
    xs = np.arange(0, w - patch + 1, stride)
    if ys.size == 0 or xs.size == 0:
        return np.empty((0, patch, patch) + im.shape[2:], im.dtype), np.empty((0, 2), int)
    coords = np.array([(y, x) for y in ys for x in xs])
    patches = np.stack([im[y:y + patch, x:x + patch] for y, x in coords])
    return patches, coords


def keep_textured(patches: np.ndarray, coords: np.ndarray, min_std: float):
    if patches.size == 0:
        return patches, coords
    stds = patches.reshape(patches.shape[0], -1).std(axis=1)
    sel = stds >= min_std
    return patches[sel], coords[sel]


# ------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-type", choices=["diskimage", "zip", "dir"], required=True)
    ap.add_argument("--input", required=True, help="Disk image, zip, or directory")
    ap.add_argument("--name", required=True, help="Corpus name recorded in the manifest")
    ap.add_argument("--out", default=None, help="Output .npy (default data/patches_<name>.npy)")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=None,
                    help="Default = patch size (non-overlapping grid)")
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--min-std", type=float, default=10.0)
    ap.add_argument("--max-per-image", type=int, default=8,
                    help="Cap on patches contributed by one source image")
    ap.add_argument("--n-target", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--colour", action="store_true", help="Keep RGB instead of grayscale")
    ap.add_argument("--keep-full-images", default=None,
                    help="Also write every decoded full-resolution image to this dir")
    args = ap.parse_args(argv)

    stride = args.stride if args.stride is not None else args.patch
    if stride < args.patch:
        print(f"[extract] refusing stride {stride} < patch {args.patch}: patches "
              f"would overlap and leak pixels across the corpus", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else PROJECT_ROOT / f"data/patches_{args.name}.npy"
    out_path = out_path if out_path.is_absolute() else PROJECT_ROOT / out_path

    full_dir = None
    if args.keep_full_images:
        full_dir = Path(args.keep_full_images)
        full_dir = full_dir if full_dir.is_absolute() else PROJECT_ROOT / full_dir
        full_dir.mkdir(parents=True, exist_ok=True)

    pools: list[np.ndarray] = []
    records: list[dict] = []

    for ident, raw in iter_source_blobs(args.source_type, input_path):
        im = decode(raw, args.min_side, args.colour)
        if im is None:
            continue
        img_idx = len(records)
        if full_dir is not None:
            Image.fromarray(im).save(full_dir / f"img{img_idx:03d}.png")
        patches, coords = keep_textured(*dice(im, args.patch, stride), args.min_std)
        if patches.shape[0] == 0:
            continue
        if patches.shape[0] > args.max_per_image:
            sel = rng.choice(patches.shape[0], args.max_per_image, replace=False)
            patches, coords = patches[sel], coords[sel]
        pools.append(patches)
        records.append({
            "source_image": img_idx,
            "identifier": ident,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "image_hw": list(im.shape[:2]),
            "mode": "RGB" if args.colour else "L",
            "n_patches": int(patches.shape[0]),
            "coords": coords.tolist(),
        })

    if not pools:
        print("[extract] no patches extracted", file=sys.stderr)
        return 1

    stack = np.concatenate(pools, axis=0)
    # per-patch cluster label: which source image each patch came from.

    cluster = np.concatenate([np.full(r["n_patches"], r["source_image"])
                              for r in records])

    if stack.shape[0] > args.n_target:
        sel = np.sort(rng.choice(stack.shape[0], args.n_target, replace=False))
        stack, cluster = stack[sel], cluster[sel]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, stack)
    np.save(out_path.with_name(out_path.stem + "_cluster.npy"), cluster)

    kept_ids, counts = np.unique(cluster, return_counts=True)
    manifest = {
        "corpus": args.name,
        "input": str(input_path),
        "protocol": {
            "patch": args.patch, "stride": stride, "overlapping": stride < args.patch,
            "min_side": args.min_side, "min_std": args.min_std,
            "max_per_image": args.max_per_image, "seed": args.seed,
            "colour": args.colour,
        },
        "n_patches": int(stack.shape[0]),
        "n_source_images": int(kept_ids.size),
        "patches_per_source_image": {int(k): int(v) for k, v in zip(kept_ids, counts)},
        "largest_cluster_share": float(counts.max() / counts.sum()),
        "images": records,
        "out": str(out_path),
    }
    out_path.with_name(out_path.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[extract] {args.name}: {stack.shape} from {kept_ids.size} source images "
          f"(largest cluster {manifest['largest_cluster_share']:.1%}) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
