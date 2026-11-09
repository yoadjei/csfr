"""Fetch and assemble the round-2 corpora.

Subcommands:
  govdocs1             download GovDocs1 zip archives (Digital Corpora, public)
  kodak                download the 24 Kodak Lossless True Color images
  jpeg-corruption-set  select baseline-sequential JPEGs for the byte-corruption
                       experiment from the GovDocs1 archives, plus the Nick Mikus
                       carved JPEGs; copies raw bytes and writes a manifest

Usage:
    python scripts/fetch_corpora.py govdocs1 --archives 003-009
    python scripts/fetch_corpora.py kodak
    python scripts/fetch_corpora.py jpeg-corruption-set --n 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

GOVDOCS_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/{:03d}.zip"
GOVDOCS_DIR = Path("B:/rsencarver/Datasets/GovDocs1/zipfiles")
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
MIKUS_DIR = Path("B:/rsencarver/Datasets/NickMikus")
USER_AGENT = "csfr-reproducibility-fetch/1.0"


def _download(url: str, dest: Path) -> int:
    if dest.exists():
        return dest.stat().st_size
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    part.replace(dest)
    return dest.stat().st_size


def _range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def cmd_govdocs1(args) -> int:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for n in _range(args.archives):
        size = _download(GOVDOCS_URL.format(n), dest / f"{n:03d}.zip")
        print(f"[govdocs1] {n:03d}.zip {size / 2**20:.0f} MB", flush=True)
    return 0


def cmd_kodak(args) -> int:
    from PIL import Image
    dest = ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    for n in range(1, 25):
        p = dest / f"kodim{n:02d}.png"
        _download(KODAK_URL.format(n), p)
        with Image.open(p) as im:
            if im.mode != "RGB" or sorted(im.size) != [512, 768]:
                raise RuntimeError(f"unexpected Kodak image {p.name}: {im.mode} {im.size}")
    print(f"[kodak] 24 images in {dest}")
    return 0


def _jpeg_kind(raw: bytes) -> str:
    """'baseline', 'progressive' or 'other' from the first SOF marker."""
    i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            return "other"
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(raw[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1):
            return "baseline"
        if marker == 0xC2:
            return "progressive"
        if marker == 0xDA:
            return "other"
        i += 2 + seg
    return "other"


def cmd_jpeg_corruption_set(args) -> int:
    from PIL import Image
    from extract_patches import carve_jpegs
    dest = ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for arc in sorted(Path(args.govdocs).glob("*.zip")):
        with zipfile.ZipFile(arc) as z:
            for name in sorted(z.namelist()):
                if len(records) >= args.n:
                    break
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                raw = z.read(name)
                if _jpeg_kind(raw) != "baseline":
                    continue
                try:
                    with Image.open(BytesIO(raw)) as im:
                        im.load()
                        w, h = im.size
                        mode = im.mode
                except Exception:
                    continue
                if min(w, h) < args.min_side or max(w, h) > args.max_side or mode not in ("L", "RGB"):
                    continue
                sha = hashlib.sha256(raw).hexdigest()
                fn = f"govdocs_{len(records):03d}.jpg"
                (dest / fn).write_bytes(raw)
                records.append({"file": fn, "source": f"{arc.name}:{name}", "sha256": sha,
                                "bytes": len(raw), "width": w, "height": h, "mode": mode,
                                "corpus": "govdocs1"})
        if len(records) >= args.n:
            break
    for disk in ["11-carve-fat", "12-carve-ext2"]:
        dd = sorted((MIKUS_DIR / disk).glob("*.dd"))
        if not dd:
            continue
        kept = 0
        for j, raw in enumerate(carve_jpegs(dd[0].read_bytes())):
            if kept >= 3:
                break
            try:
                with Image.open(BytesIO(raw)) as im:
                    im.load()
                    w, h = im.size
                    mode = im.mode
            except Exception:
                continue
            if min(w, h) < 96 or _jpeg_kind(raw) != "baseline":
                continue
            fn = f"mikus_{disk}_{kept}.jpg"
            (dest / fn).write_bytes(raw)
            records.append({"file": fn, "source": f"{dd[0].name}#carved{j:04d}",
                            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "width": w, "height": h, "mode": mode, "corpus": "nick_mikus_dftt"})
            kept += 1
    manifest = {"selection": {"govdocs_n": args.n, "min_side": args.min_side,
                              "max_side": args.max_side, "jpeg": "baseline sequential (SOF0/SOF1)",
                              "order": "archive then member name, sorted"},
                "files": records}
    (ROOT / "data/jpeg_corruption").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/jpeg_corruption/manifest.json").write_text(json.dumps(manifest, indent=2),
                                                            encoding="utf-8")
    print(f"[jpeg-set] {len(records)} files -> {dest}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("govdocs1")
    g.add_argument("--archives", default="003-009")
    g.add_argument("--dest", default=str(GOVDOCS_DIR))
    k = sub.add_parser("kodak")
    k.add_argument("--dest", default="data/kodak")
    j = sub.add_parser("jpeg-corruption-set")
    j.add_argument("--n", type=int, default=100)
    j.add_argument("--govdocs", default=str(GOVDOCS_DIR))
    j.add_argument("--dest", default="data/jpeg_corruption/src")
    j.add_argument("--min-side", type=int, default=256)
    j.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args(argv)
    return {"govdocs1": cmd_govdocs1, "kodak": cmd_kodak,
            "jpeg-corruption-set": cmd_jpeg_corruption_set}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
