"""Byte-level JPEG corruption (E9): scan location, the three corruption kinds,
tolerant decoding, and the three missing-pixel masks a carver could build."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jpeg_corruption import (KINDS, corrupt, decode_tolerant, decoder_mask, locate_scan,  # noqa: E402
                             mcu_height, offset_mask, oracle_mask)


def _jpeg(mode="L", side=256, subsampling=None, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    g = (128 + 60 * np.sin(xx / 7.0) + 40 * np.cos(yy / 5.0) + rng.normal(0, 8, (side, side)))
    g = g.clip(0, 255).astype(np.uint8)
    arr = g if mode == "L" else np.stack([g, g[::-1], g.T], -1)
    buf = io.BytesIO()
    kw = {} if subsampling is None else {"subsampling": subsampling}
    Image.fromarray(arr).save(buf, "JPEG", quality=85, **kw)
    return buf.getvalue()


def test_locate_scan():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    assert 0 < s0 < s1 == len(b) - 2
    assert b[s1:s1 + 2] == b"\xff\xd9"


def test_mcu_height_follows_chroma_subsampling():
    assert mcu_height(_jpeg("L")) == 8
    assert mcu_height(_jpeg("RGB", subsampling=0)) == 8      # 4:4:4
    assert mcu_height(_jpeg("RGB", subsampling=2)) == 16     # 4:2:0


@pytest.mark.parametrize("kind", KINDS)
def test_corrupt_changes_only_the_scan_and_decodes(kind):
    b = _jpeg()
    scan = locate_scan(b)
    out, off = corrupt(b, kind, np.random.default_rng(1), scan)
    assert scan[0] <= off < scan[1]
    assert out[:scan[0]] == b[:scan[0]]                       # headers untouched
    assert out != b and out.endswith(b"\xff\xd9")
    img, used = decode_tolerant(out)
    assert img.shape == (256, 256) and img.dtype == np.float32 and used in ("pil", "cv2", "grey")


def test_truncate_keeps_the_stated_fraction():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    out, off = corrupt(b, "truncate", np.random.default_rng(0), (s0, s1))
    frac = (off - s0) / (s1 - s0)
    assert 0.3 <= frac <= 0.9 and len(out) == off + 2


def test_masks_for_a_half_truncation():
    b = _jpeg()
    s0, s1 = locate_scan(b)
    cut = s0 + (s1 - s0) // 2
    t = b[:cut] + b"\xff\xd9"
    clean, _ = decode_tolerant(b)
    dec, _ = decode_tolerant(t)
    Mo = oracle_mask(clean, dec)
    lost = 1.0 - Mo.mean()
    assert 0.3 <= lost <= 0.7                                  # roughly the lower half
    Md = decoder_mask(dec)
    assert ((Md == 0) <= (Mo == 0)).mean() > 0.99             # grey fill blocks are truly lost
    Mf = offset_mask(clean.shape, s0, s1, cut, mcu_h=8)
    rows_ok = [((Mf[r] == 0) >= (Mo[r] == 0)).all() for r in range(clean.shape[0])]
    assert np.mean(rows_ok) >= 0.9                             # offset mask covers the loss
    for M in (Mo, Md, Mf):
        assert M.dtype == np.float32 and set(np.unique(M)) <= {0.0, 1.0}


def test_masks_handle_sizes_that_are_not_block_multiples():
    clean = np.full((37, 45), 100, np.float32)
    dec = clean.copy()
    dec[30:, :] = 128.0
    Mo = oracle_mask(clean, dec)
    assert Mo.shape == clean.shape and Mo[36, 44] == 0 and Mo[0, 0] == 1
    Md = decoder_mask(dec)
    assert Md[36, 44] == 0


def test_driver_end_to_end(tmp_path):
    import csv
    import importlib.util
    import json

    import yaml
    root = Path(__file__).resolve().parent.parent
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(_jpeg(side=96))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"files": [{"file": "a.jpg", "corpus": "synthetic", "mode": "L"}]}))
    cfg = {"experiment": "t", "manifest": str(tmp_path / "manifest.json"), "src_dir": str(src),
           "output_dir": str(tmp_path / "out"), "kinds": ["truncate"], "repeats": 1, "seed": 0,
           "max_side": 64, "conditions": ["oracle", "decoder", "offset"], "offset_margin_rows": 2,
           "solver": {"max_iter": 3000}, "csfr_variants": ["csfr"],
           "baselines": ["bilinear", "inpaint_ns"], "external": ["lama"]}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    spec = importlib.util.spec_from_file_location("jpeg_drv", root / "scripts" / "run_jpeg_corruption.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(tmp_path / "cfg.yaml"), "--workers", "2", "--bootstrap", "200"]) == 0
    out = tmp_path / "out"
    rows = list(csv.DictReader((out / "per_image.csv").open()))
    done = [r for r in rows if r["status"] == "done"]
    assert {r["method"] for r in done} == {"csfr", "bilinear", "inpaint_ns"}
    assert {r["condition"] for r in done} == {"oracle", "decoder", "offset"}
    assert any(r["method"] == "lama" and r["status"] == "pending" for r in rows)
    case = out / "cases" / "a_truncate_r0" / "oracle"
    assert np.load(case / "y.npy").shape == (1, 64, 64)
    mq = list(csv.DictReader((out / "masks_quality.csv").open()))
    assert len(mq) == 3 and all(float(r["coverage_of_oracle_loss"]) >= 0 for r in mq)
    oracle = next(r for r in mq if r["condition"] == "oracle")
    assert float(oracle["false_pos_frac"]) == 0.0 and float(oracle["false_neg_frac"]) == 0.0
    summ = list(csv.DictReader((out / "summary.csv").open()))
    assert {r["method"] for r in summ} == {"csfr", "bilinear", "inpaint_ns"}
