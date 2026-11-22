"""OCR task on GovDocs1 text crops (E13): does a reconstruction keep text readable?

Four stages; EasyOCR lives in .venv-ocr (scripts/setup_ocr_venv.sh), the
reconstruction methods in the main environment, LaMa in .venv-lama:

  .venv-ocr/Scripts/python -I scripts/run_ocr_task.py crops       # build the crop set
  python scripts/run_ocr_task.py reconstruct                      # every method
  .venv-lama/Scripts/python scripts/run_lama_cells.py --root results/ocr/cases
  .venv-ocr/Scripts/python scripts/run_ocr_task.py recognise      # OCR every output
  python scripts/run_ocr_task.py score                            # CER and summary

crops: EasyOCR detection on the clean greyscale decode of each GovDocs1 file of
the JPEG-corruption set (files with fewer than 5 detections skipped); detections
with confidence >= 0.8 and box height 24 to 80 px, at most 3 per document, up to
300 crops; each box resized to height 64 keeping its aspect and padded to
64 x 256 with its median grey (wider boxes are scaled down to width 256). The
reference string is EasyOCR's recognition of the clean 64 x 256 crop, so the
reference and the evaluation share the input domain. Writes data/ocr/crops.npy
(gitignored) and data/ocr/manifest.json.

reconstruct: CF1 to CF4 at L1, L3 and L5, seeds 0 to 2 (masks as in the sweeps),
every method; outputs under results/ocr/cases/<cell>/seed<s>/.

recognise: EasyOCR recognition of every saved output and of the corrupted
inputs (as the no-reconstruction floor); texts.json per case directory.

EasyOCR runs on the CPU unless crops and recognise are given --gpu (the Kaggle
downstream run); data/ocr/manifest.json and results/ocr/recognise_env.json
record which, with the versions.

score: character error rate (Levenshtein distance / reference length) and exact
match per crop; per family, level and method the means with 95 percent bootstrap
intervals over crops, and the paired CSFR-minus-best-classical and
CSFR-minus-LaMa CER differences with Wilcoxon p (results/ocr/{per_crop.csv,
summary.csv}).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CROP_H, CROP_W = 64, 256
CELLS = [f"CF{f}_{l}" for f in range(1, 5) for l in ("L1", "L3", "L5")]
SEEDS = [0, 1, 2]
METHODS = ["csfr", "csfr_tv_only", "bilinear", "inpaint_telea", "inpaint_ns", "dictlearn",
           "biharmonic", "guleryuz", "fsr", "shiftmap", "criminisi"]
EXTERNAL = ["lama"]


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions, deletions, substitutions of unit cost)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(len(ref), 1)


def fit_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop rows r0:r1, cols c0:c1, resize to height 64 keeping the aspect
    (width capped at 256), pad to 64 x 256 with the crop's median grey."""
    from PIL import Image
    r0, r1, c0, c1 = box
    patch = img[r0:r1, c0:c1]
    h, w = patch.shape
    nw = max(1, min(CROP_W, int(round(w * CROP_H / h))))
    im = Image.fromarray(patch.astype(np.uint8)).resize((nw, CROP_H), Image.LANCZOS)
    out = np.full((CROP_H, CROP_W), float(np.median(patch)), np.float32)
    out[:, :nw] = np.asarray(im, np.float32)
    return out


def _reader(gpu: bool = False):
    import easyocr
    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def _reader_env(gpu: bool) -> dict:
    import platform

    import easyocr
    import torch
    return {"gpu": gpu, "easyocr": getattr(easyocr, "__version__", "unknown"), "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform()}


def _recognise(reader, crop: np.ndarray) -> str:
    txt = reader.recognize(np.clip(crop, 0, 255).astype(np.uint8), horizontal_list=[[0, CROP_W, 0, CROP_H]],
                           free_list=[], detail=0)
    return " ".join(txt).strip()


def cmd_crops(args) -> int:
    from PIL import Image
    manifest = json.loads((ROOT / "data/jpeg_corruption/manifest.json").read_text())["files"]
    reader = _reader(args.gpu)
    crops, records = [], []
    for rec in manifest:
        if rec["corpus"] != "govdocs1" or len(crops) >= args.target:
            continue
        with Image.open(ROOT / "data/jpeg_corruption/src" / rec["file"]) as im:
            img = np.asarray(im.convert("L"), np.float32)
        dets = reader.readtext(img.astype(np.uint8))
        if len(dets) < 5:
            continue
        kept = 0
        for bbox, _, conf in sorted(dets, key=lambda d: -d[2]):
            xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
            r0, r1, c0, c1 = int(min(ys)), int(max(ys)), int(min(xs)), int(max(xs))
            if conf < 0.8 or not 24 <= r1 - r0 <= 80 or c1 <= c0:
                continue
            crop = fit_crop(img, (max(r0, 0), r1, max(c0, 0), c1))
            ref = _recognise(reader, crop)
            if not ref:
                continue
            crops.append(crop)
            records.append({"crop": len(crops) - 1, "file": rec["file"], "box_rows": [r0, r1],
                            "box_cols": [c0, c1], "detection_conf": float(conf), "reference": ref})
            kept += 1
            if kept >= 3 or len(crops) >= args.target:
                break
    out = ROOT / "data/ocr"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "crops.npy", np.round(np.stack(crops)).astype(np.uint8))
    (out / "manifest.json").write_text(json.dumps(
        {"selection": {"conf_min": 0.8, "height_px": [24, 80], "per_document_max": 3, "target": args.target,
                       "crop": [CROP_H, CROP_W], "reference": "EasyOCR recognition of the clean crop"},
         "environment": _reader_env(args.gpu), "crops": records}, indent=2), encoding="utf-8")
    print(f"[ocr] {len(crops)} crops from {len({r['file'] for r in records})} documents")
    return 0


def cmd_reconstruct(args) -> int:
    import importlib.util
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from csfr_solver import VARIANTS, solve
    from masks import SEVERITY, make_mask
    spec = importlib.util.spec_from_file_location("_sweep_v2", ROOT / "scripts" / "run_sweep_v2.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    from checkpoint import replace_with_retry
    torch.set_num_threads(args.threads)
    X = np.load(ROOT / "data/ocr/crops.npy").astype(np.float32)
    pool = sweep._make_pool(args.workers)
    try:
        for cell in CELLS:
            fam, lvl = cell.split("_")
            for seed in SEEDS:
                d = ROOT / "results/ocr/cases" / cell / f"seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                M = make_mask(X.shape, fam, SEVERITY[lvl], np.random.default_rng(seed))
                y = X * M
                np.save(d / "y.npy", y)
                np.save(d / "mask.npy", M)
                for m in METHODS:
                    f = d / f"x_hat_{m}.npy"
                    if f.exists():
                        continue
                    # the output marks the method done, so it appears only when written in full
                    part = d / f".partial_{f.name}"
                    if m in VARIANTS:
                        x, _ = solve(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(M), VARIANTS[m])
                        sweep.save_output(part, x.numpy(), sweep.CSFR_STORE_MAX_ERR)
                    else:
                        x, _ = sweep.run_baseline(m, y, M, seed, workers=args.workers, pool=pool)
                        sweep.save_output(part, x)
                    replace_with_retry(part, f)
                print(f"[ocr] reconstructed {cell} seed {seed}", flush=True)
    finally:
        pool.shutdown()
    return 0


def cmd_recognise(args) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from checkpoint import write_text_atomic
    reader = _reader(args.gpu)
    out = ROOT / "results/ocr"
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "recognise_env.json", json.dumps(_reader_env(args.gpu), indent=2))
    for d in sorted((ROOT / "results/ocr/cases").glob("*/seed*")):
        tpath = d / "texts.json"
        texts = json.loads(tpath.read_text()) if tpath.exists() else {}
        for f in sorted(d.glob("x_hat_*.npy")) + [d / "y.npy"]:
            name = "corrupted_input" if f.name == "y.npy" else f.stem[len("x_hat_"):]
            if name in texts:
                continue
            arr = np.load(f).astype(np.float32)
            texts[name] = [_recognise(reader, a) for a in arr]
            write_text_atomic(tpath, json.dumps(texts))
        print(f"[ocr] recognised {d.parent.name}/{d.name}", flush=True)
    return 0


def cmd_score(args) -> int:
    from scipy import stats
    manifest = json.loads((ROOT / "data/ocr/manifest.json").read_text())["crops"]
    refs = [c["reference"] for c in manifest]
    rows = []
    for d in sorted((ROOT / "results/ocr/cases").glob("*/seed*")):
        tpath = d / "texts.json"
        if not tpath.exists():
            continue
        fam, lvl = d.parent.name.split("_")
        for name, hyps in json.loads(tpath.read_text()).items():
            for i, (r, h) in enumerate(zip(refs, hyps)):
                rows.append({"family": fam, "level": lvl, "seed": int(d.name[4:]), "method": name, "crop": i,
                             "cer": cer(r, h), "exact": int(r == h)})
    out = ROOT / "results/ocr"
    with (out / "per_crop.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    classical = [m for m in METHODS if not m.startswith("csfr")]
    for fam, lvl in sorted({(r["family"], r["level"]) for r in rows}):
        g = [r for r in rows if (r["family"], r["level"]) == (fam, lvl)]
        per: dict[str, np.ndarray] = {}
        for m in sorted({r["method"] for r in g}):
            mat = {}
            for r in g:
                if r["method"] == m:
                    mat.setdefault(r["crop"], []).append(r["cer"])
            per[m] = np.array([np.mean(v) for _, v in sorted(mat.items())])     # seed-averaged per crop
        best = min((m for m in per if m in classical), key=lambda m: per[m].mean(), default=None)
        for m, v in per.items():
            rng = np.random.default_rng([2026, len(summary)])
            boots = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
            row = {"family": fam, "level": lvl, "method": m, "n_crops": len(v), "cer_mean": float(v.mean()),
                   "cer_ci_lo": float(np.quantile(boots, 0.025)), "cer_ci_hi": float(np.quantile(boots, 0.975)),
                   "exact_rate": float(np.mean([r["exact"] for r in g if r["method"] == m]))}
            if m == "csfr":
                for other, label in ((best, "best_classical"), ("lama", "lama")):
                    if other in per and len(per[other]) == len(v):
                        diff = v - per[other]
                        p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
                        row.update({f"vs_{label}": other, f"cer_diff_vs_{label}": float(diff.mean()),
                                    f"p_wilcoxon_vs_{label}": p})
            summary.append(row)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        w.writeheader()
        w.writerows(summary)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--target", type=int, default=300)
    c.add_argument("--gpu", action="store_true", help="EasyOCR on the GPU")
    r = sub.add_parser("reconstruct")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--threads", type=int, default=8)
    g = sub.add_parser("recognise")
    g.add_argument("--gpu", action="store_true", help="EasyOCR on the GPU")
    sub.add_parser("score")
    args = ap.parse_args(argv)
    return {"crops": cmd_crops, "reconstruct": cmd_reconstruct, "recognise": cmd_recognise,
            "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
