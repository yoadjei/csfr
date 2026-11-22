"""OCR task helpers: the edit distance and CER, and the crop geometry."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("ocr", ROOT / "scripts" / "run_ocr_task.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_levenshtein_known_values():
    m = _mod()
    assert m.levenshtein("kitten", "sitting") == 3
    assert m.levenshtein("", "abc") == 3 and m.levenshtein("abc", "") == 3
    assert m.levenshtein("flaw", "lawn") == 2
    assert m.levenshtein("same", "same") == 0
    assert m.levenshtein("ab", "ba") == 2               # no transposition operation


def test_cer_is_normalised_by_the_reference():
    m = _mod()
    assert m.cer("abcd", "abcd") == 0.0
    assert m.cer("abcd", "abed") == 0.25
    assert m.cer("", "x") == 1.0


def test_fit_crop_keeps_aspect_and_pads():
    m = _mod()
    img = np.zeros((100, 400), np.float32)
    img[10:42, 20:84] = 200.0                           # a 32 x 64 box
    c = m.fit_crop(img, (10, 42, 20, 84))
    assert c.shape == (64, 256)
    assert np.allclose(c[:, :128], 200.0, atol=1.0)     # height 64 -> width 128
    assert np.allclose(c[:, 128:], 200.0)               # padded with the median grey
    wide = m.fit_crop(np.full((40, 2000), 50, np.float32), (0, 40, 0, 2000))
    assert wide.shape == (64, 256)


def test_the_reader_uses_the_gpu_only_when_asked(tmp_path, monkeypatch):
    # recognise runs over results/ocr/cases; with none there it only builds the reader
    import sys
    import types
    m = _mod()
    made = []
    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(
        __version__="0", Reader=lambda langs, gpu, verbose: made.append(gpu) or object()))
    monkeypatch.setattr(m, "ROOT", tmp_path)
    assert m.main(["recognise"]) == 0
    assert m.main(["recognise", "--gpu"]) == 0
    assert made == [False, True]
    import json
    env = json.loads((tmp_path / "results" / "ocr" / "recognise_env.json").read_text())
    assert env["gpu"] is True and {"python", "platform", "torch", "easyocr"} <= set(env)
