#!/usr/bin/env bash
# Isolated environment for the OCR task (E13). EasyOCR depends on
# opencv-python-headless, which would shadow the opencv-contrib build that the
# main environment needs for FSR and shift-map, so it lives in .venv-ocr.
# Its detection and recognition models download on first use into ~/.EasyOCR.
# Override the interpreter with OCR_PYTHON=/path/to/python if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${OCR_PYTHON:-python}"
"$PY" -m venv .venv-ocr
VPY=".venv-ocr/Scripts/python.exe"
[ -x "$VPY" ] || VPY=".venv-ocr/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q easyocr==1.7.2 PyYAML
"$VPY" - <<'PYEOF'
import numpy as np
import easyocr
r = easyocr.Reader(["en"], gpu=False, verbose=False)
img = np.full((64, 256), 255, np.uint8)
print("easyocr ok", r.recognize(img, detail=0))
PYEOF
