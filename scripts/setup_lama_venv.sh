#!/usr/bin/env bash
# Isolated environment for the external-prior baseline (LaMa via
# simple-lama-inpainting). The package pins numpy<2 and an old Pillow, which
# would break the main environment, so it is installed without its declared
# dependencies into .venv-lama and given current versions of what it imports.
# Override the interpreter with LAMA_PYTHON=/path/to/python if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${LAMA_PYTHON:-python}"
"$PY" -m venv .venv-lama
VPY=".venv-lama/Scripts/python.exe"
[ -x "$VPY" ] || VPY=".venv-lama/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q --no-deps simple-lama-inpainting==0.1.2
"$VPY" -m pip install -q torch numpy pillow PyYAML fire opencv-python-headless
"$VPY" - <<'PYEOF'
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama
m = SimpleLama()
img = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
mask = Image.fromarray(np.zeros((64, 64), np.uint8))
out = m(img, mask)
print("lama ok", np.asarray(out).shape)
PYEOF
