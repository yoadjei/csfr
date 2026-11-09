"""Run one OpenCV-contrib xphoto inpainting call in a fresh interpreter.

    python src/xphoto_child.py <FLAG_NAME>   < y and M as two .npy streams
                                             > the output as one .npy stream

OpenCV-contrib's shift-map inpainting is not a function of its input alone:
after other code has run in the same process its output changes, and it also
differs between fresh processes. On DFRWS 2006 seed 0, the sweep's in-process
outputs and fresh-process outputs differed in up to 256 of 300 patches of a
cell, by up to 1.5 dB in the cell's mean hole PSNR. baselines.baseline_shiftmap
therefore runs every call through this script, so that no call depends on what
the harness ran before it. Arrays travel as .npy streams, not pickle.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # needed under -I


def main() -> int:
    import cv2
    cv2.setNumThreads(1)
    from baselines import _xphoto
    buf = io.BytesIO(sys.stdin.buffer.read())
    y = np.load(buf, allow_pickle=False)
    M = np.load(buf, allow_pickle=False)
    out = io.BytesIO()
    np.save(out, _xphoto(y, M, sys.argv[1]), allow_pickle=False)
    sys.stdout.buffer.write(out.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
