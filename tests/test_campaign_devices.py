"""scripts/run_round2_campaign.sh runs the GPU-capable downstream steps (ResNet
training, LaMa, EasyOCR) on the CPU, as on the workstation, unless the caller
asks for a device: the Kaggle run sets DOWNSTREAM_DEVICE=cuda and OCR_GPU=1.
The stage runs against stand-in interpreters that only record their arguments."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"


def _stand_in(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "{label} $*" >> "$CALLS"\n', encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _downstream_calls(tmp_path: Path, **env) -> list[str]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "run_round2_campaign.sh", repo / "scripts")
    _stand_in(tmp_path / "bin" / "python", "main")
    _stand_in(repo / ".venv-lama" / "bin" / "python", "lama")
    _stand_in(repo / ".venv-ocr" / "bin" / "python", "ocr")
    calls = tmp_path / "calls.txt"
    run_env = {k: v for k, v in os.environ.items() if k not in ("DOWNSTREAM_DEVICE", "OCR_GPU")}
    run_env.update(env, CALLS=str(calls), PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    r = subprocess.run([BASH, str(repo / "scripts" / "run_round2_campaign.sh"), "downstream"],
                       env=run_env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return [ln for ln in calls.read_text().splitlines() if "import sys" not in ln]


def _one(calls: list[str], *parts: str) -> str:
    hits = [c for c in calls if all(p in c for p in parts)]
    assert len(hits) == 1, (parts, calls)
    return hits[0]


@pytest.mark.parametrize("env, device, gpu", [({}, "cpu", False),
                                              ({"DOWNSTREAM_DEVICE": "cuda", "OCR_GPU": "1"}, "cuda", True)])
def test_the_downstream_stage_passes_the_devices_it_is_given(tmp_path, env, device, gpu):
    calls = _downstream_calls(tmp_path, **env)
    assert _one(calls, "main", "train_resnet_classifier.py").endswith(f"--device {device}")
    for root in ("results/downstream_v2/cases", "results/ocr/cases"):
        assert _one(calls, "lama", "run_lama_cells.py", root).endswith(f"--device {device}")
    for step in ("crops", "recognise"):
        assert _one(calls, "ocr", f"run_ocr_task.py {step}").endswith("--gpu") is gpu
    # the steps without a device switch are unchanged
    assert sum("run_downstream.py" in c for c in calls) == 3
    assert _one(calls, "main", "run_ocr_task.py reconstruct").endswith("reconstruct")
