"""scripts/campaign_processes.py finds the campaign by its scripts, so stopping it
reaches the Python processes below Git-bash subshells, whose Windows parent
links break, and the wrapper's own check does not count the wrapper itself."""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "scripts" / "campaign_processes.py"
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"


def _mod():
    spec = importlib.util.spec_from_file_location("campaign_processes", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("cmdline, expected", [
    (["bash.exe", "scripts/run_campaign_detached.sh", "sweeps", "lama"], True),
    (["C:\\Program Files\\Git\\usr\\bin\\bash.exe", "results\\campaign_snapshot.sh", "sweeps"], True),
    (["bash", "--norc", "results/campaign_snapshot.sh"], True),
    # the login shell that starts the wrapper, and any shell that only mentions it
    (["bash.exe", "-lc", "bash '/b/paper2_recon/scripts/run_campaign_detached.sh' sweeps"], False),
    (["bash.exe", "-c", "grep -n x scripts/run_campaign_detached.sh"], False),
    (["bash.exe", "scripts/run_round2_campaign.sh"], False),
    (["bash.exe"], False),
    ([], False),
])
def test_a_process_counts_when_it_runs_one_of_the_scripts(cmdline, expected):
    assert _mod().runs_script(cmdline, ("run_campaign_detached.sh", "campaign_snapshot.sh")) is expected


def _wait_for(path: Path, timeout=60.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.read_text().strip():
            return path.read_text()
        time.sleep(0.2)
    raise AssertionError(f"{path} never appeared")


def _scripts(tmp_path: Path):
    """A fake wrapper that starts a fake campaign script with bash, which runs a
    Python process in a subshell, which starts a Python child: the shape of the
    real wrapper -> snapshot -> sweep -> workers tree."""
    tag = uuid.uuid4().hex[:8]
    wrapper, snapshot = f"fake_wrapper_{tag}.sh", f"fake_snapshot_{tag}.sh"
    pids = tmp_path / "pids.txt"
    sweep = tmp_path / "sweep.py"
    sweep.write_text(
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open(r'{pids}', 'w').write(f'{{__import__(\"os\").getpid()}} {{c.pid}}')\n"
        "time.sleep(600)\n", encoding="utf-8")
    py = Path(sys.executable).as_posix()
    (tmp_path / snapshot).write_text(f'( "{py}" "{sweep.as_posix()}" )\nwait\n', encoding="utf-8", newline="\n")
    (tmp_path / wrapper).write_text(f'bash "{snapshot}"\n', encoding="utf-8", newline="\n")
    return (wrapper, snapshot), pids


# Children get no handle of pytest's: an inherited stdin can be invalid under
# pytest's capture on Windows, and an orphan holding stdout keeps the run open.
QUIET = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(TOOL), *args], cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)


def test_stop_reaches_the_python_processes_below_the_bash_subshells(tmp_path):
    names, pids = _scripts(tmp_path)
    top = subprocess.Popen([BASH, names[0]], cwd=tmp_path, **QUIET)
    try:
        sweep, worker = (int(x) for x in _wait_for(pids).split())
        status = _cli("status", "--scripts", *names)
        assert status.returncode == 0, status
        assert str(sweep) in status.stdout and str(worker) in status.stdout

        stop = _cli("stop", "--scripts", *names)
        assert stop.returncode == 0, stop
        for pid in (sweep, worker):
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        assert _cli("status", "--scripts", *names).returncode == 1
    finally:
        if top.poll() is None:
            top.kill()
        for p in psutil.process_iter(["cmdline"]):
            if any(n in " ".join(p.info["cmdline"] or []) for n in names) or str(tmp_path) in " ".join(p.info["cmdline"] or []):
                p.kill()


def test_a_campaign_script_checking_for_another_campaign_does_not_count_itself(tmp_path):
    tag = uuid.uuid4().hex[:8]
    me, other = f"fake_wrapper_{tag}.sh", f"fake_snapshot_{tag}.sh"
    result = tmp_path / "status.txt"
    py, tool = Path(sys.executable).as_posix(), TOOL.as_posix()
    (tmp_path / me).write_text(
        f'"{py}" "{tool}" status --scripts {me} {other} > /dev/null\necho $? > status.txt\n',
        encoding="utf-8", newline="\n")
    (tmp_path / other).write_text("sleep 600\n", encoding="utf-8", newline="\n")

    subprocess.run([BASH, me], cwd=tmp_path, timeout=120, **QUIET)
    assert result.read_text().strip() == "1"                 # alone: nothing else is running

    running = subprocess.Popen([BASH, other], cwd=tmp_path, **QUIET)
    try:
        time.sleep(1.0)
        subprocess.run([BASH, me], cwd=tmp_path, timeout=120, **QUIET)
        assert result.read_text().strip() == "0"             # the other campaign is seen
    finally:
        if running.poll() is None:
            running.kill()
        _cli("stop", "--scripts", other)
