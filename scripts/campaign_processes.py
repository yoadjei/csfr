"""Find or stop the round-2 campaign's processes.

The campaign is a tree of Git-bash and Python processes, but Git bash starts a
script through a short-lived process that exits, so the Windows parent links
inside the tree break: stopping the wrapper's PID with its tree (taskkill /T)
leaves the campaign script and the sweep it started running. The campaign is
found instead by its scripts: every bash process running the wrapper, the
campaign snapshot or the campaign script itself, and everything below each of them (Python's own parent links
are intact). The caller and its ancestors are never counted, so the wrapper can
ask whether another campaign is running.

  python scripts/campaign_processes.py status   # list them; exit 0 if any runs, 1 if none
  python scripts/campaign_processes.py stop     # stop them; exit 0 once none is left
"""
from __future__ import annotations

import argparse
import re
import sys

import psutil

# the wrapper, the snapshot it runs, and the campaign script run directly (a
# stage started outside the wrapper writes the same outputs)
SCRIPTS = ("run_campaign_detached.sh", "campaign_snapshot.sh", "run_round2_campaign.sh")


def runs_script(cmdline: list[str], scripts) -> bool:
    """True when a bash command line runs one of the scripts as its script file,
    not when a -c command only mentions one."""
    if not cmdline or not re.split(r"[\\/]", cmdline[0])[-1].lower().startswith("bash"):
        return False
    for arg in cmdline[1:]:
        if arg.startswith("-"):
            if "c" in arg.lstrip("-") and not arg.startswith("--"):
                return False                    # bash -c / -lc: the next argument is a command string
            continue
        return re.split(r"[\\/]", arg)[-1] in scripts
    return False


def _cmdline(p: psutil.Process) -> list[str]:
    try:
        return p.cmdline()
    except psutil.Error:
        return []


def find(scripts=SCRIPTS) -> tuple[list[psutil.Process], list[psutil.Process]]:
    """(the processes running the scripts, the processes below them)."""
    caller = psutil.Process()
    exclude = {caller.pid} | {a.pid for a in caller.parents()}
    tops = [p for p in psutil.process_iter() if p.pid not in exclude and runs_script(_cmdline(p), scripts)]
    below: dict[int, psutil.Process] = {}
    for p in tops:
        try:
            below.update((c.pid, c) for c in p.children(recursive=True))
        except psutil.Error:
            pass
    top_pids = {p.pid for p in tops}
    return tops, [p for pid, p in below.items() if pid not in top_pids and pid not in exclude]


def _kill(procs: list[psutil.Process], timeout: float = 30.0) -> list[psutil.Process]:
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    return alive


def stop(scripts=SCRIPTS) -> list[psutil.Process]:
    """Stop the campaign; returns the processes still alive (empty on success)."""
    tops, below = find(scripts)
    # the scripts first, so none starts its next step when a child dies; then
    # the rest, with any child they started since they were found
    alive = _kill(tops)
    rest: dict[int, psutil.Process] = {}
    for p in below:
        try:
            rest[p.pid] = p
            rest.update((c.pid, c) for c in p.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    return alive + _kill(list(rest.values()))


def _describe(p: psutil.Process) -> str:
    return f"{p.pid} {' '.join(_cmdline(p))[:160]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["status", "stop"])
    ap.add_argument("--scripts", nargs="+", default=list(SCRIPTS), help="script file names that mark the campaign")
    args = ap.parse_args(argv)
    if args.action == "status":
        tops, below = find(args.scripts)
        for p in tops + below:
            print(_describe(p))
        return 0 if tops or below else 1
    tops, below = find(args.scripts)
    if not tops and not below:
        print("no campaign process running")
        return 0
    alive = stop(args.scripts)
    for p in alive:
        print("still running:", _describe(p))
    print(f"stopped {len(tops) + len(below) - len(alive)} campaign process(es)" if not alive else "")
    return 1 if alive else 0


if __name__ == "__main__":
    sys.exit(main())
