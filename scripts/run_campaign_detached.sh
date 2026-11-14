#!/usr/bin/env bash
# Run scripts/run_round2_campaign.sh as a process that outlives the shell or
# agent session that started it. Start it through WMI so it has no parent
# console or job object to die with, for example from PowerShell:
#
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#     CommandLine = '"C:\Program Files\Git\bin\bash.exe" -lc "cd /b/paper2_recon && bash scripts/run_campaign_detached.sh sweeps lama"';
#     CurrentDirectory = 'B:\paper2_recon' }
#
# The campaign runs from a copy of the campaign script (bash reads a script as
# it runs, so editing scripts/run_round2_campaign.sh mid-run would change what
# the running copy does). Everything goes to results/campaign.log.
#
#   cat results/campaign.pid                         # Windows PID of this wrapper
#   python scripts/campaign_processes.py status      # every process of the campaign
#   python scripts/campaign_processes.py stop        # stop all of them
#
# Not taskkill /T on the PID: Git bash breaks the Windows parent links inside the
# tree, so that leaves the campaign script and the sweep running.
#
# A second start while any campaign process runs exits 3 without touching
# anything, and lists that campaign's processes in the log.
set -uo pipefail
cd "$(dirname "$0")/.."
PIDFILE=results/campaign.pid
LOG=results/campaign.log
SNAP=results/campaign_snapshot.sh
export PATH="/c/Users/adjei/AppData/Local/Python/bin:$PATH"
unset VIRTUAL_ENV

# called directly, not in $(...), so this wrapper is among the caller's ancestors
# and is not counted
if python scripts/campaign_processes.py status >> "$LOG" 2>&1; then
  echo "[detached] $(date '+%F %T') not started: a campaign is already running (processes above)" >> "$LOG"
  exit 3
fi
cat /proc/$$/winpid > "$PIDFILE"
cp scripts/run_round2_campaign.sh "$SNAP"
echo "[detached] $(date '+%F %T') start, PID $(cat "$PIDFILE"), commit $(git rev-parse --short HEAD), stages: $*" >> "$LOG"
bash "$SNAP" "$@" >> "$LOG" 2>&1
status=$?
echo "[detached] $(date '+%F %T') end, exit $status" >> "$LOG"
rm -f "$PIDFILE"
exit $status
