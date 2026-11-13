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
#   cat results/campaign.pid                                  # Windows PID of this wrapper
#   taskkill //PID "$(cat results/campaign.pid)" //T //F     # stop it and every child
#
# A second start while one runs exits 3 without touching anything.
set -uo pipefail
cd "$(dirname "$0")/.."
PIDFILE=results/campaign.pid
LOG=results/campaign.log
SNAP=results/campaign_snapshot.sh
TASKLIST=/c/Windows/System32/tasklist.exe

if [ -f "$PIDFILE" ] && "$TASKLIST" //FI "PID eq $(cat "$PIDFILE")" //NH 2>/dev/null | grep -qi bash; then
  echo "[detached] $(date '+%F %T') not started: a campaign is already running (PID $(cat "$PIDFILE"))" >> "$LOG"
  exit 3
fi
cat /proc/$$/winpid > "$PIDFILE"

export PATH="/c/Users/adjei/AppData/Local/Python/bin:$PATH"
unset VIRTUAL_ENV
cp scripts/run_round2_campaign.sh "$SNAP"
echo "[detached] $(date '+%F %T') start, PID $(cat "$PIDFILE"), commit $(git rev-parse --short HEAD), stages: $*" >> "$LOG"
bash "$SNAP" "$@" >> "$LOG" 2>&1
status=$?
echo "[detached] $(date '+%F %T') end, exit $status" >> "$LOG"
rm -f "$PIDFILE"
exit $status
