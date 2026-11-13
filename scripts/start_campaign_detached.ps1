# Start the round-2 campaign as a process that outlives the terminal or agent
# session that started it (see scripts/run_campaign_detached.sh). A copy that is
# already running is stopped first; every campaign step resumes from its saved
# outputs, so the stop costs at most the unit of work it was on.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_campaign_detached.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_campaign_detached.ps1 -Stages "analysis"
param([string]$Stages = "sweeps lama analysis forensic downstream idle")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$pidfile = Join-Path $repo "results\campaign.pid"

if (Test-Path $pidfile) {
    $old = [int](Get-Content $pidfile -Raw).Trim()
    $proc = Get-Process -Id $old -ErrorAction SilentlyContinue
    # Only a bash process is the wrapper; anything else means the PID was reused.
    if ($proc -and $proc.ProcessName -eq "bash") {
        & taskkill.exe /PID $old /T /F | Out-Null
        Write-Output "stopped the running campaign (PID $old)"
    }
    Remove-Item $pidfile
}

$wrapper = (Join-Path $repo "scripts\run_campaign_detached.sh").Replace("\", "/")
$cmd = '"C:\Program Files\Git\bin\bash.exe" -lc "bash ''' + $wrapper + ''' ' + $Stages + '"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $repo; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create returned $($r.ReturnValue)" }
Write-Output "started the campaign detached (launcher PID $($r.ProcessId)); log: results\campaign.log, PID file: results\campaign.pid"
