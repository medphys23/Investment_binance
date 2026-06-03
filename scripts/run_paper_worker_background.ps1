# Runs the paper worker in the background with log file output.
# Used by Task Scheduler at logon; safe to run manually as well.
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$DataDir = Join-Path $RepoRoot "data"
if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

$LogPath = Join-Path $DataDir "worker.log"
$Python = (Get-Command python -ErrorAction Stop).Source

$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*-m src.paper_worker*' }
if ($existing) {
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -LiteralPath $LogPath -Value "[$stamp] Skipped start: paper worker already running (PID $($existing[0].ProcessId))."
    exit 0
}

$stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -LiteralPath $LogPath -Value "[$stamp] Starting paper worker ($Python)."

& $Python -m src.paper_worker *>> $LogPath

$exitCode = $LASTEXITCODE
$stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -LiteralPath $LogPath -Value "[$stamp] Paper worker exited with code $exitCode."
exit $exitCode
