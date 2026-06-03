# Registers a Windows scheduled task to start the paper worker at user logon.
# Requires no administrator rights when installing for the current user.
param(
    [int]$LogonDelaySeconds = 60
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunScript = Join-Path $RepoRoot "scripts\run_paper_worker_background.ps1"
if (-not (Test-Path -LiteralPath $RunScript)) {
    throw "Missing script: $RunScript"
}

$TaskName = "InvestmentBinancePaperWorker"
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunScript`"" `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Trigger.Delay = "PT${LogonDelaySeconds}S"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew
$Settings.ExecutionTimeLimit = "PT0S"

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Local Binance paper bot worker (read-only public market data)." `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "  Trigger: at logon for $env:USERNAME (delay ${LogonDelaySeconds}s)"
Write-Host "  Script:  $RunScript"
Write-Host "  Log:     $RepoRoot\data\worker.log"
Write-Host ""
Write-Host "To start now without rebooting:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To remove:"
Write-Host "  .\scripts\unregister_paper_worker_startup.ps1"
