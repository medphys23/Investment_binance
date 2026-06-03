# Removes the paper worker logon scheduled task.
$ErrorActionPreference = "Stop"

$TaskName = "InvestmentBinancePaperWorker"
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "Task not found: $TaskName"
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Removed scheduled task: $TaskName"
