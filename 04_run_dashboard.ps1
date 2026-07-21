param(
    [ValidateSet("bcic2a", "hgd")]
    [string]$Model = "bcic2a",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8712,
    [int]$DashboardPort = 8000,
    [double]$DeviceSfreq = 1000,
    [double]$StepSec = 0.5,
    [string]$ChannelFile = "",
    [switch]$Cpu,
    [switch]$DryRun,
    [int[]]$CaptureBbox = @()
)

$arguments = @(
    ".\dashboard_service.py",
    "--model", $Model,
    "--host", $HostName,
    "--port", $Port,
    "--device-sfreq", $DeviceSfreq,
    "--step-sec", $StepSec
)
if ($ChannelFile) { $arguments += @("--channel-list-file", $ChannelFile) }
if ($Cpu) { $arguments += "--cpu" }
if ($DryRun) { $arguments += "--dry-run" }
if ($CaptureBbox.Count -eq 4) { $arguments += @("--capture-bbox") + $CaptureBbox }

$env:DASHBOARD_PORT = "$DashboardPort"
Write-Host "Starting BCI dashboard: http://127.0.0.1:$DashboardPort"
Write-Host "Model=$Model EEG=$HostName`:$Port"
python @arguments
