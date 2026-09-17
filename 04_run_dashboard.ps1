param(
    [string]$PythonPath = "",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8712,
    [int]$DashboardPort = 8000,
    [double]$DeviceSfreq = 1000,
    [double]$StepSec = 0.5,
    [double]$LiveBaselineSec = 30,
    [double]$MinConfidence = 0.55,
    [string]$ChannelFile = "",
    [string]$LogDir = "",
    [string]$ExperimentId = "yhc_realtime",
    [ValidateSet('lsl', 'tcp-json')][string]$GameTransport = 'lsl',
    [string]$GameTcpProfile = '',
    [string]$LslConfig = '',
    [switch]$DebugControls,
    [switch]$Cpu,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
if (-not $ChannelFile) {
    $ChannelFile = Join-Path $projectRoot "configs\neusen_w_64_channels_template.txt"
}
if (-not $LogDir) {
    $LogDir = Join-Path $projectRoot "experiment_logs"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Project Python not found: $PythonPath. Run .\00_setup_environment.ps1 first."
}

$arguments = @(
    (Join-Path $projectRoot "dashboard_service.py"),
    "--model", "yhc",
    "--host", $HostName,
    "--port", $Port,
    "--device-sfreq", $DeviceSfreq,
    "--step-sec", $StepSec,
    "--live-baseline-sec", $LiveBaselineSec,
    "--min-confidence", $MinConfidence,
    "--channel-list-file", $ChannelFile,
    "--log-dir", $LogDir,
    "--experiment-id", $ExperimentId,
    "--game-transport", $GameTransport,
    "--save-logs",
    "--auto-reconnect"
)
if ($Cpu) { $arguments += "--cpu" }
if ($DryRun) { $arguments += "--dry-run" }
if ($DebugControls) { $arguments += "--debug-controls" }
if ($GameTcpProfile) { $arguments += @('--game-tcp-profile', $GameTcpProfile) }
if ($LslConfig) { $arguments += @('--lsl-config', $LslConfig) }

$env:DASHBOARD_PORT = "$DashboardPort"
Write-Host "Starting YHC BCI dashboard backend: http://127.0.0.1:$DashboardPort"
Write-Host "EEG TCP=$HostName`:$Port, baseline=${LiveBaselineSec}s, step=${StepSec}s"
& $PythonPath @arguments
exit $LASTEXITCODE
