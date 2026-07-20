param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8712,
    [double]$DeviceSfreq = 1000,
    [double]$StepSec = 0.5,
    [string]$ChannelFile = ".\configs\hgd_required_channels.txt"
)

python .\hgd_racing_game.py `
  --host $HostName `
  --port $Port `
  --device-sfreq $DeviceSfreq `
  --step-sec $StepSec `
  --channel-list-file $ChannelFile
