# Transfer and Debug Guide

## 1. Package to Send

Send the whole `realtime博睿康` folder. Do not send only the two entry scripts.

The folder is now self-contained for model loading:

- `checkpoints/`: trained BCIC2a and HGD checkpoints
- `model_deps/`: model source files required by the checkpoints
- `configs/`: channel order templates
- `bcic2a_racing_game.py`: BCIC2a model -> LSL racing control
- `hgd_racing_game.py`: HGD model -> LSL racing control
- `realtime_common.py`: TCP receiver, preprocessing, sliding-window inference, LSL output

## 2. Environment Setup

Use Python 3.10 if possible.

```powershell
conda create -n bci_realtime python=3.10 -y
conda activate bci_realtime
cd "PATH_TO\realtime博睿康"
pip install -r .\requirements_realtime.txt
python .\01_check_environment.py
```

If GPU Torch installation is needed, install the correct PyTorch build from the official PyTorch command generator, then rerun:

```powershell
python .\01_check_environment.py
```

## 3. Dry Run

Dry run checks model loading and preprocessing without connecting to the EEG stream or the game.

```powershell
python .\bcic2a_racing_game.py --dry-run --cpu
python .\hgd_racing_game.py --dry-run --cpu
```

## 4. NeuSen W 64-Channel Setup

Edit:

```powershell
.\configs\neusen_w_64_channels_template.txt
```

The file must match the exact channel order sent by the NeuSen W TCP stream. Keep `TRG` as the last item only if the stream includes the trigger channel.

BCIC2a realtime inference uses a 22-channel subset, so it can run from a standard 64-channel stream if these channels are present:

```text
Fz, FC3, FC1, FCz, FC2, FC4,
C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CPz, CP2, CP4,
P1, Pz, P2, POz
```

HGD realtime inference requires the HGD 44-channel montage, including channels such as `FFC5h`, `FCC3h`, `CCP5h`, and `CPP1h`. If the NeuSen W stream does not contain these exact channel names, do not run the HGD script as-is. Use BCIC2a first, or add a separately validated montage-remapping/interpolation step.

## 5. Start the Racing Game Control Stream

Start the NeuSen W TCP data sender first, then run one of the scripts below.

Recommended first test with NeuSen W 64-channel stream:

```powershell
cd "PATH_TO\realtime博睿康"
.\02_run_bcic2a_neusenw_game.ps1
```

Equivalent explicit command:

```powershell
python .\bcic2a_racing_game.py `
  --host 127.0.0.1 `
  --port 8712 `
  --device-sfreq 1000 `
  --step-sec 0.5 `
  --channel-list-file .\configs\neusen_w_64_channels_template.txt
```

The script outputs an LSL stream matching the racing-game test script:

```text
name = EEGback
type = EEG
channel_count = 1
sample_format = float32
nominal_srate = 10 Hz
values = 0 / 1 / 2 / 3
```

Control mapping:

```text
0 = left
1 = right
2 = forward
3 = stop
```

## 6. Common Debug Points

- If the script reports missing channels, fix the channel order file first.
- If the game does not receive control values, confirm `pylsl` is installed and the game config still expects `EEGback|EEG`.
- If the model keeps outputting one class, check EEG units, channel order, filter range, and whether the stream is real EEG rather than impedance/test data.
- If the stream does not include `TRG`, remove the final `TRG` line from the channel file.
