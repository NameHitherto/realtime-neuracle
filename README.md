# realtime博睿康

这个目录放的是两个已经训练好的单被试 4 分类模型，以及博睿康/Neuracle TCP 实时推理脚本。

- `realtime_bcic2a_4class.py`: 调用 BCIC2a 当前最佳平均模型，使用 S03 checkpoint，类别为 `feet / left_hand / right_hand / tongue`。
- `realtime_hgd_4class.py`: 调用 HGD 当前最佳平均模型，使用 S003 checkpoint，类别为 `feet / left_hand / rest / right_hand`。
- `realtime_common.py`: 从 `demo04_original.py` 改造出的 TCP 接收、环形缓冲、滑动窗口、滤波、重采样、标准化和概率平滑代码。

运行前必须保证博睿康发送端的通道顺序与脚本中的模型通道顺序一致。原始 `demo 04.py` 默认只有 10 个枕区通道，不能直接喂给 MI 模型；BCIC2a 模型需要 22 个导联，HGD 模型需要 44 个导联。

## 先做离线自检

```powershell
cd "E:\运动想象算法\realtime博睿康"
python .\realtime_bcic2a_4class.py --dry-run
python .\realtime_hgd_4class.py --dry-run
```

## 实时运行

```powershell
cd "E:\运动想象算法\realtime博睿康"
python .\realtime_bcic2a_4class.py --host 127.0.0.1 --port 8712 --device-sfreq 1000 --step-sec 0.5
python .\realtime_hgd_4class.py --host 127.0.0.1 --port 8712 --device-sfreq 1000 --step-sec 0.5
```

## 接入赛车游戏

赛车游戏读取的是 LSL 单通道控制流，和 `bci_接入游戏_original.py` 保持一致：

- stream name: `EEGback`
- stream type: `EEG`
- channel count: `1`
- sample format: `float32`
- value mapping: `0=left`, `1=right`, `2=forward`, `3=stop`

模型类别到赛车控制的映射：

- BCIC2a: `left_hand -> 0`, `right_hand -> 1`, `feet -> 2`, `tongue -> 3`
- HGD: `left_hand -> 0`, `right_hand -> 1`, `feet -> 2`, `rest -> 3`

运行前需要安装 `pylsl`：

```powershell
conda activate pt310
pip install pylsl
```

然后运行其中一个实时控制脚本：

```powershell
cd "E:\运动想象算法\realtime博睿康"
python .\bcic2a_racing_game.py --host 127.0.0.1 --port 8712 --device-sfreq 1000 --step-sec 0.5
python .\hgd_racing_game.py --host 127.0.0.1 --port 8712 --device-sfreq 1000 --step-sec 0.5
```

这两个脚本会持续向游戏发送 `0/1/2/3`，名义输出频率为 10 Hz；模型每 `step-sec` 秒更新一次判断，在两次判断之间保持上一条控制值。

如果博睿康发送端实际通道列表不同，用 `--channel-list` 显式传入，最后一列触发通道写成 `TRG`：

```powershell
python .\realtime_bcic2a_4class.py --channel-list "Fz,FC3,FC1,FCz,FC2,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP1,CPz,CP2,CP4,P1,Pz,P2,POz,TRG"
```

HGD 模型训练时做了标准化；实时脚本默认使用窗口内 z-score。更严谨的实时部署应采集一段静息/校准数据，保存每个通道的 `mean/std` 到 `.npz`，然后通过 `--calibration-npz` 传入。
