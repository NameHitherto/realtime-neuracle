# 部署与实机排障

## 需要传输的目录

必须整体传输 `realtime-neuracle-master（带模型）`，不要只复制入口脚本。关键内容包括：

- `checkpoints/yhc_hyena_final_realtime.pth`
- `model_deps/`
- `configs/neusen_w_64_channels_template.txt`
- `dashboard_service.py`、`realtime_common.py`、`realtime_yhc_4class.py`
- `experiment_logger.py`
- `GUI/`
- `requirements_realtime.lock.txt`

`.venv` 不建议跨电脑直接复制；在目标电脑使用 Python 3.10 执行 `00_setup_environment.ps1` 重建。

## 环境验证

```powershell
cd "项目目录"
.\00_setup_environment.ps1 -PythonPath "Python3.10的完整路径"
.\.venv\Scripts\python.exe .\01_check_environment.py
```

环境检查会真实导入依赖、核对游戏配置、加载 YHC checkpoint，并完成一次 `12 × 1000` CPU 前向推理；不再只检查包名是否存在。

## Neuracle 配置

默认连接：

```text
host = 127.0.0.1
port = 8712
sampling rate = 1000 Hz
frame = 64 float32 EEG + 1 int32 TRG
```

`configs/neusen_w_64_channels_template.txt` 必须与采集软件实际发送顺序完全相同。代码会从完整64导中按以下顺序抽取：

```text
FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP4
```

如果采集软件没有发送 TRG，不能仅删除配置中的 TRG 后继续正式实验；当前 Neuracle 实时契约要求第65字段为 `int32 TRG`。

开启 Data Sending 后先测实际帧宽：

```powershell
.\.venv\Scripts\python.exe .\02_check_yhc_neuracle_stream.py
```

设备可能以 nV 输出；后端会根据幅值自动换算成 µV，并在遥测中记录检测到的源单位。

## 分层排障

### 1. 模型与后端

```powershell
.\04_run_dashboard.ps1 -Cpu -DryRun
```

访问 `http://127.0.0.1:8000/api/health`。完成后用 Ctrl+C 停止。

### 2. EEG TCP

开启采集软件 Data Sending，再运行：

```powershell
.\04_run_dashboard.ps1 -Cpu
```

GUI/API 应依次显示 `connecting → connected → calibrating → running`。如果持续 `reconnecting`：

- 检查 8712 端口是否监听；
- 检查发送端是否确为 65 字段；
- 检查防火墙和采集软件 Data Sending 状态；
- 检查通道顺序，而不仅是通道名集合。

### 3. LSL 与游戏

游戏配置必须是：

```text
EEGback|EEG
```

后端输出协议：

```text
name = EEGback
type = EEG
channel count = 1
sample format = float32
nominal rate = 10 Hz
values = 0 / 1 / 2 / 3
```

如果 GUI 中 `LSL Receiver` 未连接，说明游戏尚未找到/订阅流。先保持后端运行，再重新启动游戏。

### 4. 分类异常

若模型长期输出单一类别或概率异常饱和，依次检查：

- 12 导顺序是否完全正确；
- TCP 数值单位是 V 还是 µV，实际幅值是否合理；
- 30 秒基线期间是否真正静息；
- 电极是否松动、饱和或有强工频噪声；
- 当前被试是否为 YHC；
- 实时日志中的波形和 `live_calibration.npz` 是否合理。

### 5. 断流保护

EEG 超过2秒没有新数据时，后端将控制切换为 `3=停止`；TCP 断开后默认每2秒重连。事件列表会记录“强制停车、连接中断、数据恢复”。不要在正式实验中关闭 `--auto-reconnect` 或把默认控制改成非停止值。

## 日志恢复

每次真实运行在 `experiment_logs/时间_实验ID/` 生成：

- `eeg_raw.f32`：64导、sample-major、小端 float32；
- `triggers.i32`：小端 int32；
- `blocks.csv`：数据块与样本位置；
- `predictions.csv`：原始/平滑概率、分类、控制；
- `lsl_controls.csv`：实际发布控制；
- `events.jsonl`：连接和运行事件；
- `session.json`：通道、采样率、参数和完成状态；
- `live_calibration.npz`：本次30秒静息统计。

若进程异常退出，二进制和文本日志采用及时写盘方式，已接收部分通常仍可恢复；`session.json` 可能保持 `recording`，应结合文件大小和 `blocks.csv` 判断实际样本数。
