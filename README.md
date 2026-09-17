# YHC 实时脑电赛车系统

**2026-09-17：已适配 0911 最终客户端的实际 LSL 接口，并增加待现场确认协议的 TCP JSON 适配。**
版本差异、验证结果和未通过项见 [最终调试报告](docs/FINAL_DEBUG_REPORT.md)；比赛双机连接请按 [现场联调手册](docs/ONSITE_RUNBOOK.md) 操作。
《附件6》要求 TCP JSON，但给定最终发行包的指令入口仍是 LSL；目前没有官方 JSON 字段、端口或分帧说明，不能宣称赛事 TCP 接口已验证。

本目录是面向 YHC 单被试模型的完整实时系统：接收 NeuSen W/Neuracle TCP 脑电流，在 GUI 中显示 12 导脑电、四分类概率、当前分类、游戏控制和运行事件，并通过 LSL 实时控制 `虚拟任务竞速赛`。

## 固定实验契约

- 设备流：1000 Hz，64 个 `float32` EEG + 1 个 `int32` TRG，共 65 字段。
- 模型导联：`FC3、FC1、FCz、FC2、FC4、C3、C1、Cz、C2、C4、CP3、CP4`。
- 模型输入：4 秒，降采样至 250 Hz，形状 `12 × 1000`。
- 预处理：自动 V/µV 量纲识别、12 导 CAR、4–38 Hz 带通、重采样、按通道基线标准化。
- 类别：`rest、feet、left_hand、right_hand`。
- 游戏协议：`0=左、1=右、2=直行、3=停止/加速区蓄能`。车辆速度由官方赛道逻辑决定，stop 不保证各区域均立即静止。
- 输出：LSL `EEGback|EEG`，单通道 `float32`，10 Hz。

训练阶段每个 Trial 的四个 4 秒窗口相对 Trial 起点为 `3.0、3.5、4.0、4.5 s`。赛车部署采用等价的 4 秒滚动窗口，每 0.5 秒更新一次，从而持续输出控制，而不是等待离散 Trial 触发。

## 第一次使用

在 PowerShell 中执行：

```powershell
.\00_setup_environment.ps1
.\.venv\Scripts\python.exe .\01_check_environment.py
```

安装脚本会自动查找本机的 Python 3.10（包括已有 Conda 环境），创建项目独立的 `.venv`，按 `requirements_realtime.lock.txt` 安装 Python 依赖，并安装、构建和检查 React/Tauri GUI。它不会激活或修改 Conda 环境。若只需安装 Python，可使用 `-SkipGui`；若暂时跳过构建检查，可使用 `-SkipChecks`。

项目已固定使用 `.venv` 解释器，并关闭终端自动环境激活，因此打开新终端时不会再执行 `conda-hook.ps1` 或 `conda activate base`。首次修改配置后，请关闭旧终端并新建一个终端使设置生效。

## 正式实时实验

1. 佩戴设备并确认阻抗、参考和地线正常。
2. 在博睿康采集软件中确认发送顺序与 `configs/neusen_w_64_channels_template.txt` 完全一致，开启 `127.0.0.1:8712` Data Sending。
3. 先验证实际 TCP 帧宽：

   ```powershell
   .\.venv\Scripts\python.exe .\02_check_yhc_neuracle_stream.py
   ```

   只有检查显示 `PASSED` 才能继续。

4. 启动 GUI：

   ```powershell
   .\05_run_gui.ps1
   ```

5. GUI 自动填入本项目 `.venv` 和 `dashboard_service.py`；默认游戏在赛事电脑运行。分别配置 EEG 地址和游戏接口，保持模型为 YHC、关闭联调模式，点击“启动服务”。
6. 被试保持睁眼/闭眼状态与训练基线一致，静息 30 秒。GUI 会显示倒计时和实时 12 导波形；校准结束后自动开始分类。
7. 在赛事电脑启动游戏并进入赛道。LSL 模式确认 `EEG TCP、模型推理、游戏指令输出、LSL Receiver` 状态；TCP 模式需赛方先确认协议配置。只有现场观察四种动作与实际指令一致后，才开始比赛。LSL 接收者连接不等于车辆执行确认。

建议先用 GUI 的运行状态确认链路，再进入正式赛道。停止实验时先点击“停止游戏”，再点击“停止服务”，以确保日志完整落盘。

## 运行行为

- 实时波形以 5 Hz 刷新，显示最近 2 秒、4–38 Hz/CAR 后的 12 导 µV 信号。
- 模型每 0.5 秒推理一次，最近 3 次概率取均值；置信度低于 0.55 时输出停止。
- 分类动作：`left_hand=左转、right_hand=右转、feet=直行、rest=停止/蓄能`。已修正旧版 rest 与 feet 同时映射 2 的问题，模型权重和类别顺序没有改动。
- 30 秒实时基线会覆盖 checkpoint 中的历史基线统计；如使用命令行传入 `--live-baseline-sec 0`，才会沿用 checkpoint/校准文件统计。
- 独立线程以 10 Hz 发送；1.5 秒没有新推理结果或 EEG 超过 2 秒无数据时改发 stop。EEG 重连清空旧窗口和平滑结果；停止服务尝试连续发送 3 次 stop。断网或进程被强杀时，无法保证官方游戏收到 stop。
- 正式模式人工控制 API 返回 403；本地回放仅允许显式 `--debug-controls`。离线 `--dry-run` 不发送控制。
- 原始 64 导 EEG、TRG、预测、实际提交的 LSL/TCP 控制和事件写入 `experiment_logs/`。发送日志不代表游戏 ACK。实时基线另存为 `live_calibration.npz`。

## 仅启动后端

```powershell
.\04_run_dashboard.ps1
```

离线模型/API 自检：

```powershell
.\04_run_dashboard.ps1 -Cpu -DryRun
```

浏览器模式使用已构建的 `GUI/dist`：

```text
http://127.0.0.1:8000
```

浏览器模式只能看遥测；启动/停止 Python 和游戏必须使用 Tauri GUI。

## 实验前必须确认

- TCP 实际通道顺序，而不只是通道数量。
- 设备输出单位和波形幅值合理，无饱和、平直或大量工频噪声。
- 游戏文件 `虚拟任务竞速赛_Data/StreamingAssets/LSLInletConfig.txt` 内容为 `EEGback|EEG`。
- 30 秒基线期间没有运动、说话、眨眼集中爆发或电极松动。
- GUI 中 LSL Receiver 已连接；否则游戏尚未订阅控制流。

旧的 BCIC2A/HGD 文件作为历史代码保留，但 GUI 和 `dashboard_service.py` 已锁定为 YHC，不参与当前实验。
