# 实时实验操作手册

## 实验前 5 分钟检查

- 设备采样率为 1000 Hz。
- Data Sending 为 `127.0.0.1:8712`。
- 实际数据帧为 64 EEG + TRG，且顺序与 `configs/neusen_w_64_channels_template.txt` 一致。
- 游戏配置为 `EEGback|EEG`。
- 被试为 YHC，电极阻抗和信号质量合格。
- 关闭可能占用 8000、5173、8712 端口的旧进程。

环境快速检查：

```powershell
cd "C:\Users\Admin\Desktop\又是帅呗YHC\realtime-neuracle-master（带模型）"
.\.venv\Scripts\python.exe .\01_check_environment.py
```

必须看到最后一行：

```text
Environment check passed. YHC realtime runtime is ready.
```

开启 Data Sending 后、启动 GUI 前，再检查实际 TCP 帧宽：

```powershell
.\.venv\Scripts\python.exe .\02_check_yhc_neuracle_stream.py
```

只有看到 `PASSED` 才能进入正式实验。

## 2026-08-11 本机硬件检查结果

当前 Neuracle 8712 端口实测约 `36.0 kB/s`，对应约 `9` 个4字节字段/样本，即 **8 EEG + TRG**；项目需要约 `260 kB/s`，对应 **64 EEG + TRG**。因此当前发送配置尚不能运行 YHC 12导模型。

根因已经定位：`C:\Program Files (x86)\Neuracle\Neusen W\Conf\SystemSetting.json` 当前的 `DataServiceMontage` 为 `SSVEP`，其发送导联表是 `SSVEP-8`：POz、PO3、PO4、PO5、PO6、Oz、O1、O2。Neuracle 的记录头同时确认硬件为 `NSW3BA0064 / 1000Hz64 / EEGCap64-V3.0`，所以问题在 **Data Sending 导联表**，不在脑电帽通道数。

在 Neuracle 中把 Data Sending 的方案/导联表从 `SSVEP / SSVEP-8` 改成当前 `EEGCap64-V3.0` 的完整64通道，并确保追加 TRG；重启 Data Sending 后再次运行帧宽检查。项目模板已按本机 BDF 头的真实64通道顺序更新，不能使用旧的 CB1/CB2 模板。

设备当前原始数值为 nV 量级，后端已经支持自动换算为 µV；GUI 会显示检测到的源单位和换算后的波形。

## 启动顺序

1. 打开 Neuracle 采集并确认实时波形正常。
2. 开启 Data Sending。
3. 运行 `.\05_run_gui.ps1`。
4. 在 GUI 点击“启动服务”。
5. 让被试静息，等待30秒倒计时结束。
6. 确认12个通道均有连续合理波形，分类概率开始更新。
7. 在 GUI 点击“启动游戏”。
8. 确认 `LSL Receiver = connected`，再进入正式赛道。

## 正常状态判据

```text
EEG TCP       connected
模型推理      running
LSL 输出      running
LSL Receiver connected
健康状态      green
```

GUI 中应同时满足：

- 恰好显示 FC3、FC1、FCz、FC2、FC4、C3、C1、Cz、C2、C4、CP3、CP4；
- 四个类别概率之和约为100%；
- 当前分类、置信度和控制方向持续更新；
- `samples` 和 `predictions` 计数增长；
- 事件列表没有持续出现断流或窗口错误。

## 控制映射

| 模型类别 | 游戏值 | 动作 |
|---|---:|---|
| rest | 3 | 停止 / 加速区蓄能 |
| feet | 2 | 前进 |
| left_hand | 0 | 左移 |
| right_hand | 1 | 右移 |

0911 最终版适配后四类分别对应四种状态；rest 输出 3、feet 输出 2。低置信度、过期、断流或停止服务也会尝试发送 3。stop 不保证所有赛道区域均静止。比赛双机连接与验收以 [现场联调手册](docs/ONSITE_RUNBOOK.md) 为准。

## 异常时立即处理

- `reconnecting`：确认 Data Sending 是否开启、8712端口和帧格式是否正确。
- `stale`：检查采集软件是否暂停、网线/USB是否断开；车辆已经被切到停止。
- LSL Receiver 未连接：保持服务运行，退出并重新启动游戏。
- 波形平直/饱和：不要继续实验，先检查电极和量纲。
- 分类长期单一且高置信：停止实验，检查通道顺序和本次基线质量。

## 结束顺序

1. 停止游戏。
2. 停止模型服务。
3. 关闭 Data Sending。
4. 检查最新 `experiment_logs` 会话中的 `session.json` 状态为 `completed`。
5. 备份该会话的完整目录，不要只复制 `predictions.csv`。
