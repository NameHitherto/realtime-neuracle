# YHC GUI 与实时遥测

## GUI 展示内容

- YHC 指定的 12 导实时脑电波形；
- `rest / feet / left_hand / right_hand` 平滑概率和分类结果；
- 当前发送给游戏的 `0 / 1 / 2 / 3` 控制值；
- EEG TCP、模型、LSL outlet、游戏 LSL consumer 状态；
- 30 秒静息基线进度；
- 连接、校准、推理、断流、重连和日志事件；
- 原始数据和预测日志保存进度。

## 启动

```powershell
cd "C:\Users\Admin\Desktop\又是帅呗YHC\realtime-neuracle-master（带模型）"
.\05_run_gui.ps1
```

GUI 自动解析以下路径：

- Python：`.venv\Scripts\python.exe`
- 后端：`dashboard_service.py`
- 游戏：默认在赛事电脑运行；可选本地 0911 最终版 EXE（GUI 输入路径或设置 `BCI_GAME_EXE`）。

正式运行前先开启 Neuracle Data Sending。点击“启动服务”后，即使设备暂时未连接，后端也会保持运行并每 2 秒重试；连接后先采集 30 秒静息基线，再开始分类。

## 开发和构建

前端类型检查和构建：

```powershell
cd .\GUI
npm run lint
npm run build
```

Tauri 开发模式：

```powershell
npm run tauri dev
```

生成桌面安装包：

```powershell
npm run tauri build
```

## 后端 API

```text
GET  /api/health
GET  /api/status
GET  /api/events?limit=100
GET  /api/history?limit=120
POST /api/test/control
POST /api/runtime/stop
WS   /ws/telemetry
```

测试控制请求：

```json
{"control": 2, "duration_ms": 1000}
```

## 状态说明

- `calibrating`：正在采集静息基线，模型尚不接管车辆，持续发送停止。
- `validating`：正在用实际字节率验证发送端是否为64 EEG + TRG。
- `running`：实时推理运行中。
- `stale`：EEG 数据超时，已发送停止。
- `reconnecting`：TCP 已断开，等待自动重连。
- `LSL Receiver connected`：至少一个接收者已订阅 `EEGback|EEG`，需现场核实是否为赛事游戏。

Unity 发布包没有内部 ACK 接口，因此 GUI 的游戏状态属于推断状态：可以确认 Python 已发送、LSL 有消费者，但不能证明 Unity 内部每条控制都完成执行。
