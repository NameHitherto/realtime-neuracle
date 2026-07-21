# BCI Racing Dashboard

本目录现在包含一个本地实时诊断面板，用于观察：

- EEG 实时波形（后端降采样到显示频率）
- BCIC2a / HGD 模型概率和分类结果
- 0/1/2/3 游戏控制映射
- LSL stream 信息与 consumer 在线状态
- Unity 游戏画面快照
- 实时链路状态、健康状态和事件日志

## 安装

```powershell
conda create -n bci_realtime python=3.10 -y
conda activate bci_realtime
pip install -r .\requirements_realtime.txt
cd .\dashboard
npm install
npm run build
cd ..
python .\01_check_environment.py
```

`pylsl` 是游戏控制链路的必需依赖。没有安装时，可以运行 dry-run，但不能向官方游戏发布 LSL 控制流。

## dry-run 启动

dry-run 不连接 EEG 设备、不发布 LSL 控制，只验证模型、后端 API 和前端面板：

```powershell
.\04_run_dashboard.ps1 -Model bcic2a -Cpu -DryRun
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果 8000 端口已被占用：

```powershell
.\04_run_dashboard.ps1 -DashboardPort 8011 -Model bcic2a -Cpu -DryRun
```

## 真实设备启动

BCIC2a：

```powershell
.\04_run_dashboard.ps1 `
  -Model bcic2a `
  -HostName 127.0.0.1 `
  -Port 8712 `
  -DeviceSfreq 1000 `
  -StepSec 0.5 `
  -ChannelFile .\configs\neusen_w_64_channels_template.txt
```

HGD：

```powershell
.\04_run_dashboard.ps1 `
  -Model hgd `
  -HostName 127.0.0.1 `
  -Port 8712 `
  -DeviceSfreq 1000 `
  -StepSec 0.5 `
  -ChannelFile .\configs\hgd_required_channels.txt
```

如果只需要采集游戏窗口的一部分，可传入屏幕坐标：

```powershell
.\04_run_dashboard.ps1 -CaptureBbox 100 80 1380 800
```

坐标格式为：`left top right bottom`。未指定时使用系统屏幕截图，正式比赛建议指定游戏窗口区域或后续接入 Windows Graphics Capture。

## 开发模式

后端：

```powershell
$env:DASHBOARD_PORT=8000
python .\dashboard_service.py --dry-run --cpu
```

前端热更新：

```powershell
cd .\dashboard
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 会把 `/api`、`/ws` 和 `/media` 代理到 8000 端口。

## API

```text
GET  /api/health
GET  /api/status
GET  /api/events?limit=100
GET  /api/history?limit=120
POST /api/test/control
WS   /ws/telemetry
GET  /media/game.jpg
```

测试控制请求示例：

```json
{"control": 2, "duration_ms": 1000}
```

控制值映射保持官方协议：

```text
0 = 左移
1 = 右移
2 = 前进
3 = 停止
```

## 当前遥测边界

当前官方游戏是 Unity 发布版，项目目录没有 Unity 工程源码，因此 dashboard 默认将游戏状态标记为 `inferred`：

- `Python sent` 表示 Python/LSL 已发布的最近控制值；
- `Consumer online` 表示 LSL 至少发现一个消费者；
- 游戏画面表示客户端窗口采集结果；
- 这三项不能等同于 Unity 内部的逐条 command ACK。

如果后续取得 Unity 源码，应在游戏端增加 WebSocket 或 LSL telemetry outlet，主动回传接收指令、车速、圈数和赛道状态。
