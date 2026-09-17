# 0911 决赛双机联调手册

## 1. 先确认实际接口

当前 0911 包：`EEGback|EEG`、单通道 float32、0 左 / 1 右 / 2 直行 / 3 停止。游戏内部是 double inlet，liblsl 可转换 float32，保留既有发送格式。

附件6：TCP Socket + JSON，动作字符串 left/right/forward/stop。未给出 IP、端口、双方角色、字段、分帧、握手与 ACK；用户已确认这些需到现场调试。**不要把采集端口 8712 当成游戏端口，也不要默认发送 {"command":"left"} 就是官方协议。**

向赛事技术人员确认：

1. 赛事实际运行包是否与本地 0911 一致；比赛使用 LSL 还是另有 JSON 网关。
2. 若 LSL：每队如何隔离同名 EEGback 流，是否允许配置 SessionID/LSLInletConfig；不私自修改官方客户端。
3. 若 JSON：谁 listen/谁 connect、目标 IP/端口、四个完整消息样例、编码与分帧、是否需要身份字段/握手/ACK、是否限制发送率。
4. 游戏在掉线、stop、蓄能期间的预期动作；静息作为停止状态是否符合现场范式要求。

## 2. 准备本机

从项目根目录执行，迁移电脑时重建 `.venv`，不要复制旧 `.venv` 充当可移植环境：

```powershell
.\00_setup_environment.ps1
.\.venv\Scripts\python.exe .\01_check_environment.py
.\.venv\Scripts\python.exe .\02_check_yhc_neuracle_stream.py
.\05_run_gui.ps1
```

本机已按锁文件恢复 Python 3.10.21 / torch 2.13.0+cpu。CPU 完成模型和模拟链路验收；未验证 GPU 推理。更换电脑后重新测推理周期与指令过期情况。确认采集软件按模板发送 64 EEG float32 + TRG int32、1000 Hz；接收率正确不证明导联顺序正确。

GUI 默认勾选“游戏运行在赛事电脑”。仅本地练习时取消勾选并选择最终版 EXE，也可启动 GUI 前设置 `$env:BCI_GAME_EXE`。无需把游戏拷到 EEG 电脑才能运行环境自检。

## 3. 若赛方确认 LSL

两台电脑加入同一赛事局域网，记录各自 IPv4。优先按赛事网络配置使用有线连接；不要为排错关闭所有防火墙。按管理员要求仅放行相关 Python/官方游戏及必要网络流量。

官方 LSL 文档列出默认 UDP 16571 用于发现，TCP/UDP 16572–16604 用于数据与服务；多网卡和防火墙可能影响发现。参见 [官方网络排错](https://labstreaminglayer.readthedocs.io/info/network-connectivity.html)。这些是 LSL 端口，不是“JSON 游戏端口”。

若发现失败，复制 `configs/lsl_api.site-template.cfg`，经赛方允许后，在两端设置 `KnownPeers = {本机IP, 游戏电脑IP}`；使用相同的赛方分配 SessionID。配置在进程初始化时读取，因此改完须重启相关程序。GUI“LSL 配置文件”作用于本机后端；它不会自动修改远端游戏。KnownPeers 只是发现设置，不能替代多队隔离和身份核对。参见 [官方 LSL 配置](https://labstreaminglayer.readthedocs.io/info/lslapicfg.html)。

只读发现与观测：

```powershell
.\.venv\Scripts\python.exe tools\game_link_check.py --seconds 5
# 若启用现场配置：
.\.venv\Scripts\python.exe tools\game_link_check.py --lsl-config configs\lsl_api.site.cfg --seconds 5
# 最终版新增的状态流；同名多队时必须追加 --source-id 精确选择：
.\.venv\Scripts\python.exe tools\game_link_check.py --listen Unity_KartData --seconds 30
```

**确认恰好一个属于本队的 EEGback 输出。**同时开命令行后端、GUI 后端、旧调试程序可能产生多个同名流。`LSL Receiver connected` 表示至少一个订阅者，可能是诊断监听器，不能据此认定赛事游戏收到指令。完成监听后退出诊断工具，再核对游戏订阅和真实动作。

`Unity_KartData|Markers` 是最终版新加的单通道字符串输出，代码包含 startgame / endgame / pos / accelerateTime / boostCount；默认位置间隔 0.1 秒，具体可能被场景序列化配置覆盖。该流用于观察和保存，不参与控制决策；位置不是逐条指令 ACK。

## 4. 若赛方确认 TCP JSON

复制 `configs/game_tcp.site-template.json` 为 `configs/game_tcp.site.json`。按赛方说明填写 host、port、四个完整 JSON 对象和 framing，确认后才把 confirmed_by_organizer 改为 true。本程序提供 TCP 客户端角色；支持 UTF-8 换行、4 字节大端长度前缀或原样 JSON 三种分帧。若现场需要其他角色、握手、身份协商或 ACK，必须扩展适配并重新测，不能套用猜测。

```powershell
# 只建立 TCP，不发送控制；成功也不证明 JSON 兼容
.\.venv\Scripts\python.exe tools\game_link_check.py --tcp-host <赛方IP> --tcp-port <赛方端口>
# 已确认协议后运行：
.\04_run_dashboard.ps1 -GameTransport tcp-json -GameTcpProfile configs\game_tcp.site.json
```

也可在 GUI 选择 TCP + JSON 并填配置文件。两个出口互斥，不会同时发送 LSL 和 JSON。连接超时 0.3 秒、失败后最多每秒尝试一次；新连接先发 stop，只有建立连接后的新推理结果才能恢复驾驶。不积压旧驾驶命令；发送成功只表示本机 socket 接受了字节。

## 5. 比赛前验收顺序

1. 进入赛道前验证来源、单通道格式、四类码值、10 Hz 接收和多队隔离。
2. 正式比赛前经赛方允许，可用 `--debug-controls` 与本地 API 调试四个码；请求 `POST /api/test/control`，JSON 为 `control:0..3`、`duration_ms:100..10000`。该 API 是本系统调试入口，**不是官方游戏 JSON 协议**。不要在正式比赛人工发指令。
3. 重启为正式模式，关闭“联调模式”，确保 EEG 源为博睿康 TCP；30 秒静息基线完成后，用真实被试验证 left/right/feet/rest 的类别与对应动作。
4. 验证转向、直行、加速区 stop 蓄能及加速垫行为。stop 不保证所有区域立刻静止；没有新 EEG 时必须观察到发送 stop，车辆反应另外记录。
5. 比赛前做断流与恢复试验：断流输出变 stop；重连等待完整 4 秒新窗口及数据率检查，未完成的基线重新采集。恢复时不能立即复用旧驾驶状态。
6. 正式运行连续观察至少 15 分钟，记录耗时、丢包/异常、信号质量、命令与车辆行为。不要把本机约 20 秒模拟验收当作这项现场长稳测试。
7. 结束后停止本机服务，确认日志关闭且 EEG/TRG/predictions/controls/events 完整。关闭 GUI 会尝试停止它启动的后端和本地游戏；远端赛事游戏由赛方操作。

## 6. 仍需现场完成的项目

- 赛事 IP/端口/协议，双机和多队环境的发现、隔离、防火墙与延迟。
- 最终客户端进入赛道后的实际动作；游戏状态流采集。
- 真实 NeuSen W、阻抗/导联、受试者四分类效果和 15 分钟稳定运行。
- 网络断开或进程被强杀时，发送端无法保证 stop 抵达。现有官方 `OnStreamLost` 不会重置最后指令，这需向赛方确认接收端掉线策略。
