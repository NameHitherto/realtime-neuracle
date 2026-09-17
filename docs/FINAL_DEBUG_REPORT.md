# 0911 最终版适配与调试报告

日期：2026-09-17。结论：**已完成本机软件适配和模拟链路验证；赛事双机网络、官方游戏实际车辆执行及真实受试者闭环仍待现场验收。** 用户确认没有额外协议文档或当前可联调的赛事 IP/端口。

## 客户端差异

两个发行包各 176 个文件，161 个 SHA-256 一致、15 个不同，无新增/删除文件。变化包括主业务程序集、场景、资源与 boot.config；UnityPlayer、运行库及两个 StreamingAssets 配置未变。调试结束再次核对，两个发行包原有文件均未修改。

| 项目 | 06251432 | 0911 最终版 | 对系统影响 |
|---|---|---|---|
| 控制入口 | EEGInletForDouble → EEG_COMMAND_DATA → KartInput.ApplyEEGInput | 关键方法逻辑相同 | 继续兼容 EEGback / EEG 数值流 |
| 指令码 | 0 左、1 右、2 油门/直行、3 reverse/停止状态 | 相同 | 无需交换左右或改变码值 |
| 配置 | LSLInletConfig=EEGback\|EEG，InputConfig 首项0 | 内容完全相同 | 保持禁键盘配置；键盘开关解析方法有内部变化，不启用它 |
| 加速 | 区域匹配、停止蓄能与加速增益 | 新增 SplineBoostPad、TriggerSplineBoost、计时/冷却/次数 | 需要可独立输出 stop；不能让静息与双脚均输出2 |
| 比赛统计 | 旧统计 | 新增 AccelerateForwardTime、SplineBoostCount，结果相关方法变化 | 现场核对新规则计分和实际展示 |
| 状态输出 | 无 KartLSLOutlet 类型 | 新增 Unity_KartData / Markers 字符串流 | 可只读观测开始、结束、位置、加速时间、boost 次数 |
| 掉线处理 | OnStreamLost 仅停止拉流 | 相同，不重置最后指令 | 发送端做过期保护，但无法在断网时保证远端停车 |

最终版 GetTargetSpeed 仍存在“非匹配区域返回 baseSpeed”等分支；**stop 是控制状态，不是适用于所有区域的硬停车命令**。加速参数在构造器里的默认值也可能被 Unity 场景序列化值覆盖，本报告不把静态默认值视为实测赛道速度。

主程序集 SHA-256：

- 旧版：`0acc1326756b619c8fb05ec798dc1cfb29ebf228141998374054277290464efc`
- 最终版：`48ddf86b7bf1a6957dcca9e753b4a0fefb2583ef695901970091eeb5649e2e47`

证据：[完整差异](../outputs/final-debug/client-audit/diff.json)、[最终版关键方法 IL](../outputs/final-debug/client-audit/final-control.il.txt)、[附件6提取文本](../outputs/final-debug/client-audit/attachment6.txt)。`tools/audit_clients.py` 可复现；审计环境 `.tools-venv` 使用 dnfile 0.18.0 / dncil 1.0.2。自动方法差异列表可能包括元数据令牌编号变化，以上结论按已解析的具体方法判断，未宣称所有资源视觉差异均已还原。

## 附件6与发行包不一致

附件6明确 TCP Socket + JSON、left/right/forward/stop，但未给字段、端口、分帧或连接角色。给定最终版主业务程序集仍是 LSL 控制入口，未发现 JSON/TCP 指令接收类型引用或相应配置；Fusion 的多人游戏网络不是参赛系统的 JSON 控制口。不能推断赛场一定没有外部网关。

因此保留已核实 LSL 通道，并添加显式现场配置的 TCP JSON 客户端。模板默认不可运行，必须由赛方给出四个完整 JSON 对象及网络参数后启用。支持换行、4字节大端长度前缀、原样 JSON；不假定官方采用其中任一种。其他连接角色/握手/ACK 需现场扩展。TCP 模拟接收端测试通过不等于官方接口兼容。

## 已修复

1. 四类输出改为 rest→3、feet→2、left_hand→0、right_hand→1；保留模型权重、类别索引和预处理。
2. 游戏输出独立10 Hz线程，不再受 EEG recv 阻塞控制。推理指令1.5秒失效、EEG两秒无完整新数据、异常与停止均切换 stop；LSL 发送缓冲缩短到1秒。
3. 断流重连清除旧窗口和平滑结果，重新等待完整窗口；修复连接后首包延迟被错误算成低数据率的问题。
4. 正式模式人工 API 返回403，离线回放必须显式调试模式。dry-run 无输出，界面区分模拟推理与实际发送。非有限 EEG/概率不产生驾驶指令。
5. TCP 重连先发 stop，只接受新连接之后的新推理决定；不积压旧驾驶指令。日志区分发送失败、成功提交与未知执行状态。
6. GUI 默认远端赛事电脑；独立显示 EEG 地址、游戏接口和实际发送值，支持可选本地最终版路径。浏览器遥测可重连；GUI 退出尝试关闭它启动的服务和本地游戏。
7. 修复迁移后的 Python 环境；旧环境保留于 `outputs/final-debug/venv-before-repair`，原 pyvenv.cfg 另有备份。安装脚本可识别失效的旧解释器路径并备份后重建。当前依赖严格按已有锁文件恢复，torch 为 CPU 构建；未修改其他研究环境。

## 验证结果与边界

| 验证 | 结果 |
|---|---|
| Python 回归 | 13项通过，含真实 LSL inlet 四码、真实本地 TCP字节、重连过期、独立节拍、日志与采集 |
| 环境与模型 | Python3.10.21、锁文件依赖兼容，CPU真实checkpoint自检通过 |
| 完整模拟链路 | 合成64导TCP→真实模型→独立LSL接收；断流stop、恢复等待新窗口均通过；详细计数见下方JSON |
| HTTP 服务 | 实际8000端口：人工控制403，停止接口200并释放服务 |
| Web页面 | 实际后端页面、LSL/TCP选择、联调开关和本地/远端切换已浏览器交互；12导与概率展示可见 |
| Tauri | 前端构建、Rust测试、桌面debug构建通过；原生窗口完整启停交互未验证 |
| 官方最终游戏 | 进程无界面启动尝试期间未观测到LSL订阅；不能判为游戏控制通过 |
| 双机/真人/长稳 | 无赛事电脑及真实设备联调条件，未验证 |

模拟链路使用独立随机流名，未向官方游戏输入模拟驾驶信号。测试含特意断流，liblsl 在关闭 inlet/outlet 后出现的“stream transmission broke off / reconnecting”是测试清理时的日志，不作为成功证据。

证据文件：[回归结果](../outputs/final-debug/python-tests.txt)、[完整模拟验收](../outputs/final-debug/loopback/acceptance.json)、[实际接收样本](../outputs/final-debug/loopback/received-samples.json)、[官方无界面尝试](../outputs/final-debug/official-headless.json)、[页面截图](../output/playwright/live-dry-run.png)。模拟结果不能代表解码准确率、真实脑控或正式游戏动作验收。

复现：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\01_check_environment.py
.\.venv\Scripts\python.exe tools\acceptance_loopback.py
npm.cmd --prefix GUI run build
cargo test --manifest-path GUI\src-tauri\Cargo.toml
```

现场步骤见 [双机联调手册](ONSITE_RUNBOOK.md)。上线前必须补齐：赛方实际接口、网络隔离、四种动作/蓄能/掉线行为、真实EEG与受试者闭环、至少15分钟长稳测试。
