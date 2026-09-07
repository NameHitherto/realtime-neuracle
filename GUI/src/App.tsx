import { useEffect, useState } from 'react'
import { ProbabilityChart, SignalChart } from './charts'
import { useTelemetry } from './useTelemetry'
import type { Event, StatusValue } from './types'
import { invoke } from '@tauri-apps/api/core'

const statusLabels: Record<string, string> = {
  device_tcp: 'EEG TCP',
  inference: '模型推理',
  lsl_outlet: 'LSL 输出',
  lsl_consumer: 'LSL Receiver'
}

const controlGlyphs = ['←', '→', '↑', '■']

const actionLabels: Record<string, string> = {
  left: 'LEFT',
  right: 'RIGHT',
  forward: 'FORWARD',
  accelerate: 'SPEED UP',
  stop: 'STOP',
}

const neusenW64StreamChannels = [
  'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
  'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
  'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
  'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
  'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
  'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
  'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
  'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL', 'TRG',
] as const

type DefaultLaunchPaths = {
  python_path: string
  script_path: string
  game_path: string
}

const emptyChannels: string[] = []

function displayProjectRelativePath(value?: string): string {
  if (!value) return ''
  const normalized = value.replaceAll('\\', '/')
  const marker = '/experiment_logs/'
  const markerIndex = normalized.toLowerCase().indexOf(marker)
  if (markerIndex < 0) return value
  return normalized.slice(markerIndex + 1).replaceAll('/', '\\')
}

function StatusChip({ name, value }: { name: string; value?: StatusValue }) {
  const good = value === 'connected' || value === 'running'
  const bad = value === 'error' || value === 'stale' || value === 'reconnecting' || value === 'stopped' || value === 'unavailable'
  const dotColor = good
    ? 'bg-emerald-500'
    : bad
      ? 'bg-red-500'
      : 'bg-amber-500'
  return (
    <div className="flex items-center gap-2 min-w-[136px] px-3 py-2 rounded-lg border border-gray-200 bg-white">
      <span className={`w-2 h-2 rounded-full ${dotColor}`} />
      <div>
        <b className="block text-xs font-medium text-gray-700">{statusLabels[name] ?? name}</b>
        <small className="block text-[10px] uppercase tracking-wider text-gray-400">
          {value ?? 'unknown'}
        </small>
      </div>
    </div>
  )
}

function SectionTitle({
  eyebrow,
  title,
  meta,
}: {
  eyebrow: string
  title: string
  meta?: string
}) {
  return (
    <div className="flex justify-between items-start gap-3 px-5 py-4 border-b border-gray-100">
      <div>
        <span className="text-[10px] uppercase tracking-[0.18em] text-gray-400 font-medium">
          {eyebrow}
        </span>
        <h2 className="mt-1 text-base font-semibold text-gray-900 tracking-tight">{title}</h2>
      </div>
      {meta && (
        <span className="text-[10px] uppercase tracking-wider text-gray-400 pt-1 whitespace-nowrap">
          {meta}
        </span>
      )}
    </div>
  )
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string
  value: string
  detail?: string
}) {
  return (
    <div className="px-4 py-3 border-r border-gray-100 last:border-r-0">
      <span className="block text-[9px] uppercase tracking-[0.11em] text-gray-400">{label}</span>
      <strong className="block text-base font-semibold text-gray-900 mt-1">{value}</strong>
      {detail && <small className="block text-[9px] text-gray-400 mt-0.5">{detail}</small>}
    </div>
  )
}

function EventList({ events }: { events: Event[] }) {
  return (
    <div className="event-list max-h-[260px] overflow-auto px-5 pb-4">
      {events.length === 0 ? (
        <div className="text-gray-400 py-6 text-sm">等待运行事件…</div>
      ) : (
        events.map((event, index) => (
          <div
            key={`${event.ts}-${index}`}
            className="grid grid-cols-[50px_65px_1fr] gap-2 items-baseline py-3 border-b border-gray-100 last:border-b-0"
          >
            <span
              className={`text-[9px] uppercase tracking-wider font-medium ${
                event.level === 'error'
                  ? 'text-red-500'
                  : event.level === 'warning'
                    ? 'text-amber-500'
                    : 'text-blue-600'
              }`}
            >
              {event.level}
            </span>
            <time className="text-gray-400 text-[10px]">
              {new Date(event.ts * 1000).toLocaleTimeString()}
            </time>
            <p className="text-gray-600 text-xs">{event.message}</p>
          </div>
        ))
      )}
    </div>
  )
}

function App() {
  const { telemetry, events, connected } = useTelemetry()
  const status = telemetry.status ?? {}
  const model = telemetry.model ?? {}
  const healthLevel = telemetry.health?.level ?? 'yellow'
  const inferenceActive = status.inference === 'running'
  const serviceActive = ['starting', 'waiting', 'calibrating', 'running', 'reconnecting'].includes(
    status.inference ?? '',
  )
  const command = inferenceActive && typeof model.control === 'number' ? model.control : 3
  const activeActionName = model.action_name ?? model.control_name ?? ''

  // Service launch states
  const [pythonPath, setPythonPath] = useState('')
  const [scriptPath, setScriptPath] = useState('')
  const [eegSource, setEegSource] = useState<'neuracle' | 'local'>('neuracle')
  const [localEegFile, setLocalEegFile] = useState('')
  const [serviceRunning, setServiceRunning] = useState(false)
  const [gamePath, setGamePath] = useState('')
  const [gameRunning, setGameRunning] = useState(false)
  const [launchMessage, setLaunchMessage] = useState('')
  // Collapse states for launch panels
  const [serviceCollapsed, setServiceCollapsed] = useState(false)
  const [gameCollapsed, setGameCollapsed] = useState(false)
  // EEG display controls
  const [selectedChannels, setSelectedChannels] = useState<string[]>([])
  const [singleChannel, setSingleChannel] = useState<string | null>(null)
  const [zoomRange, setZoomRange] = useState<[number, number] | undefined>(undefined)

  const availableChannels = telemetry.eeg?.channels ?? emptyChannels
  const recordingDisplayPath = displayProjectRelativePath(telemetry.recording?.session_dir)

  useEffect(() => {
    setServiceRunning(serviceActive)
    if (!serviceActive) setServiceCollapsed(false)
  }, [serviceActive])

  useEffect(() => {
    if (availableChannels.length === 0) return
    setSelectedChannels((previous) => {
      const retained = previous.filter((channel) => availableChannels.includes(channel))
      if (retained.length === previous.length) return previous
      return retained.length > 0 ? retained : availableChannels
    })
    setSingleChannel((previous) =>
      previous && availableChannels.includes(previous) ? previous : null,
    )
  }, [availableChannels])

  useEffect(() => {
    if (!('__TAURI_INTERNALS__' in window)) {
      setLaunchMessage('浏览器只读模式：请使用 Tauri GUI 启停服务和游戏')
      return
    }
    invoke<DefaultLaunchPaths>('get_default_launch_paths')
      .then((paths) => {
        setPythonPath(paths.python_path)
        setScriptPath(paths.script_path)
        setGamePath(paths.game_path)
      })
      .catch((error) => setLaunchMessage(`默认路径解析失败: ${String(error)}`))
  }, [])

  const showMessage = (msg: string) => {
    setLaunchMessage(msg)
    setTimeout(() => setLaunchMessage(''), 3000)
  }

  const startService = async () => {
    if (!pythonPath) {
      showMessage('请先选择 Python 解释器')
      return
    }
    if (eegSource === 'local' && !localEegFile.trim()) {
      showMessage('请选择本地 BDF 文件')
      return
    }
    try {
      const args: string[] = [
        '--model',
        'yhc',
        '--channel-list',
        neusenW64StreamChannels.join(','),
        '--live-baseline-sec',
        '30',
        '--min-confidence',
        '0.55',
        '--save-logs',
        '--auto-reconnect',
      ]
      if (eegSource === 'local') {
        args.push('--eeg-source', 'local', '--local-eeg-file', localEegFile)
      } else {
        args.push('--eeg-source', 'neuracle')
      }
      await invoke('start_python_service', {
        payload: {
          python_path: pythonPath,
          script_path: scriptPath,
          args,
        },
      })
      setServiceRunning(true)
      setServiceCollapsed(true)
      showMessage('模型推理服务已启动')
    } catch (e) {
      showMessage(`启动失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const stopService = async () => {
    try {
      await invoke('stop_python_service')
      setServiceRunning(false)
      setServiceCollapsed(false)
      showMessage('模型推理服务已停止')
    } catch (e) {
      showMessage(`停止失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const startGame = async () => {
    if (!gamePath.trim()) {
      showMessage('请选择游戏可执行文件路径')
      return
    }
    try {
      await invoke('start_game_process', { payload: { game_path: gamePath } })
      setGameRunning(true)
      setGameCollapsed(true)
      showMessage('赛车游戏已启动')
    } catch (e) {
      showMessage(`启动失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const stopGame = async () => {
    try {
      await invoke('stop_game_process')
      setGameRunning(false)
      setGameCollapsed(false)
      showMessage('赛车游戏已停止')
    } catch (e) {
      showMessage(`停止失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 px-8 pt-7 pb-5 max-w-[1820px] mx-auto">
      {/* Topbar */}
      <header className="flex justify-between items-start pb-6 border-b border-gray-200">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-full border-2 border-blue-600 flex items-center justify-center text-blue-600 text-xl font-bold shadow-sm">
            ◎
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900 tracking-tight">
              Fly Drive
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-6 text-gray-400 text-[11px] uppercase tracking-wider">
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[10px] font-medium border ${
              connected
                ? 'text-blue-600 border-blue-200 bg-blue-50'
                : 'text-red-500 border-red-200 bg-red-50'
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-blue-500 animate-pulse-dot' : 'bg-red-500'}`}
            />
            {connected ? 'REALTIME LINK' : 'OFFLINE'}
          </span>
          <span>
            {new Date().toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })} ·{' '}
            {new Date().toLocaleTimeString()}
          </span>
        </div>
      </header>

      {/* Launch Control */}
      <section className="grid grid-cols-2 gap-4 py-4">
        {/* Model Service Card */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle eyebrow="LAUNCH" title="模型推理服务" meta={serviceRunning ? '运行中' : '已停止'} />
          {serviceCollapsed ? (
            <div className="p-4 flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-emerald-50 border-emerald-200 text-emerald-700">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                运行中
              </span>
              <button
                onClick={stopService}
                className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition-colors"
              >
                停止服务
              </button>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1.5">项目 Python 环境（相对项目目录）</label>
                  <input
                    type="text"
                    value={pythonPath}
                    onChange={(e) => setPythonPath(e.target.value)}
                    className="w-full px-3 py-1.5 text-xs border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                    placeholder=".venv/Scripts/python.exe"
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">脚本路径（相对项目目录）</label>
                  <input
                    type="text"
                    value={scriptPath}
                    onChange={(e) => setScriptPath(e.target.value)}
                    className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                    placeholder="dashboard_service.py"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">模型类型</label>
                  <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2">
                    <strong className="block text-sm font-medium text-blue-700">YHC（固定）</strong>
                    <span className="block text-[10px] text-blue-600 mt-0.5">四分类 · 4 秒窗口 · 0.5 秒更新</span>
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">TCP 数据帧</label>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2">
                    <strong className="block text-sm font-medium text-emerald-700">64 EEG + TRG</strong>
                    <span className="block text-[10px] text-emerald-600 mt-0.5">
                      完整接收后抽取YHC感觉运动区12导联
                    </span>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">EEG 信号源</label>
                  <select
                    value={eegSource}
                    onChange={(e) => setEegSource(e.target.value as 'neuracle' | 'local')}
                    className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 bg-white"
                  >
                    <option value="neuracle">博睿康采集软件 (TCP)</option>
                    <option value="local">本地 EEG 数据</option>
                  </select>
                </div>
                {eegSource === 'local' && (
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">本地 BDF 文件路径</label>
                    <input
                      type="text"
                      value={localEegFile}
                      onChange={(e) => setLocalEegFile(e.target.value)}
                      className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                      placeholder="path/to/data.bdf"
                    />
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={startService}
                  disabled={serviceRunning}
                  className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                >
                  启动服务
                </button>
                <button
                  onClick={stopService}
                  disabled={!serviceRunning}
                  className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                >
                  停止服务
                </button>
                {launchMessage && <span className="text-xs text-gray-500 ml-2">{launchMessage}</span>}
              </div>
            </div>
          )}
        </div>

        {/* Game Card */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle eyebrow="LAUNCH" title="赛车游戏 Demo" meta={gameRunning ? '运行中' : '已停止'} />
          {gameCollapsed ? (
            <div className="p-4 flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-emerald-50 border-emerald-200 text-emerald-700">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                运行中
              </span>
              <button
                onClick={stopGame}
                className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition-colors"
              >
                停止游戏
              </button>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              <div>
                <label className="block text-[10px] uppercase tracking-wider text-gray-400 mb-1">游戏可执行文件（相对项目目录）</label>
                <input
                  type="text"
                  value={gamePath}
                  onChange={(e) => setGamePath(e.target.value)}
                  className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                  placeholder="..\\虚拟任务竞速赛06251432\\虚拟任务竞速赛.exe"
                />
              </div>
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={startGame}
                  disabled={gameRunning}
                  className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                >
                  启动游戏
                </button>
                <button
                  onClick={stopGame}
                  disabled={!gameRunning}
                  className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                >
                  停止游戏
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Status Strip */}
      <section className="flex gap-2 flex-wrap py-4 items-stretch">
        {Object.keys(statusLabels).map((key) => (
          <StatusChip key={key} name={key} value={status[key] as StatusValue} />
        ))}
        {telemetry.calibration?.state === 'collecting' && (
          <div className="min-w-[220px] flex-1 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2">
            <div className="flex justify-between text-[10px] font-medium text-amber-700">
              <span>静息基线</span>
              <span>{Math.max(0, telemetry.calibration.remaining_seconds ?? 0).toFixed(1)} s</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-amber-100">
              <div
                className="h-full rounded-full bg-amber-500 transition-all"
                style={{
                  width: `${Math.min(
                    100,
                    ((telemetry.calibration.received_seconds ?? 0) /
                      Math.max(telemetry.calibration.required_seconds ?? 30, 0.1)) *
                      100,
                  )}%`,
                }}
              />
            </div>
          </div>
        )}
        <div
          className={`flex items-center justify-center gap-2 flex-1 min-w-[240px] px-4 py-2 rounded-lg border text-sm font-medium ${
            healthLevel === 'green'
              ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
              : healthLevel === 'red'
                ? 'text-red-700 bg-red-50 border-red-200'
                : 'text-amber-700 bg-amber-50 border-amber-200'
          }`}
        >
          <span className="text-lg">
            {healthLevel === 'green' ? '✓' : healthLevel === 'red' ? '!' : '·'}
          </span>
          {telemetry.health?.message ?? '等待服务'}
        </div>
      </section>

      {/* Hero Grid */}
      <section className="grid grid-cols-[1.15fr_1fr] gap-4 max-lg:grid-cols-1">
        {/* Signal Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle
            eyebrow="01 / SENSOR FEED"
            title="EEG live field"
            meta={`${telemetry.eeg?.sample_rate ?? '—'} Hz · ${telemetry.eeg?.source_unit ?? '—'}→µV · 2 s · TRG ${telemetry.eeg?.latest_trigger ?? '—'}`}
          />
          <div className="p-3.5">
            {/* Channel & zoom controls */}
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <button
                onClick={() => {
                  setSelectedChannels(availableChannels)
                  setSingleChannel(null)
                }}
                className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-colors ${
                  singleChannel === null && selectedChannels.length === availableChannels.length
                    ? 'bg-blue-50 border-blue-200 text-blue-600'
                    : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                }`}
              >
                全部
              </button>
              <button
                onClick={() => setSingleChannel((prev) => (prev ? null : selectedChannels[0] ?? null))}
                className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-colors ${
                  singleChannel !== null
                    ? 'bg-amber-50 border-amber-200 text-amber-600'
                    : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                }`}
              >
                单通道
              </button>
              {singleChannel && (
                <span className="text-xs font-semibold text-amber-600 mr-1">{singleChannel}</span>
              )}
              <div className="flex-1" />
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    const max = telemetry.eeg?.values?.[0]?.length ?? 100
                    const span = zoomRange ? zoomRange[1] - zoomRange[0] : max
                    const center = zoomRange ? zoomRange[0] + span / 2 : max / 2
                    const newSpan = Math.max(20, span * 0.7)
                    setZoomRange([Math.max(0, center - newSpan / 2), Math.min(max, center + newSpan / 2)])
                  }}
                  className="px-2 py-0.5 rounded text-[10px] font-medium border border-gray-200 text-gray-500 hover:bg-gray-50 transition-colors"
                >
                  放大
                </button>
                <button
                  onClick={() => {
                    const max = telemetry.eeg?.values?.[0]?.length ?? 100
                    const span = zoomRange ? zoomRange[1] - zoomRange[0] : max
                    const center = zoomRange ? zoomRange[0] + span / 2 : max / 2
                    const newSpan = Math.min(max, span * 1.4)
                    setZoomRange([Math.max(0, center - newSpan / 2), Math.min(max, center + newSpan / 2)])
                  }}
                  className="px-2 py-0.5 rounded text-[10px] font-medium border border-gray-200 text-gray-500 hover:bg-gray-50 transition-colors"
                >
                  缩小
                </button>
                <button
                  onClick={() => setZoomRange(undefined)}
                  className="px-2 py-0.5 rounded text-[10px] font-medium border border-gray-200 text-gray-500 hover:bg-gray-50 transition-colors"
                >
                  重置
                </button>
              </div>
            </div>
            {/* Channel pills */}
            <div className="flex flex-wrap gap-1.5 mb-2">
              {availableChannels.map((ch) => {
                const active = singleChannel ? singleChannel === ch : selectedChannels.includes(ch)
                return (
                  <button
                    key={ch}
                    onClick={() => {
                      if (singleChannel) {
                        setSingleChannel(ch)
                      } else {
                        setSelectedChannels((prev) =>
                          prev.includes(ch) ? prev.filter((c) => c !== ch) : [...prev, ch],
                        )
                      }
                    }}
                    className={`px-2 py-0.5 rounded-full text-[10px] border transition-colors ${
                      active
                        ? 'bg-blue-50 border-blue-200 text-blue-600'
                        : 'bg-white border-gray-200 text-gray-400 hover:bg-gray-50'
                    }`}
                  >
                    {ch}
                  </button>
                )
              })}
            </div>
            <SignalChart
              telemetry={telemetry}
              selectedChannels={singleChannel ? [singleChannel] : selectedChannels}
              singleChannel={singleChannel}
              zoomRange={zoomRange}
              onZoomChange={setZoomRange}
            />
            <div className="flex justify-between text-[10px] text-gray-400 pt-2 border-t border-gray-100 mt-2">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse-dot" />
                4–38 Hz / CAR · 自动缩放
              </span>
              <span>{telemetry.eeg?.channels?.length ?? 0} channels</span>
            </div>
          </div>
        </div>

        {/* Command Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle
            eyebrow="02 / MODEL READOUT"
            title="Classification"
            meta={model.name ?? '—'}
          />
          <div className="grid grid-cols-[68px_1fr_auto] gap-3 items-center px-5 py-5">
            <div className="w-[60px] h-[60px] rounded-full border-2 border-blue-500 flex items-center justify-center text-blue-600 text-2xl bg-blue-50/50">
              {controlGlyphs[command]}
            </div>
            <div>
              <span className="text-[10px] uppercase tracking-[0.18em] text-gray-400 font-medium">
                控制命令
              </span>
              <h3 className="text-xl font-semibold text-blue-600 mt-1 tracking-wide">
                {inferenceActive && activeActionName
                  ? actionLabels[activeActionName] ?? activeActionName.toUpperCase()
                  : 'WAITING'}
              </h3>
            </div>
            <div className="text-right">
              <strong className="block text-3xl font-light text-amber-500">
                {inferenceActive ? Math.round((model.confidence ?? 0) * 100) : 0}
                <small className="text-lg">%</small>
              </strong>
              <span className="text-[10px] uppercase tracking-wider text-gray-400">confidence</span>
            </div>
          </div>
          <ProbabilityChart telemetry={telemetry} active={inferenceActive} />
          <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100">
            <Metric label="更新周期" value="0.5 s" detail="sliding inference" />
            <Metric label="设备" value={model.device?.toUpperCase() ?? '—'} detail="torch runtime" />
            <Metric label="平滑" value="3 frame" detail="probability mean" />
          </div>
        </div>
      </section>

      {/* Lower Grid */}
      <section className="grid grid-cols-1 gap-4 mt-4 max-lg:grid-cols-1">
        {/* Events Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle eyebrow="03 / EVENT LOG" title="Trace" meta={`${events.length} recent`} />
          {telemetry.recording?.enabled && (
            <div className="mx-5 mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-700">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <strong>实验日志：{telemetry.recording.state ?? 'recording'}</strong>
                <span>
                  {telemetry.recording.samples_saved ?? 0} samples · {telemetry.recording.predictions_saved ?? 0} predictions
                </span>
              </div>
              <div className="mt-1 truncate font-mono text-[10px]" title={recordingDisplayPath}>
                {recordingDisplayPath}
              </div>
            </div>
          )}
          <EventList events={events} />
        </div>
      </section>
    </main>
  )
}

export default App
