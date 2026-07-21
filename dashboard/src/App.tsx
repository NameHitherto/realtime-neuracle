import { useMemo, useState } from 'react'
import { ProbabilityChart, SignalChart } from './charts'
import { sendManualControl, useTelemetry } from './useTelemetry'
import type { Event, StatusValue, Telemetry } from './types'

const statusLabels: Record<string, string> = {
  device_tcp: 'EEG TCP',
  inference: '模型推理',
  lsl_outlet: 'LSL 输出',
  lsl_consumer: 'LSL 消费者',
  game_process: '游戏进程',
  game_capture: '游戏画面',
}

const controlLabels = ['左移', '右移', '前进', '停止']
const controlGlyphs = ['←', '→', '↑', '■']

function StatusChip({ name, value }: { name: string; value?: StatusValue }) {
  const good = value === 'connected' || value === 'running'
  const bad = value === 'error' || value === 'stopped' || value === 'unavailable'
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
  const [controlMessage, setControlMessage] = useState('')
  const status = telemetry.status ?? {}
  const model = telemetry.model ?? {}
  const lsl = telemetry.lsl ?? {}
  const game = telemetry.game ?? {}
  const age = telemetry.ts
    ? Math.max(0, Math.round((Date.now() / 1000 - telemetry.ts) * 1000))
    : null
  const healthLevel = telemetry.health?.level ?? 'yellow'
  const command = typeof model.control === 'number' ? model.control : 3

  const sendTest = async (value: number) => {
    setControlMessage('发送中…')
    try {
      await sendManualControl(value)
      setControlMessage(`已发送 ${controlLabels[value]}`)
    } catch (error) {
      setControlMessage(error instanceof Error ? error.message : '发送失败')
    }
    window.setTimeout(() => setControlMessage(''), 2200)
  }

  const frameUrl = useMemo(() => `/media/game.jpg?seq=${telemetry.seq ?? 0}`, [telemetry.seq])

  return (
    <main className="min-h-screen bg-gray-50 px-8 pt-7 pb-5 max-w-[1820px] mx-auto">
      {/* Topbar */}
      <header className="flex justify-between items-start pb-6 border-b border-gray-200">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-full border-2 border-blue-600 flex items-center justify-center text-blue-600 text-xl font-bold shadow-sm">
            ◎
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-gray-400 font-medium">
              HCI LAB / LIVE INSTRUMENTATION
            </div>
            <h1 className="text-xl font-semibold text-gray-900 tracking-tight">
              BCI Racing <em className="text-blue-600 not-italic font-light">Control Room</em>
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

      {/* Status Strip */}
      <section className="flex gap-2 flex-wrap py-4 items-stretch">
        {Object.keys(statusLabels).map((key) => (
          <StatusChip key={key} name={key} value={status[key] as StatusValue} />
        ))}
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
      <section className="grid grid-cols-[1.15fr_1fr_1.08fr] gap-4 max-lg:grid-cols-1">
        {/* Signal Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle
            eyebrow="01 / SENSOR FEED"
            title="EEG live field"
            meta={`${telemetry.eeg?.sample_rate ?? '—'} Hz · 2 s window`}
          />
          <div className="p-3.5">
            <SignalChart telemetry={telemetry} />
            <div className="flex justify-between text-[10px] text-gray-400 pt-2 border-t border-gray-100 mt-2">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse-dot" />
                滤波后显示 · 自动缩放
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
                CURRENT COMMAND
              </span>
              <h3 className="text-xl font-semibold text-blue-600 mt-1 tracking-wide">
                {model.control_name ? model.control_name.toUpperCase() : 'WAITING'}
              </h3>
              <p className="text-xs text-gray-500 mt-1">
                {model.prediction ?? '等待有效推理窗口'}
              </p>
            </div>
            <div className="text-right">
              <strong className="block text-3xl font-light text-amber-500">
                {Math.round((model.confidence ?? 0) * 100)}
                <small className="text-lg">%</small>
              </strong>
              <span className="text-[10px] uppercase tracking-wider text-gray-400">confidence</span>
            </div>
          </div>
          <ProbabilityChart telemetry={telemetry} />
          <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100">
            <Metric label="更新周期" value="0.5 s" detail="sliding inference" />
            <Metric label="设备" value={model.device?.toUpperCase() ?? '—'} detail="torch runtime" />
            <Metric label="平滑" value="3 frame" detail="probability mean" />
          </div>
        </div>

        {/* Game Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle
            eyebrow="03 / CLIENT OBSERVATION"
            title="Game window"
            meta={game.telemetry_level ?? 'inferred'}
          />
          <div className="h-[310px] relative bg-gray-900 overflow-hidden rounded-b-none">
            <img
              src={frameUrl}
              className="w-full h-full object-cover block"
              onError={(event) => {
                event.currentTarget.style.display = 'none'
              }}
            />
            <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 gap-2 bg-gray-900/90">
              <span className="text-blue-500 text-3xl">◉</span>
              <b className="text-xs uppercase tracking-[0.14em]">WAITING FOR GAME CAPTURE</b>
              <small className="text-xs">启动 dashboard 服务后将显示本机画面</small>
            </div>
            <div className="absolute inset-0 pointer-events-none flex justify-between items-start p-3 text-[9px] uppercase tracking-wider text-white drop-shadow-md">
              <span>
                CAPTURE / {game.capture_age_ms != null ? `${game.capture_age_ms} ms` : '—'}
              </span>
              <span>NO DIRECT ACK</span>
            </div>
          </div>
          <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100">
            <div className="px-3.5 py-3">
              <span className="block text-[9px] uppercase tracking-wider text-gray-400">Python sent</span>
              <b className="block text-sm text-gray-900 mt-1">
                {typeof lsl.last_sent === 'number'
                  ? `${controlGlyphs[lsl.last_sent]} ${controlLabels[lsl.last_sent]}`
                  : '—'}
              </b>
            </div>
            <div className="px-3.5 py-3">
              <span className="block text-[9px] uppercase tracking-wider text-gray-400">Consumer</span>
              <b className={`block text-sm mt-1 ${lsl.consumer_online ? 'text-emerald-600' : 'text-amber-500'}`}>
                {lsl.consumer_online ? 'ONLINE' : 'NOT FOUND'}
              </b>
            </div>
            <div className="px-3.5 py-3">
              <span className="block text-[9px] uppercase tracking-wider text-gray-400">Stream</span>
              <b className="block text-sm text-gray-900 mt-1">
                {lsl.stream_name ?? 'EEGback'} <small className="text-gray-400 font-normal">| {lsl.stream_type ?? 'EEG'}</small>
              </b>
            </div>
          </div>
        </div>
      </section>

      {/* Lower Grid */}
      <section className="grid grid-cols-[1.72fr_1fr] gap-4 mt-4 max-lg:grid-cols-1">
        {/* Timeline Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle eyebrow="04 / OPERATOR TEST" title="Control timeline" meta="manual test only" />
          <div className="px-5 py-4 flex items-center gap-2">
            {controlLabels.map((label, index) => (
              <button
                key={label}
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                  command === index
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                }`}
                onClick={() => void sendTest(index)}
              >
                <span className="text-base mr-1.5 align-[-1px]">{controlGlyphs[index]}</span>
                {label}
              </button>
            ))}
            <span className="ml-auto text-[10px] text-gray-400">{controlMessage}</span>
          </div>
          <div className="h-[92px] mx-6 mb-2.5 relative">
            <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-200" />
            <div
              className="absolute top-[34%] h-[34%] w-0.5 bg-amber-500 shadow-sm"
              style={{ left: `${Math.min(96, ((telemetry.seq ?? 0) % 100))}%` }}
            />
            {[0, 1, 2, 3, 4, 5, 6, 7].map((tick) => (
              <span
                key={tick}
                className="absolute bottom-1.5 text-gray-400 text-[9px] -translate-x-1/2"
                style={{ left: `${tick * 14}%` }}
              >
                {tick * 5}s
              </span>
            ))}
          </div>
          <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100">
            <Metric label="Last update" value={age != null ? `${age} ms` : '—'} detail="dashboard age" />
            <Metric
              label="Last LSL"
              value={lsl.last_sent_at ? new Date(lsl.last_sent_at * 1000).toLocaleTimeString() : '—'}
              detail="published"
            />
            <Metric
              label="Frame size"
              value={game.capture_size ? `${Math.round(game.capture_size / 1024)} KB` : '—'}
              detail="jpeg snapshot"
            />
          </div>
        </div>

        {/* Events Panel */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <SectionTitle eyebrow="05 / EVENT LOG" title="Trace" meta={`${events.length} recent`} />
          <EventList events={events} />
        </div>
      </section>

      {/* Footer */}
      <footer className="flex justify-between py-5 text-xs text-gray-400 tracking-wide">
        <span>BCI Racing Instrumentation · local diagnostic build 0.1</span>
        <span>
          Telemetry truth level: <b className="text-amber-500 font-normal">{game.telemetry_level ?? 'inferred'}</b>
        </span>
      </footer>
    </main>
  )
}

export default App
