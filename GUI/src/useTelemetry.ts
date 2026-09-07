import { useCallback, useEffect, useRef, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import type { Event, Telemetry } from './types'

const empty: Telemetry = {
  status: {
    device_tcp: 'idle',
    inference: 'idle',
    lsl_outlet: 'idle',
    lsl_consumer: 'unknown',
    game_process: 'unknown',
    game_capture: 'idle',
  },
  health: { level: 'yellow', message: '连接后等待实时服务' },
  model: {
    class_names: ['rest', 'feet', 'left_hand', 'right_hand'],
    probabilities: [],
  },
  eeg: { channels: [], values: [] },
  lsl: { stream_name: 'EEGback', stream_type: 'EEG' },
  game: { telemetry_level: 'inferred' },
}

const isTauriRuntime = () => '__TAURI_INTERNALS__' in window

const actionLabels: Record<string, string> = {
  left: 'LEFT',
  right: 'RIGHT',
  forward: 'FORWARD',
  accelerate: 'SPEED UP',
  stop: 'STOP',
}

const eventKey = (event: Event) => `${event.level}:${event.ts}:${event.message}`

function mergeEvents(...groups: Event[][]): Event[] {
  const seen = new Set<string>()
  const unique = groups
    .flat()
    .sort((a, b) => b.ts - a.ts)
    .filter((event) => {
      const key = eventKey(event)
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })

  // Inference results refresh twice a second. Keep a useful live tail without
  // allowing it to push connection/errors and other system events out of Trace.
  const inferenceEvents = unique.filter((event) => event.level === 'infer').slice(0, 20)
  const systemEvents = unique.filter((event) => event.level !== 'infer').slice(0, 40 - inferenceEvents.length)
  return [...inferenceEvents, ...systemEvents].sort((a, b) => b.ts - a.ts)
}

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<Telemetry>(empty)
  const [events, setEvents] = useState<Event[]>([])
  const [connected, setConnected] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const lastInferenceEvent = useRef(0)

  const acceptTelemetry = useCallback((payload: Telemetry) => {
    setTelemetry(payload)
    setRefresh((value) => value + 1)

    const model = payload.model
    const inferenceAt = model?.last_inference_at ?? 0
    if (!model?.prediction || inferenceAt <= lastInferenceEvent.current) return

    lastInferenceEvent.current = inferenceAt
    const confidence = Math.max(0, model.confidence ?? 0)
    const controlName = model.action_name ?? model.control_name ?? '—'
    const actionLabel = actionLabels[controlName] ?? controlName
    const inferenceEvent: Event = {
      ts: inferenceAt,
      level: 'infer',
      message: `分类：${model.prediction} · ${(confidence * 100).toFixed(1)}% → ${actionLabel}`,
      details: {
        prediction: model.prediction,
        confidence,
        control: model.control,
        control_name: controlName,
      },
    }
    setEvents((current) => mergeEvents([inferenceEvent], current))
  }, [])

  const loadEvents = useCallback(async () => {
    try {
      const response = isTauriRuntime()
        ? await invoke<Event[]>('get_events', { limit: 40 })
        : await fetch('/api/events?limit=40').then((result) => {
            if (!result.ok) throw new Error(`HTTP ${result.status}`)
            return result.json() as Promise<Event[]>
          })
      setEvents((current) => mergeEvents(response, current.filter((event) => event.level === 'infer')))
    } catch {
      // WebSocket status remains the primary health indicator.
    }
  }, [])

  useEffect(() => {
    let unlistenTelemetry: UnlistenFn | null = null
    let unlistenConnected: UnlistenFn | null = null
    let socket: WebSocket | null = null
    let interval: number | undefined

    const setup = async () => {
      if (!isTauriRuntime()) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        socket = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`)
        socket.onopen = () => {
          setConnected(true)
          loadEvents()
        }
        socket.onclose = () => setConnected(false)
        socket.onerror = () => setConnected(false)
        socket.onmessage = (event) => {
          const payload = JSON.parse(event.data) as Telemetry
          acceptTelemetry(payload)
        }
        interval = window.setInterval(loadEvents, 3000)
        return
      }

      unlistenTelemetry = await listen<Telemetry>('telemetry-update', (event) => {
        acceptTelemetry(event.payload)
      })

      unlistenConnected = await listen<boolean>('telemetry-connected', (event) => {
        setConnected(event.payload)
        if (event.payload) loadEvents()
      })

      interval = window.setInterval(loadEvents, 3000)
    }

    setup()

    return () => {
      if (unlistenTelemetry) unlistenTelemetry()
      if (unlistenConnected) unlistenConnected()
      if (socket) socket.close()
      window.clearInterval(interval)
    }
  }, [acceptTelemetry, loadEvents])

  return { telemetry, events, connected, refresh }
}

export async function sendManualControl(control: number) {
  return invoke<Telemetry>('send_manual_control', {
    control,
    durationMs: 1000,
  })
}
