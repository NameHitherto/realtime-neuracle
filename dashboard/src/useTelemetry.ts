import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Event, Telemetry } from './types'

const empty: Telemetry = {
  status: { device_tcp: 'idle', inference: 'idle', lsl_outlet: 'idle', lsl_consumer: 'unknown', game_process: 'unknown', game_capture: 'idle' },
  health: { level: 'yellow', message: '连接后等待实时服务' },
  model: { class_names: ['feet', 'left_hand', 'right_hand', 'tongue'], probabilities: [] },
  eeg: { channels: [], values: [] },
  lsl: { stream_name: 'EEGback', stream_type: 'EEG' },
  game: { telemetry_level: 'inferred' },
}

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<Telemetry>(empty)
  const [events, setEvents] = useState<Event[]>([])
  const [connected, setConnected] = useState(false)
  const [refresh, setRefresh] = useState(0)

  const loadEvents = useCallback(async () => {
    try {
      const response = await fetch('/api/events?limit=40')
      if (response.ok) setEvents(await response.json())
    } catch { /* websocket status remains the primary health indicator */ }
  }, [])

  useEffect(() => {
    let socket: WebSocket | null = null
    let retry: number | undefined
    let closed = false
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const host = window.location.host
      socket = new WebSocket(`${protocol}://${host}/ws/telemetry`)
      socket.onopen = () => { setConnected(true); loadEvents() }
      socket.onmessage = (message) => {
        try { setTelemetry(JSON.parse(message.data)); setRefresh((value) => value + 1) } catch { /* ignore malformed frames */ }
      }
      socket.onclose = () => { setConnected(false); if (!closed) retry = window.setTimeout(connect, 1200) }
      socket.onerror = () => socket?.close()
    }
    connect()
    const interval = window.setInterval(loadEvents, 3000)
    return () => { closed = true; if (retry) window.clearTimeout(retry); window.clearInterval(interval); socket?.close() }
  }, [loadEvents])

  return { telemetry, events, connected, refresh }
}

export async function sendManualControl(control: number) {
  const response = await fetch('/api/test/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ control, duration_ms: 1000 }),
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}
