import { useCallback, useEffect, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
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
      const response = await invoke<Event[]>('get_events', { limit: 40 })
      setEvents(response)
    } catch { /* websocket status remains the primary health indicator */ }
  }, [])

  useEffect(() => {
    let unlistenTelemetry: UnlistenFn | null = null
    let unlistenConnected: UnlistenFn | null = null
    let interval: number | undefined

    const setup = async () => {
      // Listen for real-time telemetry updates from Rust backend
      unlistenTelemetry = await listen<Telemetry>('telemetry-update', (event) => {
        setTelemetry(event.payload)
        setRefresh((v) => v + 1)
      })

      // Listen for connection status changes
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
      window.clearInterval(interval)
    }
  }, [loadEvents])

  return { telemetry, events, connected, refresh }
}

export async function sendManualControl(control: number) {
  const response = await invoke<Telemetry>('send_manual_control', {
    control,
    durationMs: 1000,
  })
  return response
}
