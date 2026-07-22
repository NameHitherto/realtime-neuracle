export type StatusValue = 'connected' | 'running' | 'idle' | 'starting' | 'connecting' | 'stopped' | 'error' | 'unknown' | 'dry-run' | 'unavailable'

export type Telemetry = {
  seq?: number
  ts?: number
  status?: Record<string, StatusValue>
  health?: { level?: 'green' | 'yellow' | 'red'; message?: string }
  eeg?: { sample_rate?: number; channels?: string[]; values?: number[][]; display_rate?: number; last_data_at?: number }
  model?: { name?: string; class_names?: string[]; probabilities?: number[]; prediction?: string; confidence?: number; control?: number; control_name?: string; device?: string; last_inference_at?: number }
  lsl?: { stream_name?: string; stream_type?: string; sample_rate?: number; last_sent?: number; last_sent_at?: number; consumer_online?: boolean }
  game?: { telemetry_level?: string; capture_age_ms?: number; capture_error?: string; capture_size?: number }
}

export type Event = { ts: number; level: string; message: string; details?: Record<string, unknown> }
