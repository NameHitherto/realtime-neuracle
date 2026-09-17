export type StatusValue = 'connected' | 'running' | 'idle' | 'starting' | 'connecting' | 'validating' | 'calibrating' | 'waiting' | 'stale' | 'reconnecting' | 'stopped' | 'error' | 'unknown' | 'dry-run' | 'unavailable'

export type Telemetry = {
  seq?: number
  ts?: number
  status?: Record<string, StatusValue>
  health?: { level?: 'green' | 'yellow' | 'red'; message?: string }
  eeg?: { sample_rate?: number; received_sample_rate?: number; channels?: string[]; values?: number[][]; display_rate?: number; unit?: string; filtered?: boolean; source_unit?: string; source_scale_to_uv?: number; last_data_at?: number; latest_trigger?: number }
  model?: { name?: string; class_names?: string[]; probabilities?: number[]; prediction?: string; confidence?: number; control?: number; control_name?: string; action_name?: string; device?: string; last_inference_at?: number }
  lsl?: { stream_name?: string; stream_type?: string; sample_rate?: number; last_sent?: number; last_sent_at?: number; consumer_online?: boolean }
  game?: { telemetry_level?: string; capture_age_ms?: number; capture_error?: string; capture_size?: number; transport?: string; last_sent?: number | null; last_sent_at?: number | null; peer_connected?: boolean; error?: string | null; delivery?: string; mode?: string }
  recording?: { enabled?: boolean; state?: string; session_dir?: string; samples_saved?: number; predictions_saved?: number }
  calibration?: { state?: 'collecting' | 'completed'; required_seconds?: number; received_seconds?: number; remaining_seconds?: number; completed_at?: number }
}

export type Event = { ts: number; level: string; message: string; details?: Record<string, unknown> }
