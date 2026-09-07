import { useEffect, useMemo, useRef } from 'react'
import * as echarts from 'echarts'
import type { Telemetry } from './types'

const palette = ['#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#06b6d4', '#ec4899']
const emptyValues: number[][] = []
const emptyChannels: string[] = []
const emptyProbabilities: number[] = []

const classLabels: Record<string, string> = {
  rest: '静息',
  feet: '双脚',
  left_hand: '左手',
  right_hand: '右手',
}

function formatProbability(value: number): string {
  const percent = value * 100
  if (percent <= 0) return '0%'
  if (percent < 1) return '<1%'
  if (percent < 10) return `${percent.toFixed(1)}%`
  return `${Math.round(percent)}%`
}

export function SignalChart({
  telemetry,
  selectedChannels,
  singleChannel,
  zoomRange,
  onZoomChange,
}: {
  telemetry: Telemetry
  selectedChannels?: string[]
  singleChannel?: string | null
  zoomRange?: [number, number]
  onZoomChange?: (range: [number, number]) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const values = telemetry.eeg?.values ?? emptyValues
  const channels = telemetry.eeg?.channels ?? emptyChannels

  useEffect(() => {
    if (!ref.current) return
    chartRef.current = echarts.init(ref.current)
    const resize = () => chartRef.current?.resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      chartRef.current?.dispose()
      chartRef.current = null
    }
  }, [])

  // Filter channels based on selectedChannels and singleChannel
  const displayChannels = useMemo(
    () =>
      singleChannel
        ? [singleChannel]
        : selectedChannels && selectedChannels.length > 0
          ? selectedChannels.filter((ch) => channels.includes(ch))
          : channels,
    [channels, selectedChannels, singleChannel],
  )

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const isSingle = !!singleChannel
    const channelIndices = displayChannels.map((ch) => channels.indexOf(ch)).filter((i) => i >= 0)

    // High electrode impedance can leave only tiny post-CAR/filter amplitudes.
    // The old fixed divisor plus three-decimal rounding flattened those samples
    // to zero. Use one robust gain for the visible window so relative channel
    // amplitudes remain comparable. This only changes display coordinates.
    const centeredRows = channelIndices.map((chIndex) => {
      const row = values[chIndex] ?? []
      const finiteValues = row.filter(Number.isFinite)
      const mean = finiteValues.length > 0
        ? finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
        : 0
      return row.map((value) => (Number.isFinite(value) ? value - mean : 0))
    })
    const absoluteValues = centeredRows
      .flatMap((row) => row.map((value) => Math.abs(value)))
      .sort((a, b) => a - b)
    const robustIndex = Math.max(0, Math.ceil(absoluteValues.length * 0.98) - 1)
    const robustAmplitude = absoluteValues.length > 0
      ? Math.max(absoluteValues[robustIndex], 1e-6)
      : 1
    const channelSpacing = 3
    const laneAmplitude = 1.05

    const series = channelIndices.map((chIndex, i) => {
      const channel = channels[chIndex]
      const data = (values[chIndex] ?? []).map((value, point) => [
        point,
        isSingle
          ? (Number.isFinite(value) ? value : 0)
          : Math.max(
              -1.3,
              Math.min(1.3, centeredRows[i][point] / robustAmplitude * laneAmplitude),
            ) + i * channelSpacing,
      ])
      return {
        name: channel,
        type: 'line' as const,
        showSymbol: false,
        smooth: false,
        data,
        lineStyle: { width: 1.2, color: palette[i % palette.length] },
      }
    })

    const xAxis = {
      type: 'value' as const,
      show: isSingle || !!zoomRange,
      min: zoomRange ? zoomRange[0] : 0,
      axisLabel: { color: '#6b7280', fontSize: 10 },
      splitLine: { show: false },
    }

    const yAxis = {
      type: 'value' as const,
      show: isSingle,
      axisLabel: { color: '#6b7280', fontSize: 10 },
      splitLine: { lineStyle: { color: '#e5e7eb' } },
    }

    chart.setOption(
      {
        animation: false,
        backgroundColor: 'transparent',
        grid: {
          left: isSingle ? 48 : 8,
          right: 16,
          top: 10,
          bottom: isSingle || !!zoomRange ? 28 : 18,
          containLabel: false,
        },
        tooltip: {
          trigger: 'axis',
          backgroundColor: '#ffffff',
          borderColor: '#e5e7eb',
          textStyle: { color: '#111827' },
        },
        dataZoom: [
          { type: 'inside' as const },
          { type: 'slider' as const, bottom: 4, height: 16, show: true },
        ],
        xAxis,
        yAxis,
        series,
      },
      true,
    )

    if (onZoomChange) {
      chart.off('dataZoom')
      chart.on('dataZoom', () => {
        const opt = chart.getOption()
        const xAxisOpt = (opt.xAxis as unknown as { min?: number; max?: number }[])[0]
        if (xAxisOpt && typeof xAxisOpt.min === 'number' && typeof xAxisOpt.max === 'number') {
          onZoomChange([xAxisOpt.min, xAxisOpt.max])
        }
      })
    }
  }, [channels, values, displayChannels, singleChannel, zoomRange, onZoomChange])

  return (
    <div>
      <div ref={ref} className="signal-chart" />
      <div className="flex gap-3.5 flex-wrap text-[10px] text-gray-500 px-1 pt-2 pb-2">
        {displayChannels.map((channel, i) => (
          <span key={channel}>
            <i
              className="w-[7px] h-[7px] inline-block mr-1 rounded-sm"
              style={{ background: palette[i % palette.length] }}
            />
            {channel}
          </span>
        ))}
      </div>
    </div>
  )
}

export function ProbabilityChart({ telemetry, active = true }: { telemetry: Telemetry; active?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!ref.current) return
    chartRef.current = echarts.init(ref.current)
    const resize = () => chartRef.current?.resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      chartRef.current?.dispose()
      chartRef.current = null
    }
  }, [])

  const labels = telemetry.model?.class_names ?? emptyChannels
  const probs = active ? telemetry.model?.probabilities ?? emptyProbabilities : emptyProbabilities
  const safeProbabilities = labels.map((_, index) => {
    const value = Number(probs[index] ?? 0)
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
  })
  // Telemetry also refreshes for EEG and LSL updates. Only repaint this chart
  // when its own labels/probabilities change, otherwise an animation/update can
  // be restarted several times between the model's 0.5 s inference frames.
  const probabilityUpdateKey = `${labels.join('\u0000')}|${safeProbabilities.join(',')}`
  const probabilityDataRef = useRef({ labels, probabilities: safeProbabilities })
  probabilityDataRef.current = { labels, probabilities: safeProbabilities }

  useEffect(() => {
    const probabilityData = probabilityDataRef.current
    chartRef.current?.setOption(
      {
        animation: false,
        animationDuration: 0,
        animationDurationUpdate: 0,
        grid: { left: 88, right: 18, top: 10, bottom: 18 },
        xAxis: {
          type: 'value',
          max: 1,
          splitLine: { lineStyle: { color: '#e5e7eb' } },
          axisLabel: {
            color: '#6b7280',
            formatter: (v: number) => `${Math.round(v * 100)}%`,
          },
        },
        yAxis: {
          type: 'category',
          data: probabilityData.labels.map((label) => classLabels[label] ?? label),
          axisLabel: { color: '#374151' },
          axisLine: { show: false },
          axisTick: { show: false },
        },
        series: [
          {
            type: 'bar',
            id: 'yhc-probabilities',
            data: probabilityData.probabilities.map((value, index) => ({
              value,
              itemStyle: {
                color: palette[index % palette.length],
                borderRadius: value === 0 ? 0 : [0, 4, 4, 0],
              },
            })),
            barWidth: 14,
            label: {
              show: true,
              position: 'right',
              color: '#111827',
              formatter: (params: { value: number }) => formatProbability(params.value),
            },
          },
        ],
      },
      true,
    )
  }, [probabilityUpdateKey])

  return <div ref={ref} className="probability-chart" />
}
