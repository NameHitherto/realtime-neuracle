import { useEffect, useMemo, useRef } from 'react'
import * as echarts from 'echarts'
import type { Telemetry } from './types'

const palette = ['#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#06b6d4', '#ec4899']

export function SignalChart({ telemetry }: { telemetry: Telemetry }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const values = telemetry.eeg?.values ?? []
  const channels = telemetry.eeg?.channels ?? []

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

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    const series = channels.map((channel, index) => ({
      name: channel,
      type: 'line' as const,
      showSymbol: false,
      smooth: false,
      data: (values[index] ?? []).map((value, point) => [
        point,
        Number((value / 20 + index * 3).toFixed(3)),
      ]),
      lineStyle: { width: 1.2, color: palette[index % palette.length] },
    }))
    chart.setOption(
      {
        animation: false,
        backgroundColor: 'transparent',
        grid: { left: 8, right: 16, top: 10, bottom: 18, containLabel: false },
        tooltip: {
          trigger: 'axis',
          backgroundColor: '#ffffff',
          borderColor: '#e5e7eb',
          textStyle: { color: '#111827' },
        },
        xAxis: { type: 'value', show: false, min: 0 },
        yAxis: { type: 'value', show: false },
        series,
      },
      true,
    )
  }, [channels, values])

  return (
    <div>
      <div ref={ref} className="signal-chart" />
      <div className="flex gap-3.5 flex-wrap text-[10px] text-gray-500 px-1 pt-2 pb-2">
        {channels.map((channel, i) => (
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

export function ProbabilityChart({ telemetry }: { telemetry: Telemetry }) {
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

  const labels = telemetry.model?.class_names ?? []
  const probs = telemetry.model?.probabilities ?? []

  useEffect(() => {
    chartRef.current?.setOption(
      {
        animationDuration: 300,
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
          data: labels,
          axisLabel: { color: '#374151' },
          axisLine: { show: false },
          axisTick: { show: false },
        },
        series: [
          {
            type: 'bar',
            data: probs.map((value, index) => ({
              value,
              itemStyle: {
                color: palette[index % palette.length],
                borderRadius: [0, 4, 4, 0],
              },
            })),
            barWidth: 14,
            label: {
              show: true,
              position: 'right',
              color: '#111827',
              formatter: (params: { value: number }) => `${Math.round(params.value * 100)}%`,
            },
          },
        ],
      },
      true,
    )
  }, [labels, probs])

  return <div ref={ref} className="probability-chart" />
}
