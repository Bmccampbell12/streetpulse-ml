import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDashboardStats } from '../services/api'
import type { AlertLevel, DashboardAlert, DashboardStats, SeverityEntry, TrendDir, TrendPoint } from '../types'
import { LABELS } from '../types'

interface DashboardPageProps {
  onBack: () => void
}

// ---------------------------------------------------------------------------
// Severity colour config
// ---------------------------------------------------------------------------
const SEVERITY_COLORS: Record<string, string> = {
  pothole: '#ef4444',
  crack: '#f97316',
  manhole: '#a78bfa',
  normal: '#22c55e',
}

const LABEL_BG: Record<string, string> = {
  pothole: 'bg-red-600/20 text-red-300',
  crack: 'bg-orange-600/20 text-orange-300',
  manhole: 'bg-violet-600/20 text-violet-300',
  normal: 'bg-emerald-600/20 text-emerald-300',
}

const ALERT_STYLES: Record<AlertLevel, { bar: string; badge: string; icon: string }> = {
  critical: {
    bar: 'border-red-700 bg-red-950/70',
    badge: 'bg-red-700 text-white',
    icon: '⚠',
  },
  warning: {
    bar: 'border-amber-700 bg-amber-950/60',
    badge: 'bg-amber-700 text-white',
    icon: '!',
  },
  info: {
    bar: 'border-cyan-700 bg-cyan-950/50',
    badge: 'bg-cyan-700 text-white',
    icon: 'i',
  },
}

const TREND_ICON: Record<TrendDir, string> = {
  up: '↑',
  down: '↓',
  flat: '→',
}
const TREND_COLOR: Record<TrendDir, string> = {
  up: 'text-red-400',
  down: 'text-emerald-400',
  flat: 'text-slate-400',
}

// ---------------------------------------------------------------------------
// SVG Sparkline
// ---------------------------------------------------------------------------
interface SparklineProps {
  points: number[]
  color: string
  width?: number
  height?: number
}

function Sparkline({ points, color, width = 120, height = 40 }: SparklineProps) {
  if (points.length < 2) {
    return <svg width={width} height={height} />
  }
  const max = Math.max(...points, 1)
  const step = width / (points.length - 1)
  const coords = points.map((v, i) => [i * step, height - (v / max) * (height - 4) - 2] as const)
  const d = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const area =
    `${d} L${(coords.at(-1)![0]).toFixed(1)},${height} L0,${height} Z`
  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`sg-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#sg-${color.replace('#', '')})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Multi-line trend chart
// ---------------------------------------------------------------------------
interface TrendChartProps {
  data: TrendPoint[]
}

function TrendChart({ data }: TrendChartProps) {
  const W = 720
  const H = 160
  const PAD = { top: 12, right: 12, bottom: 28, left: 32 }
  const chartW = W - PAD.left - PAD.right
  const chartH = H - PAD.top - PAD.bottom
  const n = data.length

  if (n === 0) return <div className="flex h-40 items-center justify-center text-slate-500 text-sm">No trend data yet</div>

  const maxVal = Math.max(...LABELS.flatMap((lbl) => data.map((d) => (d as unknown as Record<string, number>)[lbl] ?? 0)), 1)
  const xPos = (i: number) => PAD.left + (i / Math.max(n - 1, 1)) * chartW
  const yPos = (v: number) => PAD.top + chartH - (v / maxVal) * chartH

  // Y gridlines — use frac as key to guarantee uniqueness regardless of maxVal.
  const grid = [0, 0.25, 0.5, 0.75, 1.0].map((frac) => ({
    frac,
    y: PAD.top + chartH * (1 - frac),
    label: Math.round(frac * maxVal),
  }))

  // X axis labels (every ~7 days)
  const xLabels = data
    .map((d, i) => ({ i, date: d.date }))
    .filter((_, i) => i % 7 === 0 || i === n - 1)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
      {/* Grid */}
      {grid.map(({ frac, y, label }) => (
        <g key={frac}>
          <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="#1e293b" strokeWidth="1" />
          <text x={PAD.left - 4} y={y + 4} textAnchor="end" className="fill-slate-500" fontSize="9">
            {label}
          </text>
        </g>
      ))}

      {/* Lines per label */}
      {LABELS.map((lbl) => {
        const color = SEVERITY_COLORS[lbl] ?? '#64748b'
        const pts = data.map((d) => (d as unknown as Record<string, number>)[lbl] ?? 0)
        const pathD = pts
          .map((v, i) => `${i === 0 ? 'M' : 'L'}${xPos(i).toFixed(1)},${yPos(v).toFixed(1)}`)
          .join(' ')
        return <path key={lbl} d={pathD} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" opacity="0.9" />
      })}

      {/* X axis labels */}
      {xLabels.map(({ i, date }) => (
        <text key={date} x={xPos(i)} y={H - 4} textAnchor="middle" className="fill-slate-500" fontSize="8">
          {date.slice(5)}
        </text>
      ))}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Alert card
// ---------------------------------------------------------------------------
function AlertCard({ alert }: { alert: DashboardAlert }) {
  const [expanded, setExpanded] = useState(false)
  const { bar, badge, icon } = ALERT_STYLES[alert.level]
  return (
    <div className={`rounded-lg border p-3 ${bar} cursor-pointer`} onClick={() => setExpanded((e) => !e)}>
      <div className="flex items-start gap-2">
        <span className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold ${badge}`}>
          {icon}
        </span>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-slate-100">
            {alert.message}
          </span>
          {alert.label && (
            <span className={`ml-2 rounded px-1.5 py-0.5 text-xs font-semibold ${LABEL_BG[alert.label] ?? 'bg-slate-700 text-slate-300'}`}>
              {alert.label}
            </span>
          )}
          {expanded && (
            <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">{alert.detail}</p>
          )}
        </div>
        <span className="text-slate-500 text-xs shrink-0">{expanded ? '▲' : '▼'}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Severity row
// ---------------------------------------------------------------------------
function SeverityRow({ entry, max }: { entry: SeverityEntry; max: number }) {
  const fillPct = max > 0 ? (entry.count / max) * 100 : 0
  const color = SEVERITY_COLORS[entry.label] ?? '#64748b'
  const trendClass = TREND_COLOR[entry.trend]
  const sparkPoints = [entry.prior_count, (entry.prior_count + entry.recent_count) / 2, entry.recent_count]

  return (
    <tr className="border-b border-slate-800 hover:bg-slate-800/30 transition">
      <td className="py-3 pl-3 pr-2 w-28">
        <span className={`inline-block rounded px-2 py-0.5 text-xs font-semibold capitalize ${LABEL_BG[entry.label] ?? 'bg-slate-700 text-slate-300'}`}>
          {entry.label}
        </span>
      </td>
      <td className="py-3 px-2">
        <div className="flex items-center gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-2 w-2 rounded-full"
              style={{ background: i < entry.severity ? color : '#1e293b' }}
            />
          ))}
        </div>
      </td>
      <td className="py-3 px-2">
        <div className="flex items-center gap-2">
          <div className="h-2 w-28 rounded-full bg-slate-800 overflow-hidden">
            <div className="h-2 rounded-full transition-all" style={{ width: `${fillPct}%`, background: color }} />
          </div>
          <span className="text-sm text-slate-200 tabular-nums w-10">{entry.count}</span>
        </div>
      </td>
      <td className="py-3 px-2 text-sm tabular-nums text-slate-300">
        <span className={`font-medium ${trendClass}`}>{TREND_ICON[entry.trend]}</span>{' '}
        {entry.recent_count}
      </td>
      <td className="py-3 px-2 hidden md:table-cell">
        <Sparkline points={sparkPoints} color={color} />
      </td>
      <td className="py-3 pr-3 text-sm tabular-nums text-slate-400 text-right">
        {entry.avg_confidence != null ? `${(entry.avg_confidence * 100).toFixed(1)}%` : '—'}
      </td>
    </tr>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function DashboardPage({ onBack }: DashboardPageProps) {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const refreshTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  const load = useCallback(async () => {
    try {
      const data = await fetchDashboardStats()
      setStats(data)
      setError(null)
    } catch {
      setError('Could not reach backend. Is the server running on port 8000?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    refreshTimer.current = setInterval(() => void load(), 30_000)
    return () => {
      if (refreshTimer.current !== null) clearInterval(refreshTimer.current)
    }
  }, [load])

  const criticalCount = stats?.alerts.filter((a) => a.level === 'critical').length ?? 0
  const maxCount = stats ? Math.max(...stats.severity_ranking.map((r) => r.count), 1) : 1

  return (
    <div className="min-h-screen bg-[#090b12] text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/60 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500"
          >
            Home
          </button>
          <h1 className="flex-1 text-lg font-semibold tracking-tight">StreetPulse Decision Dashboard</h1>
          {criticalCount > 0 && (
            <span className="rounded-full bg-red-700 px-2.5 py-0.5 text-xs font-bold text-white animate-pulse">
              {criticalCount} critical
            </span>
          )}
          <button
            type="button"
            onClick={() => void load()}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:border-slate-500"
          >
            Refresh
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 space-y-6">
        {loading && (
          <div className="flex h-48 items-center justify-center text-slate-500">Loading stats…</div>
        )}
        {error && (
          <div className="rounded-xl border border-red-800 bg-red-950/50 p-4 text-sm text-red-300">{error}</div>
        )}

        {stats && (
          <>
            {/* Summary cards */}
            <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <SummaryCard label="Total Labeled" value={stats.summary.total_labeled.toString()} />
              <SummaryCard
                label="Model Version"
                value={stats.summary.model_version === 'unknown' ? '—' : stats.summary.model_version}
                small
              />
              <SummaryCard
                label="Correction Rate"
                value={`${(stats.summary.correction_rate * 100).toFixed(1)}%`}
                accent={stats.summary.correction_rate > 0.15 ? 'orange' : 'green'}
              />
              <SummaryCard
                label="Alerts"
                value={stats.alerts.length.toString()}
                accent={criticalCount > 0 ? 'red' : stats.alerts.length > 0 ? 'orange' : 'green'}
              />
            </section>

            {/* Alerts */}
            {stats.alerts.length > 0 && (
              <section>
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-slate-400">Alerts</h2>
                <div className="space-y-2">
                  {/* Sort: critical first, then warning, then info */}
                  {[...stats.alerts]
                    .sort((a, b) => {
                      const order = { critical: 0, warning: 1, info: 2 }
                      return order[a.level] - order[b.level]
                    })
                    .map((alert, i) => (
                      <AlertCard key={`${alert.code}-${alert.label}-${i}`} alert={alert} />
                    ))}
                </div>
              </section>
            )}

            {/* Severity ranking */}
            <section>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-slate-400">Severity Ranking</h2>
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-xs text-slate-500 uppercase tracking-wide">
                      <th className="py-2 pl-3 pr-2 text-left">Class</th>
                      <th className="py-2 px-2 text-left">Severity</th>
                      <th className="py-2 px-2 text-left">Total count</th>
                      <th className="py-2 px-2 text-left">Last 7 days</th>
                      <th className="py-2 px-2 text-left hidden md:table-cell">Trend</th>
                      <th className="py-2 pr-3 text-right">Avg conf.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.severity_ranking.map((entry) => (
                      <SeverityRow key={entry.label} entry={entry} max={maxCount} />
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Trends chart */}
            <section>
              <h2 className="mb-1 text-sm font-semibold uppercase tracking-widest text-slate-400">
                Detection Trends — last 30 days
              </h2>
              <div className="mb-3 flex gap-4 text-xs text-slate-400">
                {LABELS.map((lbl) => (
                  <span key={lbl} className="flex items-center gap-1">
                    <span className="inline-block h-2 w-5 rounded-full" style={{ background: SEVERITY_COLORS[lbl] }} />
                    <span className="capitalize">{lbl}</span>
                  </span>
                ))}
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
                <TrendChart data={stats.trends} />
              </div>
            </section>

            {/* Per-class breakdown cards */}
            <section>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-slate-400">Class Breakdown</h2>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {LABELS.map((lbl) => {
                  const count = stats.summary.by_class[lbl] ?? 0
                  const total = stats.summary.total_labeled
                  const pct = total > 0 ? Math.round((count / total) * 100) : 0
                  const color = SEVERITY_COLORS[lbl]
                  return (
                    <div key={lbl} className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
                      <div className="flex justify-between items-start">
                        <span className={`rounded px-2 py-0.5 text-xs font-semibold capitalize ${LABEL_BG[lbl]}`}>
                          {lbl}
                        </span>
                        <span className="text-lg font-bold text-white">{count}</span>
                      </div>
                      <div className="mt-3 h-1.5 rounded-full bg-slate-800">
                        <div
                          className="h-1.5 rounded-full transition-all"
                          style={{ width: `${pct}%`, background: color }}
                        />
                      </div>
                      <p className="mt-1 text-xs text-slate-500 text-right">{pct}% of total</p>
                    </div>
                  )
                })}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Small helper component
// ---------------------------------------------------------------------------
interface SummaryCardProps {
  label: string
  value: string
  small?: boolean
  accent?: 'red' | 'orange' | 'green'
}

function SummaryCard({ label, value, small = false, accent }: SummaryCardProps) {
  const valueColor =
    accent === 'red'
      ? 'text-red-400'
      : accent === 'orange'
        ? 'text-amber-400'
        : accent === 'green'
          ? 'text-emerald-400'
          : 'text-white'
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`mt-1 font-bold ${valueColor} ${small ? 'text-sm break-all' : 'text-2xl'}`}>{value}</p>
    </div>
  )
}
