import { CartesianGrid, ErrorBar, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { signed } from '@/lib/api'

// palette: dataviz reference instance (light). Categorical slots in fixed order; status colours reserved for verdicts.
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
export const INK = { primary: '#0b0b0b', secondary: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7' }
export const UP = '#2a78d6' // diverging blue: arm better
export const DOWN = '#e34948' // diverging red: arm worse
export const VERDICT_COLOR: Record<string, string> = {
  STRONG_WIN: '#0ca30c', PROMISING: '#0ca30c', INCONCLUSIVE: '#fab219', REJECTED: '#d03b3b',
}
export const tick = { fontSize: 11, fill: INK.muted }

export function VerdictBadge({ v }: { v?: string }) {
  if (!v) return null
  const cls: Record<string, string> = {
    STRONG_WIN: 'bg-emerald-600 text-white',
    PROMISING: 'bg-emerald-100 text-emerald-900',
    REJECTED: 'bg-red-100 text-red-800',
    INCONCLUSIVE: 'bg-amber-100 text-amber-900',
  }
  const icon: Record<string, string> = { STRONG_WIN: '▲', PROMISING: '▲', REJECTED: '✕', INCONCLUSIVE: '?' }
  return <Badge className={cls[v] ?? ''} variant="secondary">{icon[v] ?? ''} {v}</Badge>
}

export const OracleBadge = () => (
  <Badge variant="destructive" title="diagnostic: true. GT를 쓰는 상한 실험">ORACLE / 배포 불가</Badge>
)

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}

export function Swatch({ color, hollow, label }: { color: string; hollow?: boolean; label: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className={hollow ? 'inline-block size-2.5 rotate-45 border-2 bg-white' : 'inline-block size-2.5 rounded-full'}
        style={hollow ? { borderColor: color } : { background: color }} />
      {label}
    </span>
  )
}

export type ForestRow = { label: string; est: number; lo?: number | null; hi?: number | null; color: string; hollow?: boolean; note?: string }

function Marker(p: any) {
  const { cx, cy, payload } = p
  if (cx == null || cy == null) return null
  if (payload.hollow) {
    return <path d={`M${cx},${cy - 6}L${cx + 6},${cy}L${cx},${cy + 6}L${cx - 6},${cy}Z`} fill="#fff" stroke={payload.color} strokeWidth={2} />
  }
  return <circle cx={cx} cy={cy} r={5} fill={payload.color} stroke="#fff" strokeWidth={2} />
}

/** ~6 round ticks covering [lo, hi] (1/2/5 x 10^k steps) */
export function niceTicks(lo: number, hi: number, n = 6) {
  const raw = (hi - lo) / n || 0.01
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw)!
  const out: number[] = []
  for (let k = Math.floor(lo / step); k <= Math.max(Math.ceil(hi / step), Math.floor(lo / step) + 1); k++) out.push(+(k * step).toPrecision(10))
  return out
}

/** powers of ten spanning [lo, hi] for log axes */
export const logTicks = (lo: number, hi: number) => {
  const out: number[] = []
  for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k++) out.push(10 ** k)
  return out
}

/** Horizontal forest plot: point estimate with CI whiskers, vertical line at 0. */
export function Forest({ rows, labelWidth = 170, digits = 4 }: { rows: ForestRow[]; labelWidth?: number; digits?: number }) {
  const data = rows.map((r, i) => ({ ...r, y: i, err: r.lo != null && r.hi != null ? [r.est - r.lo, r.hi - r.est] : [0, 0] }))
  const xs = rows.flatMap((r) => [r.est, r.lo ?? r.est, r.hi ?? r.est, 0])
  const ticks = niceTicks(Math.min(...xs), Math.max(...xs))
  return (
    <ResponsiveContainer width="100%" height={rows.length * 28 + 50}>
      <ScatterChart margin={{ left: 4, right: 24, top: 8, bottom: 8 }}>
        <CartesianGrid horizontal={false} stroke={INK.grid} />
        <XAxis type="number" dataKey="est" domain={[ticks[0], ticks[ticks.length - 1]]} ticks={ticks} tick={tick} tickFormatter={(v) => signed(v, 3)} stroke={INK.axis} />
        <YAxis type="number" dataKey="y" domain={[-0.5, rows.length - 0.5]} ticks={rows.map((_, i) => i)} reversed interval={0}
          tickFormatter={(i) => rows[i]?.label ?? ''} width={labelWidth} tick={{ ...tick, fill: INK.secondary }} stroke={INK.axis} />
        <ReferenceLine x={0} stroke={INK.secondary} />
        <Tooltip cursor={false} content={({ active, payload }) => {
          const r = active && payload?.[0]?.payload
          if (!r) return null
          return (
            <div className="rounded border bg-background px-2 py-1 text-xs shadow">
              <div className="font-mono font-medium">{r.label}</div>
              <div className="tabular-nums">{signed(r.est, digits)}{r.lo != null && <> [{signed(r.lo, digits)}, {signed(r.hi, digits)}]</>}</div>
              {r.note && <div className="text-muted-foreground">{r.note}</div>}
            </div>
          )
        }} />
        <Scatter data={data} shape={Marker} isAnimationActive={false}>
          <ErrorBar dataKey="err" direction="x" width={5} stroke={INK.secondary} strokeWidth={1.5} isAnimationActive={false} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}

/** rows from {name: {dJF|delta|mean, CI_video|CI}} report blocks, sorted by estimate */
export function forestRows(obj: Record<string, any>, est: string[], ci: string[], color: (k: string, v: any) => string = () => SERIES[0]): ForestRow[] {
  return Object.entries(obj ?? {})
    .map(([k, v]) => {
      const e = est.map((f) => v?.[f]).find((x) => typeof x === 'number')
      const c = ci.map((f) => v?.[f]).find(Array.isArray)
      return e == null ? null : { label: k, est: e, lo: c?.[0], hi: c?.[1], color: color(k, v) }
    })
    .filter((r): r is NonNullable<typeof r> => r !== null)
    .sort((a, b) => b.est - a.est)
}
