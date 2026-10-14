import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, q, signed, useFetch, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { stemOf } from './Arms'
import { Help } from '@/HelpTip'
import { DOWN, INK, SERIES, tick } from './charts'

type Frame = { i: number; JF: number | null; gt_empty: boolean; rescued: boolean; coabs: boolean; gen: string | null }
type Series = { machine: string; arm: string; path: string; frames: Frame[] }
type UnitData = { stem: string; series: Series[]; baseline: Record<string, { path: string; frames: Frame[] }>; candidates: { path: string; rows: Record<string, string>[] } | null }

const BASE_COLOR: Record<string, string> = { runpod: '#52514e', ubuntu: '#898781' }
const COABS = '#e34948'

function spans(frames: Frame[], pred: (f: Frame) => boolean) {
  const out: [number, number][] = []
  for (const f of frames) {
    if (!pred(f)) continue
    const last = out[out.length - 1]
    if (last && last[1] === f.i - 1) last[1] = f.i
    else out.push([f.i, f.i])
  }
  return out
}

const mean = (xs: (number | null)[]) => { const v = xs.filter((x): x is number => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null }

export default function Units({ stem, nav }: { stem?: string; nav: Nav }) {
  const { data: list } = useFetch<Record<string, string[]>>('/api/units', 30_000)
  const { data: summary } = useFetch<any>('/api/arms/summary', 30_000)
  const stems = Object.keys(list ?? {})
  const current = stem && stems.includes(stem) ? stem : stems[0]
  const { data: u, error } = useFetch<UnitData>(current ? `/api/unit/${q(current)}` : null, 30_000)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [frame, setFrame] = useState<string>('')
  const [genFilter, setGenFilter] = useState('')
  const diag = (arm: string) => !!summary?.arms?.[arm]?.diagnostic

  const view = useMemo(() => {
    if (!u) return null
    const keys = u.series.map((s) => `${s.machine}/${s.arm}`)
    const ref = u.baseline.runpod ?? u.baseline.ubuntu ?? u.series[0]
    const pts = new Map<number, any>()
    const at = (i: number) => pts.get(i) ?? (pts.set(i, { i }), pts.get(i))
    for (const [m, b] of Object.entries(u.baseline)) for (const f of b.frames) at(f.i)[`base:${m}`] = f.JF
    u.series.forEach((s, k) => {
      const base = new Map((u.baseline[s.machine]?.frames ?? []).map((f) => [f.i, f.JF]))
      for (const f of s.frames) {
        const p = at(f.i)
        p[keys[k]] = f.JF
        p[`d:${keys[k]}`] = f.JF != null && base.get(f.i) != null ? f.JF - base.get(f.i)! : null
        if (f.rescued) p[`r:${keys[k]}`] = f.gen ?? true
      }
    })
    const data = [...pts.values()].sort((a, b) => a.i - b.i)
    const stats = u.series.map((s, k) => ({
      key: keys[k], arm: s.arm, machine: s.machine, path: s.path, color: SERIES[k % 8], dashed: k >= 8,
      jf: mean(s.frames.map((f) => f.JF)), d: mean(data.map((p) => p[`d:${keys[k]}`] ?? null).filter((x) => x != null)),
      rescued: s.frames.filter((f) => f.rescued).length,
    }))
    return { data, stats, empty: ref ? spans(ref.frames, (f) => f.gt_empty) : [], coabs: ref ? spans(ref.frames, (f) => f.coabs) : [], nCoabs: ref?.frames.filter((f) => f.coabs).length ?? 0 }
  }, [u])

  const metrics = (summary?.per_unit ?? []).filter((r: any) => stemOf(r.unit) === current)
  const cands = (u?.candidates?.rows ?? []).filter((r) => (frame === '' || r.video_idx === frame) && (!genFilter || r.gen?.toLowerCase().includes(genFilter.toLowerCase())))
  const candCols = u?.candidates?.rows[0] ? Object.keys(u.candidates.rows[0]).filter((c) => c !== 'unit' && c !== 'video') : []
  const visible = view?.stats.filter((s) => !hidden.has(s.key)) ?? []
  const shade = view && (
    <>
      {view.empty.map(([a, b]) => <ReferenceArea key={`e${a}`} x1={a - 0.5} x2={b + 0.5} fill="#9ca3af" fillOpacity={0.2} ifOverflow="hidden" />)}
      {frame !== '' && <ReferenceLine x={Number(frame)} stroke={INK.primary} strokeDasharray="2 2" />}
    </>
  )
  const rescueDot = (key: string, color: string) => (p: any) =>
    p.payload?.[`r:${key}`] ? <circle key={`${key}${p.index}`} cx={p.cx} cy={p.cy} r={4} fill={color} stroke="#fff" strokeWidth={1.5} /> : <g key={`${key}${p.index}`} />
  const pick = (e: any) => e?.activeLabel != null && setFrame(String(e.activeLabel))

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm">unit{' '}
          <select className="max-w-lg rounded-md border bg-background px-2 py-1 font-mono text-sm" value={current ?? ''} onChange={(e) => { setFrame(''); nav('units', e.target.value) }}>
            {stems.map((s) => <option key={s} value={s}>{s} ({list![s].length})</option>)}
          </select>
        </label>
        <Help k="units.unit" />
        {error && <span className="text-sm text-red-600">{error}</span>}
        <span className="text-xs text-muted-foreground">30초마다 갱신 · 차트를 누르면 후보 표를 그 프레임으로 필터링</span>
        <span className="text-xs">실패 사례 조사 방법<Help k="units.howto" /></span>
      </div>

      {view && (
        <Card>
          <CardHeader>
            <CardTitle className="font-mono">{current}<Help k="units.chart_jf" /></CardTitle>
            <CardDescription>
              각 arm을 같은 기계의 SGGL과 비교 (runpod → runpod/runs/SGGL, ubuntu → diag) · 회색 음영: gt_empty · 빨간 띠: co-absence (score_G ≤ 0이고 score_L ≤ 0, {view.nCoabs} frames) · 굵은 점: rescued 프레임<Help k="units.shading" />
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Help k="units.baseline" className="ml-0" />
              {Object.entries(u!.baseline).map(([m, b]) => (
                <span key={m} className="flex items-center gap-1 rounded border px-2 py-0.5" title={b.path}>
                  <span className="inline-block h-1 w-4 rounded" style={{ background: BASE_COLOR[m], opacity: 0.6 }} />SGGL {m} (기준선)
                </span>
              ))}
              {view.stats.map((s) => (
                <button key={s.key} title={s.path} onClick={() => setHidden((h) => { const n = new Set(h); if (n.has(s.key)) n.delete(s.key); else n.add(s.key); return n })}
                  className={cn('flex items-center gap-1 rounded border px-2 py-0.5 font-mono', hidden.has(s.key) && 'opacity-40')}>
                  <span className="inline-block h-0.5 w-4" style={{ background: s.dashed ? `repeating-linear-gradient(90deg, ${s.color} 0 3px, transparent 3px 5px)` : s.color }} />
                  {s.key}{diag(s.arm) ? ' ◇ORACLE' : ''}
                </button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={view.data} margin={{ right: 16, left: 0 }} onClick={pick}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="i" type="number" domain={['dataMin', 'dataMax']} tick={tick} stroke={INK.axis} />
                <YAxis domain={[0, 1]} tick={tick} stroke={INK.axis} label={{ value: 'J&F', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip formatter={(v, n) => [num(v as number, 3), n]} labelFormatter={(l) => `frame ${l}`} />
                {shade}
                {view.coabs.map(([a, b]) => <ReferenceArea key={`c${a}`} x1={a - 0.5} x2={b + 0.5} y1={0} y2={0.04} fill={COABS} fillOpacity={0.7} ifOverflow="hidden" />)}
                {Object.keys(u!.baseline).map((m) => (
                  <Line key={m} dataKey={`base:${m}`} name={`SGGL ${m}`} stroke={BASE_COLOR[m]} strokeWidth={5} strokeOpacity={0.35} dot={false} isAnimationActive={false} connectNulls />
                ))}
                {visible.map((s) => (
                  <Line key={s.key} dataKey={s.key} name={s.key} stroke={s.color} strokeWidth={1.5} strokeDasharray={s.dashed ? '5 3' : undefined}
                    dot={rescueDot(s.key, s.color)} activeDot={{ r: 3 }} isAnimationActive={false} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <div className="text-sm font-medium">프레임별 Δ J&F (arm − 같은 기계 SGGL)<Help k="units.chart_delta" /></div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={view.data} margin={{ right: 16, left: 0 }} onClick={pick}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="i" type="number" domain={['dataMin', 'dataMax']} tick={tick} stroke={INK.axis} />
                <YAxis tick={tick} stroke={INK.axis} tickFormatter={(v) => signed(v, 2)} />
                <Tooltip formatter={(v, n) => [signed(v as number, 3), String(n).replace('d:', 'Δ ')]} labelFormatter={(l) => `frame ${l}`} />
                {shade}
                <ReferenceLine y={0} stroke={INK.secondary} />
                {visible.map((s) => (
                  <Line key={s.key} dataKey={`d:${s.key}`} name={s.key} stroke={s.color} strokeWidth={1.5} strokeDasharray={s.dashed ? '5 3' : undefined}
                    dot={false} isAnimationActive={false} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <Table className="text-xs">
              <TableHeader>
                <TableRow>{([['arm', 'metrics'], ['mean J&F', 'm.jf'], ['mean Δ (frames)', 'm.dframes'], ['rescued frames', 'm.rescued'], ['d (report)', 'm.d'], ['dJ', 'm.dJ'], ['dF', 'm.dF'], ['오재획득 arm/SGGL', 'm.fp'], ['latency arm/SGGL', 'm.lat'], ['set', 'm.set']] as const).map(([h, k]) => <TableHead key={h}>{h}<Help k={`units.${k}`} /></TableHead>)}</TableRow>
              </TableHeader>
              <TableBody>
                {view.stats.map((s) => {
                  const m = metrics.find((r: any) => r.arm === s.arm && r.machine === s.machine)
                  return (
                    <TableRow key={s.key}>
                      <TableCell className="font-mono"><span className="mr-1 inline-block size-2 rounded-full" style={{ background: s.color }} />{s.key}</TableCell>
                      <TableCell className="tabular-nums">{num(s.jf)}</TableCell>
                      <TableCell className="tabular-nums" style={{ color: s.d != null && s.d < -1e-4 ? DOWN : undefined }}>{signed(s.d)}</TableCell>
                      <TableCell className="tabular-nums">{s.rescued}</TableCell>
                      <TableCell className="tabular-nums">{signed(m?.d)}</TableCell>
                      <TableCell className="tabular-nums">{signed(m?.dJ)}</TableCell>
                      <TableCell className="tabular-nums">{signed(m?.dF)}</TableCell>
                      <TableCell className="tabular-nums">{m ? `${m.fp_arm} / ${m.fp_base}` : '—'}</TableCell>
                      <TableCell className="tabular-nums">{m ? `${num(m.lat, 3)} / ${num(m.lat_base, 3)}` : '—'}</TableCell>
                      <TableCell>{m?.set ?? '—'}{m?.tiny ? ' · tiny' : ''}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
            {!u!.series.length && <p className="text-sm text-muted-foreground">이 unit의 arm 실행 결과 없음 (기준선만 있음)</p>}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Stage-1 후보 (candidates)<Help k="units.candidates" /></CardTitle>
          <CardDescription>{u?.candidates ? <span className="font-mono">{u.candidates.path}</span> : '이 unit의 .cand.csv 없음'} · {cands.length} / {u?.candidates?.rows.length ?? 0}행</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Input className="w-40" placeholder="frame (video_idx)" value={frame} onChange={(e) => setFrame(e.target.value.trim())} />
            <Input className="w-48" placeholder="생성기 필터 (예: A_cos)" value={genFilter} onChange={(e) => setGenFilter(e.target.value)} />
            <Help k="units.cand_filter" />
            {frame !== '' && <button className="rounded border px-2 py-1 text-xs" onClick={() => setFrame('')}>프레임 필터 해제</button>}
            <span className="text-xs text-muted-foreground">
              후보가 있는 프레임: {[...new Set((u?.candidates?.rows ?? []).map((r) => r.video_idx))].slice(0, 40).map((i) => (
                <button key={i} className={cn('mx-0.5 underline', i === frame && 'font-bold')} onClick={() => setFrame(i)}>{i}</button>
              ))}
            </span>
          </div>
          <div className="max-h-[600px] overflow-auto">
            <Table className="text-xs">
              <TableHeader><TableRow>{candCols.map((c) => <TableHead key={c}>{c}<Help k={`units.cand.${c}`} /></TableHead>)}</TableRow></TableHeader>
              <TableBody>
                {cands.slice(0, 1000).map((r, j) => (
                  <TableRow key={j} className={Number(r.J) > 0.5 ? 'bg-emerald-50' : undefined}>
                    {candCols.map((c) => <TableCell key={c} className={cn('py-1 tabular-nums', c === 'gen' && 'font-mono')}>{/^-?\d+\.\d{4,}/.test(r[c]) ? num(r[c], 3) : r[c]}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div className="text-xs text-muted-foreground">초록 행은 J &gt; 0.5 (stage-1 recall 기준)</div>
        </CardContent>
      </Card>
    </div>
  )
}
