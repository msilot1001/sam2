import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, signed, useFetch, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Help } from '@/HelpTip'
import { DOWN, INK, OracleBadge, SERIES, Stat, UP, VerdictBadge, logTicks, tick } from './charts'

export const stemOf = (unit: string) => unit.replace(/\//g, '_')

type Col = { key: string; label: string; h?: string; get: (a: string, r: any) => any; fmt?: (v: any) => React.ReactNode }
const COLS: Col[] = [
  { key: 'arm', label: 'arm', get: (a) => a },
  { key: 'verdict', label: 'verdict', get: (_, r) => r.verdict, fmt: (v) => <VerdictBadge v={v} /> },
  { key: 'n_units', label: 'units', get: (_, r) => r.n_units },
  { key: 'n_videos', label: 'videos', get: (_, r) => r.n_videos },
  { key: 'dJF', label: 'ΔJ&F', get: (_, r) => r.dJF, fmt: (v) => <b>{signed(v)}</b> },
  { key: 'lo', label: 'CI lo', h: 'ci', get: (_, r) => r.CI_video?.[0], fmt: (v) => signed(v) },
  { key: 'hi', label: 'CI hi', h: 'ci', get: (_, r) => r.CI_video?.[1], fmt: (v) => signed(v) },
  { key: 'dJ', label: 'ΔJ', get: (_, r) => r.dJ, fmt: (v) => signed(v) },
  { key: 'dF', label: 'ΔF', get: (_, r) => r.dF, fmt: (v) => signed(v) },
  { key: 'up', label: 'vid ↑0.01', get: (_, r) => r['videos_up_0.01'] },
  { key: 'down', label: 'vid ↓0.01', get: (_, r) => r['videos_down_0.01'] },
  { key: 'lovo_min', label: 'LOVO min', get: (_, r) => r.lovo_min, fmt: (v) => signed(v) },
  { key: 'without_miami', label: 'w/o miami', get: (_, r) => r.without_miami, fmt: (v) => signed(v) },
  { key: 'tiny_dJF', label: 'tiny ΔJ&F', get: (_, r) => r.tiny_dJF, fmt: (v) => signed(v) },
  { key: 'worst', label: 'worst unit', get: (_, r) => r.worst_unit?.[1], fmt: (v) => signed(v, 3) },
  { key: 'best', label: 'best unit', get: (_, r) => r.best_unit?.[1], fmt: (v) => signed(v, 3) },
  { key: 'rescued', label: 'rescued', get: (_, r) => r.rescued_frames },
  { key: 'fp', label: '오재획득 arm', get: (_, r) => r.false_reacq_frames },
  { key: 'fpb', label: '오재획득 SGGL', get: (_, r) => r.false_reacq_frames_SGGL },
  { key: 'lat', label: 'latency s', get: (_, r) => r.latency_s_mean, fmt: (v) => num(v, 3) },
  { key: 'latb', label: 'SGGL s', get: (_, r) => r.latency_s_SGGL, fmt: (v) => num(v, 3) },
  { key: 'vram', label: 'VRAM GB', get: (_, r) => r.peak_vram_gb, fmt: (v) => num(v, 2) },
  { key: 'machines', label: 'machine', get: (_, r) => Object.entries(r.machines ?? {}).map(([m, n]) => `${m}:${n}`).join(' ') },
]

// stops the help dialog's clicks (they bubble through the portal) from sorting / navigating
const stop = (e: React.MouseEvent) => e.stopPropagation()

function cmp(x: any, y: any) {
  if (x == null) return 1
  if (y == null) return -1
  return typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))
}

export default function Arms({ sel, nav }: { sel?: string; nav: Nav }) {
  const { data, error, at } = useFetch<any>('/api/arms/summary', 30_000)
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: 'dJF', dir: -1 })
  const causal: Record<string, any> = data?.causal ?? {}
  const arms: Record<string, any> = data?.arms ?? {}
  const names = Object.keys(causal).filter((k) => !k.includes('@'))
  const vk = Object.keys(causal).filter((k) => k.includes('@'))
  const col = COLS.find((c) => c.key === sort.key) ?? COLS[4]
  const sorted = [...names].sort((a, b) => cmp(col.get(a, causal[a]), col.get(b, causal[b])) * sort.dir)
  const current = sel && causal[sel] ? sel : sorted[0]
  if (!data) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">causal_report.json · causal_per_unit.csv · arms.json · 30초마다 갱신 ({at?.toLocaleTimeString()})</div>
      <Card>
        <CardHeader><CardTitle>모든 causal arm 지표<Help k="arms.table" /></CardTitle><CardDescription>열 머리글 클릭 = 정렬 · 행 클릭 = 아래에서 상세 보기</CardDescription></CardHeader>
        <CardContent>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                {COLS.map((c) => (
                  <TableHead key={c.key} className="cursor-pointer select-none whitespace-nowrap"
                    onClick={() => setSort({ key: c.key, dir: sort.key === c.key ? (-sort.dir as 1 | -1) : -1 })}>
                    {c.label}<span onClick={stop}><Help k={`arms.col.${c.h ?? c.key}`} /></span>{sort.key === c.key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((a) => (
                <TableRow key={a} className={cn('cursor-pointer', a === current && 'bg-muted')} onClick={() => nav('arms', a)}>
                  {COLS.map((c) => {
                    const v = c.get(a, causal[a])
                    return <TableCell key={c.key} className={cn('tabular-nums', c.key === 'arm' && 'font-mono')}>
                      {c.key === 'arm' ? <span className="flex items-center gap-1">{a}{arms[a]?.diagnostic && <Badge variant="destructive">ORACLE</Badge>}</span> : c.fmt ? c.fmt(v) : (v ?? '—')}
                    </TableCell>
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {vk.length > 0 && <VkTable causal={causal} keys={vk} />}
        </CardContent>
      </Card>
      {current && <ArmDetail arm={current} r={causal[current]} spec={arms[current]} perUnit={(data.per_unit as any[]).filter((u) => u.arm === current)} nav={nav} />}
      <Compare names={names} causal={causal} perUnit={data.per_unit} nav={nav} />
    </div>
  )
}

function VkTable({ causal, keys }: { causal: Record<string, any>; keys: string[] }) {
  return (
    <div className="mt-4">
      <div className="mb-1 text-sm font-medium">VKITTI (@VK) — scene별<Help k="arms.vk" /></div>
      <Table className="text-xs">
        <TableHeader><TableRow><TableHead>arm</TableHead><TableHead>verdict</TableHead><TableHead>ΔJ&F<Help k="arms.vk.dJF" /></TableHead><TableHead>per_scene<Help k="arms.vk.per_scene" /></TableHead></TableRow></TableHeader>
        <TableBody>
          {keys.map((k) => (
            <TableRow key={k}>
              <TableCell className="font-mono">{k}</TableCell><TableCell>{causal[k].verdict ? <VerdictBadge v={causal[k].verdict} /> : causal[k].non_regression_pass != null && <Badge variant="outline">{causal[k].non_regression_pass ? '비회귀 통과' : '비회귀 실패'}</Badge>}</TableCell>
              <TableCell>{signed(causal[k].dJF)}</TableCell>
              <TableCell className="font-mono">{Object.entries(causal[k].per_scene ?? {}).map(([s, v]) => `${s}: ${signed(Number(v))}`).join(' · ')}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function Spec({ spec }: { spec: any }) {
  if (!spec) return <p className="text-sm text-muted-foreground">arms.json에 정의 없음</p>
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {Object.entries(spec).filter(([k]) => k !== 'name').map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-mono text-xs text-muted-foreground">{k}<Help k={`arms.spec.${k}`} /></dt>
          <dd className="min-w-0">
            {Array.isArray(v) ? (v.length ? <div className="flex flex-wrap gap-1">{v.map((g) => <Badge key={g} variant="outline" className="font-mono">{g}</Badge>)}</div> : <span className="text-muted-foreground">[]</span>)
              : typeof v === 'object' ? <code className="text-xs">{JSON.stringify(v)}</code> : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function ArmDetail({ arm, r, spec, perUnit, nav }: { arm: string; r: any; spec: any; perUnit: any[]; nav: Nav }) {
  const pv = Object.entries((r.per_video ?? {}) as Record<string, number>).map(([video, v]) => ({ video, v })).sort((a, b) => b.v - a.v)
  const pts = perUnit.filter((u) => u.area0 > 0).map((u) => ({ ...u, stem: stemOf(u.unit) }))
  const lt = pts.length ? logTicks(Math.min(...pts.map((p) => p.area0)), Math.max(...pts.map((p) => p.area0))) : [1e-4, 1]
  const counts = [
    { k: 'rescued 프레임', arm: r.rescued_frames, SGGL: 0 },
    { k: '오재획득 프레임', arm: r.false_reacq_frames, SGGL: r.false_reacq_frames_SGGL },
  ]
  const toUnit = (p: any) => p?.payload?.stem && nav('units', p.payload.stem)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 font-mono">{arm}<Help k={`arms.arm.${arm}`} /> <VerdictBadge v={r.verdict} /> {spec?.diagnostic && <OracleBadge />}</CardTitle>
        <CardDescription>ΔJ&F {signed(r.dJF)} · CI [{signed(r.CI_video?.[0])}, {signed(r.CI_video?.[1])}] · {r.n_units} units / {r.n_videos} videos<Help k="arms.detail.summary" /></CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 lg:grid-cols-2">
        <section>
          <h3 className="mb-2 text-sm font-medium">frozen spec (arms.json)<Help k="arms.spec" /></h3>
          <Spec spec={spec} />
          <h3 className="mt-6 mb-2 text-sm font-medium">지연 시간 · VRAM<Help k="arms.detail.cost" /></h3>
          <div className="grid grid-cols-3 gap-4">
            <Stat label="latency arm (s/frame)" value={num(r.latency_s_mean, 3)} sub={`SGGL ${num(r.latency_s_SGGL, 3)} (${signed(r.latency_s_mean - r.latency_s_SGGL, 3)})`} />
            <Stat label="추가 SAM2 호출/frame" value={num(r.extra_sam2_calls_per_frame, 3)} />
            <Stat label="peak VRAM" value={`${num(r.peak_vram_gb, 2)} GB`} />
          </div>
          <h3 className="mt-6 mb-2 text-sm font-medium">rescue / 오재획득 (arm vs SGGL)<Help k="arms.detail.counts" /></h3>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={counts} layout="vertical" margin={{ left: 8, right: 24 }} barGap={2}>
              <CartesianGrid horizontal={false} stroke={INK.grid} />
              <XAxis type="number" tick={tick} allowDecimals={false} stroke={INK.axis} />
              <YAxis type="category" dataKey="k" width={110} tick={{ ...tick, fill: INK.secondary }} stroke={INK.axis} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="arm" name={arm} fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={14} isAnimationActive={false} />
              <Bar dataKey="SGGL" name="SGGL" fill={INK.muted} radius={[0, 4, 4, 0]} maxBarSize={14} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <div className="text-xs text-muted-foreground">co-absence 프레임 {r.coabs_frames} · co-absence ΔJ&F 합 {signed(r.coabs_dJF_framesum, 3)} · tiny unit {r.n_tiny}개 ΔJ&F {signed(r.tiny_dJF)}</div>
        </section>
        <section>
          <h3 className="mb-1 text-sm font-medium">video별 ΔJ&F (정렬)<Help k="arms.detail.perVideo" /></h3>
          <div className="flex gap-3 text-xs text-muted-foreground">
            <span><span className="mr-1 inline-block size-2 rounded-sm" style={{ background: UP }} />arm 우세</span>
            <span><span className="mr-1 inline-block size-2 rounded-sm" style={{ background: DOWN }} />SGGL 우세</span>
          </div>
          <ResponsiveContainer width="100%" height={Math.max(160, pv.length * 16 + 40)}>
            <BarChart data={pv} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid horizontal={false} stroke={INK.grid} />
              <XAxis type="number" tick={tick} tickFormatter={(v) => signed(v, 3)} stroke={INK.axis} />
              <YAxis type="category" dataKey="video" width={170} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
              <ReferenceLine x={0} stroke={INK.secondary} />
              <Tooltip formatter={(v) => signed(Number(v))} />
              <Bar dataKey="v" name="ΔJ&F" radius={2} maxBarSize={11} isAnimationActive={false}>
                {pv.map((e) => <Cell key={e.video} fill={e.v >= 0 ? UP : DOWN} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="lg:col-span-2">
          <h3 className="mb-1 text-sm font-medium">unit별 Δ (d) vs 첫 프레임 면적 area0 (로그 축) — 점 클릭 = Unit 탐색<Help k="arms.detail.scatter" /></h3>
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span><span className="mr-1 inline-block size-2 rounded-full" style={{ background: SERIES[0] }} />일반 unit</span>
            <span><span className="mr-1 inline-block size-2 rounded-full" style={{ background: SERIES[1] }} />tiny unit</span>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ left: 8, right: 24, top: 8, bottom: 16 }}>
              <CartesianGrid stroke={INK.grid} />
              <XAxis type="number" dataKey="area0" scale="log" domain={[lt[0], lt[lt.length - 1]]} ticks={lt} tick={tick} stroke={INK.axis}
                tickFormatter={(v) => Number(v).toExponential(0)} label={{ value: 'area0 (log)', position: 'insideBottom', offset: -8, fontSize: 11 }} />
              <YAxis type="number" dataKey="d" tick={tick} tickFormatter={(v) => signed(v, 2)} stroke={INK.axis} />
              <ReferenceLine y={0} stroke={INK.secondary} />
              <Tooltip cursor={false} content={({ active, payload }) => {
                const u = active && payload?.[0]?.payload
                return u ? (
                  <div className="rounded border bg-background px-2 py-1 text-xs shadow">
                    <div className="font-mono">{u.unit}</div>
                    <div>d {signed(u.d)} · dJ {signed(u.dJ)} · dF {signed(u.dF)}</div>
                    <div>area0 {num(u.area0, 5)} · {u.machine} · {u.set}{u.tiny ? ' · tiny' : ''}</div>
                    <div>rescued {u.rescued} · 오재획득 {u.fp_arm}/{u.fp_base}</div>
                  </div>
                ) : null
              }} />
              <Scatter name="일반" data={pts.filter((p) => !p.tiny)} fill={SERIES[0]} onClick={toUnit} isAnimationActive={false} className="cursor-pointer" />
              <Scatter name="tiny" data={pts.filter((p) => p.tiny)} fill={SERIES[1]} onClick={toUnit} isAnimationActive={false} className="cursor-pointer" />
            </ScatterChart>
          </ResponsiveContainer>
        </section>
      </CardContent>
    </Card>
  )
}

function Compare({ names, causal, perUnit, nav }: { names: string[]; causal: Record<string, any>; perUnit: any[]; nav: Nav }) {
  const [picked, setPicked] = useState<string[]>([])
  const pick = picked.filter((p) => names.includes(p))
  const chosen = pick.length ? pick : names.filter((n) => causal[n].verdict === 'PROMISING').slice(0, 2)
  const toggle = (a: string) => setPicked(chosen.includes(a) ? chosen.filter((x) => x !== a) : [...chosen, a].slice(-3))

  const videos = useMemo(() => {
    const vs = new Set(chosen.flatMap((a) => Object.keys(causal[a]?.per_video ?? {})))
    return [...vs].map((v) => Object.fromEntries([['video', v], ...chosen.map((a) => [a, causal[a].per_video?.[v] ?? null])]))
      .sort((x, y) => (y[chosen[0]] ?? 0) - (x[chosen[0]] ?? 0))
  }, [chosen.join(), causal]) // eslint-disable-line

  const units = useMemo(() => {
    const by: Record<string, Record<string, any>> = {}
    for (const u of perUnit) if (chosen.includes(u.arm)) (by[u.unit] ??= {})[u.arm] = u
    return Object.entries(by).map(([unit, m]) => {
      const ds = chosen.map((a) => m[a]?.d ?? null)
      const diffs = ds.slice(1).map((d) => (d != null && ds[0] != null ? d - ds[0] : null))
      return { unit, ds, diffs, spread: Math.max(0, ...diffs.map((x) => Math.abs(x ?? 0))), tiny: Object.values(m)[0]?.tiny }
    }).sort((a, b) => b.spread - a.spread)
  }, [chosen.join(), perUnit]) // eslint-disable-line

  return (
    <Card>
      <CardHeader>
        <CardTitle>arm 대 arm 비교<Help k="arms.compare" /></CardTitle>
        <CardDescription>2–3개 선택 (색 = 선택 순서) · video별 ΔJ&F와 unit별 Δ 차이 (arm k − 첫 번째 arm)</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {names.map((a) => {
            const i = chosen.indexOf(a)
            return (
              <button key={a} onClick={() => toggle(a)}
                className={cn('flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-xs', i >= 0 && 'ring-2')}
                style={i >= 0 ? { borderColor: SERIES[i], ['--tw-ring-color' as any]: SERIES[i] } : undefined}>
                {i >= 0 && <span className="inline-block size-2 rounded-full" style={{ background: SERIES[i] }} />}{a}
              </button>
            )
          })}
        </div>
        {chosen.length > 0 && (
          <div className="grid gap-6 xl:grid-cols-2">
            <ResponsiveContainer width="100%" height={Math.max(200, videos.length * (chosen.length * 6 + 8) + 50)}>
              <BarChart data={videos} layout="vertical" margin={{ left: 8, right: 24 }} barGap={1}>
                <CartesianGrid horizontal={false} stroke={INK.grid} />
                <XAxis type="number" tick={tick} tickFormatter={(v) => signed(v, 3)} stroke={INK.axis} />
                <YAxis type="category" dataKey="video" width={170} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
                <ReferenceLine x={0} stroke={INK.secondary} />
                <Tooltip formatter={(v) => signed(Number(v))} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {chosen.map((a, i) => <Bar key={a} dataKey={a} fill={SERIES[i]} maxBarSize={6} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
            <div className="max-h-[900px] overflow-auto">
              <Table className="text-xs">
                <TableHeader>
                  <TableRow>
                    <TableHead>unit</TableHead>
                    {chosen.map((a, i) => <TableHead key={a}><span className="mr-1 inline-block size-2 rounded-full" style={{ background: SERIES[i] }} />d {a}</TableHead>)}
                    {chosen.slice(1).map((a) => <TableHead key={a}>{a} − {chosen[0]}<Help k="arms.compare.diff" /></TableHead>)}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {units.map((u) => (
                    <TableRow key={u.unit} className="cursor-pointer" onClick={() => nav('units', stemOf(u.unit))}>
                      <TableCell className="font-mono">{u.unit}{u.tiny ? ' · tiny' : ''}</TableCell>
                      {u.ds.map((d, i) => <TableCell key={i} className="tabular-nums">{signed(d)}</TableCell>)}
                      {u.diffs.map((d, i) => (
                        <TableCell key={i} className="tabular-nums font-medium" style={{ color: d == null || Math.abs(d) < 1e-4 ? undefined : d > 0 ? UP : DOWN }}>{signed(d)}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
