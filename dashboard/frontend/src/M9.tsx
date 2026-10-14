import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, signed, useFetch, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { DOWN, Forest, INK, SERIES, Stat, Swatch, UP, tick, type ForestRow } from './charts'
import { Help } from '@/HelpTip'

const DS_LABEL: Record<string, string> = { mosev2: 'MOSEv2', vost: 'VOST', lvosv2: 'LVOS v2' }
const DS_COLOR: Record<string, string> = { pooled: INK.primary, mosev2: SERIES[0], vost: SERIES[1], lvosv2: SERIES[2] }
const CATS = ['OBSERVABILITY', 'SCALE', 'POSITION', 'PRESENCE_ABSENT_CLAIM', 'PRESENCE_WRONG_TARGET']
const ARM_ORDER = ['FLOWVEL', 'FLOWCARRY', 'SCALE', 'JOINT', 'PCCARRY']
const dsName = (k: string) => DS_LABEL[k] ?? k

type Status = 'DONE' | 'IN_PROGRESS' | 'NOT_RUN' | 'BLOCKED'
const STATUS_CLS: Record<Status, string> = {
  DONE: 'bg-emerald-100 text-emerald-900', IN_PROGRESS: 'bg-blue-100 text-blue-900', NOT_RUN: 'bg-muted text-muted-foreground', BLOCKED: 'bg-red-100 text-red-800',
}
function StatusBadge({ s }: { s: Status }) {
  return <Badge variant="secondary" className={STATUS_CLS[s]}>{s === 'IN_PROGRESS' && '● '}{s}</Badge>
}
const status = (has: boolean, started: boolean): Status => (has ? 'DONE' : started ? 'IN_PROGRESS' : 'NOT_RUN')

/** generic table over an array of flat-ish objects (nested values shown as JSON) */
function ObjTable({ rows, first }: { rows: Record<string, any>[]; first?: string }) {
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].sort((a, b) => (a === first ? -1 : b === first ? 1 : 0))
  return (
    <div className="overflow-x-auto">
      <Table className="text-xs">
        <TableHeader><TableRow>{cols.map((c) => <TableHead key={c}>{c}</TableHead>)}</TableRow></TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i}>{cols.map((c) => <TableCell key={c} className={cn('tabular-nums', c === first && 'font-mono')}>{fmt(r[c])}</TableCell>)}</TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
const fmt = (v: any) => (v == null ? '—' : typeof v === 'number' ? num(v, 3) : typeof v === 'object' ? JSON.stringify(v) : String(v))

/** log-binned histogram rows {bin, <ds>: n} over all datasets */
function histogram(man: Record<string, any>, key: string, base: number, step: number, label: (lo: number) => string) {
  const vals = Object.values(man).flatMap((m: any) => m.units.map((u: any) => u[key])).filter((v: any) => typeof v === 'number' && v > 0)
  if (!vals.length) return []
  const lg = (v: number) => Math.log(v) / Math.log(base)
  const k0 = Math.floor(lg(Math.min(...vals)) / step), k1 = Math.floor(lg(Math.max(...vals)) / step)
  const rows = Array.from({ length: k1 - k0 + 1 }, (_, j) => ({ bin: label(base ** ((k0 + j) * step)) } as Record<string, any>))
  for (const [ds, m] of Object.entries(man)) {
    for (const r of rows) r[ds] = 0
    for (const u of (m as any).units) if (u[key] > 0) rows[Math.floor(lg(u[key]) / step) - k0][ds]++
  }
  return rows
}

function Hist({ rows, ds, xLabel }: { rows: any[]; ds: string[]; xLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={rows} margin={{ left: 0, right: 8, bottom: 16 }} barGap={1}>
        <CartesianGrid vertical={false} stroke={INK.grid} />
        <XAxis dataKey="bin" tick={tick} stroke={INK.axis} interval={0} angle={-30} textAnchor="end" height={44}
          label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 11, fill: INK.secondary }} />
        <YAxis allowDecimals={false} tick={tick} stroke={INK.axis} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" />
        {ds.map((d) => <Bar key={d} dataKey={d} name={dsName(d)} fill={DS_COLOR[d] ?? SERIES[3]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  )
}

function Datasets({ d }: { d: any }) {
  const man: Record<string, any> = d.manifests ?? {}
  const ds = Object.keys(man)
  const preDs: Record<string, string> = Object.fromEntries(Object.entries(d.prereg?.datasets ?? {}).filter(([k]) => !/pumavos/i.test(k))) as any
  const missing = Object.keys(preDs).filter((k) => !man[k])
  const derived = (m: any) => ({
    grow: m.units.filter((u: any) => u.growth_max > 2).length,
    absent: m.units.filter((u: any) => u.absent_frames > 0).length,
  })
  const area = histogram(man, 'area0_frac', 10, 0.25, (lo) => `${(lo * 100).toPrecision(2)}%`)
  const grow = histogram(man, 'growth_max', 2, 1, (lo) => `×${num(lo, 3)}`)
  return (
    <Card>
      <CardHeader>
        <CardTitle>데이터셋 범위 (고정 manifest)<Help k="m9.datasets" /></CardTitle>
        <CardDescription>outputs/milestone9/manifests/*_units.json · GT만 보고 SAM 2 실행 전에 고정<Help k="m9.frozenManifest" /> · dev/conf는 영상 단위 분할<Help k="m9.split" /></CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              {['데이터셋', '상태', '훑은 영상', '평가 단위', '단위가 있는 영상', 'dev / conf', '극소 (<0.1%)', '총 frame', '커지는 객체 (>2×)', '부재 frame 있는 단위'].map((h) => <TableHead key={h}>{h}</TableHead>)}
            </TableRow>
          </TableHeader>
          <TableBody>
            {ds.map((k) => {
              const m = man[k], x = derived(m)
              return (
                <TableRow key={k}>
                  <TableCell className="font-medium">{dsName(k)}<Help k={`m9.ds.${k}`} /></TableCell>
                  <TableCell><StatusBadge s="DONE" /></TableCell>
                  <TableCell className="tabular-nums">{m.n_videos_scanned}</TableCell>
                  <TableCell className="tabular-nums font-semibold">{m.n_units}</TableCell>
                  <TableCell className="tabular-nums">{m.n_unit_videos}</TableCell>
                  <TableCell className="tabular-nums">{m.split_counts?.dev ?? '—'} / {m.split_counts?.conf ?? '—'}</TableCell>
                  <TableCell className="tabular-nums">{m.n_tiny}</TableCell>
                  <TableCell className="tabular-nums">{m.n_frames_total?.toLocaleString()}</TableCell>
                  <TableCell className="tabular-nums">{x.grow}</TableCell>
                  <TableCell className="tabular-nums">{x.absent}</TableCell>
                </TableRow>
              )
            })}
            {missing.map((k) => (
              <TableRow key={k}>
                <TableCell className="font-medium">{dsName(k)}<Help k={`m9.ds.${k}`} /></TableCell>
                <TableCell><StatusBadge s={/BLOCKED/i.test(preDs[k]) ? 'BLOCKED' : 'NOT_RUN'} /></TableCell>
                <TableCell colSpan={8} className="text-muted-foreground">{preDs[k]}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="grid gap-4 xl:grid-cols-2">
          <div>
            <div className="text-sm font-medium">prompt 면적 비율 area0_frac (로그 구간)<Help k="m9.histArea" /></div>
            <Hist rows={area} ds={ds} xLabel="area0 (frame 면적 대비, 구간 하한)" />
          </div>
          <div>
            <div className="text-sm font-medium">최대 성장 growth_max (×2 구간)<Help k="m9.histGrowth" /></div>
            <Hist rows={grow} ds={ds} xLabel="최대 면적 / prompt 면적 (구간 하한)" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function Progress({ d }: { d: any }) {
  const man: Record<string, any> = d.manifests ?? {}
  const runs: Record<string, Record<string, number>> = d.runs ?? {}
  return (
    <Card>
      <CardHeader><CardTitle>실행 현황 (결과 CSV 수 / manifest 단위 수)<Help k="m9.runs" /></CardTitle><CardDescription>runpod/diag = RunPod SGGL 기준선 궤적 · ubuntu/SGGL = Ubuntu 기준선 · 나머지는 arm</CardDescription></CardHeader>
      <CardContent className="space-y-1">
        {Object.keys(runs).length === 0 && <p className="text-sm text-muted-foreground">아직 결과 없음</p>}
        {Object.entries(runs).map(([key, byDs]) => Object.keys(man).filter((ds) => byDs[ds] || !/diag|SGGL/.test(key)).map((ds) => {
          const n = byDs[ds] ?? 0, tot = man[ds].n_units
          return (
            <div key={key + ds} className="flex items-center gap-2 text-xs">
              <span className="w-44 truncate font-mono">{key}</span>
              <span className="w-16">{dsName(ds)}</span>
              <div className="h-1.5 flex-1 rounded bg-muted">
                <div className={cn('h-1.5 rounded', n >= tot ? 'bg-[#0ca30c]' : 'bg-[#2a78d6]')} style={{ width: `${Math.min(100, (100 * n) / tot)}%` }} />
              </div>
              <span className="w-16 text-right tabular-nums">{n}/{tot}</span>
            </div>
          )
        }))}
      </CardContent>
    </Card>
  )
}

function Prereg({ d }: { d: any }) {
  const p = d.prereg
  const arms: Record<string, any> = d.arms ?? {}
  if (!p) return <Card><CardHeader><CardTitle>사전 등록</CardTitle><CardDescription><StatusBadge s="NOT_RUN" /> preregistration.json 없음</CardDescription></CardHeader></Card>
  const List = ({ xs }: { xs: string[] }) => <ul className="list-disc space-y-0.5 pl-5">{xs.map((x) => <li key={x}>{x}</li>)}</ul>
  return (
    <Card>
      <CardHeader>
        <CardTitle>사전 등록 (preregistration.json)<Help k="m9.prereg" /></CardTitle>
        <CardDescription>고정 {p.created_kst} KST · SAM 2 출력 전 고정: {String(p.created_before_any_m9_sam2_output_on_new_data)} · 시작 {p.branch_start}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 text-sm xl:grid-cols-2">
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">arm<Help k="m9.armsTable" /></div>
            <Table className="text-xs">
              <TableHeader><TableRow><TableHead>arm</TableHead><TableHead>고정</TableHead><TableHead>정의 (arms.json why)</TableHead></TableRow></TableHeader>
              <TableBody>
                {Object.entries(p.arms ?? {}).map(([a, v]) => (
                  <TableRow key={a}>
                    <TableCell className="font-mono">{a}<Help k={`m9.arm.${a}`} /></TableCell>
                    <TableCell>{String(v)}</TableCell>
                    <TableCell className="max-w-md whitespace-normal break-words text-muted-foreground">
                      {arms[a]?.why ?? '—'}
                      {arms[a]?.scale && <div className="font-mono break-all">scale {JSON.stringify(arms[a].scale)} · center={arms[a].center}</div>}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div><div className="font-medium">분할 사용 규칙<Help k="m9.split" /></div><p className="text-muted-foreground">{p.use_of_splits}</p></div>
          <div><div className="font-medium">단위 규칙<Help k="m9.frozenManifest" /></div><p className="text-muted-foreground">{p.unit_rule}</p></div>
          <div>
            <div className="font-medium">Stage-1 관문<Help k="m9.stage1Gates" /></div>
            <List xs={Object.entries(p.stage1_gates ?? {}).map(([k, v]) => `${k}: ${v}`)} />
          </div>
        </div>
        <div className="space-y-3">
          <div>
            <div className="font-medium">FLOWVEL 일반화 기준 (모두 충족)<Help k="m9.flowvelCriteria" /></div>
            <List xs={p.flowvel_generalizes_if ?? []} />
          </div>
          <div className="space-y-1">
            <div className="font-medium">새 arm 판정<Help k="m9.newArmCriteria" /></div>
            <List xs={[`SCALE: ${p.scale_success_if}`, `JOINT: ${p.joint_synergy_if}`, `PCCARRY: ${p.cheap_motion_success_if}`, `중단: ${p.kill_rules}`]} />
          </div>
          <div><div className="font-medium">주 지표 · 통계<Help k="m9.endpoint" /></div><p className="text-muted-foreground">{p.primary_endpoint}</p><p className="text-muted-foreground">{p.statistics}</p></div>
          <div>
            <div className="font-medium">실패 분류 순서<Help k="m9.taxonomy" /></div>
            <List xs={[...(p.failure_taxonomy?.order ?? []), `false_reacquisition: ${p.failure_taxonomy?.false_reacquisition}`, `dominant: ${p.failure_taxonomy?.dominant}`]} />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function Stage1({ d }: { d: any }) {
  const ds = Object.keys(d.manifests ?? {})
  const s1: Record<string, any> = d.stage1 ?? {}
  return (
    <Card>
      <CardHeader>
        <CardTitle>Stage 1 최소 진단 — 데이터셋별<Help k="m9.stage1" /></CardTitle>
        <CardDescription>reports/stage1_&lt;dataset&gt;.json · recall: GT가 있는 co-absence frame 중 후보 하나라도 J &gt; 0.5 · 원시 FP: GT가 없는 frame에서 후보가 '있음'을 낸 비율 (순위화 전)</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 xl:grid-cols-2">
        {ds.map((k) => {
          const r = s1[k]
          const started = (d.runs?.['runpod/diag']?.[k] ?? 0) > 0
          if (!r) return <div key={k}><div className="mb-1 font-medium">{dsName(k)} <StatusBadge s={status(false, started)} /></div><p className="text-sm text-muted-foreground">리포트 아직 없음{started ? ' (RunPod 진단 진행 중)' : k === 'vost' ? ' (VOST 생성기 진단은 중단됨. COMPUTE_AND_THROUGHPUT_AUDIT.md 참고)' : ''}</p></div>
          const gens = [...(r.generators ?? [])].sort((a: any, b: any) => b.recall - a.recall)
          const tests: ForestRow[] = Object.entries(r.paired_tests ?? {}).filter(([, v]: any) => v && typeof v.delta === 'number').map(([lab, v]: [string, any]) => {
            const sig = v.CI && (v.CI[0] > 0 || v.CI[1] < 0)
            return { label: lab, est: v.delta, lo: v.CI?.[0], hi: v.CI?.[1], color: !sig ? INK.muted : v.delta > 0 ? UP : DOWN, note: `${v.n_frames} frames · ${v.n_videos} videos` }
          })
          return (
            <div key={k} className="space-y-3">
              <div className="font-medium">{dsName(k)} <StatusBadge s="DONE" /></div>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="units / videos" value={`${r.n_units ?? '—'} / ${r.n_videos ?? '—'}`} sub={`${r.frames ?? '—'} frames`} />
                <Stat label="co-absence GT 있음 / 없음" value={`${r.coabs_gt_present ?? '—'} / ${r.coabs_gt_empty ?? '—'}`} sub={`trigger ${JSON.stringify(r.trigger_counts ?? {})}`} />
                <Stat label="oracle any-generator recall" value={num(r.oracle_any_generator_recall, 3)} sub={r.oracle_any_generator_videos != null ? `${r.oracle_any_generator_videos} videos` : undefined} />
                <Stat label="latency sggl / rescue" value={`${num(r.latency_s?.sggl, 3)} s`} sub={`rescue/trigger ${num(r.latency_s?.rescue_per_trigger, 2)} s`} />
              </div>
              {gens.length ? (
                <ResponsiveContainer width="100%" height={gens.length * 24 + 60}>
                  <BarChart data={gens} layout="vertical" margin={{ left: 8, right: 24 }} barGap={1}>
                    <CartesianGrid horizontal={false} stroke={INK.grid} />
                    <XAxis type="number" domain={[0, 1]} tick={tick} stroke={INK.axis} />
                    <YAxis type="category" dataKey="gen" width={100} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
                    <Tooltip formatter={(v) => num(v as number, 3)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="recall" name="recall" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={8} isAnimationActive={false} />
                    <Bar dataKey="fp_rate_gt_empty" name="원시 FP (GT 없음)" fill={SERIES[1]} radius={[0, 4, 4, 0]} maxBarSize={8} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <p className="text-sm text-muted-foreground">생성기 후보 없음 (SGGL 궤적만으로 만든 리포트)</p>}
              {gens.length > 0 && <ObjTable rows={gens} first="gen" />}
              {tests.length > 0 && <div><div className="text-sm font-medium">짝지은 검정<Help k="m9.pairedTests" /></div><Forest rows={tests} labelWidth={220} digits={3} /></div>}
              {!!r.score_rule_replay?.length && <div><div className="text-sm font-medium">score &gt; 0 규칙 재생<Help k="m9.ruleReplay" /></div><ObjTable rows={r.score_rule_replay} first="gen" /></div>}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function Taxonomy({ d }: { d: any }) {
  const ds = Object.keys(d.manifests ?? {})
  const tx: Record<string, any> = d.taxonomy ?? {}
  const cats = [...CATS, ...new Set(Object.values(tx).flatMap((t: any) => (t?.categories ?? []).map((c: any) => c.category)).filter((c) => !CATS.includes(c)))]
  const bars = ds.filter((k) => tx[k]).map((k) => ({ ds: `${dsName(k)} (${tx[k].dominant})`, ...Object.fromEntries((tx[k].categories ?? []).map((c: any) => [c.category, c.loss_share])) }))
  const started = (k: string) => (d.runs?.['runpod/diag']?.[k] ?? 0) + (d.runs?.['ubuntu/SGGL']?.[k] ?? 0) + (d.runs?.['runpod/SGGL']?.[k] ?? 0) > 0
  return (
    <Card>
      <CardHeader>
        <CardTitle>실패 분류 (H4 관문) — 놓친 손실의 범주별 비중<Help k="m9.taxonomy" /></CardTitle>
        <CardDescription>reports/taxonomy_&lt;dataset&gt;.json · 대상: 같은 기계 SGGL 궤적에서 GT가 있는데 J &lt; 0.5인 frame · 막대 = 범주별 (1 − J&F) 합의 비중 · 괄호 = 지배 범주<Help k="m9.dominant" /></CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {ds.map((k) => (
            <span key={k} className="flex items-center gap-1">{dsName(k)} <StatusBadge s={status(!!tx[k], started(k))} />
              {tx[k] && <Badge variant="outline" className="font-mono">dominant: {tx[k].dominant}</Badge>}</span>
          ))}
        </div>
        {bars.length > 0 && (
          <>
            <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
              {cats.map((c, i) => <span key={c} className="flex items-center gap-0.5"><Swatch color={SERIES[i % 8]} label={c} /><Help k={`m9.cat.${c}`} /></span>)}
            </div>
            <ResponsiveContainer width="100%" height={bars.length * 44 + 50}>
              <BarChart data={bars} layout="vertical" margin={{ left: 8, right: 24 }}>
                <CartesianGrid horizontal={false} stroke={INK.grid} />
                <XAxis type="number" domain={[0, 1]} tick={tick} stroke={INK.axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <YAxis type="category" dataKey="ds" width={220} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
                <Tooltip formatter={(v, n) => [`${(100 * (v as number)).toFixed(1)}%`, n]} />
                {cats.map((c, i) => <Bar key={c} dataKey={c} stackId="s" fill={SERIES[i % 8]} stroke="#fff" strokeWidth={1} maxBarSize={28} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
        {ds.filter((k) => tx[k]).map((k) => {
          const t = tx[k]
          return (
            <div key={k} className="space-y-2">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
                <Stat label={`${dsName(k)} frames`} value={t.frames ?? '—'} sub={`GT 있음 ${t.gt_present_frames ?? '—'}`} />
                <Stat label="놓친 frame" value={t.missed_frames ?? '—'} sub={`손실 합 ${num(t.missed_loss, 1)}`} />
                <div className="flex items-start gap-1"><Stat label="거짓 재획득 frame" value={t.false_reacq_frames ?? '—'} sub={`GT 없는 frame ${t.gt_empty_frames ?? '—'}`} /><Help k="m9.falseReacq" /></div>
                <Stat label="SGGL 작은 단위 J&F" value={num(t.sggl_small_unit_JF_mean, 3)} />
                <Stat label="dominant" value={t.dominant} />
              </div>
              <ObjTable rows={t.categories ?? []} first="category" />
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

const CAUSAL_COLS: [string, string, (r: any) => React.ReactNode][] = [
  ['units / videos', 'n_units', (r) => `${r.n_units} / ${r.n_videos}`],
  ['ΔJ&F', 'dJF', (r) => signed(r.dJF)],
  ['CI (video)', 'CI_video', (r) => r.CI_video ? `[${signed(r.CI_video[0])}, ${signed(r.CI_video[1])}]` : '—'],
  ['ΔJ / ΔF', 'dJ', (r) => `${signed(r.dJ, 3)} / ${signed(r.dF, 3)}`],
  ['영상 ↑/↓ 0.01', 'videos_up', (r) => `${r['videos_up_0.01'] ?? '—'} / ${r['videos_down_0.01'] ?? '—'}`],
  ['sign p', 'sign_test_p', (r) => num(r.sign_test_p, 3)],
  ['Holm p', 'holm_p', (r) => num(r.holm_p, 3)],
  ['LOVO min', 'lovo_min', (r) => signed(r.lovo_min)],
  ['상위2 비중', 'top2', (r) => num(r.top2_share_of_net_gain, 2)],
  ['최악 단위', 'worst_unit', (r) => Array.isArray(r.worst_unit) ? <span title={r.worst_unit[0]}>{signed(r.worst_unit[1], 3)}</span> : '—'],
  ['< −0.05', 'n_lt', (r) => r['n_units_lt_-0.05']],
  ['tiny ΔJ&F', 'tiny_dJF', (r) => signed(r.tiny_dJF)],
  ['co-abs Δ/frame', 'coabs', (r) => signed(r.coabs_dJF_per_frame, 3)],
  ['거짓 재획득 arm/SGGL', 'fp', (r) => Array.isArray(r.false_reacq_arm_vs_base) ? r.false_reacq_arm_vs_base.join(' / ') : '—'],
  ['latency s', 'latency_s', (r) => num(r.latency_s, 3)],
  ['VRAM GB', 'peak_vram_gb', (r) => num(r.peak_vram_gb, 2)],
]

function Causal({ d, nav }: { d: any; nav: Nav }) {
  const c: Record<string, Record<string, any>> | null = d.causal
  const arms = c ? Object.keys(c).sort((a, b) => (ARM_ORDER.indexOf(a) + 99) % 99 - (ARM_ORDER.indexOf(b) + 99) % 99) : []
  const started = Object.entries(d.runs ?? {}).some(([k, v]: [string, any]) => !/diag|SGGL/.test(k) && Object.values(v).some((n: any) => n > 0))
  const keyOrder = (k: string) => (k === 'pooled' ? '0' : k.includes(':') ? `2${k}` : `1${k}`)
  const rows: ForestRow[] = arms.flatMap((a) => Object.keys(c![a]).sort((x, y) => keyOrder(x).localeCompare(keyOrder(y))).map((k) => {
    const r = c![a][k]
    return { label: `${a} · ${k}`, est: r.dJF, lo: r.CI_video?.[0], hi: r.CI_video?.[1], color: DS_COLOR[k.split(':')[0]] ?? SERIES[3], hollow: k.includes(':'), note: `${r.n_units} units / ${r.n_videos} videos` }
  })).filter((r) => typeof r.est === 'number')
  return (
    <Card>
      <CardHeader>
        <CardTitle>Causal — ΔJ&F (arm − 같은 기계 SGGL), 작은 단위만<Help k="m9.causal" /></CardTitle>
        <CardDescription>reports/causal.json · 점 = unit 평균 ΔJ&F, 수염 = 영상 군집 bootstrap 95% CI <StatusBadge s={status(!!c, started)} /></CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!c && <p className="text-sm text-muted-foreground">causal.json 아직 없음. arm 실행이 끝나면 scripts/m9_analysis.py causal이 만든다.</p>}
        {c && (
          <>
            <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
              {['pooled', ...Object.keys(d.manifests ?? {})].map((k) => <Swatch key={k} color={DS_COLOR[k] ?? SERIES[3]} label={k === 'pooled' ? '전체(pooled)' : dsName(k)} />)}
              <Swatch color={INK.secondary} hollow label="빈 마름모 = dev/conf 분할 행" />
              <Help k="m9.forest" />
            </div>
            <Forest rows={rows} labelWidth={230} />
            <div className="overflow-x-auto">
              <Table className="text-xs">
                <TableHeader><TableRow><TableHead>arm · 범위</TableHead>{CAUSAL_COLS.map(([h, k]) => <TableHead key={h}>{h}<Help k={`m9.col.${k}`} /></TableHead>)}</TableRow></TableHeader>
                <TableBody>
                  {arms.flatMap((a) => Object.entries(c[a]).sort(([x], [y]) => keyOrder(x).localeCompare(keyOrder(y))).map(([k, r]) => (
                    <TableRow key={a + k} className={k === 'pooled' ? 'font-medium' : undefined}>
                      <TableCell className="font-mono">{a} · {k}</TableCell>
                      {CAUSAL_COLS.map(([h, , f]) => <TableCell key={h} className="tabular-nums">{f(r) ?? '—'}</TableCell>)}
                    </TableRow>
                  )))}
                </TableBody>
              </Table>
            </div>
          </>
        )}
        {d.causal_units?.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs">단위별 Δ CSV:
            {d.causal_units.map((p: string) => <button key={p} className="font-mono underline" onClick={() => nav('raw', p)}>{p.split('/').pop()}</button>)}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function M9({ nav }: { nav: Nav }) {
  const { data: d, error, at } = useFetch<any>('/api/m9', 30_000)
  if (!d) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">
        <span className="mr-2 text-sm font-semibold text-foreground">M9 교차 데이터셋 검증<Help k="m9.about" /></span>
        30초마다 자동 갱신 · 마지막 갱신 {at?.toLocaleTimeString()} {error && <span className="text-red-600">· 오류: {error}</span>}
        <button className="ml-3 underline" onClick={() => nav('reports', 'docs/m9/RESEARCH_PLAN_AND_PREREGISTRATION.md')}>연구 계획 문서</button>
        <button className="ml-3 underline" onClick={() => nav('units')}>Unit 탐색 (M9 단위 포함)</button>
      </div>
      <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
        <Datasets d={d} />
        <Progress d={d} />
      </div>
      <Causal d={d} nav={nav} />
      <Taxonomy d={d} />
      <Stage1 d={d} />
      <Prereg d={d} />
    </div>
  )
}
