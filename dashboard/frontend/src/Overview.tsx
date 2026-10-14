import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, signed, useFetch } from '@/lib/api'

const EXPECTED_UNITS = 64
const UP = '#2a78d6' // diverging: blue = arm better
const DOWN = '#eb6834' // orange = arm worse

export function VerdictBadge({ v }: { v?: string }) {
  if (!v) return null
  const cls: Record<string, string> = {
    STRONG_WIN: 'bg-emerald-600 text-white',
    PROMISING: 'bg-emerald-100 text-emerald-900',
    REJECTED: 'bg-red-100 text-red-800',
    INCONCLUSIVE: 'bg-amber-100 text-amber-900',
  }
  return <Badge className={cls[v] ?? ''} variant="secondary">{v}</Badge>
}

const OracleBadge = () => (
  <Badge variant="destructive" title="diagnostic: true — GT를 사용하는 상한선 실험">ORACLE / 배포 불가</Badge>
)

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}

export default function Overview() {
  const { data: d, error, at } = useFetch<any>('/api/overview', 30_000)
  const [sel, setSel] = useState<string | null>(null)
  if (!d) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  const arms = d.arms ?? {}
  const causal: Record<string, any> = d.causal ?? {}
  const s1 = d.stage1 ?? {}
  const sync = d.sync ?? {}
  const targets = [...new Set(Object.keys(sync).filter((k) => k.includes(':')).map((k) => k.split(':').slice(1).join(':')))]

  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">
        30초마다 자동 갱신 · 마지막 갱신 {at?.toLocaleTimeString()} {error && <span className="text-red-600">· 오류: {error}</span>}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>Causal 진행 상황</CardTitle><CardDescription>arm별 완료 unit 수 (기대값 {EXPECTED_UNITS})</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(d.runs as Record<string, Record<string, string[]>>).map(([m, byArm]) => (
              <div key={m}>
                <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">{m === 'runpod' ? 'RunPod 4090' : 'Ubuntu 2080'}</div>
                {Object.keys(byArm).length === 0 && <div className="text-sm text-muted-foreground">없음</div>}
                {Object.entries(byArm).map(([arm, units]) => (
                  <div key={arm} className="flex items-center gap-2 text-sm">
                    <span className="w-36 truncate font-mono">{arm}</span>
                    <div className="h-1.5 flex-1 rounded bg-muted">
                      <div className="h-1.5 rounded bg-[#2a78d6]" style={{ width: `${Math.min(100, (100 * units.length) / EXPECTED_UNITS)}%` }} />
                    </div>
                    <span className="w-14 text-right tabular-nums">{units.length}/{EXPECTED_UNITS}</span>
                    {arms[arm]?.diagnostic && <Badge variant="destructive">ORACLE</Badge>}
                  </div>
                ))}
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Stage-1 진단 요약</CardTitle><CardDescription>stage1_report.json</CardDescription></CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            <Stat label="units / videos" value={`${s1.n_units} / ${s1.n_videos}`} sub={`${s1.frames} frames`} />
            <Stat label="co-absence 트리거" value={s1.trigger_counts?.coabs} sub={`GT 있음 ${s1.coabs_gt_present} · GT 없음 ${s1.coabs_gt_empty}`} />
            <Stat label="oracle any-generator recall" value={num(s1.oracle_any_generator_recall, 3)} />
            <Stat label="reference recall (max Gp/Lp)" value={num(s1.reference_recall_max_Gp_Lp, 3)} />
            <Stat label="SGGL latency / frame" value={`${num(s1.latency_s?.sggl_per_frame, 3)} s`} sub={`rescue(all gens)/trigger ${num(s1.latency_s?.rescue_all_generators_per_trigger, 2)} s`} />
            <Stat label="peak VRAM" value={`${num(s1.peak_vram_gb, 2)} GB`} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>동기화 상태</CardTitle><CardDescription>SYNC_STATUS.json · 갱신 {sync.updated}</CardDescription></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex flex-wrap gap-1">
              {Object.entries(d.ledger as Record<string, number>).map(([k, v]) => (
                <Badge key={k} variant={/FAIL|CONFLICT/i.test(k) ? 'destructive' : 'outline'}>{k}: {v}</Badge>
              ))}
            </div>
            {targets.map((t) => {
              const ok = sync[`last_ok:${t}`], fail = sync[`last_fail:${t}`]
              const failing = fail && (!ok || fail.slice(0, 19) > ok)
              return (
                <div key={t} className="border-t pt-1">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    {t.replace('outputs/milestone8/', '')}
                    <Badge variant={failing ? 'destructive' : 'secondary'}>{failing ? '실패' : 'OK'}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">마지막 OK {ok ?? '—'} · 신규 {sync[`last_new:${t}`] ?? 0} · 충돌 {sync[`conflicts:${t}`] ?? 0}</div>
                  {fail && <div className="truncate text-xs text-red-700" title={fail}>마지막 실패 {fail}</div>}
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Causal 결과 (arm vs machine-matched SGGL)</CardTitle>
          <CardDescription>causal_report.json · 행을 클릭하면 frozen spec과 video별 ΔJ&F</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                {['arm', 'verdict', 'ΔJ&F', 'CI (video)', 'videos ↑/↓ 0.01', 'LOVO min', 'worst unit', 'rescued', '오재획득 arm/SGGL', 'latency arm/SGGL (s)', 'units', 'machine'].map((h) => <TableHead key={h}>{h}</TableHead>)}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(causal).map(([arm, r]) => (
                <TableRow key={arm} className="cursor-pointer" onClick={() => setSel(arm)}>
                  <TableCell className="font-mono">
                    <div className="flex items-center gap-2">{arm}{arms[arm]?.diagnostic && <OracleBadge />}</div>
                  </TableCell>
                  <TableCell><VerdictBadge v={r.verdict} /></TableCell>
                  <TableCell className="tabular-nums">{signed(r.dJF)}</TableCell>
                  <TableCell className="tabular-nums">[{signed(r.CI_video?.[0])}, {signed(r.CI_video?.[1])}]</TableCell>
                  <TableCell className="tabular-nums">{r['videos_up_0.01']} / {r['videos_down_0.01']}</TableCell>
                  <TableCell className="tabular-nums">{signed(r.lovo_min)}</TableCell>
                  <TableCell className="max-w-56 truncate font-mono text-xs" title={r.worst_unit?.[0]}>{r.worst_unit?.[0]} ({signed(r.worst_unit?.[1], 3)})</TableCell>
                  <TableCell className="tabular-nums">{r.rescued_frames}</TableCell>
                  <TableCell className="tabular-nums">{r.false_reacq_frames} / {r.false_reacq_frames_SGGL}</TableCell>
                  <TableCell className="tabular-nums">{num(r.latency_s_mean, 3)} / {num(r.latency_s_SGGL, 3)}</TableCell>
                  <TableCell className="tabular-nums">{r.n_units} ({r.n_videos} vid)</TableCell>
                  <TableCell className="text-xs">{Object.entries(r.machines ?? {}).map(([m, n]) => `${m}:${n}`).join(' ')}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Stage-1 generator recall (co-absence)</CardTitle><CardDescription>stage1_generators.csv</CardDescription></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>{(d.generators[0] ? Object.keys(d.generators[0]) : []).map((h) => <TableHead key={h}>{h}</TableHead>)}</TableRow>
            </TableHeader>
            <TableBody>
              {d.generators.map((g: Record<string, string>) => (
                <TableRow key={g.gen}>
                  {Object.entries(g).map(([k, v]) => (
                    <TableCell key={k} className={k === 'gen' ? 'font-mono' : 'tabular-nums'}>
                      {k === 'adv_recall' ? <Badge variant={v === 'True' ? 'default' : 'outline'}>{v}</Badge> : num(v, 3)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <ArmDialog arm={sel} spec={sel ? arms[sel] : null} report={sel ? causal[sel] : null} onClose={() => setSel(null)} />
    </div>
  )
}

function ArmDialog({ arm, spec, report, onClose }: { arm: string | null; spec: any; report: any; onClose: () => void }) {
  const data = Object.entries((report?.per_video ?? {}) as Record<string, number>)
    .map(([video, v]) => ({ video, v }))
    .sort((a, b) => b.v - a.v)
  return (
    <Dialog open={!!arm} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-mono">
            {arm} <VerdictBadge v={report?.verdict} /> {spec?.diagnostic && <OracleBadge />}
          </DialogTitle>
        </DialogHeader>
        <div className="text-sm font-medium">video별 ΔJ&F (arm − SGGL)</div>
        <div className="flex gap-3 text-xs text-muted-foreground">
          <span><span className="mr-1 inline-block size-2 rounded-sm bg-[#2a78d6]" />arm 우세</span>
          <span><span className="mr-1 inline-block size-2 rounded-sm bg-[#eb6834]" />SGGL 우세</span>
        </div>
        <ResponsiveContainer width="100%" height={Math.max(160, data.length * 18 + 40)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
            <CartesianGrid horizontal={false} stroke="#e5e5e5" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="video" width={170} tick={{ fontSize: 11 }} interval={0} />
            <Tooltip formatter={(v) => signed(Number(v))} />
            <Bar dataKey="v" name="ΔJ&F" radius={2} maxBarSize={12}>
              {data.map((e) => <Cell key={e.video} fill={e.v >= 0 ? UP : DOWN} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="text-sm font-medium">frozen spec (arms.json)</div>
        <pre className="overflow-x-auto rounded bg-muted p-3 text-xs">{spec ? JSON.stringify(spec, null, 2) : 'arms.json에 정의 없음'}</pre>
      </DialogContent>
    </Dialog>
  )
}
