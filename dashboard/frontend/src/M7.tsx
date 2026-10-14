import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, signed, useFetch, type Nav } from '@/lib/api'
import { Help } from '@/HelpTip'
import { DOWN, Forest, INK, SERIES, UP, forestRows, tick } from './charts'

const isPilot = (v: any) => v && typeof v === 'object' && typeof v.dJF_vs_SGGL === 'number'
const ciColor = (_: string, v: any) => {
  const c = v?.CI_video ?? v?.CI ?? v?.CI_by_unit_descriptive
  return !c ? SERIES[0] : c[0] > 0 ? UP : c[1] < 0 ? DOWN : INK.muted
}

function PilotTable({ rows }: { rows: [string, any][] }) {
  return (
    <Table className="text-xs">
      <TableHeader>
        <TableRow>{([['pilot', 'pilot'], ['units / videos', 'units'], ['ΔJ&F vs SGGL', 'dJF'], ['CI (video)', 'ci'], ['vid ↑/↓', 'vid'], ['unit ↑/↓', 'unit'], ['LOVO min', 'lovo'], ['w/o miami', 'miami'], ['worst unit', 'worst'], ['best unit', 'best'], ['Δ vs P5', 'p5']] as const).map(([h, k]) => <TableHead key={h}>{h}<Help k={`m7.col.${k}`} /></TableHead>)}</TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(([k, r]) => (
          <TableRow key={k}>
            <TableCell className="font-mono">{k}<Help k={`m7.pilot.${k}`} /></TableCell>
            <TableCell>{r.n_units} / {r.n_videos}</TableCell>
            <TableCell className="tabular-nums font-medium">{signed(r.dJF_vs_SGGL)}</TableCell>
            <TableCell className="tabular-nums">[{signed(r.CI_video?.[0])}, {signed(r.CI_video?.[1])}]</TableCell>
            <TableCell>{r['videos_up_0.01']} / {r['videos_down_0.01']}</TableCell>
            <TableCell>{r['units_up_0.01']} / {r['units_down_0.01']}</TableCell>
            <TableCell className="tabular-nums">{signed(r.lovo_min)}</TableCell>
            <TableCell className="tabular-nums">{signed(r.without_miami)}</TableCell>
            <TableCell className="max-w-48 truncate font-mono" title={r.worst_unit?.[0]}>{r.worst_unit?.[0]} ({signed(r.worst_unit?.[1], 3)})</TableCell>
            <TableCell className="max-w-48 truncate font-mono" title={r.best_unit?.[0]}>{r.best_unit?.[0]} ({signed(r.best_unit?.[1], 3)})</TableCell>
            <TableCell className="tabular-nums">{signed(r.dJF_vs_P5)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function PilotCard({ title, desc, obj, hk }: { title: string; desc: string; obj: Record<string, any>; hk: string }) {
  const pilots = Object.entries(obj ?? {}).filter(([, v]) => isPilot(v))
  if (!pilots.length) return null
  return (
    <Card>
      <CardHeader><CardTitle>{title}<Help k={hk} /></CardTitle><CardDescription>{desc} · 색: CI가 0보다 크면 파랑, 0보다 작으면 빨강, 0을 포함하면 회색<Help k="m7.forest" /></CardDescription></CardHeader>
      <CardContent className="space-y-3">
        <Forest rows={forestRows(Object.fromEntries(pilots), ['dJF_vs_SGGL'], ['CI_video'], ciColor)} labelWidth={140} />
        <PilotTable rows={pilots.sort((a, b) => b[1].dJF_vs_SGGL - a[1].dJF_vs_SGGL)} />
      </CardContent>
    </Card>
  )
}

export default function M7({ nav }: { nav: Nav }) {
  const { data, error } = useFetch<any>('/api/m7', 60_000)
  if (!data) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  const R = data.reports
  const teDeltas = Object.fromEntries(Object.entries(R.teval_report ?? {}).filter(([k, v]: [string, any]) => k.includes('-') && v?.mean != null))
  const vk = Object.entries(R.vk_report ?? {}).filter(([, v]: [string, any]) => v && typeof v === 'object' && typeof v.mean === 'number') as [string, any][]
  const rp = R.replay_report ?? {}
  const tax = Object.entries(rp.taxonomy ?? {}).map(([k, v]: [string, any]) => ({ k, loss: v.SGGL_JF_loss_sum, frames: v.frames_primary })).sort((a, b) => b.loss - a.loss)
  const m3r = R.m3r_report

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader><CardTitle>M7 요약<Help k="m7.summary" /></CardTitle><CardDescription>최종 결정 KEEP_SGGL · M3-vel 도전자 · outputs/milestone7/*.json · 판정 규칙<Help k="m7.rule" /></CardDescription></CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {data.docs.map((d: string) => (
            <button key={d} onClick={() => nav('reports', d)} className="rounded border px-2 py-0.5 font-mono text-xs hover:bg-muted">{d.split('/').pop()}</button>
          ))}
        </CardContent>
      </Card>

      {isPilot(m3r) && (
        <Card>
          <CardHeader><CardTitle>M3-vel-r 개정판 재실행 (m3r_report)<Help k="m7.m3r" /></CardTitle><CardDescription>M7 도전자 M3-vel의 사전 등록 개정판 · 원본 M3-vel은 M8의 M3vel arm으로 재현함</CardDescription></CardHeader>
          <CardContent><PilotTable rows={[['M3-vel-r', m3r]]} /></CardContent>
        </Card>
      )}
      <PilotCard title="Causal pilots" desc="causal_pilots_report.json" obj={R.causal_pilots_report} hk="m7.causal" />
      <PilotCard title="T-head 평가 (T0–T4)" desc="teval_report.json" obj={R.teval_report} hk="m7.teval" />
      {Object.keys(teDeltas).length > 0 && (
        <Card>
          <CardHeader><CardTitle>T-head 단계 간 차이<Help k="m7.tdelta" /></CardTitle><CardDescription>teval_report.json의 Tk−Tj (mean, CI)</CardDescription></CardHeader>
          <CardContent><Forest rows={forestRows(teDeltas, ['mean'], ['CI'], ciColor)} labelWidth={80} /></CardContent>
        </Card>
      )}
      {vk.length > 0 && (
        <Card>
          <CardHeader><CardTitle>VKITTI (vk_report)<Help k="m7.vk" /></CardTitle><CardDescription>{R.vk_report.note} · units {R.vk_report.units} · frames {R.vk_report.frames}</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            <Forest rows={forestRows(Object.fromEntries(vk), ['mean'], ['CI_by_unit_descriptive'], ciColor)} labelWidth={120} />
            <Table className="text-xs">
              <TableHeader><TableRow>{([['비교', 'cmp'], ['mean', 'mean'], ['CI (unit, descriptive)', 'ci'], ['scene별', 'scene'], ['↑/↓ 0.01', 'updown'], ['worst', 'worst']] as const).map(([h, c]) => <TableHead key={h}>{h}<Help k={`m7.vk.col.${c}`} /></TableHead>)}</TableRow></TableHeader>
              <TableBody>
                {vk.map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell className="font-mono">{k}<Help k={`m7.vkrow.${k}`} /></TableCell>
                    <TableCell className="tabular-nums font-medium">{signed(v.mean)}</TableCell>
                    <TableCell className="tabular-nums">[{signed(v.CI_by_unit_descriptive?.[0])}, {signed(v.CI_by_unit_descriptive?.[1])}]</TableCell>
                    <TableCell className="font-mono">{Object.entries(v.by_scene ?? {}).map(([s, x]) => `${s.replace('vkitti/', '')}: ${signed(Number(x))}`).join(' · ')}</TableCell>
                    <TableCell>{v['up_0.01']} / {v['down_0.01']}</TableCell>
                    <TableCell className="font-mono">{v.worst?.[0]} ({signed(v.worst?.[1], 3)})</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      <PilotCard title="Replay pilots" desc={`replay_report.json · replay fidelity: max |ΔJ| ${num(rp.replay_fidelity?.max_abs_J_diff_SGGL_replay_vs_live, 5)}`} obj={rp.pilots} hk="m7.replay" />
      <div className="grid gap-4 xl:grid-cols-2">
        {rp.oracles_small_vs_SGGL && (
          <Card>
            <CardHeader><CardTitle>Oracle 상한 (small units vs SGGL)<Help k="m7.oracle" /></CardTitle><CardDescription>replay_report.oracles_small_vs_SGGL</CardDescription></CardHeader>
            <CardContent><Forest rows={forestRows(rp.oracles_small_vs_SGGL, ['mean'], ['CI'], ciColor)} labelWidth={90} /></CardContent>
          </Card>
        )}
        {tax.length > 0 && (
          <Card>
            <CardHeader><CardTitle>실패 분류 (taxonomy)<Help k="m7.taxonomy" /></CardTitle><CardDescription>클래스별 SGGL J&F 손실 합 (replay_report.taxonomy) · 두 view 모두 틀림, 동시 부재<Help k="m7.coabsence" /></CardDescription></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={tax.length * 24 + 40}>
                <BarChart data={tax} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid horizontal={false} stroke={INK.grid} />
                  <XAxis type="number" tick={tick} stroke={INK.axis} />
                  <YAxis type="category" dataKey="k" width={170} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
                  <Tooltip formatter={(v, n) => [num(v as number, 2), n]} />
                  <Bar dataKey="loss" name="J&F 손실 합" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
