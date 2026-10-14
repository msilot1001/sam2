import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { num, signed, useFetch } from '@/lib/api'
import { DOWN, Forest, INK, SERIES, Swatch, UP, tick } from './charts'
import { Help } from '@/HelpTip'

const RANKERS = ['R0_score', 'R1_trainfree', 'R2_logistic', 'R3_set']
// sequential blue ramp (light -> dark), step 100..700
const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

export default function Stage1() {
  const { data, error, at } = useFetch<any>('/api/stage1', 30_000)
  if (!data) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  const s1 = data.report ?? {}
  const gens = [...(s1.generators_coabs ?? [])].sort((a: any, b: any) => b.recall - a.recall)
  const loc = Object.entries(s1.localisation_coabs ?? {}).map(([gen, v]: [string, any]) => ({ gen, ...v }))
  const tests = Object.entries(s1.paired_tests ?? {}).map(([k, v]: [string, any]) => {
    const sig = v.CI && (v.CI[0] > 0 || v.CI[1] < 0)
    return { label: k, est: v.delta, lo: v.CI?.[0], hi: v.CI?.[1], color: !sig ? INK.muted : v.delta > 0 ? UP : DOWN, note: `${v.n_frames} frames · ${v.n_videos} videos${sig ? ' · CI가 0을 제외' : ''}` }
  }).sort((a, b) => b.est - a.est)
  const sets = Object.keys(data.rankers ?? {}).filter((s) => data.rankers[s])
  const rk = sets.map((s) => Object.fromEntries([['set', s], ...RANKERS.map((r) => [r, data.rankers[s][r]?.pooled_heldout?.net_dJ_frames ?? null])]))
  const hits: Record<string, Record<string, number>> = s1.hits_by_video ?? {}
  const hitGens = Object.keys(hits)
  const videos = [...new Set(hitGens.flatMap((g) => Object.keys(hits[g])))].sort()
  const hmax = Math.max(1, ...hitGens.flatMap((g) => Object.values(hits[g])))

  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">stage1_report.json · stage1_generators.csv · rankers_*.json · 30초마다 갱신 ({at?.toLocaleTimeString()})<Help k="stage1.about" /></div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>생성기 recall vs 거짓 양성 (co-absence 프레임)<Help k="stage1.recallChart" /></CardTitle>
            <CardDescription>recall: GT가 있는 co-absence 프레임에서 후보 중 하나라도 J &gt; 0.5인 비율 · 원시 FP: GT가 없는 프레임에서 후보가 '존재'를 출력한 비율 (순위화 전) · 점선: 기준 recall (max Gp/Lp)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={gens.length * 22 + 60}>
              <BarChart data={gens} layout="vertical" margin={{ left: 8, right: 24 }} barGap={1}>
                <CartesianGrid horizontal={false} stroke={INK.grid} />
                <XAxis type="number" domain={[0, 1]} tick={tick} stroke={INK.axis} />
                <YAxis type="category" dataKey="gen" width={110} tick={{ ...tick, fill: INK.secondary }} interval={0} stroke={INK.axis} />
                <Tooltip formatter={(v) => num(v as number, 3)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine x={s1.reference_recall_max_Gp_Lp} stroke={INK.secondary} strokeDasharray="4 3"
                  label={{ value: 'ref (max Gp/Lp)', fontSize: 10, position: 'top', fill: INK.secondary }} />
                <Bar dataKey="recall" name="recall" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={8} isAnimationActive={false} />
                <Bar dataKey="fp_rate_gt_empty" name="FP rate (GT 없음)" fill={SERIES[1]} radius={[0, 4, 4, 0]} maxBarSize={8} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle>위치 정확도 (localisation, co-absence)<Help k="stage1.locChart" /></CardTitle><CardDescription>localisation_coabs의 loc1 / loc3 (top-1 / top-3 위치 정확도)</CardDescription></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={loc.length * 40 + 60}>
                <BarChart data={loc} layout="vertical" margin={{ left: 8, right: 24 }} barGap={1}>
                  <CartesianGrid horizontal={false} stroke={INK.grid} />
                  <XAxis type="number" domain={[0, 1]} tick={tick} stroke={INK.axis} />
                  <YAxis type="category" dataKey="gen" width={80} tick={{ ...tick, fill: INK.secondary }} stroke={INK.axis} />
                  <Tooltip formatter={(v) => num(v as number, 3)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="loc1" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />
                  <Bar dataKey="loc3" fill={SERIES[2]} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>ranker의 pooled held-out 순이득<Help k="stage1.rankerChart" /></CardTitle><CardDescription>net ΔJ frames (REP→DEV와 DEV→REP 교차 적합을 합한 값) · 0보다 크면 이득</CardDescription></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rk} margin={{ left: 8, right: 24 }} barGap={2}>
                  <CartesianGrid vertical={false} stroke={INK.grid} />
                  <XAxis dataKey="set" tick={{ ...tick, fill: INK.secondary }} stroke={INK.axis} />
                  <YAxis tick={tick} stroke={INK.axis} />
                  <ReferenceLine y={0} stroke={INK.secondary} />
                  <Tooltip formatter={(v) => num(v as number, 2)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {RANKERS.map((r, i) => <Bar key={r} dataKey={r} fill={SERIES[i]} maxBarSize={18} isAnimationActive={false} />)}
                </BarChart>
              </ResponsiveContainer>
              <RankerTable rankers={data.rankers} sets={sets} />
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>짝지은 반증 검정 (paired falsification tests)<Help k="stage1.tests" /></CardTitle><CardDescription>delta와 bootstrap CI · CI가 0을 제외하면 파랑(+) 또는 빨강(−), 아니면 회색</CardDescription></CardHeader>
        <CardContent>
          <div className="mb-2 flex gap-4 text-xs text-muted-foreground">
            <Swatch color={UP} label="유의 +" /><Swatch color={DOWN} label="유의 −" /><Swatch color={INK.muted} label="CI가 0 포함" />
          </div>
          <Forest rows={tests} labelWidth={230} />
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {tests.map((t) => <span key={t.label} className="font-mono">{t.label}<Help k={`stage1.test.${t.label}`} /></span>)}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>video별 hit 수 히트맵 (생성기 × video)<Help k="stage1.hits" /></CardTitle><CardDescription>hits_by_video · 진할수록 많음 (최대 {hmax})</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="border-separate border-spacing-0.5 text-xs">
            <thead>
              <tr><th />{videos.map((v) => <th key={v} className="px-1 font-mono font-normal text-muted-foreground">{v}</th>)}</tr>
            </thead>
            <tbody>
              {hitGens.map((g) => (
                <tr key={g}>
                  <td className="whitespace-nowrap pr-2 font-mono text-muted-foreground">{g}<Help k={`stage1.gen.${g}`} /></td>
                  {videos.map((v) => {
                    const x = hits[g][v] ?? 0
                    const step = x === 0 ? -1 : Math.min(RAMP.length - 1, Math.floor((x / hmax) * RAMP.length))
                    return (
                      <td key={v} title={`${g} · ${v}: ${x}`} className="h-6 min-w-14 rounded text-center tabular-nums"
                        style={{ background: step < 0 ? '#f0efec' : RAMP[step], color: step >= 3 ? '#fff' : INK.primary }}>{x}</td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>생성기 표<Help k="stage1.genTable" /></CardTitle><CardDescription>co-absence 지표와 disagree/regular 트리거 recall</CardDescription></CardHeader>
        <CardContent>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>{['gen', 'recall', 'mean_best_J', 'top_J', 'videos_hit', 'units_hit', 'fp_rate_gt_empty', 'adv_recall', 'disagree recall', 'regular recall'].map((h) => <TableHead key={h}>{h}<Help k={`stage1.col.${h.split(' ')[0]}`} /></TableHead>)}</TableRow>
            </TableHeader>
            <TableBody>
              {gens.map((g: any) => (
                <TableRow key={g.gen}>
                  <TableCell className="whitespace-nowrap font-mono">{g.gen}<Help k={`stage1.gen.${g.gen}`} /></TableCell>
                  {['recall', 'mean_best_J', 'top_J', 'videos_hit', 'units_hit', 'fp_rate_gt_empty'].map((k) => <TableCell key={k} className="tabular-nums">{num(g[k], 3)}</TableCell>)}
                  <TableCell>{g.adv_recall ? '✓' : ''}</TableCell>
                  <TableCell className="tabular-nums">{num(s1.disagree_recall?.[g.gen], 3)}</TableCell>
                  <TableCell className="tabular-nums">{num(s1.regular_recall?.[g.gen], 3)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

const RCOLS = [['set', 'set'], ['ranker', 'ranker'], ['pooled net ΔJ', 'pooled'], ['rescued', 'rescued'], ['FP frames', 'fp'], ['fitREP→DEV net', 'fitREP'], ['fitDEV→REP net', 'fitDEV']]

function RankerTable({ rankers, sets }: { rankers: any; sets: string[] }) {
  return (
    <Table className="mt-3 text-xs">
      <TableHeader>
        <TableRow>{RCOLS.map(([h, k]) => <TableHead key={h}>{h}<Help k={`stage1.rcol.${k}`} /></TableHead>)}</TableRow>
      </TableHeader>
      <TableBody>
        {sets.flatMap((s) => RANKERS.map((r) => {
          const x = rankers[s][r] ?? {}
          return (
            <TableRow key={s + r}>
              <TableCell className="font-mono">{s}</TableCell><TableCell className="whitespace-nowrap font-mono">{r}<Help k={`stage1.ranker.${r}`} /></TableCell>
              <TableCell className="tabular-nums">{signed(x.pooled_heldout?.net_dJ_frames, 2)}</TableCell>
              <TableCell className="tabular-nums">{num(x.pooled_heldout?.rescued_frames, 1)}</TableCell>
              <TableCell className="tabular-nums">{num(x.pooled_heldout?.fp_frames, 1)}</TableCell>
              <TableCell className="tabular-nums">{signed(x.fitREP?.heldout_eval?.net_dJ_frames, 2)}</TableCell>
              <TableCell className="tabular-nums">{signed(x.fitDEV?.heldout_eval?.net_dJ_frames, 2)}</TableCell>
            </TableRow>
          )
        }))}
      </TableBody>
    </Table>
  )
}
