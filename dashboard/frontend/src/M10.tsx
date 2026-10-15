import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { fileUrl, num, signed, useFetch, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Forest, INK, SERIES, Swatch, tick, type ForestRow } from './charts'
import { Help } from '@/HelpTip'

const DS = ['mosev2', 'vost', 'lvosv2']
const DS_LABEL: Record<string, string> = { mosev2: 'MOSEv2', vost: 'VOST', lvosv2: 'LVOS v2' }
const DS_COLOR: Record<string, string> = { mosev2: SERIES[0], vost: SERIES[1], lvosv2: SERIES[2] }
const ARM_LABEL: Record<string, string> = {
  OR1off: '존재 (출력만, 오프라인)', OR5off: '후보 선택 best-of-7 (오프라인)', OR6off: '대상 영역 판별 (오프라인)',
  OR1M: '존재 → 기억 기록', OR2: '위치 (GT 중심)', OR3: '크기 (GT 크기)', OR4: '기하 (위치+크기)',
  OR5S: '후보 선택 (온라인 출력)', OR5M: '후보 선택 + 기억', OR6: '대상 영역 판별 + 기록 차단',
  OR7: '기억 기록 허용 (GT)', OR7W: '대상 밖 기록만 차단', OR7R: '대조: 같은 횟수 무작위 차단',
  OR8: '완벽한 기억 (GT 재부호화)', OR8C0: 'frame 0 기억만',
}
const STATES = ['correct', 'partial', 'empty', 'wrong_region', 'wrong_object', 'false_present']
const POOL = ['none', 'own_view_misselect', 'own_view_suppressed', 'other_view_output', 'other_view_hyp']
const ds = (k: string) => DS_LABEL[k] ?? k
const dsLabel = ds
const pct = (v: any) => (typeof v === 'number' ? `${(100 * v).toFixed(1)}%` : '—')

function Progress({ d }: { d: any }) {
  const runs: Record<string, Record<string, number>> = d.runs ?? {}
  const tot: Record<string, number> = d.dev_units ?? {}
  return (
    <Card>
      <CardHeader><CardTitle>실행 현황<Help k="m10.progress" /></CardTitle><CardDescription>결과 CSV 수 / dev 단위 수 · runpod = 단일 oracle (RTX 4090), ubuntu/&lt;모델&gt; = 용량 진단 (RTX 2080, VOST부터)<Help k="m10.capacity" /></CardDescription></CardHeader>
      <CardContent className="space-y-1">
        {Object.keys(runs).length === 0 && <p className="text-sm text-muted-foreground">아직 결과 없음</p>}
        {Object.entries(runs).map(([key, by]) => DS.filter((k) => by[k] || key.startsWith('runpod')).map((k) => {
          const n = by[k] ?? 0, t = tot[k] ?? 1
          return (
            <div key={key + k} className="flex items-center gap-2 text-xs">
              <span className="w-32 truncate font-mono">{key}</span>
              <span className="w-16">{ds(k)}</span>
              <div className="h-1.5 flex-1 rounded bg-muted">
                <div className={cn('h-1.5 rounded', n >= t ? 'bg-[#0ca30c]' : 'bg-[#2a78d6]')} style={{ width: `${Math.min(100, (100 * n) / t)}%` }} />
              </div>
              <span className="w-14 text-right tabular-nums">{n}/{t}</span>
            </div>
          )
        }))}
      </CardContent>
    </Card>
  )
}

function OracleTable({ d }: { d: any }) {
  const t = d.reports?.oracle_table
  if (!t) return <Card><CardHeader><CardTitle>Oracle 상한표<Help k="m10.oracleTable" /></CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">oracle_table.json 아직 없음 (scripts/m10_analysis.py table)</CardContent></Card>
  const arms = Object.keys(t.arms ?? {})
  const rows: ForestRow[] = arms.flatMap((a) => DS.filter((k) => t.arms[a].per_dataset?.[k]).map((k) => {
    const r = t.arms[a].per_dataset[k]
    return { label: `${a} · ${ds(k)}`, est: r.dJF, lo: r.ci?.[0], hi: r.ci?.[1], color: DS_COLOR[k], hollow: t.arms[a].kind !== 'online-rollout',
             note: `${r.n_units}단위 / ${r.n_videos}영상${r.complete ? '' : ' (진행 중)'}` }
  }))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Oracle 상한표 — ΔJ&F (oracle − OR0), dev<Help k="m10.oracleTable" /></CardTitle>
        <CardDescription>reports/oracle_table.json · 영상 동일 가중 · 수염 = 영상 군집 bootstrap 95% CI · 빈 마름모 = 오프라인(사후, rollout 없음): 온라인 arm과 같은 성격으로 비교하지 말 것</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          {DS.map((k) => <Swatch key={k} color={DS_COLOR[k]} label={ds(k)} />)}
          <Swatch color={INK.secondary} hollow label="오프라인 oracle" /><Help k="m10.forest" />
        </div>
        <Forest rows={rows} labelWidth={170} />
        <div className="overflow-x-auto">
          <Table className="text-xs">
            <TableHeader><TableRow>
              <TableHead>arm</TableHead><TableHead>뜻</TableHead><TableHead>종류<Help k="m10.col.kind" /></TableHead>
              {DS.map((k) => <TableHead key={k}>{ds(k)} ΔJ&F [CI] · 회복<Help k="m10.col.recovered" /></TableHead>)}
              <TableHead>기억 기록률 G/L<Help k="m10.col.admit" /></TableHead>
            </TableRow></TableHeader>
            <TableBody>
              <TableRow className="text-muted-foreground"><TableCell className="font-mono">OR0<Help k="m10.arm.OR0" /></TableCell><TableCell>SGGL 기준</TableCell><TableCell>기준</TableCell>
                {DS.map((k) => <TableCell key={k}>{t.baseline?.[k] ? `J&F ${num(t.baseline[k].JF_video_equal, 3)} · ${t.baseline[k].units}단위` : '—'}</TableCell>)}<TableCell /></TableRow>
              {arms.map((a) => {
                const r = t.arms[a]
                return (
                  <TableRow key={a}>
                    <TableCell className="font-mono whitespace-nowrap">{a}<Help k={`m10.arm.${a}`} /></TableCell>
                    <TableCell>{ARM_LABEL[a] ?? ''}</TableCell>
                    <TableCell><Badge variant="outline">{r.kind === 'online-rollout' ? '온라인' : '오프라인'}</Badge></TableCell>
                    {DS.map((k) => {
                      const x = r.per_dataset?.[k]
                      return <TableCell key={k} className="tabular-nums">{x ? <span className={cn(x.ci?.[0] > 0 && 'font-semibold text-[#2a78d6]', x.ci?.[1] < 0 && 'text-[#e34948]')}>
                        {signed(x.dJF, 3)} [{signed(x.ci?.[0], 3)}, {signed(x.ci?.[1], 3)}] · {pct(x.recovered_frac)}{x.complete ? '' : ' *'}</span> : 'NOT_RUN'}</TableCell>
                    })}
                    <TableCell className="tabular-nums">{r.admit_rate ? `${pct(r.admit_rate.G)} / ${pct(r.admit_rate.L)}` : '—'}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          <p className="mt-1 text-xs text-muted-foreground">회복 = ΔJ&F / (1 − OR0 J&F) · * = 아직 일부 단위만 완료</p>
        </div>
      </CardContent>
    </Card>
  )
}

function Stacked({ title, desc, data, keys, hk }: { title: string; desc: string; data: any[]; keys: string[]; hk: string }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}<Help k={hk} /></CardTitle><CardDescription>{desc}</CardDescription></CardHeader>
      <CardContent>
        {data.length === 0 ? <p className="text-sm text-muted-foreground">아직 없음</p> : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid stroke={INK.grid} horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={tick} />
              <YAxis type="category" dataKey="ds" width={70} tick={tick} />
              <Tooltip formatter={(v, n) => [pct(v), n]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {keys.map((k, i) => <Bar key={k} dataKey={k} stackId="a" fill={SERIES[i % SERIES.length]} />)}
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

function Recovery({ d }: { d: any }) {
  const s = d.reports?.states?.recovery
  if (!s) return null
  return (
    <Card>
      <CardHeader><CardTitle>어느 상태의 손실을 회복하는가<Help k="m10.recovery" /></CardTitle><CardDescription>reports/states.json · OR0 상태별 (1 − J&F) 합 중 arm이 되찾은 비율 (음수 = 더 나빠짐)</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader><TableRow><TableHead>arm · 데이터셋</TableHead>{STATES.map((k) => <TableHead key={k}>{k}</TableHead>)}<TableHead><Help k="m10.lossStates" /></TableHead></TableRow></TableHeader>
          <TableBody>
            {Object.entries(s).flatMap(([a, by]: [string, any]) => Object.entries(by).map(([k, r]: [string, any]) => (
              <TableRow key={a + k}><TableCell className="font-mono whitespace-nowrap">{a}<Help k={`m10.arm.${a}`} /> · {ds(k)}</TableCell>
                {STATES.map((st) => <TableCell key={st} className="tabular-nums">{pct(r.recovered_frac_by_state?.[st])}</TableCell>)}</TableRow>
            )))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Episodes({ d }: { d: any }) {
  const e = d.reports?.episodes
  if (!e) return null
  const arms = Object.keys(e).filter((a) => a !== 'OR0_all')
  const cell = (x: any) => x ? `${x.episodes} · ${num(x.mean_dur, 1)}f · ${num(x.loss_sum, 0)}` : '—'
  return (
    <Card>
      <CardHeader><CardTitle>이탈 에피소드 (arm vs 같은 단위의 OR0)<Help k="m10.episodes" /></CardTitle><CardDescription>reports/episodes.json · 이탈 = 출력이 대상 밖(precision &lt; 0.5)인 연속 frame · 칸 = 에피소드 수 · 평균 길이 · 손실 합</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader><TableRow><TableHead>arm · 데이터셋</TableHead><TableHead>단위</TableHead><TableHead>arm 이탈</TableHead><TableHead>OR0 이탈</TableHead><TableHead>arm 손실 구간</TableHead><TableHead>OR0 손실 구간</TableHead></TableRow></TableHeader>
          <TableBody>
            {arms.flatMap((a) => Object.entries(e[a]).map(([k, r]: [string, any]) => (
              <TableRow key={a + k}><TableCell className="font-mono whitespace-nowrap">{a}<Help k={`m10.arm.${a}`} /> · {ds(k)}</TableCell><TableCell>{r.units}</TableCell>
                <TableCell>{cell(r.arm?.drift)}</TableCell><TableCell>{cell(r.OR0?.drift)}</TableCell>
                <TableCell>{cell(r.arm?.loss)}</TableCell><TableCell>{cell(r.OR0?.loss)}</TableCell></TableRow>
            )))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Or9({ d }: { d: any }) {
  const o = d.reports?.or9
  if (!o) return null
  return (
    <Card>
      <CardHeader><CardTitle>OR9 분할 용량 상한 (GT box prompt, 이미지 모드)<Help k="m10.or9" /></CardTitle><CardDescription>reports/or9.json · GT가 있는 frame만 · 같은 frame의 OR0와 비교 · 추적 결과가 아니라 frame별 분할 능력의 상한</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader><TableRow><TableHead>데이터셋</TableHead><TableHead>단위 / frame</TableHead><TableHead>OR0 J&F</TableHead><TableHead>box 전체 frame</TableHead><TableHead>box + 완벽 crop</TableHead><TableHead>Δ 전체 [CI]</TableHead><TableHead>Δ crop [CI]</TableHead><TableHead>극소 (crop)</TableHead></TableRow></TableHeader>
          <TableBody>{DS.filter((k) => o[k]).map((k) => (
            <TableRow key={k}><TableCell>{ds(k)}</TableCell><TableCell>{o[k].units} / {o[k].frames}</TableCell><TableCell>{num(o[k].full.JF0, 3)}</TableCell>
              <TableCell>{num(o[k].full.JF_oracle, 3)}</TableCell><TableCell>{num(o[k].crop.JF_oracle, 3)}</TableCell>
              <TableCell>{signed(o[k].full.dJF, 3)} [{signed(o[k].full.ci?.[0], 3)}, {signed(o[k].full.ci?.[1], 3)}]</TableCell>
              <TableCell>{signed(o[k].crop.dJF, 3)} [{signed(o[k].crop.ci?.[0], 3)}, {signed(o[k].crop.ci?.[1], 3)}]</TableCell>
              <TableCell>{num(o[k].crop.tiny_JF_oracle, 3)}</TableCell></TableRow>
          ))}</TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Interactions({ d }: { d: any }) {
  const x = d.reports?.interactions
  if (!x || !Object.keys(x).length) return null
  const ci = (c: any) => (c ? `[${signed(c[0], 3)}, ${signed(c[1], 3)}]` : '')
  return (
    <Card>
      <CardHeader><CardTitle>조합: 독립 · 조건부 · 상호작용<Help k="m10.interactions" /></CardTitle><CardDescription>reports/interactions.json · 네 arm 모두 끝난 같은 단위만 · 영상 동일 가중</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader><TableRow><TableHead>조합</TableHead><TableHead>데이터셋</TableHead><TableHead>단위</TableHead><TableHead>ΔA</TableHead><TableHead>ΔB</TableHead><TableHead>ΔA+B</TableHead><TableHead>B | A (조건부)</TableHead><TableHead>A | B (조건부)</TableHead><TableHead>상호작용</TableHead></TableRow></TableHeader>
          <TableBody>{Object.entries(x).flatMap(([k, r]: [string, any]) => DS.filter((ds) => r[ds]).map((ds) => (
            <TableRow key={k + ds}>
              <TableCell className="font-mono whitespace-nowrap">{k}<Help k={`m10.arm.${k}`} /> = {r.A} + {r.B}</TableCell><TableCell>{dsLabel(ds)}</TableCell><TableCell>{r[ds].units}</TableCell>
              <TableCell>{signed(r[ds].dA, 3)}</TableCell><TableCell>{signed(r[ds].dB, 3)}</TableCell>
              <TableCell>{signed(r[ds].dAB, 3)} {ci(r[ds].ci_AB)}</TableCell>
              <TableCell>{signed(r[ds].cond_B_given_A, 3)} {ci(r[ds].ci_cond_B_given_A)}</TableCell>
              <TableCell>{signed(r[ds].cond_A_given_B, 3)} {ci(r[ds].ci_cond_A_given_B)}</TableCell>
              <TableCell>{signed(r[ds].interaction, 3)} {ci(r[ds].ci_interaction)}</TableCell>
            </TableRow>
          )))}</TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Gate({ d }: { d: any }) {
  const g = d.reports?.selector_gate
  if (!g) return null
  const rules = Object.keys(g.rules ?? {})
  return (
    <Card>
      <CardHeader>
        <CardTitle>개선 단계: GT 없는 선택기 오프라인 관문 (수정 5)<Help k="m10.gate" /></CardTitle>
        <CardDescription>reports/selector_gate.json · OR0F 궤적 · 6개 원시 가설만 (빈 mask 제외) · 통과 기준: τ*에서 MOSEv2·VOST 각각 ΔJ CI 하한 &gt; 0, 스위치 정밀도 ≥ 0.6, 이탈 frame ΔJ &gt; 0, 맞던 frame ΔJ ≥ −0.005 · 선택된 규칙: <b>{g.chosen ?? '없음'}</b></CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader><TableRow><TableHead>규칙</TableHead><TableHead>τ* (정밀도 0.70)</TableHead><TableHead>판정</TableHead>
            {DS.map((k) => <TableHead key={k}>{ds(k)} τ=0: ΔJ · 스위치 · 정밀도 / 악화</TableHead>)}<TableHead>합동 정밀도 곡선 (τ: 스위치, 정밀도)</TableHead></TableRow></TableHeader>
          <TableBody>{rules.map((r) => {
            const x = g.rules[r]
            return (
              <TableRow key={r}>
                <TableCell className="font-mono whitespace-nowrap">{r}<Help k={`m10.rule.${r}`} /></TableCell>
                <TableCell>{x.tau_star == null ? '도달 못 함' : num(x.tau_star, 3)}</TableCell>
                <TableCell><Badge variant={x.pass ? 'default' : 'outline'}>{x.pass ? 'PASS' : 'FAIL'}</Badge></TableCell>
                {DS.map((k) => {
                  const e = x.at_tau0?.[k]
                  return <TableCell key={k} className="tabular-nums whitespace-nowrap">{e ? `${signed(e.dJ_present, 3)} · ${e.switches} · ${num(e.precision, 2)} / ${num(e.worse_share, 2)}` : '—'}</TableCell>
                })}
                <TableCell className="tabular-nums text-[11px]">{(x.curve ?? []).filter((_: any, i: number) => i % 4 === 0).map((c: any) => `${num(c.tau, 2)}: ${c.switches}, ${num(c.precision_pooled, 2)}`).join(' · ')}</TableCell>
              </TableRow>
            )
          })}</TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Json({ title, obj }: { title: string; obj: any }) {
  return (
    <details className="rounded-md border p-2 text-xs">
      <summary className="cursor-pointer font-medium">{title}</summary>
      <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap">{JSON.stringify(obj, null, 1)}</pre>
    </details>
  )
}

export default function M10({ nav }: { nav: Nav }) {
  const { data: d, error, at } = useFetch<any>('/api/m10', 30_000)
  if (!d) return <p className="p-6 text-muted-foreground">{error ?? '불러오는 중…'}</p>
  const R = d.reports ?? {}
  const loss = R.states?.OR0_loss_share ?? {}
  const pool = R.pool ?? {}
  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">
        <span className="mr-2 text-sm font-semibold text-foreground">M10 Oracle 병목 분해<Help k="m10.about" /></span>
        30초마다 자동 갱신 · 마지막 갱신 {at?.toLocaleTimeString()} {error && <span className="text-red-600">· 오류: {error}</span>}
        {(d.docs ?? []).map((p: string) => <button key={p} className="ml-3 underline" onClick={() => nav('reports', p)}>{p.split('/').pop()}</button>)}
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Progress d={d} />
        <Card>
          <CardHeader><CardTitle>기준선 재현 (OR0 vs M9 SGGL)<Help k="m10.parity" /></CardTitle><CardDescription>reports/parity.json · 같은 frame J&F 차이</CardDescription></CardHeader>
          <CardContent className="space-y-1 text-xs">
            {Object.entries(R.parity ?? {}).map(([k, r]: [string, any]) => (
              <div key={k}>{ds(k)}: {r.frames} frame · 최대 |Δ| {num(r.max_abs, 6)} · 같은 단위 {r.units_identical}/{r.units}</div>
            ))}
            {!R.parity && <p className="text-muted-foreground">아직 없음</p>}
          </CardContent>
        </Card>
      </div>
      <Gate d={d} />
      <OracleTable d={d} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Stacked title="OR0 손실의 상태별 비중" desc="reports/states.json · (1 − J&F) 합 기준 · GT 기반 사후 분류"
          data={DS.filter((k) => loss[k]).map((k) => ({ ds: ds(k), ...loss[k].loss_share }))} keys={STATES} hk="m10.lossStates" />
        <Stacked title="놓친 frame: 정답 후보는 어디에 있나" desc="reports/pool.json · GT 있는데 J < 0.5인 frame의 손실 · none = 7개 후보 모두 J < 0.5 (생성 실패)"
          data={DS.filter((k) => pool[k]).map((k) => ({ ds: ds(k), ...pool[k].classes_loss_share }))} keys={POOL} hk="m10.pool" />
      </div>
      <Recovery d={d} />
      <Episodes d={d} />
      <Or9 d={d} />
      <Interactions d={d} />
      {(d.figures ?? []).length > 0 && (
        <Card><CardHeader><CardTitle>실패 에피소드 그림<Help k="m10.figures" /></CardTitle></CardHeader>
          <CardContent className="space-y-3">{d.figures.map((p: string) => (
            <div key={p}><div className="font-mono text-xs">{p}</div><img src={fileUrl(p)} className="max-w-full border" loading="lazy" /></div>
          ))}</CardContent></Card>
      )}
      <Card>
        <CardHeader><CardTitle>최근 로그<Help k="m10.logs" /></CardTitle></CardHeader>
        <CardContent className="space-y-2">{(d.logs ?? []).map((l: any) => (
          <div key={l.path}><button className="font-mono text-xs underline" onClick={() => nav('raw', l.path)}>{l.path}</button>
            <pre className="max-h-32 overflow-auto bg-muted p-2 text-[11px]">{l.tail}</pre></div>
        ))}</CardContent>
      </Card>
      <div className="text-xs text-muted-foreground">사전 등록과 수정<Help k="m10.prereg" /></div>
      <Json title="사전 등록 (preregistration.json)" obj={d.prereg} />
      {Object.entries(d.amendments ?? {}).map(([k, v]) => <Json key={k} title={k} obj={v} />)}
    </div>
  )
}
