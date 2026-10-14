import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { num, signed, useFetch, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Forest, OracleBadge, Stat, Swatch, VERDICT_COLOR, VerdictBadge, type ForestRow } from './charts'
import { Help } from '@/HelpTip'

const EXPECTED_UNITS = 64
const MACHINE_LABEL: Record<string, string> = { runpod: 'RunPod 4090', ubuntu: 'Ubuntu 2080' }

type Task = {
  id: string; title: string; description?: string; machine?: string; status: string
  progress?: { glob: string; total: number; done: number | null; percent: number | null }; depends_on?: string[]
}
const STATUS_ORDER = ['RUNNING', 'QUEUED', 'BLOCKED', 'DONE']
const STATUS_CLS: Record<string, string> = {
  RUNNING: 'bg-blue-100 text-blue-900', QUEUED: 'bg-muted text-foreground', BLOCKED: 'bg-red-100 text-red-800', DONE: 'bg-emerald-100 text-emerald-900',
}
const STATUS_KO: Record<string, string> = { RUNNING: '실행 중', QUEUED: '대기', BLOCKED: '차단됨', DONE: '완료' }

const HStat = ({ k, ...p }: { k: string } & Parameters<typeof Stat>[0]) => (
  <div className="flex items-start gap-1"><Stat {...p} /><Help k={k} /></div>
)

function TaskRow({ t, byId }: { t: Task; byId: Record<string, Task> }) {
  const p = t.progress
  return (
    <div className={cn('space-y-1 border-t first:border-t-0', t.status === 'QUEUED' ? 'py-1' : 'py-2')}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary" className={STATUS_CLS[t.status]}>{t.status === 'RUNNING' && '● '}{STATUS_KO[t.status] ?? t.status}</Badge>
        <span className="font-medium">{t.title}</span>
        <span className="font-mono text-xs text-muted-foreground">{t.id}</span>
        {t.machine && <span className="text-xs text-muted-foreground">· {t.machine}</span>}
      </div>
      {t.description && <div className="text-sm text-muted-foreground">{t.description}</div>}
      {p && (t.status !== 'QUEUED' || (p.done ?? 0) > 0) && (
        <div className="flex items-center gap-2 text-xs">
          <div className="h-2 flex-1 rounded bg-muted" title={p.glob}>
            <div className={cn('h-2 rounded', t.status === 'DONE' ? 'bg-[#0ca30c]' : 'bg-[#2a78d6]')} style={{ width: `${Math.min(100, p.percent ?? 0)}%` }} />
          </div>
          <span className="w-28 text-right tabular-nums">{p.done ?? '?'} / {p.total} ({num(p.percent, 1)}%)</span>
        </div>
      )}
      {!!t.depends_on?.length && (
        <div className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
          선행 작업:<Help k="overview.taskDeps" />
          {t.depends_on.map((d) => (
            <Badge key={d} variant="outline" className={byId[d]?.status === 'DONE' ? 'text-emerald-800' : ''}>
              {d}{byId[d] ? ` · ${STATUS_KO[byId[d].status] ?? byId[d].status}` : ''}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

const MS_DIR: Record<string, string> = { m8: 'outputs/milestone8', m9: 'outputs/milestone9' }

export function Tasks({ nav, ms = 'm8' }: { nav: Nav; ms?: 'm8' | 'm9' }) {
  const { data, error, at } = useFetch<any>(`/api/tasks?ms=${ms}`, 15_000)
  const tasks: Task[] = data?.tasks?.tasks ?? []
  const byId = Object.fromEntries(tasks.map((t) => [t.id, t]))
  const groups = STATUS_ORDER.map((s) => [s, tasks.filter((t) => t.status === s)] as const)
  const other = tasks.filter((t) => !STATUS_ORDER.includes(t.status))
  return (
    <Card className="border-[#2a78d6]/40">
      <CardHeader>
        <CardTitle>현재 작업 — {ms.toUpperCase()}{ms === 'm9' && <span className="font-normal text-muted-foreground"> (교차 데이터셋 검증)</span>}<Help k={ms === 'm9' ? 'm9.tasks' : 'overview.tasks'} /></CardTitle>
        <CardDescription>
          TASKS.json · 갱신 {data?.tasks?.updated_kst ?? '—'} · 진행률은 결과 파일 수를 실시간으로 센 값<Help k="overview.taskProgress" /> · 15초마다 갱신 ({at?.toLocaleTimeString()})
          {error && <span className="text-red-600"> · 오류: {error}</span>}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 xl:grid-cols-[3fr_2fr]">
        <div>
          {data && !data.tasks && <p className="text-sm text-muted-foreground">{MS_DIR[ms]}/TASKS.json 없음</p>}
          {data?.tasks?.note && <p className="mb-2 text-xs text-muted-foreground">{data.tasks.note}</p>}
          {groups.map(([s, ts]) => ts.length === 0 ? null : s === 'DONE' ? (
            <details key={s} className="mt-2">
              <summary className="cursor-pointer text-sm font-medium">완료 {ts.length}건<Help k="overview.taskStatus" /></summary>
              {ts.map((t) => <TaskRow key={t.id} t={t} byId={byId} />)}
            </details>
          ) : (
            <div key={s} className="mb-2">
              <div className="text-xs font-semibold uppercase text-muted-foreground">{STATUS_KO[s]} ({ts.length})<Help k="overview.taskStatus" /></div>
              {ts.map((t) => <TaskRow key={t.id} t={t} byId={byId} />)}
            </div>
          ))}
          {other.map((t) => <TaskRow key={t.id} t={t} byId={byId} />)}
        </div>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-semibold uppercase text-muted-foreground">최신 로그<Help k="overview.logs" /></div>
          {(data?.logs ?? []).map((l: any) => (
            <div key={l.path}>
              <button className="font-mono text-xs underline" onClick={() => nav('raw', l.path)}>{l.path.replace(`${MS_DIR[ms]}/`, '')}</button>
              <span className="ml-2 text-xs text-muted-foreground">{new Date(l.mtime * 1000).toLocaleTimeString()}</span>
              <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted p-2 text-[11px] leading-snug">{l.tail}</pre>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default function Overview({ nav }: { nav: Nav }) {
  const { data: d, error, at } = useFetch<any>('/api/overview', 30_000)
  if (!d) return <div className="space-y-4 p-4"><Tasks nav={nav} ms="m9" /><Tasks nav={nav} /><p className="text-muted-foreground">{error ?? '불러오는 중…'}</p></div>
  const arms = d.arms ?? {}
  const causal: Record<string, any> = Object.fromEntries(Object.entries(d.causal ?? {}).filter(([k]) => !k.includes('@')))
  const s1 = d.stage1 ?? {}
  const sync = d.sync ?? {}
  const targets = [...new Set(Object.keys(sync).filter((k) => k.includes(':')).map((k) => k.split(':').slice(1).join(':')))]
  const promising = Object.entries(causal).filter(([, r]) => r.verdict === 'PROMISING' || r.verdict === 'STRONG_WIN').sort((a, b) => b[1].dJF - a[1].dJF)
  const best = promising.find(([a]) => !arms[a]?.diagnostic)
  const bestAny = promising[0]
  const forest: ForestRow[] = Object.entries(causal)
    .map(([a, r]) => ({
      label: `${a} (${r.n_units})`, est: r.dJF, lo: r.CI_video?.[0], hi: r.CI_video?.[1],
      color: VERDICT_COLOR[r.verdict] ?? '#898781', hollow: !!arms[a]?.diagnostic,
      note: `${r.verdict}${arms[a]?.diagnostic ? ' · ORACLE (배포 불가)' : ''} · ${Object.entries(r.machines ?? {}).map(([m, n]) => `${m}:${n}`).join(' ')}`,
    }))
    .sort((a, b) => b.est - a.est)
  const runs = d.runs as Record<string, Record<string, string[]>>

  return (
    <div className="space-y-4 p-4">
      <div className="text-xs text-muted-foreground">
        <span className="mr-2 text-sm font-semibold text-foreground">M8 연구 로드맵<Help k="overview.roadmap" /></span>
        30초마다 자동 갱신 · 마지막 갱신 {at?.toLocaleTimeString()} {error && <span className="text-red-600">· 오류: {error}</span>}
      </div>

      <Tasks nav={nav} ms="m9" />
      <Tasks nav={nav} />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader><CardTitle>현재 기준 알고리즘 (incumbent)<Help k="overview.incumbent" /></CardTitle><CardDescription>M7 최종 결정: KEEP_SGGL</CardDescription></CardHeader>
          <CardContent><Stat label="incumbent" value="SGGL" sub="모든 arm을 같은 기계의 SGGL과 비교 (machine-matched)" /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>최고 causal 도전자<Help k="overview.bestChallenger" /></CardTitle><CardDescription>배포 가능한 PROMISING arm 중 ΔJ&F가 가장 큰 것</CardDescription></CardHeader>
          <CardContent className="space-y-2">
            {best ? (
              <button className="text-left" onClick={() => nav('arms', best[0])}>
                <Stat label={best[0]} value={signed(best[1].dJF)} sub={<>CI [{signed(best[1].CI_video?.[0])}, {signed(best[1].CI_video?.[1])}] · {best[1].n_units} units</>} />
              </button>
            ) : <div className="text-sm text-muted-foreground">PROMISING arm 없음</div>}
            {bestAny && bestAny !== best && (
              <div className="text-xs text-muted-foreground">진단용(ORACLE)까지 넣으면: <span className="font-mono">{bestAny[0]}</span> {signed(bestAny[1].dJF)}<Help k="overview.oracleNote" /></div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Causal 진행 상황<Help k="overview.progress" /></CardTitle><CardDescription>arm별 완료한 unit 수 (기준 {EXPECTED_UNITS})</CardDescription></CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(runs).map(([m, byArm]) => (
              <div key={m}>
                <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">{MACHINE_LABEL[m] ?? m}</div>
                {Object.keys(byArm).length === 0 && <div className="text-sm text-muted-foreground">없음</div>}
                {Object.entries(byArm).map(([arm, units]) => (
                  <div key={arm} className="flex items-center gap-2 text-xs">
                    <span className="w-32 truncate font-mono" title={arm}>{arm}{arms[arm]?.diagnostic ? ' ◇' : ''}</span>
                    <div className="h-1.5 flex-1 rounded bg-muted">
                      <div className={cn('h-1.5 rounded', units.length >= EXPECTED_UNITS ? 'bg-[#0ca30c]' : 'bg-[#2a78d6]')}
                        style={{ width: `${Math.min(100, (100 * units.length) / EXPECTED_UNITS)}%` }} />
                    </div>
                    <span className="w-12 text-right tabular-nums">{units.length}/{EXPECTED_UNITS}</span>
                  </div>
                ))}
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>동기화 상태<Help k="overview.sync" /></CardTitle><CardDescription>SYNC_STATUS.json · {sync.updated}<Help k="overview.syncTargets" /></CardDescription></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex flex-wrap gap-1">
              {Object.entries(d.ledger as Record<string, number>).map(([k, v]) => (
                <Badge key={k} variant={/FAIL|CONFLICT/i.test(k) ? 'destructive' : 'outline'}>{k}: {v}</Badge>
              ))}
              <Help k="overview.ledger" />
            </div>
            {targets.map((t) => {
              const ok = sync[`last_ok:${t}`], fail = sync[`last_fail:${t}`]
              const failing = fail && (!ok || fail.slice(0, 19) > ok)
              return (
                <div key={t} className="border-t pt-1">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    {t.replace('outputs/milestone8/', '')}
                    <Badge variant={failing ? 'destructive' : 'secondary'}>{failing ? '✕ 실패' : '✓ OK'}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">OK {ok ?? '—'} · 신규 {sync[`last_new:${t}`] ?? 0} · 충돌 {sync[`conflicts:${t}`] ?? 0}</div>
                  {failing && <div className="truncate text-xs text-red-700" title={fail}>{fail}</div>}
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Causal arms forest plot — ΔJ&F (arm − machine-matched SGGL)<Help k="overview.forest" /></CardTitle>
          <CardDescription>점은 평균 ΔJ&F, 수염은 video-cluster bootstrap CI · 이름 옆 괄호는 unit 수 · 클릭하면 Causal arms 페이지로 이동</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
            <Swatch color={VERDICT_COLOR.PROMISING} label="▲ PROMISING" />
            <Swatch color={VERDICT_COLOR.INCONCLUSIVE} label="? INCONCLUSIVE" />
            <Swatch color={VERDICT_COLOR.REJECTED} label="✕ REJECTED" />
            <Swatch color="#52514e" hollow label="◇ 빈 마름모는 diagnostic/ORACLE (GT를 써서 배포 불가)" />
            <Help k="overview.verdicts" />
          </div>
          <div className="cursor-pointer" onClick={() => nav('arms')}><Forest rows={forest} labelWidth={200} /></div>
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(causal).map(([a, r]) => (
              <button key={a} onClick={() => nav('arms', a)} className="flex items-center gap-1 rounded border px-2 py-0.5 text-xs hover:bg-muted">
                <span className="font-mono">{a}</span><VerdictBadge v={r.verdict} />{arms[a]?.diagnostic && <OracleBadge />}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Stage-1 진단 요약<Help k="overview.stage1" /></CardTitle><CardDescription>stage1_report.json · 자세한 내용은 Stage 1 페이지</CardDescription></CardHeader>
        <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-6">
          <HStat k="overview.s1Units" label="units / videos" value={`${s1.n_units} / ${s1.n_videos}`} sub={`${s1.frames} frames`} />
          <HStat k="overview.s1Coabs" label="co-absence 트리거" value={s1.trigger_counts?.coabs} sub={`GT 있음 ${s1.coabs_gt_present} · 없음 ${s1.coabs_gt_empty}`} />
          <HStat k="overview.s1OracleRecall" label="oracle any-generator recall" value={num(s1.oracle_any_generator_recall, 3)} />
          <HStat k="overview.s1RefRecall" label="reference recall (max Gp/Lp)" value={num(s1.reference_recall_max_Gp_Lp, 3)} />
          <HStat k="overview.s1Latency" label="SGGL latency / frame" value={`${num(s1.latency_s?.sggl_per_frame, 3)} s`} sub={`rescue(all)/trigger ${num(s1.latency_s?.rescue_all_generators_per_trigger, 2)} s`} />
          <HStat k="overview.s1Vram" label="peak VRAM" value={`${num(s1.peak_vram_gb, 2)} GB`} />
        </CardContent>
      </Card>
    </div>
  )
}
