import { useEffect, useMemo, useState } from 'react'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import markdown from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { bytes, fileUrl, num, q, useFetch, type Entry, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Markdown } from './Reports'
import { Help } from '@/HelpTip'

for (const [n, l] of Object.entries({ python, bash, json, markdown })) SyntaxHighlighter.registerLanguage(n, l)

const ext = (p: string) => p.split('.').pop()!.toLowerCase()
const parent = (p: string) => p.split('/').slice(0, -1).join('/')
const BIN = ['npz', 'npy', 'pt']
const CODE_ROOTS = ['scripts', 'src/spatial', 'tests', 'AGENTS.md']

/** directory browser over the server's allow-listed roots */
function Browser({ roots, selected, onSelect }: { roots: (p: string) => boolean; selected?: string; onSelect: (p: string) => void }) {
  const [dir, setDir] = useState(selected ? parent(selected) : '')
  const [filter, setFilter] = useState('')
  const { data: top } = useFetch<Entry[]>('/api/ls')
  const { data: list, error } = useFetch<Entry[]>(`/api/ls?path=${q(dir)}`, 30_000)
  const rootPaths = (top ?? []).map((e) => e.path)
  useEffect(() => { if (selected && parent(selected) !== dir && !rootPaths.includes(selected)) setDir(parent(selected)) }, [selected]) // eslint-disable-line
  const items = (dir ? list : top?.filter((e) => roots(e.path))) ?? []
  const shown = items.filter((e) => e.name.toLowerCase().includes(filter.toLowerCase()))
  return (
    <aside className="flex w-80 shrink-0 flex-col border-r">
      <div className="space-y-2 p-2">
        <div className="flex items-center gap-2 text-xs">
          <button disabled={!dir} className="rounded border px-2 py-0.5 disabled:opacity-40"
            onClick={() => { setDir(rootPaths.includes(dir) ? '' : parent(dir)); setFilter('') }}>↑ 상위</button>
          <span className="truncate font-mono" title={dir}>{dir || '(허용된 루트)'}</span><Help k="glossary.ui.hidden_paths" />
        </div>
        <Input placeholder={`필터… (${items.length}개)`} value={filter} onChange={(e) => setFilter(e.target.value)} />
        {error && <div className="text-xs text-red-600">{error}</div>}
      </div>
      <ScrollArea className="min-h-0 flex-1">
        {shown.map((e) => (
          <button key={e.path} title={e.path}
            onClick={() => (e.dir ? (setDir(e.path), setFilter('')) : onSelect(e.path))}
            className={cn('flex w-full gap-2 px-3 py-0.5 text-left text-sm hover:bg-muted', e.path === selected && 'bg-muted font-medium')}>
            <span className="flex-1 truncate">{e.dir ? '📁 ' : ''}{e.name}</span>
            {!e.dir && <span className="shrink-0 text-xs text-muted-foreground">{bytes(e.size)}</span>}
          </button>
        ))}
      </ScrollArea>
    </aside>
  )
}

export function RawData({ path, nav }: { path?: string; nav: Nav }) {
  return (
    <div className="flex h-full">
      <Browser roots={(p) => !CODE_ROOTS.includes(p)} selected={path} onSelect={(p) => nav('raw', p)} />
      <div className="min-w-0 flex-1 overflow-auto p-4">{path ? <FileView path={path} nav={nav} /> : <p className="text-muted-foreground">왼쪽에서 파일을 선택하세요.</p>}</div>
    </div>
  )
}

export function Code({ path, nav }: { path?: string; nav: Nav }) {
  const { data, error } = useFetch<string>(path ? fileUrl(path) : null)
  return (
    <div className="flex h-full">
      <Browser roots={(p) => CODE_ROOTS.includes(p)} selected={path} onSelect={(p) => nav('code', p)} />
      <div className="min-w-0 flex-1 overflow-auto">
        <div className="sticky top-0 z-10 border-b bg-background px-4 py-2 font-mono text-xs">{path ?? '파일을 선택하세요 (읽기 전용)'}<Help k="glossary.ui.code" /></div>
        {error && <p className="p-4 text-red-600">{error}</p>}
        {path && data != null && (
          <SyntaxHighlighter language={{ py: 'python', sh: 'bash', json: 'json', md: 'markdown' }[ext(path)] ?? 'text'}
            style={oneLight} showLineNumbers customStyle={{ margin: 0, fontSize: 12, background: 'transparent' }}>
            {data}
          </SyntaxHighlighter>
        )}
      </div>
    </div>
  )
}

function FileView({ path, nav }: { path: string; nav: Nav }) {
  const e = ext(path)
  const kind = BIN.includes(e) ? 'file_binary' : e === 'csv' || e === 'tsv' ? 'file_csv' : e === 'json' ? 'file_json' : ['md', 'png', 'jpg', 'jpeg', 'gif', 'webp'].includes(e) ? 'file_view' : 'file_tail'
  const head = <div className="mb-3 flex items-center gap-2 font-mono text-xs text-muted-foreground">{path} <a className="underline" href={fileUrl(path)} target="_blank" rel="noreferrer">원본</a><Help k={`glossary.ui.${kind}`} /></div>
  if (BIN.includes(e)) return <div>{head}<p>바이너리 파일 (.{e}) — 크기만 표시하며 내용은 제공하지 않습니다.</p></div>
  if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(e)) return <div>{head}<img src={fileUrl(path)} alt={path} className="max-w-full" /></div>
  if (e === 'csv' || e === 'tsv') return <div>{head}<CsvView path={path} /></div>
  if (e === 'json') return <div>{head}<JsonFile path={path} /></div>
  if (e === 'md') return <div>{head}<MdFile path={path} nav={nav} /></div>
  return <div>{head}<TailView path={path} /></div>
}

function MdFile({ path, nav }: { path: string; nav: Nav }) {
  const { data } = useFetch<string>(fileUrl(path))
  return data == null ? null : <Markdown path={path} text={data} nav={nav} />
}

function TailView({ path }: { path: string }) {
  const [n, setN] = useState(500)
  const { data, error, at } = useFetch<string>(`/api/file?path=${q(path)}&tail=${n}`, 15_000)
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
        마지막 {n}줄 · 15초마다 갱신 ({at?.toLocaleTimeString()})
        {[200, 500, 2000, 10000].map((k) => <button key={k} className={cn('rounded border px-2', k === n && 'bg-muted')} onClick={() => setN(k)}>{k}</button>)}
      </div>
      {error && <p className="text-red-600">{error}</p>}
      <pre className="overflow-x-auto rounded bg-muted p-3 text-xs leading-relaxed">{data}</pre>
    </div>
  )
}

function JsonFile({ path }: { path: string }) {
  const { data, error } = useFetch<any>(fileUrl(path), 30_000)
  if (error) return <p className="text-red-600">{error}</p>
  return data === null ? null : <div className="font-mono text-xs"><JsonNode v={data} depth={0} /></div>
}

function JsonNode({ k, v, depth }: { k?: string; v: any; depth: number }) {
  const label = k !== undefined && <span className="text-[#2a78d6]">{k}: </span>
  if (v === null || typeof v !== 'object') {
    return <div className="pl-4">{label}<span className={typeof v === 'string' ? 'text-emerald-700' : 'text-amber-800'}>{JSON.stringify(v)}</span></div>
  }
  const entries = Object.entries(v)
  if (Array.isArray(v) && entries.length <= 12 && v.every((x) => x === null || typeof x !== 'object')) {
    return <div className="pl-4">{label}{JSON.stringify(v)}</div>
  }
  return (
    <details open={depth < 1} className="pl-4">
      <summary className="cursor-pointer">{label}<span className="text-muted-foreground">{Array.isArray(v) ? `[${entries.length}]` : `{${entries.length}}`}</span></summary>
      {entries.map(([ck, cv]) => <JsonNode key={ck} k={ck} v={cv} depth={depth + 1} />)}
    </details>
  )
}

type CsvData = { columns: string[]; rows: string[][]; total: number; offset: number; summary: any[] }
const PAGE = 2000

function CsvView({ path }: { path: string }) {
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState<{ i: number; dir: 1 | -1 } | null>(null)
  useEffect(() => { setOffset(0); setSort(null) }, [path])
  const { data, error } = useFetch<CsvData>(`/api/csv?path=${q(path)}&offset=${offset}&limit=${PAGE}`, 30_000)
  const rows = useMemo(() => {
    if (!data || !sort) return data?.rows ?? []
    const { i, dir } = sort
    return [...data.rows].sort((a, b) => {
      const x = Number(a[i]), y = Number(b[i])
      const c = a[i] !== '' && b[i] !== '' && !isNaN(x) && !isNaN(y) ? x - y : (a[i] ?? '').localeCompare(b[i] ?? '')
      return c * dir
    })
  }, [data, sort])
  if (error) return <p className="text-red-600">{error}</p>
  if (!data) return null
  return (
    <div className="space-y-3">
      <details>
        <summary className="cursor-pointer text-sm font-medium">열 요약 ({data.columns.length}개 열, 전체 {data.total}행 기준)</summary>
        <Table>
          <TableHeader><TableRow>{['column', 'n', 'mean', 'min', 'max', 'unique'].map((h) => <TableHead key={h}>{h}</TableHead>)}</TableRow></TableHeader>
          <TableBody>
            {data.summary.map((s) => (
              <TableRow key={s.col}>
                <TableCell className="font-mono">{s.col}</TableCell><TableCell>{s.n}</TableCell>
                <TableCell>{num(s.mean)}</TableCell><TableCell>{num(s.min)}</TableCell><TableCell>{num(s.max)}</TableCell><TableCell>{s.unique ?? ''}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </details>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        행 {data.offset + 1}–{data.offset + data.rows.length} / {data.total} · 열 머리글 클릭 = 정렬 (현재 페이지)
        <button disabled={offset === 0} className="rounded border px-2 disabled:opacity-40" onClick={() => setOffset(Math.max(0, offset - PAGE))}>이전</button>
        <button disabled={offset + PAGE >= data.total} className="rounded border px-2 disabled:opacity-40" onClick={() => setOffset(offset + PAGE)}>다음</button>
      </div>
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            {data.columns.map((c, i) => (
              <TableHead key={i} className="cursor-pointer select-none" onClick={() => setSort({ i, dir: sort?.i === i && sort.dir === 1 ? -1 : 1 })}>
                {c}{sort?.i === i ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, j) => (
            <TableRow key={j}>{r.map((v, i) => <TableCell key={i} className="max-w-64 truncate py-1 tabular-nums" title={v}>{/^-?\d+\.\d{5,}/.test(v) ? num(v) : v}</TableCell>)}</TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
