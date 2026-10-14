import { useEffect, useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { q, useFetch, type Nav } from '@/lib/api'
import Overview from './Overview'
import Arms from './Arms'
import Stage1 from './Stage1'
import Units from './Units'
import M7 from './M7'
import M9 from './M9'
import Reports, { Figures } from './Reports'
import { Code, RawData } from './Explorer'
import Glossary from './Glossary'
import { Help } from '@/HelpTip'

const TABS = [
  ['overview', '개요'], ['m9', 'M9 교차 검증'], ['arms', 'Causal arms'], ['stage1', 'Stage 1'], ['units', 'Unit 탐색'], ['m7', 'M7'],
  ['figures', '그림'], ['reports', '리포트'], ['raw', '원시 데이터'], ['code', '코드'], ['glossary', '용어집'],
] as const

// location hash = "#tab" or "#tab:<path|arm|unit>" (bookmarkable, back button works)
const parse = () => {
  const [tab, ...rest] = decodeURIComponent(location.hash.slice(1)).split(':')
  return { tab: tab || 'overview', path: rest.join(':') || undefined }
}

export const tabFor = (path: string) => (path.endsWith('.md') ? 'reports' : /\.(py|sh)$/.test(path) || /^(scripts|src|tests)\//.test(path) ? 'code' : 'raw')

function Search({ nav }: { nav: Nav }) {
  const [text, setText] = useState('')
  const [term, setTerm] = useState('')
  const [open, setOpen] = useState(false)
  useEffect(() => { const t = setTimeout(() => setTerm(text.trim()), 200); return () => clearTimeout(t) }, [text])
  const { data } = useFetch<{ total: number; hits: string[] }>(term.length >= 2 ? `/api/search?q=${q(term)}` : null)
  return (
    <div className="relative w-72 shrink-0" onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setOpen(false)}>
      <Input placeholder="문서·파일 이름 검색…" value={text} onFocus={() => setOpen(true)}
        onChange={(e) => { setText(e.target.value); setOpen(true) }} onKeyDown={(e) => e.key === 'Escape' && setOpen(false)} />
      {open && term.length >= 2 && data && (
        <div className="absolute right-0 z-50 mt-1 max-h-[70vh] w-[36rem] overflow-auto rounded-md border bg-background p-1 shadow-lg">
          <div className="px-2 py-1 text-xs text-muted-foreground">{data.total}개 일치{data.total > data.hits.length ? ` (상위 ${data.hits.length}개)` : ''}</div>
          {data.hits.map((h) => (
            <button key={h} className="block w-full truncate rounded px-2 py-1 text-left font-mono text-xs hover:bg-muted"
              onClick={() => { nav(tabFor(h), h); setOpen(false) }}>
              <span className="mr-2 text-muted-foreground">{tabFor(h) === 'reports' ? '문서' : tabFor(h) === 'code' ? '코드' : '데이터'}</span>{h}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [loc, setLoc] = useState(parse)
  const [last, setLast] = useState<Record<string, string | undefined>>({})
  useEffect(() => {
    const h = () => setLoc(parse())
    addEventListener('hashchange', h)
    return () => removeEventListener('hashchange', h)
  }, [])
  useEffect(() => { if (loc.path) setLast((l) => ({ ...l, [loc.tab]: loc.path })) }, [loc])
  const nav: Nav = (tab, path) => { location.hash = path ? `${tab}:${path}` : tab }
  const path = loc.path ?? last[loc.tab]
  return (
    <Tabs value={loc.tab} onValueChange={(t) => nav(String(t), last[String(t)])} className="flex h-screen flex-col gap-0">
      <header className="flex items-center gap-4 border-b px-4 py-2">
        <h1 className="text-sm font-semibold whitespace-nowrap">SAM 2 연구 사이트 <span className="font-normal text-muted-foreground">M7 · M8 · M9</span></h1>
        <div className="min-w-0 flex-1 overflow-x-auto">
          <TabsList>{TABS.map(([v, l]) => <TabsTrigger key={v} value={v}>{l}</TabsTrigger>)}</TabsList>
        </div>
        <div className="flex shrink-0 items-center"><Search nav={nav} /><Help k="glossary.ui.search" /></div>
      </header>
      <main className="min-h-0 flex-1 overflow-auto">
        <TabsContent value="overview"><Overview nav={nav} /></TabsContent>
        <TabsContent value="m9"><M9 nav={nav} /></TabsContent>
        <TabsContent value="arms"><Arms sel={path} nav={nav} /></TabsContent>
        <TabsContent value="stage1"><Stage1 /></TabsContent>
        <TabsContent value="units"><Units stem={path} nav={nav} /></TabsContent>
        <TabsContent value="m7"><M7 nav={nav} /></TabsContent>
        <TabsContent value="figures" className="h-full"><Figures /></TabsContent>
        <TabsContent value="reports" className="h-full"><Reports path={path} nav={nav} /></TabsContent>
        <TabsContent value="raw" className="h-full"><RawData path={path} nav={nav} /></TabsContent>
        <TabsContent value="code" className="h-full"><Code path={path} nav={nav} /></TabsContent>
        <TabsContent value="glossary"><Glossary /></TabsContent>
      </main>
    </Tabs>
  )
}
