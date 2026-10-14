import { useEffect, useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Overview from './Overview'
import Reports, { Figures } from './Reports'
import { Code, Compare, RawData } from './Explorer'

const TABS = [
  ['overview', '개요'], ['reports', '리포트'], ['figures', '그림'], ['raw', '원시 데이터'], ['compare', 'Unit 비교'], ['code', '코드'],
] as const

// location hash = "#tab" or "#tab:repo/relative/path" (bookmarkable, back button works)
const parse = () => {
  const [tab, ...rest] = decodeURIComponent(location.hash.slice(1)).split(':')
  return { tab: tab || 'overview', path: rest.join(':') || undefined }
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
  const nav = (tab: string, path?: string) => { location.hash = path ? `${tab}:${path}` : tab }
  const path = loc.path ?? last[loc.tab]
  return (
    <Tabs value={loc.tab} onValueChange={(t) => nav(String(t), last[String(t)])} className="flex h-screen flex-col gap-0">
      <header className="flex items-center gap-4 border-b px-4 py-2">
        <h1 className="text-sm font-semibold whitespace-nowrap">SAM 2 연구 대시보드 <span className="font-normal text-muted-foreground">M7 · M8</span></h1>
        <TabsList>{TABS.map(([v, l]) => <TabsTrigger key={v} value={v}>{l}</TabsTrigger>)}</TabsList>
      </header>
      <main className="min-h-0 flex-1 overflow-auto">
        <TabsContent value="overview"><Overview /></TabsContent>
        <TabsContent value="reports" className="h-full"><Reports path={path} nav={nav} /></TabsContent>
        <TabsContent value="figures" className="h-full"><Figures /></TabsContent>
        <TabsContent value="raw" className="h-full"><RawData path={path} nav={nav} /></TabsContent>
        <TabsContent value="compare"><Compare /></TabsContent>
        <TabsContent value="code" className="h-full"><Code path={path} nav={nav} /></TabsContent>
      </main>
    </Tabs>
  )
}
