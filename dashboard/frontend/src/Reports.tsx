import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { bytes, fileUrl, useFetch, type Entry, type Nav } from '@/lib/api'
import { cn } from '@/lib/utils'

/** resolve a relative link inside a doc to a repo-relative path */
export const resolveRel = (base: string, href: string) =>
  decodeURIComponent(new URL(href, 'http://x/' + base).pathname.slice(1))

export function Markdown({ path, text, nav }: { path: string; text: string; nav: Nav }) {
  return (
    <article className="prose prose-sm max-w-none prose-table:text-xs prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href = '', children }) => {
            if (/^[a-z]+:|^#/i.test(href)) return <a href={href} target="_blank" rel="noreferrer">{children}</a>
            const target = resolveRel(path, href.split('#')[0])
            const tab = target.endsWith('.md') ? 'reports' : /\.(py|sh)$/.test(target) ? 'code' : 'raw'
            return <a href="#" onClick={(e) => { e.preventDefault(); nav(tab, target) }}>{children}</a>
          },
          img: ({ src = '', alt }) => (
            <img alt={alt} src={/^[a-z]+:/i.test(String(src)) ? String(src) : fileUrl(resolveRel(path, String(src)))} />
          ),
          table: ({ children }) => <div className="overflow-x-auto"><table>{children}</table></div>,
        }}
      >
        {text}
      </ReactMarkdown>
    </article>
  )
}

export default function Reports({ path, nav }: { path?: string; nav: Nav }) {
  const { data: docs } = useFetch<string[]>('/api/docs')
  const [filter, setFilter] = useState('')
  const current = path ?? docs?.[0]
  const { data: text, error } = useFetch<string>(current ? fileUrl(current) : null)
  const shown = (docs ?? []).filter((d) => d.toLowerCase().includes(filter.toLowerCase()))
  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r">
        <div className="p-2"><Input placeholder="문서 검색…" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>
        <ScrollArea className="min-h-0 flex-1">
          {shown.map((d, i) => {
            const group = d.split('/').slice(0, -1).join('/') || '(root)'
            const prev = i ? shown[i - 1].split('/').slice(0, -1).join('/') || '(root)' : null
            return (
              <div key={d}>
                {group !== prev && <div className="px-3 pt-3 pb-1 text-xs font-semibold text-muted-foreground">{group}</div>}
                <button onClick={() => nav('reports', d)}
                  className={cn('block w-full truncate px-3 py-1 text-left text-sm hover:bg-muted', d === current && 'bg-muted font-medium')}>
                  {d.split('/').pop()}
                </button>
              </div>
            )
          })}
        </ScrollArea>
      </aside>
      <ScrollArea className="min-w-0 flex-1">
        <div className="mx-auto max-w-5xl p-6">
          <div className="mb-3 font-mono text-xs text-muted-foreground">{current}</div>
          {error ? <p className="text-red-600">{error}</p> : current && text != null && <Markdown path={current} text={text} nav={nav} />}
        </div>
      </ScrollArea>
    </div>
  )
}

export function Figures() {
  const { data } = useFetch<Entry[]>('/api/images', 30_000)
  const [open, setOpen] = useState<Entry | null>(null)
  const groups = Object.groupBy(data ?? [], (e) => e.path.split('/').slice(0, -1).join('/'))
  return (
    <ScrollArea className="h-full">
      <div className="space-y-6 p-4">
        {Object.entries(groups).map(([dir, imgs]) => (
          <section key={dir}>
            <h2 className="mb-2 font-mono text-sm text-muted-foreground">{dir} ({imgs!.length})</h2>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
              {imgs!.map((e) => (
                <button key={e.path} onClick={() => setOpen(e)} className="overflow-hidden rounded-lg border text-left hover:ring-2 hover:ring-ring">
                  <img src={fileUrl(e.path)} alt={e.name} loading="lazy" className="aspect-video w-full bg-muted object-contain" />
                  <div className="truncate p-2 text-xs">{e.name} <span className="text-muted-foreground">· {bytes(e.size)}</span></div>
                </button>
              ))}
            </div>
          </section>
        ))}
        {data?.length === 0 && <p className="text-muted-foreground">이미지 없음</p>}
      </div>
      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-h-[95vh] overflow-auto sm:max-w-[95vw]">
          <DialogHeader><DialogTitle className="font-mono text-sm">{open?.path}</DialogTitle></DialogHeader>
          {open && <img src={fileUrl(open.path)} alt={open.name} className="mx-auto max-w-full" />}
        </DialogContent>
      </Dialog>
    </ScrollArea>
  )
}
