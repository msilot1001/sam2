import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Input } from '@/components/ui/input'
import { Help } from '@/HelpTip'
import { help } from '@/help/glossary'

const GROUPS: Record<string, string> = {
  sam2: 'SAM 2 기초', sggl: 'SGGL', metric: '지표·통계', data: '데이터', stage1: 'Stage 1 진단',
  causal: '인과 실행 (Stage 2/3)', family: '후보 계열', verdict: '판정·사전 등록', ops: '운영',
}
const ENTRIES = Object.entries(help).filter(([k]) => !k.startsWith('glossary.ui.'))

/** searchable list of every glossary.* help entry (same source as the "?" popovers) */
export default function Glossary() {
  const [text, setText] = useState('')
  const terms = text.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = ENTRIES.filter(([, h]) => { const s = (h.title + '\n' + h.body).toLowerCase(); return terms.every((t) => s.includes(t)) })
  const groups = Object.groupBy(shown, ([k]) => k.split('.')[1])
  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div className="flex items-center gap-2">
        <Input className="max-w-md" placeholder={`용어 검색… (${ENTRIES.length}개 항목)`} value={text} onChange={(e) => setText(e.target.value)} />
        <Help k="glossary.ui.glossary" />
        <span className="text-xs text-muted-foreground">{shown.length}개 표시</span>
      </div>
      <nav className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
        {shown.map(([k, h]) => <a key={k} href={`#glossary`} className="text-muted-foreground hover:text-foreground hover:underline"
          onClick={(e) => { e.preventDefault(); document.getElementById(k)?.scrollIntoView({ behavior: 'smooth' }) }}>{h.title}</a>)}
      </nav>
      {Object.entries(groups).map(([g, items]) => (
        <section key={g} className="space-y-4">
          <h2 className="border-b pb-1 text-base font-semibold">{GROUPS[g] ?? g}</h2>
          {items!.map(([k, h]) => (
            <article key={k} id={k} className="scroll-mt-4 rounded-lg border p-4">
              <h3 className="mb-2 font-semibold">{h.title}</h3>
              <div className="prose prose-sm max-w-none dark:prose-invert [&_table]:text-xs [&_td]:border [&_th]:border [&_td]:px-2 [&_th]:px-2">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{h.body}</ReactMarkdown>
              </div>
            </article>
          ))}
        </section>
      ))}
      {!shown.length && <p className="text-muted-foreground">일치하는 용어가 없습니다.</p>}
    </div>
  )
}
