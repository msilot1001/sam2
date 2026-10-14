// "?" help button: opens a dialog with a detailed markdown explanation from src/help/*.ts (key = "<page>.<element>").
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { HELP } from "@/help"

export function Help({ k, className = "" }: { k: string; className?: string }) {
  const h = HELP[k]
  if (!h) return null
  return (
    <Dialog>
      <DialogTrigger
        className={"inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-muted-foreground/40 text-[11px] font-semibold text-muted-foreground hover:bg-muted hover:text-foreground align-middle ml-1 " + className}
        aria-label={`설명: ${h.title}`}
        title={h.title}
      >
        ?
      </DialogTrigger>
      <DialogContent className="sm:max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{h.title}</DialogTitle>
        </DialogHeader>
        <div className="prose prose-sm max-w-none dark:prose-invert [&_table]:text-xs [&_td]:border [&_th]:border [&_td]:px-2 [&_th]:px-2">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{h.body}</ReactMarkdown>
        </div>
      </DialogContent>
    </Dialog>
  )
}
