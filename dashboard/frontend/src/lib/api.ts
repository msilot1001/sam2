import { useEffect, useState } from 'react'

export async function get<T = any>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`)
  return (r.headers.get('content-type') || '').includes('json') ? r.json() : (r.text() as any)
}

export const q = (path: string) => encodeURIComponent(path)
export const fileUrl = (path: string) => `/api/file?path=${q(path)}`

/** fetch url, refetch every `everyMs` (0 = once) or when url changes */
export function useFetch<T = any>(url: string | null, everyMs = 0) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [at, setAt] = useState<Date | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (!url) return
    let live = true
    const load = () =>
      get<T>(url)
        .then((d) => live && (setData(d), setError(null), setAt(new Date())))
        .catch((e) => live && setError(String(e)))
    load()
    const id = everyMs ? setInterval(load, everyMs) : undefined
    return () => { live = false; clearInterval(id) }
  }, [url, everyMs, tick])
  return { data, error, at, reload: () => setTick((t) => t + 1) }
}

export function num(v: unknown, d = 4): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  return Number.isInteger(n) ? String(n) : n.toFixed(d)
}

export const signed = (v: number | null | undefined, d = 4) =>
  v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(d)

export function bytes(n: number) {
  const u = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (n >= 1024 && i < 3) { n /= 1024; i++ }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`
}

export type Entry = { name: string; path: string; dir: boolean; size: number; mtime: number; binary: boolean }
export type Nav = (tab: string, path?: string) => void
