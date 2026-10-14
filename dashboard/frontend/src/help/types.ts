export type HelpEntry = { title: string; body: string } // body = Korean markdown (GFM tables allowed)
export type HelpMap = Record<string, HelpEntry>
