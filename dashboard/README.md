# Research dashboard (local, read-only)

```bash
cd /home/sojak/sam2/dashboard/frontend && npm install && npm run build   # -> frontend/dist
cd /home/sojak/sam2 && uv run python dashboard/server.py                 # http://127.0.0.1:8765
uv run python dashboard/test_server.py                                   # path-guard self-check
```

- `server.py`: stdlib only, binds 127.0.0.1, serves only the allow-listed repo paths (see `ALLOW`); never serves `.env*`, `tcache/`, `checkpoints/`, `data/`, or `.npz/.npy/.pt` (those are listed with their size only).
- Running in tmux: `m8:dashboard`. To restart after a rebuild: `tmux respawn-window -k -t m8:dashboard`.
- Frontend dev with hot reload: `npm run dev` (proxies `/api` to :8765).
