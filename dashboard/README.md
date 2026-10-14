# Research site (local, read-only)

```bash
cd /home/sojak/sam2/dashboard/frontend && npm install && npm run build   # -> frontend/dist
cd /home/sojak/sam2 && uv run python dashboard/server.py                 # http://127.0.0.1:8765
uv run python dashboard/test_server.py                                   # path-guard + token-auth self-check
```

- Running in tmux: `m8:dashboard`. Restart after a rebuild: `tmux respawn-window -k -t m8:dashboard`.
- Frontend dev with hot reload: `npm run dev` (proxies `/api` to :8765).

## Access from another machine

Default bind is `127.0.0.1` (no token). Two ways in from elsewhere:

**SSH tunnel (preferred; nothing listens on the LAN):**

```bash
ssh -N -L 8765:127.0.0.1:8765 sojak@<this-host>     # then open http://127.0.0.1:8765 locally
```

**LAN bind with a token** (`DASH_HOST` non-loopback ⇒ `DASH_TOKEN` of ≥ 16 chars is required, otherwise the server refuses to start):

```bash
cd /home/sojak/sam2
export DASH_TOKEN=$(openssl rand -hex 16); echo "$DASH_TOKEN"
DASH_HOST=192.168.1.21 DASH_TOKEN=$DASH_TOKEN uv run python dashboard/server.py
# browser, once:  http://192.168.1.21:8765/?token=<DASH_TOKEN>   -> sets an HttpOnly SameSite=Strict cookie (30 days) and redirects
# scripts:        curl -H "Authorization: Bearer $DASH_TOKEN" http://192.168.1.21:8765/api/overview
```

With a token set, every request (pages, API, images) needs the cookie, the `Bearer` header or `?token=`; otherwise 401.
Tokens are compared with `hmac.compare_digest`. Plain HTTP: the token is visible to anyone sniffing the LAN, so prefer the tunnel on untrusted networks.
Without a token, requests whose `Host` header is not `127.0.0.1`/`localhost`/`[::1]` get 403 (DNS-rebinding guard).

## What is served

- `server.py` (stdlib only) serves only the allow-listed repo paths (`ALLOW`); never `.env*`, `.ssh`, `tcache/`, `checkpoints/`, `data/`, or `.npz/.npy/.pt` (those are listed with their size only).
- Aggregated endpoints (parsed files cached by mtime): `/api/overview`, `/api/tasks` (TASKS.json with live glob progress + newest log tails), `/api/arms/summary`, `/api/units`, `/api/unit/<stem>` (all arms' frame series + machine-matched SGGL baselines + stage-1 candidates), `/api/stage1`, `/api/m7`, `/api/search?q=`; generic `/api/ls`, `/api/csv`, `/api/file`, `/api/docs`, `/api/images`.
- Pages: 개요 (tasks, status, causal forest plot) · Causal arms · Stage 1 · Unit 탐색 · M7 · 그림 · 리포트 · 원시 데이터 · 코드, plus a global file-name search in the header.

**Public link (Cloudflare Tunnel `dash`, user setup 2026-10-14):** https://research.sojak.io → `http://192.168.1.21:8765`. The config is `/etc/cloudflared/config.yml` (systemd `cloudflared`). The same token is required: open `/?token=<dashboard/.token>` once. Credentials (`/etc/cloudflared/<tunnel-id>.json`, `~/.cloudflared/cert.pem`) are secrets; only their paths are documented, in `docs/GPU_WORKER_OPERATIONS.md`.
