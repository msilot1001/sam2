# GPU worker operations (RunPod / Vast.ai)

Policy summary: see [`../AGENTS.md`](../AGENTS.md) §1. Ubuntu is the only source of truth; workers are disposable.

## Current worker (2026-10-14)

| Item | Value |
|---|---|
| Provider / GPU | RunPod, RTX 4090 24 GB, 48 vCPU visible, RAM 251 GB, DC EU-RO-1 |
| SSH | Direct TCP. Address in `.gpu_worker.env` (gitignored). The `ssh.runpod.io` gateway is PTY-only, so it cannot carry tar streams |
| Repo / venv | `/workspace/sam2-m7-test`, `/workspace/.venvs/m7` (torch 2.11.0+cu128, sam2 @ 2b90b9f with `_C`) |
| Volume | `/workspace` is a RunPod network volume (MooseFS). Cold imports are slow (`import torch` ≈ 24 s). Untar needs `--no-same-owner` |
| Setup script | `scripts/ops/setup_runpod_worker.sh` (on the pod: `/workspace/setup_m8_pod.sh`). It is idempotent and resumes the original `setup_m7_runpod.sh` from step 4. It pins the CUDA torchvision build (a dependency replaced it with the PyPI build, which lacks `torchvision::nms`) |
| Run environment | `scripts/ops/worker_env.sh` → `/workspace/m8env.sh`. Jobs run in `/root/w`: data is copied to the container disk (`/workspace/localize.sh`; 1.5× faster than reading from the volume). Code, outputs and checkpoints are symlinks to `/workspace/sam2-m7-test`, so outputs always land on the volume |
| Job wrapper | `scripts/ops/worker_run.sh <log> <cmd…>` → `/workspace/podrun.sh`, run inside `tmux` session `m8` on the pod |
| Ubuntu sync | tmux session `m8`, window `sync`: `gpu_sync.py loop 300 outputs/milestone8/runpod` (log: `outputs/milestone8/logs/gpu_sync.log`) |

## Verified on 2026-10-14 (sync self-test with small files, `outputs/milestone8/runpod/_synctest`)

| Case | Result |
|---|---|
| 2 new files (text + 300 kB random) | Pulled. sha256 identical on both sides. Ledger rows `NEW` |
| Same remote path with changed content | The Ubuntu copy is kept. The remote version is stored as `hello.txt.conflict-<ts>` (`CONFLICT`) |
| `audit` with unsynced remote files / after the pull | exit 1 with the files listed / exit 0 |
| Worker unreachable | 3 attempts with backoff, then `GAVE UP … (worker copy untouched)`. The failure is recorded in `SYNC_STATUS.json`; no success is claimed |

## Procedure

1. **Code:** `uv run python scripts/gpu_sync.py push-code`.
   - Sends `src scripts tests configs pyproject.toml` plus the M8 pre-registration and arm JSONs.
   - Writes `CODE_SYNC.json` on the worker, recording the Ubuntu HEAD and the number of uncommitted paths.
2. **Data:** send only what the experiment needs.
   - DAVIS 2017: downloaded on the worker, then md5-checked against Ubuntu.
   - PUMaVOS: A/B clips only, sent with `tar -h`. **C is never sent.**
   - SegTrack: the 14 clips that are used.
   - Verification: an md5 list from each side, compared with `LC_ALL=C sort -k2`.
3. **Runs:** launch each job in a worker tmux window, writing to `outputs/milestone8/runpod/<experiment>/`.
4. **Results:** an Ubuntu tmux window runs `gpu_sync.py loop 300 outputs/milestone8/runpod`. It writes to the ledger and the status file.
5. **Shutdown:** `gpu_sync.py audit outputs/milestone8/runpod` must exit 0 before the pod is stopped or deleted.

## Research site (dashboard)

- tmux `m8:dashboard`, `http://192.168.1.21:8765` (internal network, user decision 2026-10-14). Token auth: `dashboard/.token` (mode 600, gitignored); open `/?token=<token>` once.
- Default bind without `DASH_HOST` is 127.0.0.1; a non-loopback bind refuses to start without `DASH_TOKEN`. See `dashboard/README.md`.
- Current launch (tmux `m8:dashboard`): `DASH_HOST=192.168.1.21 DASH_TOKEN=$(cat dashboard/.token) uv run python dashboard/server.py`.

### External access: Cloudflare Tunnel (set up by the user, 2026-10-14)

| Item | Value |
|---|---|
| Public URL | https://research.sojak.io (the same token auth applies: open `/?token=<token>` once; without it → 401, verified) |
| Tunnel | `dash`, ID `8f2e9b77-a7bc-4980-91a8-3770f33dc924` |
| Config | `/etc/cloudflared/config.yml` (copied from `~/.cloudflared/config.yml`) |
| Ingress | `research.sojak.io` → `http://192.168.1.21:8765`; everything else → `http_status:404`. The dashboard listens on 192.168.1.21 only, so `127.0.0.1:8765` would NOT work as an ingress target |
| Service | systemd `cloudflared` (installed with `sudo cloudflared service install`); `systemctl status cloudflared` |
| Secrets (paths only, never copy contents into docs or git) | tunnel credentials `/etc/cloudflared/8f2e9b77-a7bc-4980-91a8-3770f33dc924.json` (root, mode 400), origin cert `~/.cloudflared/cert.pem`, dashboard token `dashboard/.token` |

If the dashboard is restarted with a different `DASH_HOST` or port, update the ingress `service:` line and restart: `sudo systemctl restart cloudflared`.
