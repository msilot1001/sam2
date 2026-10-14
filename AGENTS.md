# Agent operating rules for this repository

## 1. Ubuntu is the only source of truth for research results (highest priority)

The local Ubuntu server (`/home/sojak/sam2`) is the ONLY official store of research results. Cloud GPU instances
(RunPod, Vast.ai, ...) are compute workers only; their disks and network volumes are never a permanent store.

1. **Keep on Ubuntu permanently:**
   - Results: CSV, JSON and NPZ files, evaluation results, graphs.
   - Trained weights and checkpoints, training logs.
   - Pre-registrations, experiment configs, seeds.
   - Run provenance: the git commit and the environment info.
   - Reports and everything else needed to reproduce the research.
2. **Workers** run SAM 2 inference, training and other GPU jobs. Temporary files and caches may live on the worker. Results must be pulled to Ubuntu as soon as they are complete. During long training, checkpoints are synced periodically.
3. **Sync** with `scripts/gpu_sync.py`, run on Ubuntu (it pulls; the worker never pushes).
   - Every GPU workflow includes a pull step: `gpu_sync.py loop <sec> <dir>` in an Ubuntu tmux window.
   - Each file is verified by size and sha256 against a manifest made on the worker, and logged in `outputs/SYNC_LEDGER.tsv`.
   - Failures are retried and recorded in `outputs/SYNC_STATUS.json`. A failed sync is never reported as a success.
   - Existing Ubuntu files are never overwritten. Different content is kept as `*.conflict-<ts>` and reported.
   - Worker outputs go to machine-specific experiment directories, e.g. `outputs/milestone8/runpod/...`.
4. **Before stopping or deleting a worker:** run `gpu_sync.py audit <dirs>`; it must exit 0, meaning every worker file is verified on Ubuntu.
   - Never delete worker results that are not listed in the worker's `<dir>/.ubuntu_verified.tsv`.
   - The research must be continuable from Ubuntu files alone.
5. **If Ubuntu is unreachable**, keep results on the worker and retry. Never claim data is saved before the ledger says so.
6. **Role split:** GPU compute on the worker; analysis, reports, commits and permanent storage on Ubuntu. Long jobs run inside `tmux` sessions on both machines.

Details: `docs/GPU_WORKER_OPERATIONS.md`.

## 2. Research hygiene (carried over from M5–M8)

- PUMaVOS C is sealed: never copy it to a worker, never inspect its masks, never tune on it.
- VKITTI Scene02/18/20 are not used.
- Compare only against a baseline from the same machine; GPUs differ slightly in floating point.
- Pre-register before any output, never overwrite frozen files, and disclose amendments.
- Never print or commit secrets: SSH private keys, API keys, tokens.
- Never merge experiment branches into master automatically.
