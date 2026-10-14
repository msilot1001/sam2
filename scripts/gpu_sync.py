"""Ubuntu-side sync for cloud GPU workers (RunPod / Vast). Ubuntu is the ONLY source of truth for research results.

The worker never pushes; Ubuntu pulls (works behind NAT, and a pull that fails leaves the worker copy untouched).

    uv run python scripts/gpu_sync.py push-code                      # code + tests + prereg/arms + small refs -> worker
    uv run python scripts/gpu_sync.py pull outputs/milestone8/runpod # results worker -> Ubuntu (verified, no overwrite)
    uv run python scripts/gpu_sync.py loop 300 outputs/milestone8/runpod   # pull every 300 s until killed (run in tmux)
    uv run python scripts/gpu_sync.py audit outputs/milestone8/runpod      # is EVERY worker file safely on Ubuntu?

Worker address comes from `.gpu_worker.env` (gitignored): WORKER_HOST, WORKER_PORT, WORKER_KEY, WORKER_REPO.
Pull protocol per directory:
  1. worker prints a manifest (relative path, size, sha256) of every file under the directory;
  2. one tar stream worker -> Ubuntu staging (outputs/_sync_staging/<ts>);
  3. every staged file is checked against the manifest (size + sha256); failures are retried (3x, backoff);
  4. merge into the destination WITHOUT overwriting: identical file -> skip; new file -> move in; different content
     -> kept as <name>.conflict-<ts> and reported (never replaces the Ubuntu copy);
  5. each verified file is appended to outputs/SYNC_LEDGER.tsv; the verified list is written back to the worker as
     <dir>/.ubuntu_verified.tsv (only files listed there may ever be deleted on the worker);
  6. outputs/SYNC_STATUS.json records the last success / failure (a failure is never reported as success).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs" / "SYNC_LEDGER.tsv"
STATUS = ROOT / "outputs" / "SYNC_STATUS.json"
STAGING = ROOT / "outputs" / "_sync_staging"
CODE = ["src", "scripts", "tests", "configs", "pyproject.toml"]


def env():
    e = {}
    for line in (ROOT / ".gpu_worker.env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = os.path.expanduser(v.strip())
    return e


def ssh(e, cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                           "-i", e["WORKER_KEY"], "-p", e["WORKER_PORT"], f"root@{e['WORKER_HOST']}", cmd], **kw)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def status(**kw):
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATUS)


def push_code(extra=""):
    e = env()
    items = CODE + [str(p.relative_to(ROOT)) for p in (ROOT / "outputs/milestone8").glob("*.json")
                    if p.name.startswith(("preregistration", "arms"))] + extra.split()
    items = [i for i in items if i and (ROOT / i).exists()]
    tar = subprocess.Popen(["tar", "cf", "-", "--exclude=__pycache__", *items], cwd=ROOT, stdout=subprocess.PIPE)
    r = ssh(e, f"mkdir -p {e['WORKER_REPO']} && cd {e['WORKER_REPO']} && tar xf - --no-same-owner", stdin=tar.stdout)
    tar.wait()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    stamp = json.dumps({"ubuntu_head": head, "uncommitted_paths": len(dirty.splitlines()),
                        "pushed": time.strftime("%Y-%m-%d %H:%M:%S"), "items": items})
    ssh(e, f"cat > {e['WORKER_REPO']}/CODE_SYNC.json", input=stamp.encode())
    if r.returncode or tar.returncode:
        raise SystemExit("push-code FAILED")
    print("push-code ok:", stamp)


def _manifest(e, rdir):
    # files modified in the last minute are still being written: they are picked up by a later pull
    cmd = (f"cd {e['WORKER_REPO']} && [ -d {rdir} ] && find {rdir} -type f ! -name .ubuntu_verified.tsv -mmin +1 "
           f"-printf '%s\\t%p\\n' | while IFS=$'\\t' read s p; do printf '%s\\t%s\\t%s\\n' \"$p\" \"$s\" "
           f"\"$(sha256sum \"$p\" | cut -d' ' -f1)\"; done")
    r = ssh(e, cmd, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"manifest failed: {r.stderr.strip()}")
    out = {}
    for line in r.stdout.splitlines():
        if line.count("\t") != 2:
            continue
        p, s, h = line.split("\t")
        out[p] = (int(s), h)
    return out


def _verified_local():
    done = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 7 and f[6] in ("NEW", "SAME", "CONFLICT", "APPENDED"):     # CONFLICT: saved as *.conflict-<ts>
                done.add((f[2], f[4]))                         # (remote path, sha256)
    return done


def pull_once(rdir, ldir=None):
    e = env()
    ldir = ldir or rdir
    man = _manifest(e, rdir)
    done = _verified_local()
    todo = {p: v for p, v in man.items() if (p, v[1]) not in done}
    if not todo:
        return 0, 0
    ts = time.strftime("%Y%m%d-%H%M%S")
    stage = STAGING / ts
    stage.mkdir(parents=True, exist_ok=True)
    lst = "\n".join(todo) + "\n"
    with subprocess.Popen(["tar", "xf", "-"], cwd=stage, stdin=subprocess.PIPE) as tx:
        r = ssh(e, f"cd {e['WORKER_REPO']} && tar cf - -T -", input=lst.encode(), stdout=tx.stdin)
        tx.stdin.close()
    if r.returncode:
        raise RuntimeError("tar stream failed")
    new = conf = 0
    rows = []
    if not LEDGER.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text("time\tworker\tremote\tlocal\tsha256\tsize\tstatus\n")
    retry = 0
    for p, (size, h) in todo.items():
        sp = stage / p
        if not sp.exists() or sp.stat().st_size != size or sha(sp) != h:
            retry += 1                                        # changed during transfer: not saved, retried next pull
            print(f"  RETRY (changed during transfer): {p}", flush=True)
            continue
        dst = ROOT / ldir / Path(p).relative_to(rdir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            old = dst.read_bytes()
            if sha(dst) == h:
                st = "SAME"
            elif sp.read_bytes()[:len(old)] == old:           # append-only growth (logs): Ubuntu bytes kept as prefix
                shutil.move(sp, dst)
                st = "APPENDED"
            else:
                dst = dst.with_name(dst.name + f".conflict-{ts}")
                shutil.move(sp, dst)
                st, conf = "CONFLICT", conf + 1
        else:
            shutil.move(sp, dst)
            st, new = "NEW", new + 1
        rows.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{e['WORKER_HOST']}:{e['WORKER_PORT']}\t{p}\t"
                    f"{dst.relative_to(ROOT)}\t{h}\t{size}\t{st}")
    with open(LEDGER, "a") as f:
        f.write("\n".join(rows) + "\n")
    shutil.rmtree(stage, ignore_errors=True)
    ver = "".join(f"{p}\t{h}\n" for p, (s, h) in man.items() if (p, h) in _verified_local())
    ssh(e, f"cat > {e['WORKER_REPO']}/{rdir}/.ubuntu_verified.tsv", input=ver.encode())
    return new, conf


def pull(rdir, ldir=None, tries=3):
    for k in range(tries):
        try:
            n, c = pull_once(rdir, ldir)
            status(**{f"last_ok:{rdir}": time.strftime("%Y-%m-%d %H:%M:%S"), f"last_new:{rdir}": n,
                      f"conflicts:{rdir}": c})
            print(f"pull {rdir}: {n} new, {c} conflicts", flush=True)
            return n
        except Exception as ex:                                   # noqa: BLE001 - report and retry
            print(f"pull {rdir} attempt {k + 1} FAILED: {ex}", flush=True)
            status(**{f"last_fail:{rdir}": f"{time.strftime('%Y-%m-%d %H:%M:%S')} {ex}"})
            time.sleep(30 * (k + 1))
    print(f"pull {rdir}: GAVE UP after {tries} attempts (worker copy untouched)", flush=True)
    return -1


def loop(interval, *dirs):
    while True:
        for d in dirs:
            pull(d)
        time.sleep(int(interval))


def audit(*dirs):
    """Lists worker files NOT yet verified on Ubuntu. Exit code 0 only if every file is safely on Ubuntu."""
    e = env()
    done = _verified_local()
    missing = []
    for d in dirs:
        missing += [p for p, (s, h) in _manifest(e, d).items() if (p, h) not in done]
    print(f"unsynced files: {len(missing)}")
    for p in missing[:50]:
        print("  ", p)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    fn = {"push-code": push_code, "pull": pull, "loop": loop, "audit": audit}[sys.argv[1]]
    fn(*sys.argv[2:])
