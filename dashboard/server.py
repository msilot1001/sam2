"""Local read-only research dashboard backend (stdlib only).

Run:  cd /home/sojak/sam2 && uv run python dashboard/server.py
Serves the JSON API under /api and the built frontend (dashboard/frontend/dist) on 127.0.0.1:8765.
"""
import csv
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
DIST = (REPO / "dashboard/frontend/dist").resolve()
M8 = "outputs/milestone8"
ALLOW = ["docs/m7", "docs/m8", "docs/GPU_WORKER_OPERATIONS.md", "AGENTS.md", "outputs/milestone7",
         M8, "outputs/SYNC_LEDGER.tsv", "outputs/SYNC_STATUS.json", "scripts", "src/spatial", "tests"]
ROOTS = [(a, (REPO / a).resolve()) for a in ALLOW]
DENY_PARTS = {"tcache", "checkpoints", "data", ".ssh", "__pycache__", ".git"}
BINARY = {".npz", ".npy", ".pt"}  # listed with size only, never served
MACHINES = {"ubuntu": f"{M8}/runs", "runpod": f"{M8}/runpod/runs"}
MAX_BYTES = 8 << 20
TYPES = {".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
csv.field_size_limit(1 << 24)


class Denied(Exception):
    pass


def denied_name(parts):
    return any(p in DENY_PARTS or p.endswith(".env") or p.startswith(".env") for p in parts)


def safe(rel):
    """Repo-relative path -> resolved Path inside an allow-listed root, else raise Denied."""
    rel = (rel or "").strip()
    if not rel or "\x00" in rel or rel.startswith(("/", "~")) or ".." in Path(rel).parts:
        raise Denied(rel)
    p = (REPO / rel).resolve()  # resolves symlinks; the containment check below runs on the real path
    for _, root in ROOTS:
        if p == root or root in p.parents:
            if denied_name(p.relative_to(root).parts) or denied_name(Path(rel).parts):
                raise Denied(rel)
            return p
    raise Denied(rel)


def rel(p):
    return p.relative_to(REPO).as_posix()


def is_listable(p):
    try:
        safe(rel(p))
        return True
    except (Denied, ValueError):
        return False


def entry(p):
    st = p.stat()
    return {"name": p.name, "path": rel(p), "dir": p.is_dir(), "size": st.st_size, "mtime": st.st_mtime,
            "binary": p.suffix in BINARY}


def ls(path):
    if not path:
        return [entry(r) for a, r in ROOTS if r.exists()]
    p = safe(path)
    kids = [c for c in p.iterdir() if is_listable(c)]
    return [entry(c) for c in sorted(kids, key=lambda c: (not c.is_dir(), c.name))]


def read_table(p, offset=0, limit=2000):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t" if p.suffix == ".tsv" else ","))
    cols, body = (rows[0], rows[1:]) if rows else ([], [])
    summary = []
    for i, c in enumerate(cols):
        vals = [r[i] for r in body if i < len(r) and r[i] != ""]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            nums = None
        s = {"col": c, "n": len(vals)}
        if nums:
            s.update(mean=sum(nums) / len(nums), min=min(nums), max=max(nums))
        else:
            s["unique"] = len(set(vals))
        summary.append(s)
    return {"columns": cols, "rows": body[offset:offset + limit], "total": len(body), "offset": offset,
            "summary": summary}


def load_json(r):
    p = REPO / r
    return json.loads(p.read_text()) if p.exists() else None


def overview():
    runs = {}
    for m, d in MACHINES.items():
        base = REPO / d
        runs[m] = {a.name: sorted(f.name for f in a.glob("*.csv"))
                   for a in sorted(base.iterdir()) if a.is_dir()} if base.exists() else {}
    ledger = {}
    led = REPO / "outputs/SYNC_LEDGER.tsv"
    if led.exists():
        for r in csv.DictReader(open(led, newline=""), delimiter="\t"):
            ledger[r.get("status")] = ledger.get(r.get("status"), 0) + 1
    gens = REPO / M8 / "stage1_generators.csv"
    return {"arms": load_json(f"{M8}/arms.json"), "causal": load_json(f"{M8}/causal_report.json"),
            "stage1": load_json(f"{M8}/stage1_report.json"),
            "generators": list(csv.DictReader(open(gens, newline=""))) if gens.exists() else [],
            "runs": runs, "sync": load_json("outputs/SYNC_STATUS.json"), "ledger": ledger}


def docs():
    out = []
    for d in ["docs/m8", "docs/m7"]:
        out += [rel(p) for p in sorted((REPO / d).glob("*.md"))]
    return out + [f for f in ["docs/GPU_WORKER_OPERATIONS.md", "AGENTS.md"] if (REPO / f).exists()]


def images():
    return [entry(p) for d in [f"{M8}/viz", "outputs/milestone7/viz"]
            for p in sorted((REPO / d).rglob("*.png")) if is_listable(p)]


def tail(p, n):
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - (2 << 20)))
        lines = f.read().decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = (body if isinstance(body, str) else json.dumps(body, default=str)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith("/api/"):
                return self.api(u.path[5:], q)
            return self.static(u.path)
        except Denied:
            self.send(403, {"error": "forbidden"})
        except (FileNotFoundError, NotADirectoryError):
            self.send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 - surface to UI, keep serving
            self.send(500, {"error": repr(e)})

    def api(self, name, q):
        if name == "overview":
            return self.send(200, overview())
        if name == "ls":
            return self.send(200, ls(q.get("path", "")))
        if name == "docs":
            return self.send(200, docs())
        if name == "images":
            return self.send(200, images())
        p = safe(q.get("path", ""))
        if not p.is_file():
            raise FileNotFoundError
        if name == "csv":
            return self.send(200, read_table(p, int(q.get("offset", 0)), min(int(q.get("limit", 2000)), 20000)))
        if name == "file":
            if p.suffix in BINARY:
                raise Denied(p)
            if "tail" in q:
                return self.send(200, tail(p, int(q["tail"])), "text/plain; charset=utf-8")
            if p.stat().st_size > MAX_BYTES:
                return self.send(413, {"error": "too large; use tail"})
            ctype = TYPES.get(p.suffix.lower(), "text/plain; charset=utf-8")  # never html/svg -> no script
            return self.send(200, p.read_bytes(), ctype)
        self.send(404, {"error": "unknown endpoint"})

    def static(self, path):
        p = (DIST / path.lstrip("/")).resolve()
        if not (DIST in p.parents and p.is_file()):
            p = DIST / "index.html"  # SPA fallback
        self.send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"dashboard on http://127.0.0.1:{port}  (repo {REPO})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
