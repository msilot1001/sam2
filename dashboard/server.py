"""Local read-only research dashboard backend (stdlib only).

Run:  cd /home/sojak/sam2 && uv run python dashboard/server.py [port]
Serves the JSON API under /api and the built frontend (dashboard/frontend/dist) on 127.0.0.1:8765.
LAN:  DASH_HOST=<lan-ip> DASH_TOKEN=$(openssl rand -hex 16) uv run python dashboard/server.py
      (a non-loopback DASH_HOST refuses to start without a DASH_TOKEN of >= 16 chars)
"""
import copy
import csv
import hmac
import ipaddress
import json
import math
import mimetypes
import os
import re
import sys
import time
from functools import lru_cache
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse

REPO = Path(__file__).resolve().parent.parent
DIST = (REPO / "dashboard/frontend/dist").resolve()
M8 = "outputs/milestone8"
M9 = "outputs/milestone9"
ALLOW = ["docs/m7", "docs/m8", "docs/m9", "docs/GPU_WORKER_OPERATIONS.md", "AGENTS.md", "outputs/milestone7",
         M8, M9, "outputs/SYNC_LEDGER.tsv", "outputs/SYNC_STATUS.json", "scripts", "src/spatial", "tests"]
ROOTS = [(a, (REPO / a).resolve()) for a in ALLOW]
DENY_PARTS = {"tcache", "checkpoints", "data", ".ssh", "__pycache__", ".git"}
BINARY = {".npz", ".npy", ".pt"}  # listed with size only, never served
# per milestone: run dirs per machine, machine-matched SGGL baseline and stage-1 candidate paths ({ds} = stem prefix)
MILESTONES = {
    "m8": {"dir": M8, "runs": {"ubuntu": f"{M8}/runs", "runpod": f"{M8}/runpod/runs"},
           "base": {"runpod": [f"{M8}/runpod/runs/SGGL/{{stem}}.csv"], "ubuntu": [f"{M8}/diag/{{stem}}.csv"]},
           "cand": [f"{M8}/diag/{{stem}}.cand.csv", f"{M8}/runpod/diag/{{stem}}.cand.csv"],
           "logs": [f"{M8}/logs/*.log", f"{M8}/runpod/logs/queue*.log"]},
    "m9": {"dir": M9, "runs": {"ubuntu": f"{M9}/runs", "runpod": f"{M9}/runpod/runs"},
           # same order as scripts/m9_analysis.py _unit_deltas: diag rows, else the plain SGGL arm
           "base": {"runpod": [f"{M9}/runpod/diag/{{ds}}/{{stem}}.csv", f"{M9}/runpod/runs/SGGL/{{stem}}.csv"],
                    "ubuntu": [f"{M9}/diag/{{ds}}/{{stem}}.csv", f"{M9}/runs/SGGL/{{stem}}.csv"]},
           "cand": [f"{M9}/runpod/diag/{{ds}}/{{stem}}.cand.csv", f"{M9}/diag/{{ds}}/{{stem}}.cand.csv"],
           "logs": [f"{M9}/*.log", f"{M9}/logs/*.log", f"{M9}/runpod/logs/*.log"]},
}
MACHINES = MILESTONES["m8"]["runs"]
M9_PREFIXES = ("mosev2__", "vost__", "lvos")
MAX_BYTES = 8 << 20
TYPES = {".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
csv.field_size_limit(1 << 24)
HOST = os.environ.get("DASH_HOST", "127.0.0.1")
TOKEN = os.environ.get("DASH_TOKEN", "")
COOKIE = "dash_token"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


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


def size_only(p):
    """Listed with size only, never served (binaries, the 7 MB MOSEv2 meta dump)."""
    return p.suffix in BINARY or p.name.endswith("_meta_train.json")


def entry(p):
    st = p.stat()
    return {"name": p.name, "path": rel(p), "dir": p.is_dir(), "size": st.st_size, "mtime": st.st_mtime,
            "binary": size_only(p)}


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
    for d in ["docs/m9", "docs/m8", "docs/m7"]:
        out += [rel(p) for p in sorted((REPO / d).glob("*.md"))]
    return out + [f for f in ["docs/GPU_WORKER_OPERATIONS.md", "AGENTS.md"] if (REPO / f).exists()]


def images():
    return [entry(p) for d in [M9, f"{M8}/viz", "outputs/milestone7/viz"]
            for p in sorted((REPO / d).rglob("*.png")) if is_listable(p)]


def tail(p, n):
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - (2 << 20)))
        lines = f.read().decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:])


# ---- mtime-keyed caches: parsed files are reused until the file changes ----
@lru_cache(maxsize=1024)
def _rows(path, mtime_ns):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=64)
def _json(path, mtime_ns):
    return json.loads(Path(path).read_text())


def rows(p):
    p = Path(p)
    return _rows(str(p), p.stat().st_mtime_ns) if p.exists() else []


def jsonf(p):
    p = Path(p)
    return _json(str(p), p.stat().st_mtime_ns) if p.exists() else None


def finite(o):
    """Browser JSON.parse rejects NaN/Infinity (Python json.dump writes them): map non-finite floats to null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def dumps(o):
    """The one JSON encoder for every response (API payloads and served .json files)."""
    return json.dumps(finite(o), default=str, allow_nan=False)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def arm_dirs(ms="m8"):
    """[(machine, arm, dir)] for every arm run directory of a milestone (incl. SGGL baseline dirs)."""
    out = []
    for m, d in MILESTONES[ms]["runs"].items():
        base = REPO / d
        if base.exists():
            out += [(m, a.name, a) for a in sorted(base.iterdir()) if a.is_dir()]
    return out


def ms_of(stem):
    return "m9" if stem.startswith(M9_PREFIXES) else "m8"


def ms_paths(templates, stem):
    return [REPO / t.format(stem=stem, ds=stem.split("__")[0]) for t in templates]


def baseline_paths(machine, stem, ms="m8"):
    return ms_paths(MILESTONES[ms]["base"][machine], stem)


def baseline_path(machine, stem, ms="m8"):
    """first existing baseline candidate (else the first one)"""
    c = baseline_paths(machine, stem, ms)
    return next((p for p in c if p.exists()), c[0])


def arms_summary():
    per_unit = [typed(r) for r in rows(REPO / M8 / "causal_per_unit.csv")]
    return {"arms": jsonf(REPO / M8 / "arms.json"), "causal": jsonf(REPO / M8 / "causal_report.json"),
            "per_unit": per_unit,
            "runs": {f"{m}/{a}": len(list(d.glob("*.csv"))) for m, a, d in arm_dirs()}}


def units():
    """{stem: ["machine/arm", ...]} over all per-frame run CSVs (M8 + M9, incl. the M9 RunPod diag baselines)."""
    out = {}
    for ms in MILESTONES:
        for m, a, d in arm_dirs(ms):
            for f in d.glob("*.csv"):
                out.setdefault(f.stem, []).append(f"{m}/{a}")
    for f in REPO.glob(f"{M9}/runpod/diag/*/*.csv"):
        if not f.name.endswith(".cand.csv"):
            out.setdefault(f.stem, []).append("runpod/diag")
    return dict(sorted(out.items()))


EXTRA = ["crop_side", "gt_in_crop", "gt_long_over_side", "pc_dx", "pc_dy"]  # M9 run columns, plotted if present


def frames(p):
    out = []
    for r in rows(p):
        g, l = fnum(r.get("score_G")), fnum(r.get("score_L"))
        out.append({"i": int(r["video_idx"]), "JF": fnum(r.get("JF")), "J": fnum(r.get("J")), "F": fnum(r.get("F")),
                    "gt_empty": r.get("gt_empty") == "True", "rescued": r.get("rescued") == "True",
                    "coabs": g is not None and l is not None and g <= 0 and l <= 0, "gen": r.get("gen") or None,
                    **{c: fnum(r[c]) for c in EXTRA if r.get(c) not in (None, "")}})
    return out


def typed(r):
    return {k: (v if k in ("unit", "video", "machine", "set", "arm", "dataset") else
                v == "True" if v in ("True", "False") else fnum(v)) for k, v in r.items()}


def m9_unit_metrics(stem):
    out = []
    for p in sorted((REPO / M9 / "reports").glob("causal_units_*.csv")):
        out += [{"arm": p.stem[len("causal_units_"):], **typed(r)} for r in rows(p)
                if (r.get("unit") or "").replace("/", "_") == stem]
    return out


STEM = re.compile(r"^[\w.-]+$")


def unit(stem):
    if not STEM.match(stem) or ".." in stem:
        raise Denied(stem)
    ms = ms_of(stem)
    series, base = [], {}
    for m, a, d in arm_dirs(ms):
        f = d / f"{stem}.csv"
        if f.exists() and f not in baseline_paths(m, stem, ms):  # a baseline is not an arm
            series.append({"machine": m, "arm": a, "path": rel(f), "frames": frames(f)})
    for m in MILESTONES[ms]["base"]:
        b = baseline_path(m, stem, ms)
        if b.exists():
            base[m] = {"path": rel(b), "frames": frames(b)}
    cand = [p for p in ms_paths(MILESTONES[ms]["cand"], stem) if p.exists()]
    if not series and not base:
        raise FileNotFoundError(stem)
    return {"stem": stem, "milestone": ms, "series": series, "baseline": base,
            "metrics": m9_unit_metrics(stem) if ms == "m9" else None,
            "candidates": {"path": rel(cand[0]), "rows": rows(cand[0])} if cand else None}


RANKER_SETS = ["GLOBAL", "LOCAL", "POOL", "pool25"]


def stage1():
    gens = REPO / M8 / "stage1_generators.csv"
    return {"report": jsonf(REPO / M8 / "stage1_report.json"), "generators": rows(gens),
            "rankers": {s: jsonf(REPO / M8 / f"rankers_{s}.json") for s in RANKER_SETS}}


M7 = "outputs/milestone7"
M7_REPORTS = ["causal_pilots_report", "teval_report", "vk_report", "m3r_report", "replay_report"]


def m7():
    return {"reports": {n: jsonf(REPO / M7 / f"{n}.json") for n in M7_REPORTS},
            "docs": [rel(p) for p in sorted((REPO / "docs/m7").glob("*.md"))]}


def safe_glob(pattern):
    """Count files matching a repo-relative glob, only inside allow-listed paths."""
    if not pattern or pattern.startswith(("/", "~")) or ".." in pattern:
        raise Denied(pattern)
    return sum(1 for p in REPO.glob(pattern) if p.is_file() and is_listable(p))


def tasks(ms="m8"):
    t = copy.deepcopy(jsonf(REPO / MILESTONES[ms]["dir"] / "TASKS.json"))  # the cached object must not be mutated
    if not t:
        return None
    for task in t.get("tasks", []):
        pr = task.get("progress")
        if pr and pr.get("glob"):
            try:
                done = safe_glob(pr["glob"])
            except Denied:
                done = None
            total = pr.get("total") or 0
            pr.update(done=done, percent=round(100 * done / total, 1) if done is not None and total else None)
    return t


def latest_logs(ms="m8", n=4, lines=15):
    logs = [p for g in MILESTONES[ms]["logs"] for p in REPO.glob(g) if is_listable(p)]
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": rel(p), "mtime": p.stat().st_mtime, "tail": tail(p, lines)} for p in logs[:n]]


UNIT_KEYS = ["video", "obj_id", "split", "area0_frac", "growth_max", "absent_frames", "tiny", "n_frames_from_prompt"]


def ds_count(files):
    """{dataset: n} over per-unit CSVs (stem prefix before '__'), ignoring .cand.csv"""
    c = {}
    for f in files:
        if not f.name.endswith(".cand.csv"):
            ds = f.stem.split("__")[0]
            c[ds] = c.get(ds, 0) + 1
    return c


def m9():
    man = {}
    for p in sorted((REPO / M9 / "manifests").glob("*_units.json")):
        j = jsonf(p)
        man[j.get("dataset", p.stem[:-6])] = {**{k: v for k, v in j.items() if k not in ("units", "videos", "rule")},
                                               "path": rel(p), "units": [{k: u.get(k) for k in UNIT_KEYS} for u in j["units"]]}
    rep = REPO / M9 / "reports"
    by_ds = lambda pre: {p.stem[len(pre):]: jsonf(p) for p in sorted(rep.glob(f"{pre}*.json"))}
    runs = {f"{m}/{a}": ds_count(d.glob("*.csv")) for m, a, d in arm_dirs("m9")}
    runs["runpod/diag"] = ds_count(REPO.glob(f"{M9}/runpod/diag/*/*.csv"))
    return {"prereg": jsonf(REPO / M9 / "preregistration.json"), "arms": jsonf(REPO / M9 / "arms.json"),
            "manifests": man, "stage1": by_ds("stage1_"), "taxonomy": by_ds("taxonomy_"),
            "causal": jsonf(rep / "causal.json"), "runs": runs,
            "causal_units": [rel(p) for p in sorted(rep.glob("causal_units_*.csv"))]}


_index = {"at": 0.0, "files": []}


def search(q, limit=200):
    """Case-insensitive substring match on repo-relative file names in the allow-listed roots."""
    if time.time() - _index["at"] > 60:  # ponytail: full re-walk once a minute; fine for ~10^4 files
        files = []
        for _, root in ROOTS:
            if root.is_file():
                files.append(rel(root))
            for dirpath, dirs, names in os.walk(root):
                dirs[:] = [d for d in dirs if not denied_name([d])]
                files += [rel(Path(dirpath) / n) for n in names if not denied_name([n])]
        _index.update(at=time.time(), files=sorted(set(files)))
    terms = q.lower().split()
    hits = [f for f in _index["files"] if terms and all(t in f.lower() for t in terms)]
    hits.sort(key=lambda f: (not f.endswith(".md"), len(f)))
    return {"total": len(hits), "hits": hits[:limit]}


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind(host, token):
    """Refuse a non-loopback bind without a strong token."""
    if not is_loopback(host) and len(token) < 16:
        raise SystemExit(f"refusing to bind {host!r}: set DASH_TOKEN (>= 16 chars, e.g. $(openssl rand -hex 16))")


def token_ok(v):
    return bool(TOKEN) and bool(v) and hmac.compare_digest(v.encode(), TOKEN.encode())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = (body if isinstance(body, str) else dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, u, q):
        """True if the request may proceed; otherwise a response (401/403/302) has been sent."""
        if not TOKEN:  # loopback-only mode: reject foreign Host headers (DNS rebinding)
            host = self.headers.get("Host", "")
            name = host[:host.find("]") + 1] if host.startswith("[") else host.split(":")[0]
            if name not in LOCAL_HOSTS:
                self.send(403, {"error": "bad host"})
                return False
            return True
        if "token" in q:
            if not token_ok(q["token"]):
                self.send(401, {"error": "unauthorized"})
                return False
            rest = urlencode([(k, v) for k, v in parse_qsl(u.query) if k != "token"])
            self.send_response(302)
            self.send_header("Location", u.path + (f"?{rest}" if rest else ""))
            self.send_header("Set-Cookie", f"{COOKIE}={TOKEN}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        try:
            jar = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            jar = {}
        auth = self.headers.get("Authorization", "")
        if token_ok(jar[COOKIE].value if COOKIE in jar else "") or token_ok(auth[7:] if auth.startswith("Bearer ") else ""):
            return True
        self.send(401, "401 unauthorized: open this site once with ?token=<DASH_TOKEN>\n", "text/plain; charset=utf-8")
        return False

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if not self.authorized(u, q):
            return
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
        if name == "arms/summary":
            return self.send(200, arms_summary())
        if name == "units":
            return self.send(200, units())
        if name.startswith("unit/"):
            return self.send(200, unit(name[5:]))
        if name == "stage1":
            return self.send(200, stage1())
        if name == "m7":
            return self.send(200, m7())
        if name == "m9":
            return self.send(200, m9())
        if name == "tasks":
            ms = q.get("ms", "m8")
            if ms not in MILESTONES:
                raise Denied(ms)
            return self.send(200, {"tasks": tasks(ms), "logs": latest_logs(ms)})
        if name == "search":
            return self.send(200, search(q.get("q", "")))
        p = safe(q.get("path", ""))
        if not p.is_file():
            raise FileNotFoundError
        if size_only(p):
            raise Denied(p)
        if name == "csv":
            return self.send(200, read_table(p, int(q.get("offset", 0)), min(int(q.get("limit", 2000)), 20000)))
        if name == "file":
            if "tail" in q:
                return self.send(200, tail(p, int(q["tail"])), "text/plain; charset=utf-8")
            if p.stat().st_size > MAX_BYTES:
                return self.send(413, {"error": "too large; use tail"})
            if p.suffix.lower() == ".json":  # report files may hold NaN (json.dump default): re-encode via dumps()
                return self.send(200, jsonf(p))
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
    check_bind(HOST, TOKEN)  # LAN/0.0.0.0 only with a token (see README)
    auth = "token required" if TOKEN else "no token (loopback only)"
    print(f"dashboard on http://{HOST}:{port}  ({auth}; repo {REPO})", flush=True)
    ThreadingHTTPServer((HOST, port), H).serve_forever()
