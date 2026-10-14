"""Path-guard + token-auth self-check:  uv run python dashboard/test_server.py"""
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402
from server import Denied, safe  # noqa: E402

for bad in ["../.gpu_worker.env", ".gpu_worker.env", "/etc/passwd", "~/.ssh/id_rsa", "outputs/milestone8/../../AGENTS.md",
            "outputs/milestone7/tcache", "checkpoints", "data", "pyproject.toml", "scripts/.env", "",
            "data/mosev2", "data/vost/JPEGImages", "outputs/milestone9/data/x", "outputs/milestone9/../../data/vost",
            "docs/m9/../../data", "outputs/milestone10/x", "docs/m10"]:
    try:
        safe(bad)
        raise AssertionError(f"allowed: {bad!r}")
    except Denied:
        pass
for ok in ["AGENTS.md", "docs/m8", "outputs/milestone8/arms.json", "scripts/m8.py", "outputs/SYNC_STATUS.json",
           "docs/m9", "outputs/milestone9/TASKS.json", "outputs/milestone9/manifests/vost_units.json"]:
    assert safe(ok).exists(), ok
meta = "outputs/milestone9/manifests/mosev2_meta_train.json"
assert server.size_only(server.REPO / meta) and server.size_only(server.REPO / "x/a.vec.npy")
assert not server.size_only(server.REPO / "outputs/milestone9/manifests/mosev2_units.json")
assert server.ms_of("vost__vost_1169_peel_onion__obj1") == "m9" and server.ms_of("davis__davis_drone__obj1") == "m8"
assert server.baseline_path("runpod", "mosev2__mosev2_x__obj1", "m9").as_posix().endswith(
    "outputs/milestone9/runpod/diag/mosev2/mosev2__mosev2_x__obj1.csv")
v = "vost__vost_1169_peel_onion__obj1"  # no Ubuntu diag for VOST -> falls back to the plain SGGL arm rows
assert server.baseline_path("ubuntu", v, "m9").as_posix().endswith(f"outputs/milestone9/runs/SGGL/{v}.csv")
assert server.unit(v)["series"] == [] and "ubuntu" in server.unit(v)["baseline"]
assert server.safe_glob("data/*/*") == 0 and server.safe_glob("outputs/milestone9/manifests/*.json") > 0
for bad in ["../x", "/etc/passwd", "outputs/milestone8/../../.gpu_worker.env"]:
    try:
        server.safe_glob(bad)
        raise AssertionError(f"glob allowed: {bad!r}")
    except Denied:
        pass
try:
    server.unit("../../etc")
    raise AssertionError("unit stem traversal allowed")
except Denied:
    pass

# bind policy: non-loopback needs a >= 16-char token
server.check_bind("127.0.0.1", "")
server.check_bind("::1", "")
server.check_bind("192.168.1.21", "a" * 32)
for host, tok in [("192.168.1.21", ""), ("0.0.0.0", "short"), ("example.org", "")]:
    try:
        server.check_bind(host, tok)
        raise AssertionError(f"bind allowed: {host} {tok!r}")
    except SystemExit:
        pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a):
        return None


def get(url, headers=None):
    try:
        r = urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, headers=headers or {}))
        return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_address[1]}"
try:
    server.TOKEN = ""  # loopback mode: open, but a foreign Host header is refused (DNS rebinding)
    assert get(f"{base}/api/docs")[0] == 200
    assert get(f"{base}/api/docs", {"Host": "evil.example:80"})[0] == 403
    assert get(f"{base}/api/file?path=.gpu_worker.env")[0] == 403
    assert get(f"{base}/api/m9")[0] == 200
    assert get(f"{base}/api/tasks?ms=m9")[0] == 200 and get(f"{base}/api/tasks")[0] == 200
    assert get(f"{base}/api/tasks?ms=../x")[0] == 403
    assert get(f"{base}/api/file?path=outputs/milestone9/TASKS.json")[0] == 200
    assert get(f"{base}/api/file?path=docs/m9/DATASET_AUDIT.md")[0] == 200
    for bad in [meta, "data/vost/Annotations", "outputs/milestone9/../../data/x"]:
        assert get(f"{base}/api/file?path={bad}")[0] == 403, bad
        assert get(f"{base}/api/csv?path={bad}")[0] == 403, bad
    assert get(f"{base}/api/ls?path=data")[0] == 403
    import json as _json
    listing = _json.loads(urllib.request.urlopen(f"{base}/api/ls?path=outputs/milestone9/manifests").read())
    assert [e["binary"] for e in listing if e["path"] == meta] == [True], listing  # size only
    m9 = _json.loads(urllib.request.urlopen(f"{base}/api/m9").read())
    assert "pumavos" not in _json.dumps(m9["manifests"]).lower()
    assert all("frames" not in u for d in m9["manifests"].values() for u in d["units"])

    server.TOKEN = tok = "t" * 32
    assert get(f"{base}/api/docs")[0] == 401
    assert get(f"{base}/")[0] == 401
    assert get(f"{base}/api/docs?token=wrong")[0] == 401
    assert get(f"{base}/api/docs", {"Authorization": "Bearer wrong"})[0] == 401
    assert get(f"{base}/api/docs", {"Cookie": "dash_token=wrong"})[0] == 401
    code, h = get(f"{base}/api/csv?path=x&token={tok}")
    assert code == 302 and h["Location"] == "/api/csv?path=x", h
    assert "HttpOnly" in h["Set-Cookie"] and "SameSite=Strict" in h["Set-Cookie"], h
    cookie = h["Set-Cookie"].split(";")[0]
    assert get(f"{base}/api/docs", {"Cookie": cookie})[0] == 200
    assert get(f"{base}/api/docs", {"Authorization": f"Bearer {tok}"})[0] == 200
    assert get(f"{base}/api/file?path=.gpu_worker.env", {"Cookie": cookie})[0] == 403  # token never bypasses the guard
    assert get(f"{base}/api/file?path={meta}", {"Cookie": cookie})[0] == 403
    assert get(f"{base}/api/m9")[0] == 401 and get(f"{base}/api/tasks?ms=m9")[0] == 401
finally:
    httpd.shutdown()
print("ok")
