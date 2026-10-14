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
            "outputs/milestone7/tcache", "checkpoints", "data", "pyproject.toml", "scripts/.env", ""]:
    try:
        safe(bad)
        raise AssertionError(f"allowed: {bad!r}")
    except Denied:
        pass
for ok in ["AGENTS.md", "docs/m8", "outputs/milestone8/arms.json", "scripts/m8.py", "outputs/SYNC_STATUS.json"]:
    assert safe(ok).exists(), ok
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
finally:
    httpd.shutdown()
print("ok")
