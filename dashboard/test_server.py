"""Path-guard self-check:  uv run python dashboard/test_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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
print("ok")
