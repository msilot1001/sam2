"""Update the live task ledger shown on the dashboard: uv run python scripts/m8_tasks.py <id>=<STATUS> [...]"""
import json
import sys
import time
from pathlib import Path

P = Path("outputs/milestone8/TASKS.json")
t = json.loads(P.read_text())
for a in sys.argv[1:]:
    i, st = a.split("=")
    next(x for x in t["tasks"] if x["id"] == i)["status"] = st
t["updated_kst"] = time.strftime("%Y-%m-%d %H:%M")
tmp = P.with_suffix(".tmp")
tmp.write_text(json.dumps(t, ensure_ascii=False, indent=1))
tmp.replace(P)
