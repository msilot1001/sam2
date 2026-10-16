"""Check every number written in the paper/report docs against the report json files.

    uv run python scripts/paper_check.py          # list mismatches (exit 1 if any)
    uv run python scripts/paper_check.py --fix    # rewrite doc-side CI triples to the json values, then list again

Two checks:
1. CI triples "x [lo, hi]" (signed or not, '-' or '−'): must equal (to 3 decimals) some json record (point, ci).
   A record = a dict with "ci" (any numeric sibling as the point) or "ci_<k>" (point "<k>" or "d<k>").
   Mismatch kinds: 'ci' (point matches, CI differs: fixable), 'point' (CI matches, point differs by <= 0.002: fixable),
   'unmatched' (no record near: reported, not fixed).
2. Bare 3-decimal numbers in markdown table cells (outside triples): must equal (to 3 decimals, either sign) some json
   number. Reported only (derived numbers such as sums or ratios of json values land here and are read by hand).
"""
import json
import re
import sys
from pathlib import Path

DOCS = sorted(Path("docs/paper").glob("table*.md")) + [Path("docs/m10") / f for f in
                                                       ("CORE_CONCLUSIONS.md", "FINAL_REPORT.md", "BASELINES.md", "HEAD_RESULTS.md")]
JSONS = sorted(Path("outputs/milestone10/reports").glob("*.json")) + [
    Path("outputs/milestone61/comparison_report.json"), Path("outputs/milestone7/replay_report.json"), Path("outputs/milestone7/vk_report.json")]
NUM = r"[+\-−]?\d+\.\d{3}"
TRIPLE = re.compile(rf"({NUM})\s*\[\s*({NUM}),\s*({NUM})\s*\]")
BARE = re.compile(rf"(?<![\d.])({NUM})(?![\d])")


def val(s):
    return float(s.replace("−", "-"))


def records():
    """(point, lo, hi, source, path keys) for every CI in the json files, plus the sign-flipped copy (a doc may report
    P5 - B0 where the json has B0 - P5); and the set of all json numbers (abs, 3 decimals)."""
    recs, nums = [], set()

    def walk(o, src, path):
        if isinstance(o, dict):
            num = {k: v for k, v in o.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for k, v in o.items():
                if k.lower().startswith("ci") and isinstance(v, list) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
                    pts = [num[x] for x in (k[3:], "d" + k[3:]) if x in num] or list(num.values())
                    keys = frozenset(t for x in path for t in [x.lower()] + re.split(r"[^a-z0-9_+]+", x.lower()) if t) | {src.lower()}
                    for pt in pts:
                        recs.append((pt, v[0], v[1], src, keys))
                        recs.append((-pt, -v[1], -v[0], src + " (sign-flipped)", keys | {"flipped"}))
            for k, v in o.items():
                walk(v, src, path + [str(k)])
        elif isinstance(o, list):
            for v in o:
                walk(v, src, path)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            nums.add(round(abs(o), 3))

    for f in JSONS:
        walk(json.loads(f.read_text()), f.name, [f.stem])
    return recs, nums


DS = {"mosev2": "mosev2", "vost": "vost", "lvos": "lvosv2"}


def context(lines, ln, pos):
    """Tokens that name the estimate: the row's first cell (arm / rule / model) and the column header's dataset."""
    line = lines[ln]
    toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_+]*", line.split("|")[1] if line.lstrip().startswith("|") else line))
    toks = {t.lower() for t in toks}
    ds = None
    if line.lstrip().startswith("|"):
        col = line[:pos].count("|") - 1
        k = ln
        while k > 0 and lines[k].lstrip().startswith("|"):
            k -= 1
        hdr = lines[k + 1].split("|")[1:-1] if lines[k + 1].lstrip().startswith("|") else []
        h = hdr[col].lower() if 0 <= col < len(hdr) else ""
        ds = next((v for key, v in DS.items() if key in h), None)
        hdr_toks = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_+]*", h)} - set(DS) - {"ci", "j", "f", "jf"}
    if ds is None:
        ds = next((v for key, v in DS.items() if key in line.lower()), None)
    return toks, ds, (hdr_toks if line.lstrip().startswith("|") else set())


def ok(v, x):
    return abs(v - x) <= 0.0005 + 1e-9


def fmt(v, signed, uminus):
    """3 decimals in the doc's style: explicit '+' if the doc triple used one, '−' if the doc used it."""
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s.replace("-", "−") if uminus else s


def check(fix=False):
    recs, nums = records()
    out = []
    for doc in DOCS:
        text = doc.read_text()
        lines = text.splitlines()
        new = text
        for ln, line in enumerate(lines):
            for m in TRIPLE.finditer(line):
                x, lo, hi = map(val, m.groups())
                if any(ok(p, x) and ok(l, lo) and ok(h, hi) for p, l, h, _, _ in recs):
                    continue
                toks, ds, ht = context(lines, ln, m.start())
                score = lambda keys: 2 * bool(toks & keys) + (ds in keys if ds else 0) + len(ht & keys)
                cand = [(-score(k), not ok(p, x), abs(p - x) + abs(l - lo) + abs(h - hi), "ci" if ok(p, x) else "point", p, l, h, src)
                        for p, l, h, src, k in recs if abs(p - x) <= 0.0025 and abs(l - lo) + abs(h - hi) < 0.012]
                if not cand:
                    out.append({"doc": str(doc), "line": ln + 1, "kind": "unmatched", "text": m.group(0)})
                    continue
                cand.sort(key=lambda c: c[:3])
                top = [c for c in cand if c[:2] == cand[0][:2]]
                sc, _, _, kind, p, l, h, src = cand[0]
                amb = len({(round(c[4], 3), round(c[5], 3), round(c[6], 3)) for c in top}) > 1
                sg, um = "+" in m.group(0), "−" in m.group(0)
                to = f"{fmt(p, sg, um)} [{fmt(l, sg, um)}, {fmt(h, sg, um)}]"
                out.append({"doc": str(doc), "line": ln + 1, "kind": kind + (" (ambiguous)" if amb else ""), "text": m.group(0),
                            "json": to, "src": src, "context_score": -sc})
                if fix and not amb:
                    new = new.replace(m.group(0), to)
            cells = line.split("|")[1:-1] if line.lstrip().startswith("|") else []
            for c in cells:
                c = TRIPLE.sub("", c)
                for m in BARE.finditer(c):
                    if round(abs(val(m.group(1))), 3) not in nums:
                        out.append({"doc": str(doc), "line": ln + 1, "kind": "bare", "text": m.group(1), "cell": c.strip()[:60]})
        if fix and new != text:
            doc.write_text(new)
    return out


if __name__ == "__main__":
    fix = "--fix" in sys.argv
    res = check(fix)
    if fix:
        res = check()
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    kinds = {}
    for r in res:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("summary", kinds, file=sys.stderr)
    sys.exit(1 if res else 0)
