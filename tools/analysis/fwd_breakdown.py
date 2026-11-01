#!/usr/bin/env python3
"""Kernel time breakdown of the last forward in an nsys sqlite export of llama-bench prefill.

Forwards are separated by GPU gaps longer than --gap ms. Prints per-kernel share of the last forward and its idle time.
    python3 fwd_breakdown.py prof.sqlite [--gap 50]
"""
import sqlite3
import sys
from collections import defaultdict

db = sys.argv[1]
gap_ms = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 50.0
con = sqlite3.connect(db)
rows = con.execute(
    "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
).fetchall()
fwd, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > gap_ms * 1e6:
        fwd.append(cur)
        cur = []
    cur.append(r)
fwd.append(cur)
print(f"{len(fwd)} segments, last 4: " + ", ".join(f"{(f[-1][1] - f[0][0]) / 1e6:.0f} ms" for f in fwd[-4:]))
last = fwd[-1]
t0, t1 = last[0][0], max(r[1] for r in last)
span = t1 - t0
by = defaultdict(int)
busy, edge = 0, t0
for s, e, name in last:
    by[name] += e - s
    if e > edge:
        busy += e - max(s, edge)
        edge = e
print(f"last forward {span / 1e6:.1f} ms, idle {(span - busy) / span * 100:.1f} %")
for name, t in sorted(by.items(), key=lambda x: -x[1])[:14]:
    print(f"{t / span * 100:5.1f} %  {t / 1e6:8.1f} ms  {name[:90]}")
