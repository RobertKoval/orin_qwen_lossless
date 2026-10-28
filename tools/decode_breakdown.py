#!/usr/bin/env python3
"""Per-kernel time of one decode step in an nsys sqlite export of llama-bench tg (--cuda-graph-trace=node).

A step is a run of kernels with GPU gaps shorter than --gap µs; only runs with the dominant kernel count are kept
(one CUDA graph per token). Prints the step span, the GPU idle inside a step, the gap between steps and the
per-kernel time per step (mean over steps).
    python3 decode_breakdown.py prof.sqlite [--gap 50] [--top 30]
"""
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


db = sys.argv[1]
gap_ns = arg("--gap", 50.0) * 1e3
top = arg("--top", 30)
con = sqlite3.connect(db)
rows = con.execute(
    "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
).fetchall()

runs, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > gap_ns:
        runs.append(cur)
        cur = []
    cur.append(r)
runs.append(cur)
# the step length that covers the most kernels (single kernels between steps are more frequent than steps)
n = max(Counter(len(r) for r in runs).items(), key=lambda x: x[0] * x[1])[0]
steps = [i for i, r in enumerate(runs) if len(r) == n]
print(f"{len(rows)} kernels, {len(runs)} runs, {len(steps)} steps of {n} kernels")

span, busy, between = [], [], []
by = defaultdict(int)
for i in steps:
    st = runs[i]
    t0, t1 = st[0][0], max(r[1] for r in st)
    b, edge = 0, t0
    for s, e, name in st:
        by[name] += e - s
        if e > edge:
            b += e - max(s, edge)
            edge = e
    span.append(t1 - t0)
    busy.append(b)
    if i + 1 < len(runs) and len(runs[i + 1]) == n:
        between.append(runs[i + 1][0][0] - t1)
    elif i + 1 < len(runs):
        # kernels outside the graph (sampling, copies) sit between two steps
        j = i + 1
        while j < len(runs) and len(runs[j]) != n:
            j += 1
        if j < len(runs):
            between.append(runs[j][0][0] - t1)

med = statistics.median
print(f"step span {med(span) / 1e6:.3f} ms (median), busy {med(busy) / 1e6:.3f} ms, "
      f"idle inside {med([s - b for s, b in zip(span, busy)]) / 1e6:.3f} ms, "
      f"between steps {med(between) / 1e6:.3f} ms")
calls = Counter(name for _, _, name in runs[steps[0]])
total = sum(by.values()) / len(steps)
print(f"kernel time per step {total / 1e6:.3f} ms; per kernel name (per step):")
print(f"{'ms':>7} {'%':>5} {'calls':>5} {'µs/call':>8}  name")
for name, t in sorted(by.items(), key=lambda x: -x[1])[:top]:
    t /= len(steps)
    print(f"{t / 1e6:7.3f} {t / total * 100:5.1f} {calls[name]:5d} {t / calls[name] / 1e3:8.2f}  {name[:80]}")
rest = sorted(by.items(), key=lambda x: -x[1])[top:]
if rest:
    t = sum(v for _, v in rest) / len(steps)
    print(f"{t / 1e6:7.3f} {t / total * 100:5.1f} {sum(calls[k] for k, _ in rest):5d} {'':>8}  ({len(rest)} other kernels)")
