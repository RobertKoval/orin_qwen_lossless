#!/usr/bin/env python3
"""Kernel time by name in an nsys sqlite (whole capture): total ms, calls, share.  kern_summary.py prof.sqlite [top=25]"""
import collections, re, sqlite3, sys
c = sqlite3.connect(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
rows = c.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName").fetchall()
t0, t1 = min(r[0] for r in rows), max(r[1] for r in rows)
by, n = collections.Counter(), collections.Counter()
for s, e, name in rows:
    k = re.sub(r"\(.*", "", name).replace("void ", "").replace("bf16n::kern::", "")
    k = re.sub(r"<.*", "<>", k) if len(k) > 90 else k
    by[k] += e - s; n[k] += 1
tot = sum(by.values())
print(f"span {(t1 - t0) / 1e6:.1f} ms, kernel time {tot / 1e6:.1f} ms")
for k, v in by.most_common(top):
    print(f"{v / 1e6:9.2f} ms {100 * v / tot:5.1f} % {n[k]:6d}  {k[:110]}")
