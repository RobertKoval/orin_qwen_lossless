#!/usr/bin/env python3
"""CUDA runtime API calls (host side) within one decode step of an nsys sqlite: per-thread call counts/time and the
long calls, to locate host gaps. A step = interval between successive bursts of k_hp_mmv_cm (target forwards).
  host_calls.py prof.sqlite [step index, default middle]"""
import collections, sqlite3, statistics, sys

c = sqlite3.connect(sys.argv[1])
S = dict(c.execute("select id, value from StringIds"))
rt = c.execute("select start, end, nameId, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
k = c.execute("select k.start, k.end, k.correlationId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
              "join StringIds s on s.id=k.shortName order by k.start").fetchall()
cm = [x for x in k if x[3].startswith("k_hp_mmv_cm")]
starts = [cm[0][0]]
for a, b in zip(cm, cm[1:]):
    if b[0] - a[1] > 8e6:
        starts.append(b[0])
print("steps", len(starts), "median period ms %.2f" % statistics.median([(b - a) / 1e6 for a, b in zip(starts, starts[1:])]))
tids = collections.Counter(r[3] for r in rt)
main = tids.most_common(1)[0][0]
i = int(sys.argv[2]) if len(sys.argv) > 2 else len(starts) // 2
t0, t1 = starts[i], starts[i + 1]
calls = [r for r in rt if t0 <= r[0] < t1]
agg = collections.defaultdict(lambda: [0, 0.0])
for r in calls:
    key = ("main" if r[3] == main else "t%d" % (r[3] % 100000), S[r[2]])
    agg[key][0] += 1
    agg[key][1] += (r[1] - r[0]) / 1e6
for (th, n), (cnt, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
    print(f"{th:7s} {n[:44]:44s} {cnt:5d} {ms:8.2f} ms")
kc = {x[2]: x for x in k}
print("--- calls > 0.25 ms (offset ms, dur ms); and kernel launches: first/last offset")
for r in calls:
    if r[1] - r[0] > 0.25e6:
        print(f"{(r[0] - t0) / 1e6:8.2f} {(r[1] - r[0]) / 1e6:7.2f} {S[r[2]][:44]}")
