#!/usr/bin/env python3
"""Per-kernel GPU durations from an nsys sqlite export.

    kern_stats.py prof.sqlite [--nvtx] [--match REGEX]

--nvtx: attribute kernels to the NVTX range their launch (runtime API call) falls in (bench_ops.py names one range per
case) and print the median duration per (range, kernel). Without it, per kernel name over the whole run (ggml_ref.cpp
runs one case per process; the median ignores the warm-up's first launches).
"""
import re
import sqlite3
import statistics
import sys
from collections import defaultdict


def main():
    db = sys.argv[1]
    nvtx = "--nvtx" in sys.argv
    pat = re.compile(sys.argv[sys.argv.index("--match") + 1]) if "--match" in sys.argv else None
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    kern = con.execute(
        "SELECT k.start, k.end, k.correlationId, k.demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL k ORDER BY k.start"
    ).fetchall()
    by = defaultdict(list)
    if nvtx:
        launch = dict(con.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
        ranges = con.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL").fetchall()
        for s, e, cid, nid in kern:
            t = launch.get(cid)
            r = next((txt for rs, re_, txt in ranges if t is not None and rs <= t <= re_), None)
            if r is not None:
                by[(r, short(names[nid]))].append(e - s)
    else:
        for s, e, cid, nid in kern:
            by[("", short(names[nid]))].append(e - s)
    for (r, k), d in sorted(by.items(), key=lambda kv: (kv[0][0], -sum(kv[1]))):
        if pat and not pat.search(k):
            continue
        print(f"{r:45s} {statistics.median(d) / 1e3:9.2f} us  n={len(d):5d}  {k}")


def short(n):
    n = n.replace("bf16n::kern::", "").replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    n = n.replace("__nv_bfloat16", "bf16").replace("(int)", "").replace("(bool)1", "true").replace("(bool)0", "false")
    return re.sub(r"\(.*", "", n).replace("void ", "")


if __name__ == "__main__":
    main()
