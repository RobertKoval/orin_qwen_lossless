#!/usr/bin/env python3
"""Summarise py-spy raw (collapsed) stacks of the engine core: self time and inclusive time per function, only for
the busy thread.  pyspy_summary.py out/<tag>-pyspy.txt [n=40] [filter substring for inclusive view]"""
import collections, sys

lines = [l.rsplit(" ", 1) for l in open(sys.argv[1]) if l.strip()]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
threads = collections.Counter()
for st, c in lines:
    threads[st.split(";")[0]] += int(c)
main = "<module> (<string>:1)" if "<module> (<string>:1)" in threads else threads.most_common(1)[0][0]
tot = threads[main]
self_t, incl = collections.Counter(), collections.Counter()
for st, c in lines:
    fr = st.split(";")
    if fr[0] != main:
        continue
    c = int(c)
    self_t[fr[-1]] += c
    for f in set(fr[1:]):
        incl[f] += c
print("threads:", [(t[:40], v) for t, v in threads.most_common(4)], "main samples", tot)
print("== inclusive")
for f, v in incl.most_common(n):
    if len(sys.argv) > 3 and sys.argv[3] not in f:
        continue
    print(f"{100 * v / tot:6.1f}%  {f[:150]}")
print("== self")
for f, v in self_t.most_common(25):
    print(f"{100 * v / tot:6.1f}%  {f[:150]}")
