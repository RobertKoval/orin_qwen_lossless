#!/usr/bin/env python3
"""Inclusive time tree under a frame of the engine-core main thread from py-spy raw stacks, in ms per decode step.
  pyspy_tree.py out/<tag>-pyspy.txt <root substring> <ms per sample * steps scaling: total_seconds> <steps> [depth=3] [min%=0.5]"""
import collections, sys

path, root, secs, steps = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
depth = int(sys.argv[5]) if len(sys.argv) > 5 else 3
minpct = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
lines = [l.rsplit(" ", 1) for l in open(path) if l.strip()]
main = "<module> (<string>:1)"
tot = sum(int(c) for st, c in lines if st.split(";")[0] == main)
ms_per_sample = secs * 1000 / tot
tree = collections.Counter()
for st, c in lines:
    fr = st.split(";")
    if fr[0] != main:
        continue
    for i, f in enumerate(fr):
        if root in f:
            sub = fr[i + 1:i + 1 + depth]
            for d in range(len(sub) + 1):
                tree[tuple(sub[:d])] += int(c)
            break
base = tree[()]
print(f"{root}: {base * ms_per_sample / steps:.2f} ms/step ({100 * base / tot:.1f}% of main thread)")
for k, v in sorted(tree.items()):
    if k and 100 * v / base >= minpct:
        print("  " * len(k) + f"{v * ms_per_sample / steps:6.2f}  {k[-1][:110]}")
