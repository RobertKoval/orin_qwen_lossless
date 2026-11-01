#!/usr/bin/env python3
"""Per-step decode timeline from an nsys sqlite (vLLM + BF16N, MTP or plain).
Groups consecutive kernels into phases separated by GPU idle gaps > --gap µs, labels each phase by its dominant
kernels, and prints one representative step (median-length) plus per-phase medians over all steps.
A step starts at the target forward, recognised as the first phase with >= 60 expert-GEMV kernels (k_hp_mmv_cm).
  step_timeline.py prof.sqlite [--gap 20] [--show 1]"""
import collections, re, sqlite3, statistics, sys

db = sys.argv[1]
gap_us = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 20.0
c = sqlite3.connect(db)
rows = c.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
                 "ORDER BY k.start").fetchall()
phases, cur, end = [], [rows[0]], rows[0][1]
for r in rows[1:]:
    if r[0] - end > gap_us * 1e3:
        phases.append((cur, r[0] - end)); cur = []
    cur.append(r); end = max(end, r[1])
phases.append((cur, 0))


def label(ks):
    n = collections.Counter(re.sub(r"<.*", "", k[2]) for k in ks)
    cm = n.get("k_hp_mmv_cm", 0)
    lm = sum(1 for k in ks if k[2].startswith("k_hp_mmv") and k[1] - k[0] > 3e6)
    if cm >= 60:
        return "target"
    if cm > 0:
        return "draft-layer" + ("+lm_head" if lm else "")
    if lm:
        return "draft-lm_head"
    top = ",".join(f"{k}x{v}" for k, v in n.most_common(2))
    return "small[" + top[:60] + "]"


steps, s = [], None
by_phase = collections.defaultdict(collections.Counter)
for ks, g in phases:
    lab = label(ks)
    if lab == "target":
        s = []; steps.append(s)
    if s is not None:
        key = "small" if lab.startswith("small") else lab
        for k in ks:
            by_phase[key][re.sub(r"<.*", "", k[2])[:50]] += k[1] - k[0]
    if s is not None:
        s.append((lab, (ks[-1][1] - ks[0][0]) / 1e6, sum(k[1] - k[0] for k in ks) / 1e6, len(ks), g / 1e6))
full = [x for x in steps[1:-1]]
spans = [sum(p[1] + p[4] for p in st) for st in full]
print(f"{len(full)} steps, step wall (kernels + gaps) median {statistics.median(spans):.2f} ms")
rep = sorted(full, key=lambda st: abs(sum(p[1] + p[4] for p in st) - statistics.median(spans)))[0]
print(f"{'phase':40s} {'span ms':>8s} {'busy ms':>8s} {'kern':>5s} {'gap after ms':>12s}")
for p in rep:
    print(f"{p[0][:40]:40s} {p[1]:8.2f} {p[2]:8.2f} {p[3]:5d} {p[4]:12.2f}")
agg = collections.defaultdict(list)
for st in full:
    tot = collections.Counter()
    for p in st:
        tot[p[0] if not p[0].startswith("small") else "small kernels"] += p[2]
        tot["gaps"] += p[4]
    for k, v in tot.items():
        agg[k].append(v)
print("per-step medians:", {k: round(statistics.median(v), 2) for k, v in agg.items()})

if "--kernels" in sys.argv:
    for ph, cnt in by_phase.items():
        print(f"== {ph}: per-step kernel ms")
        for k, v in cnt.most_common(12):
            print(f"   {v / len(steps) / 1e6:7.2f}  {k}")
