#!/usr/bin/env python3
"""Compare two run_prompts.sh outputs: identical text, bitwise-identical top-5 logprobs, first top-1 divergence."""
import json
import statistics
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
top1 = []
for i, (x, y) in enumerate(zip(a, b)):
    n = min(len(x["probs"]), len(y["probs"]))
    div = next((k for k in range(n) if x["probs"][k][0][0] != y["probs"][k][0][0]), None)
    for px, py in zip(x["probs"], y["probs"]):
        dy = dict(py)
        tok, lp = px[0]
        if tok in dy:
            top1.append(abs(lp - dy[tok]))
    print("prompt%d: text_identical=%s logprobs_bitwise_identical=%s first_top1_divergence=%s tok/s %.2f vs %.2f" % (
        i, x["content"] == y["content"], x["probs"] == y["probs"], div,
        x["timings"]["predicted_per_second"], y["timings"]["predicted_per_second"]))
if top1:
    s = sorted(top1)
    print("top-1 |dlogprob|: median %.2e  p99 %.2e  max %.2e  over %d tokens" % (
        statistics.median(s), s[int(0.99 * (len(s) - 1))], s[-1], len(s)))
