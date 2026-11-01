#!/usr/bin/env python3
"""Expert GEMV of a multi-token decode batch (MTP verification, T tokens x 8 experts) with the (token, slot) pairs in
token order (as routed) vs sorted by expert, so that pairs sharing an expert run back to back and the second decode
finds the expert's plane in L2 (4 MB on Orin). Real layer-0 experts; routing with a given fraction of distinct
experts (0.64 measured in M6). Same per-pair arithmetic, so the outputs must be bitwise equal.
  bench_expert_order.py [T=4] [distinct=0.64]"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from bench_ops import GGUF, load  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from vllm_bf16n import ops  # noqa: E402
from vllm_bf16n.policy import choose  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.64
torch.cuda.set_per_process_memory_fraction(0.4)
ops.init()
r = GGUFReader(GGUF)
W = {}
for n in ("gate", "down"):
    w = load(r, f"blk.0.ffn_{n}_exps.weight")
    c = choose(w.shape[-2], w.shape[-1], w.shape[0])
    W[n] = ops.upload(*ops.encode(w, c.format, c.R, c.chunk_major), "cuda")
g = torch.Generator().manual_seed(0)
best = None
for p in [i / 20 for i in range(21)]:
    rows = [torch.randperm(256, generator=g)[:8].tolist()]
    for _ in range(T - 1):
        row = [e for e in rows[-1] if torch.rand(1, generator=g).item() < p]
        for e in torch.randperm(256, generator=g).tolist():
            if len(row) == 8: break
            if e not in row: row.append(e)
        rows.append(row)
    d = len({e for row in rows for e in row}) / (8 * T)
    if best is None or abs(d - target) < abs(best[0] - target): best = (d, rows)
d, rows = best
ids = torch.tensor(rows, dtype=torch.int32, device="cuda")
flat = ids.reshape(-1)
perm = torch.argsort(flat, stable=True)
ids_s = flat[perm].view(-1, 1)
tok = perm // 8
x = torch.randn(T, 2048, generator=g).to(torch.bfloat16).cuda()
xd = torch.randn(T, 8, 512, generator=g).cuda()


def tm(f, n=200):
    for _ in range(20): f()
    torch.cuda.synchronize(); e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    e[0].record()
    for _ in range(n): f()
    e[1].record(); torch.cuda.synchronize()
    return e[0].elapsed_time(e[1]) / n * 1e3


print(f"T={T} distinct={d:.3f}")
a = ops.mmv(x, W["gate"], ids, 512); b = ops.mmv(x[tok], W["gate"], ids_s, 512)
inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device="cuda")
assert torch.equal(a.view(-1, 512), b.view(-1, 512)[inv])
print(f"gate: token order {tm(lambda: ops.mmv(x, W['gate'], ids, 512)):6.1f} us, "
      f"expert order {tm(lambda: ops.mmv(x[tok], W['gate'], ids_s, 512)):6.1f} us")
xs = xd.view(-1, 512)
a = ops.mmv(xd, W["down"], ids, 2048); b = ops.mmv(xs[perm], W["down"], ids_s, 2048)
assert torch.equal(a.view(-1, 2048), b.view(-1, 2048)[inv])
print(f"down: token order {tm(lambda: ops.mmv(xd, W['down'], ids, 2048)):6.1f} us, "
      f"expert order {tm(lambda: ops.mmv(xs[perm], W['down'], ids_s, 2048)):6.1f} us")
