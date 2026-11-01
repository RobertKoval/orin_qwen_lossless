#!/usr/bin/env python3
"""Dense HUFPAIR GEMV time vs rows-per-unit R and token count, on the real merged vLLM projections of Qwen3.6
(GDN in_proj_qkvz 12288 x 2048, attention qkv 9216 x 2048 incl. the output gate, out projections 2048 x 4096).
With MTP 3 the target's dense layers run at T = 4, so the policy's T = 1 wave model may not pick the best R there.
  bench_dense_R.py [T list=1,4]"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from bench_ops import GGUF, load  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from vllm_bf16n import ops  # noqa: E402

Ts = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "1,4").split(",")]
torch.cuda.set_per_process_memory_fraction(0.4)
ops.init()
r = GGUFReader(GGUF)
mats = {"lm_head 248320x2048": load(r, "output.weight")} if os.environ.get("LMHEAD") else {
    "qkvz 12288x2048": torch.cat([load(r, "blk.0.attn_qkv.weight"), load(r, "blk.0.attn_gate.weight")]),
    "attn qkv 9216x2048": torch.cat([load(r, "blk.3.attn_q.weight"), load(r, "blk.3.attn_k.weight"), load(r, "blk.3.attn_v.weight")]),
    "ssm_out 2048x4096": load(r, "blk.0.ssm_out.weight"),
    "attn_out 2048x4096": load(r, "blk.3.attn_output.weight"),
}


def tm(f, n=200):
    for _ in range(30): f()
    torch.cuda.synchronize()
    e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    e[0].record()
    for _ in range(n): f()
    e[1].record(); torch.cuda.synchronize()
    return e[0].elapsed_time(e[1]) / n * 1e3


for name, w in mats.items():
    rows = w.shape[0]
    for R in [int(x) for x in os.environ.get("RS", "16,8,4,2").split(",")]:
        if rows % R:
            continue
        W = ops.upload(*ops.encode(w, ops.FMT_HUFPAIR, R, False), "cuda")
        res = []
        for T in Ts:
            x = torch.randn(T, w.shape[1], device="cuda").to(torch.bfloat16)
            res.append(f"T={T} {tm(lambda: ops.linear(x, W)):6.1f} us")
        print(f"{name:20s} R={R:2d} " + "  ".join(res), flush=True)
