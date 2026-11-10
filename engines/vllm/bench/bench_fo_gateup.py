#!/usr/bin/env python3
"""T-060 b2: routed gate/up experts chunk-major (R = 8, k_hp_mm_ids GLU) vs fragment order (R = 16, k_fo_mm_glu) on a
real layer (BF16 GGUF), all experts, the same routing for both. Prefill: the batched gate/up + SWIGLU kernel at the
given token counts, uniform routing and a skewed one (a fixed per-expert logit bias, std SKEW), kernel time per call
from a CUDA graph. Accuracy: max |y - ref| / max |ref| against an FP64 reference over sampled (token, slot) pairs.
    bench_fo_gateup.py [layer=20] [prefill tokens=512,1072,2048,4288,6432]"""
import os
import sys

import numpy as np
import torch

GGUF = os.environ.get("BF16N_GGUF", "/media/external_storage/bf16n-models/qwen3.6-35b-a3b/qwen3.6-35b-a3b-bf16.gguf")
REPS = int(os.environ.get("REPS", "3"))
SKEW = float(os.environ.get("SKEW", "1.0"))
E, TOPK = 256, 8


def load(reader, name):
    t = next(t for t in reader.tensors if t.name == name)
    shape = [int(x) for x in reversed(t.shape.tolist())]  # GGUF ne is innermost first
    return torch.from_numpy(np.asarray(t.data).view(np.uint16).reshape(shape).copy()).view(torch.bfloat16)


def graph_ms(fn, reps=REPS):
    """Kernel time per call: a CUDA graph of reps calls, replayed after a 1 s warm-up, median of 7 replays."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    import time
    t0 = time.time()
    while time.time() - t0 < 1.0:
        g.replay()
        torch.cuda.synchronize()
    ms = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b) / reps)
    return sorted(ms)[3]


def main():
    from gguf import GGUFReader

    from vllm_bf16n import ops

    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pre = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,1072,2048,4288,6432").split(",")]
    ops.init()
    reader = GGUFReader(GGUF)
    gate = load(reader, f"blk.{layer}.ffn_gate_exps.weight")  # [E, 512, 2048]
    up = load(reader, f"blk.{layer}.ffn_up_exps.weight")
    N, H = gate.shape[1], gate.shape[2]
    enc = {}
    for tag, R, cm, fo in (("cm", 8, True, False), ("fo", 16, False, True)):
        eg = ops.upload(*ops.encode(gate, ops.FMT_HUFPAIR, R, cm, fo), "cuda")
        eu = ops.upload(*ops.encode(up, ops.FMT_HUFPAIR, R, cm, fo), "cuda")
        assert ops.mm_ids_supported(eg, eu), tag
        enc[tag] = (eg, eu)
        print(f"{tag}: gate {eg.nbytes / 2**20:.1f} MiB, up {eu.nbytes / 2**20:.1f} MiB, round trip "
              f"{ops.roundtrip_equal(eg, gate) and ops.roundtrip_equal(eu, up)}", flush=True)
    g = torch.Generator().manual_seed(1234)
    bias = torch.randn(E, generator=g) * SKEW

    def routing(T, skew):
        logits = torch.rand(T, E, generator=g).log().neg().log().neg()  # Gumbel: top-k of logits = sampling without replacement
        if skew:
            logits = logits + bias
        return logits.topk(TOPK, dim=1).indices.to(torch.int32).cuda()

    def ref_err(y, ids, x, n=24):
        worst = 0.0
        for i in range(n):
            t, k = (i * 7919) % ids.shape[0], i % TOPK
            e = ids[t, k].item()
            xv = x[t].double().cpu()
            gv = gate[e].double() @ xv
            uv = up[e].double() @ xv
            r = torch.nn.functional.silu(gv) * uv
            worst = max(worst, ((y[t, k].double().cpu() - r).abs().max() / r.abs().max()).item())
        return worst

    for T in pre:
        for skew in (False, True):
            ids = routing(T, skew)
            cnt = torch.bincount(ids.flatten().long(), minlength=E)
            x = torch.randn(T, H, generator=g).to(torch.bfloat16).cuda()
            t = {tag: graph_ms(lambda: ops.moe_mm_ids(x, ids, eg, E, up=eu, out_dtype=torch.bfloat16))
                 for tag, (eg, eu) in enc.items()}
            ys = {tag: ops.moe_mm_ids(x, ids, eg, E, up=eu, out_dtype=torch.float32) for tag, (eg, eu) in enc.items()}
            err = {tag: ref_err(y, ids, x) for tag, y in ys.items()}
            print(f"T={T:5d} {'skew' if skew else 'unif'} (per expert mean {cnt.float().mean():.0f} max {cnt.max()}): "
                  f"cm {t['cm']:6.2f} ms, fo {t['fo']:6.2f} ms ({t['fo'] / t['cm'] - 1:+.1%}); err cm {err['cm']:.1e} "
                  f"fo {err['fo']:.1e}", flush=True)


if __name__ == "__main__":
    main()
