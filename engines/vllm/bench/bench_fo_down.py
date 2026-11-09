#!/usr/bin/env python3
"""T-023: down experts row-major vs fragment order on a real layer (BF16 GGUF), same uniform routing for both.
Decode: expert GEMV with FP32 per-slot activations, T tokens x 8 slots, kernel time per call from a CUDA graph of
REPS calls. Prefill: the batched kernel (BF16 per-slot activations), and the fused gate/up kernel for reference.
Prints ms per call and the largest |difference| between the layouts relative to sum |w x| (both FP32 sums).
    bench_fo_down.py [layer=20] [decode tokens=1,4] [prefill tokens=2048,4288]"""
import os
import sys

import numpy as np
import torch

GGUF = os.environ.get("BF16N_GGUF")
REPS = int(os.environ.get("REPS", "50"))
E, TOPK = 256, 8


def load(reader, name):
    t = next(t for t in reader.tensors if t.name == name)
    shape = [int(x) for x in reversed(t.shape.tolist())]  # GGUF ne is innermost first
    return torch.from_numpy(np.asarray(t.data).view(np.uint16).reshape(shape).copy()).view(torch.bfloat16)


def graph_ms(fn, reps=REPS):
    """Kernel time per call: a CUDA graph of reps calls, replayed after a 2 s warm-up, median of 7 replays."""
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
    while time.time() - t0 < 2.0:
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
    dec = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4").split(",")]
    pre = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2048,4288").split(",")]
    if not GGUF:
        raise SystemExit("Set BF16N_GGUF to a local BF16 GGUF checkpoint")
    torch.cuda.set_per_process_memory_fraction(0.5)
    ops.init()
    reader = GGUFReader(GGUF)
    down = load(reader, f"blk.{layer}.ffn_down_exps.weight")  # [E, 2048, 512]
    H, N = down.shape[1], down.shape[2]
    rm = ops.upload(*ops.encode(down, ops.FMT_HUFPAIR, 16, False, False), "cuda")
    fo = ops.upload(*ops.encode(down, ops.FMT_HUFPAIR, 16, False, True), "cuda")
    assert fo.meta.frag_order and not rm.meta.frag_order
    print(f"layer {layer} down {tuple(down.shape)}: row-major {rm.nbytes / 2**20:.1f} MiB, fragment order "
          f"{fo.nbytes / 2**20:.1f} MiB, round trip {ops.roundtrip_equal(fo, down)}")
    wd = down.cuda()
    g = torch.Generator().manual_seed(1234)

    def routing(T):
        return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32).cuda()

    def compare(y_rm, y_fo, ids, a):
        """max |y_fo - y_rm| / sum |w a| over a sample of slots (FP64 S)."""
        T = ids.shape[0]
        worst = 0.0
        for t in range(0, T, max(1, T // 16)):
            for k in range(TOPK):
                W = wd[ids[t, k]].double()
                S = W.abs() @ a[t, k].double().abs()
                d = (y_fo[t, k].double() - y_rm[t, k].double()).abs()
                worst = max(worst, (d / (S + 1e-30)).max().item())
        return worst

    for T in dec:
        ids = routing(T)
        a = torch.randn(T, TOPK, N, generator=g).cuda()
        t_rm = graph_ms(lambda: ops.mmv(a, rm, ids, H))
        t_fo = graph_ms(lambda: ops.mmv(a, fo, ids, H))
        d = compare(ops.mmv(a, rm, ids, H), ops.mmv(a, fo, ids, H), ids, a)
        print(f"GEMV  T={T:5d} ({T * TOPK} slots): row-major {t_rm * 1e3:7.1f} us, fragment order {t_fo * 1e3:7.1f} us "
              f"({t_fo / t_rm - 1:+.1%}), max diff / S {d:.2e}")
    if pre:
        gate = load(reader, f"blk.{layer}.ffn_gate_exps.weight")
        up = load(reader, f"blk.{layer}.ffn_up_exps.weight")
        eg = ops.upload(*ops.encode(gate, ops.FMT_HUFPAIR, 8, True), "cuda")
        eu = ops.upload(*ops.encode(up, ops.FMT_HUFPAIR, 8, True), "cuda")
        del gate, up
    for T in pre:
        ids = routing(T)
        a = torch.randn(T, TOPK, N, generator=g).to(torch.bfloat16).cuda()
        x = torch.randn(T, H, generator=g).to(torch.bfloat16).cuda()
        t_rm = graph_ms(lambda: ops.moe_mm_ids(a, ids, rm, E), reps=5)
        t_fo = graph_ms(lambda: ops.moe_mm_ids(a, ids, fo, E), reps=5)
        t_gu = graph_ms(lambda: ops.moe_mm_ids(x, ids, eg, E, up=eu, out_dtype=torch.bfloat16), reps=5)
        d = compare(ops.moe_mm_ids(a, ids, rm, E), ops.moe_mm_ids(a, ids, fo, E), ids, a)
        print(f"batch T={T:5d}: down row-major {t_rm:6.2f} ms, fragment order {t_fo:6.2f} ms ({t_fo / t_rm - 1:+.1%}); "
              f"gate/up {t_gu:6.2f} ms; max diff / S {d:.2e}")


if __name__ == "__main__":
    main()
