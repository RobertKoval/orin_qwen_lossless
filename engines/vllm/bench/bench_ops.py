#!/usr/bin/env python3
"""M1 microbenchmark: torch.ops.bf16n at production shapes on the same real tensors as the llama.cpp reference
(bench/xengine/ggml_ref.cpp), read from the BF16 GGUF. Each case runs `reps` back-to-back launches after a 2 s
warm-up (GPU clock governor); run under nsys and take per-kernel durations (engines/vllm/bench/kern_stats.py):

    nsys profile -o ours --force-overwrite true python engines/vllm/bench/bench_ops.py [case ...]

Cases are named <tensor>@<tokens>[:f32] (f32 = FP32 activations, the llama.cpp input type, to isolate BF16 staging).
NVTX ranges carry the case names so kernel times can be attributed per case.
"""
import os
import sys
import time

import numpy as np
import torch

GGUF = os.environ.get("BF16N_GGUF")
REPS = int(os.environ.get("REPS", "50"))
E, TOPK = 256, 8

DEFAULT_CASES = [
    "blk.0.attn_qkv.weight@1", "blk.0.attn_qkv.weight@1:f32", "blk.0.attn_gate.weight@1",
    "blk.0.attn_gate.weight@1:f32", "blk.0.ssm_out.weight@1", "blk.0.ssm_out.weight@1:f32",
    "blk.0.ffn_gate_exps.weight@1", "blk.0.ffn_gate_exps.weight@1:f32", "blk.0.ffn_down_exps.weight@1",
    "blk.0.ffn_down_exps.weight@1:f32", "output.weight@1", "output.weight@1:f32", "blk.0.ssm_alpha.weight@1",
    "blk.0.ssm_alpha.weight@1:f32", "blk.0.ffn_down_shexp.weight@1", "blk.0.ffn_down_shexp.weight@1:f32",
    "blk.0.ffn_gate_up_shexp.weight@1", "blk.0.ffn_gate_up_shexp.weight@1:f32",
    "blk.0.attn_qkv.weight@2", "blk.0.attn_qkv.weight@2:f32", "blk.0.ffn_gate_exps.weight@2",
    "blk.0.ffn_gate_up_exps.weight@512", "blk.0.ffn_gate_up_exps.weight@2048", "blk.0.ffn_down_exps.weight@512",
    "blk.0.ffn_down_exps.weight@2048",
]


def load(reader, name):
    t = next(t for t in reader.tensors if t.name == name)
    shape = [int(x) for x in reversed(t.shape.tolist())]  # GGUF ne is innermost first
    raw = np.asarray(t.data).view(np.uint16).reshape(shape)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def main():
    from gguf import GGUFReader

    from vllm_bf16n import ops
    from vllm_bf16n.policy import choose

    cases = sys.argv[1:] or DEFAULT_CASES
    if not GGUF:
        raise SystemExit("Set BF16N_GGUF to a local BF16 GGUF checkpoint")
    reader = GGUFReader(GGUF)
    ops.init()
    cache = {}

    def enc(name):
        if name not in cache:
            if name.endswith("gate_up_shexp.weight"):  # vLLM's merged [gate; up] NIBBLE matrix
                w = torch.cat([load(reader, name.replace("gate_up", "gate")), load(reader, name.replace("gate_up", "up"))])
            else:
                w = load(reader, name)
            n_mat = w.shape[0] if w.dim() == 3 else 1
            c = choose(w.shape[-2], w.shape[-1], n_mat)
            plane, aux, meta = ops.encode(w, c.format, c.R, c.chunk_major)
            cache[name] = ops.upload(plane, aux, meta, "cuda")
        return cache[name]

    g = torch.Generator().manual_seed(1234)
    for case in cases:
        spec, _, xt = case.partition(":")
        name, _, T = spec.partition("@")
        T = int(T)
        xdt = torch.float32 if xt == "f32" else torch.bfloat16
        if "gate_up_exps" in name:
            fn = run_gate_up(ops, enc(name.replace("gate_up", "gate")), enc(name.replace("gate_up", "up")), T, g)
        else:
            fn = run(ops, enc(name), name, T, xdt, g)
        stream = torch.cuda.current_stream()
        t0 = time.time()
        while time.time() - t0 < 2.0:  # warm-up, as ggml_ref (clocks are pinned, but keep the conditions equal)
            fn()
            stream.synchronize()
        # Every launch is followed by cudaStreamSynchronize, as in ggml_ref (ggml_backend_synchronize). Waiting with
        # cudaDeviceSynchronize (torch.cuda.synchronize) under the default context flags blocks the host thread, and
        # that alone slows long bandwidth-bound kernels on Orin by up to ~35 % (lm_head 4.22 -> 5.1-5.8 ms; M1 finding).
        torch.cuda.nvtx.range_push(case)
        times = []
        for _ in range(REPS):
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            fn()
            ev1.record()
            stream.synchronize()
            times.append(ev0.elapsed_time(ev1))
        torch.cuda.nvtx.range_pop()
        times.sort()
        print(f"{case:45s} {times[len(times) // 2] * 1e3:9.1f} us/call (events, median)", flush=True)


def routing(T, g):
    if os.environ.get("BF16N_IDS_SEQ"):  # the deterministic routing of ggml_ref: slot j of all tokens -> expert j % E
        return (torch.arange(T * TOPK) % E).view(T, TOPK).to(torch.int32).cuda()
    return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32).cuda()


def run(ops, w, name, T, xdt, g):
    m = w.meta
    if "_exps" in name:
        n_e = E
        rpm = m.rows // n_e
        ids = routing(T, g)
        if "down" in name:  # one activation row per routed slot
            x = torch.randn(T, TOPK, m.K, generator=g).to(xdt).cuda()
        else:
            x = torch.randn(T, m.K, generator=g).to(xdt).cuda()
        if T > ops.GEMV_MAX_TOKENS:
            xb = x.to(torch.bfloat16)
            return lambda: ops.moe_mm_ids(xb, ids, w, n_e)
        return lambda: ops.mmv(x, w, ids, rpm)
    x = torch.randn(T, m.K, generator=g).to(xdt).cuda()
    if "gate_up_shexp" in name:
        return lambda: ops.mmv_glu(x, w, out_dtype=torch.bfloat16)
    return lambda: ops.mmv(x, w, out_dtype=torch.bfloat16 if xdt == torch.bfloat16 else torch.float32)


def run_gate_up(ops, gate, up, T, g):
    ids = routing(T, g)
    x = torch.randn(T, gate.meta.K, generator=g).to(torch.bfloat16).cuda()
    return lambda: ops.moe_mm_ids(x, ids, gate, E, up=up, out_dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
