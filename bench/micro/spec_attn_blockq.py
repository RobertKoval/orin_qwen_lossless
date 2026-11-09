"""T-063 microbench: vLLM's split-KV ("3D") Triton attention for one Qwen3.6 attention layer (16 query heads, 2 KV
heads, head_dim 256) at nq query tokens of one sequence: vLLM's launch (16 segments, BLOCK_M = 16 rows = 2 tokens per
program) against 16 / 32 / 64 segments and, for nq > 2, vllm_bf16n.triton_spec's BLOCK_Q = next_pow2(nq) tokens per
program. Prints kernel time (median of 30) and the largest |difference| from vLLM's launch (0 = bitwise equal).
  python3 spec_attn_blockq.py [contexts...]   (default 65536 200000)
"""
import sys

import torch

HQ, HKV, D = 16, 2, 256
BS = 1072  # production block size (attention layers)
SEGMENTS = 16


def time_ms(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[len(ts) // 2]


def main():
    from vllm.v1.attention.backends import triton_attn
    from vllm_bf16n import triton_spec

    triton_spec.install()
    ua = triton_attn.unified_attention  # the vllm_bf16n wrapper
    lens = [int(x) for x in sys.argv[1:]] or [65536, 200000]
    dev = "cuda"
    torch.manual_seed(0)
    for L in lens:
        nb = (L + BS - 1) // BS
        kc = (torch.randn(nb, BS, HKV, D, device=dev) * 0.5).to(torch.bfloat16)
        vc = (torch.randn(nb, BS, HKV, D, device=dev) * 0.5).to(torch.bfloat16)
        bt = torch.arange(nb, device=dev, dtype=torch.int32)[None]
        used = torch.tensor([L], device=dev, dtype=torch.int32)
        bufs = {n: (torch.empty(64, HQ, n, D, device=dev), torch.empty(64, HQ, n, device=dev),
                    torch.empty(64, HQ, n, device=dev)) for n in (16, 32, 64)}
        for nq in (1, 2, 4):
            q = (torch.randn(nq, HQ, D, device=dev) * 0.5).to(torch.bfloat16)
            cu = torch.tensor([0, nq], device=dev, dtype=torch.int32)
            outs = {}
            times = {}
            runs = [("stock", 1, SEGMENTS)]  # max_seqlen_q = 1: vLLM's own 3D launch (BLOCK_M 16)
            for nseg in (16, 32, 64):
                runs += [(f"{nseg} segments, BLOCK_Q 2", 1, nseg)] if nseg != SEGMENTS else []
                runs += [(f"{nseg} segments, BLOCK_Q {1 << (nq - 1).bit_length()}", nq, nseg)] if nq > 2 else []
            for name, mq, nseg in runs:
                out = torch.empty_like(q)
                so, sm, se = bufs[nseg]

                def call():
                    ua(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=mq, seqused_k=used, max_seqlen_k=L,
                       softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                       q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
                       num_par_softmax_segments=nseg, softmax_segm_output=so, softmax_segm_max=sm,
                       softmax_segm_expsum=se)
                times[name] = time_ms(call)
                outs[name] = out.clone()
            for name in times:
                if name == "stock":
                    continue
                d = (outs["stock"].float() - outs[name].float()).abs().max().item()
                print(f"L={L:6d} nq={nq}: stock {times['stock']:.3f} ms | {name}: {times[name]:.3f} ms "
                      f"({times[name] / times['stock'] - 1:+.1%}), max |diff| {d:.1e}", flush=True)
        del kc, vc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
