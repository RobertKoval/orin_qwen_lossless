"""T-101 microbench: one prefill chunk of one Qwen3.6 attention layer (16 query heads, 2 KV heads, head_dim 256, causal,
queries at the end of the context, BF16 paged KV in vLLM's TRITON_ATTN layout [blocks, 2, 1072, 2, 256]):
  fi        FlashInfer BatchPrefillWithPagedKVCacheWrapper planned as production does (vllm_bf16n.triton_spec.fi_plan)
  fa2       vLLM's FA2 (vllm_flash_attn, fa_version=2) over the same block table
  fa2-contig  the same FA2 kernel on contiguous K / V (the paging cost); fa2-p16: FA2 on the cache as 16-token pages;
  fa2-gather  index_select the pages into a contiguous K / V scratch, then fa2-contig (gather included in the time)
  fi-ragged FlashInfer's ragged (contiguous K / V) prefill: FlashInfer's paging cost
  fa2-route the production implementation, vllm_bf16n.triton_spec.fa2_prefill (--kv-per-call KV heads per FA2
            call, default all; FP32 Triton merge)
  fa2-seg   the cached prefix in --seg-pages segments (gathered into a small scratch, FA2 non-causal) plus the chunk's own
            keys (causal), partial outputs merged by LSE in FP32; fa2-seg-nomerge: the FA2 calls and gathers alone
  triton    vLLM's unified_attention (2D launch, what TRITON_ATTN runs without the FlashInfer route)
Prints kernel time (median of REPS), TFLOPS (causal FLOPs), max |difference| from fi over the whole output and the
largest error of each kernel against an FP32 reference on sampled query rows.
  python3 prefill_attn.py [--nq 6432] [--reps 10] [--kernels fi,fa2,...] [context ...]   (context = cached + nq tokens;
  default 109344 = the 17th 6432-token chunk of the T-100 111K prefill)
"""
import argparse
import os

try:  # next to production the OOM killer must take this process, not the EngineCore
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
except OSError:
    pass

import torch

HQ, HKV, D = 16, 2, 256
BS = 1072  # production block size of the attention layers (mamba align mode)
SCALE = D ** -0.5


def time_ms(fn, reps):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[len(ts) // 2]


def causal_flops(nq, L):
    prior = L - nq  # query i sees prior + i + 1 keys; QK^T and PV are 2 * D FLOPs per (query, key, head) each
    return 4 * D * HQ * (nq * prior + nq * (nq + 1) // 2)


def reference_rows(q, k, v, L, nq, rows):
    """FP32 attention for the sampled query rows (contiguous k, v: [L, HKV, D])."""
    out = torch.empty(len(rows), HQ, D, device=q.device)
    for g in range(HKV):  # one KV head in FP32 at a time (memory next to production)
        kf, vf = k[:, g].float(), v[:, g].float()
        for j, r in enumerate(rows):
            n = L - nq + r + 1
            for h in range(g * (HQ // HKV), (g + 1) * (HQ // HKV)):
                s = (kf[:n] @ q[r, h].float()) * SCALE
                out[j, h] = torch.softmax(s, 0) @ vf[:n]
        del kf, vf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("context", nargs="*", type=int)
    ap.add_argument("--nq", type=int, default=6432)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--kernels", default="fi,fa2,fa2-contig,triton")
    ap.add_argument("--seg-pages", type=int, default=30, help="fa2-seg: pages per prefix segment (30 = 32160 tokens)")
    ap.add_argument("--kv-per-call", type=int, default=HKV, help="fa2-route: KV heads per FA2 call")
    a = ap.parse_args()
    kernels = a.kernels.split(",")
    dev = "cuda"
    torch.manual_seed(0)
    for L in a.context or [109344]:
        nq = a.nq
        nb = (L + BS - 1) // BS
        kv = (torch.randn(nb, 2, BS, HKV, D, device=dev) * 0.5).to(torch.bfloat16)
        bt = torch.randperm(nb, device=dev, dtype=torch.int64).to(torch.int32)[None]  # pages out of order
        q = (torch.randn(nq, HQ, D, device=dev) * 0.5).to(torch.bfloat16)
        k_lin = kv[bt[0].long(), 0].reshape(-1, HKV, D)[:L]  # the sequence's keys / values in order
        v_lin = kv[bt[0].long(), 1].reshape(-1, HKV, D)[:L]
        rows = [0, 1, nq // 2, nq - 2, nq - 1]
        ref = reference_rows(q, k_lin, v_lin, L, nq, rows)
        if not any(k in ("fa2-contig", "fi-ragged", "fa2-route") or k.startswith("fa2-seg") for k in kernels):
            del k_lin, v_lin
            k_lin = v_lin = None
        cu_q = torch.tensor([0, nq], device=dev, dtype=torch.int32)
        used = torch.tensor([L], device=dev, dtype=torch.int32)
        calls = {}
        if "fi" in kernels:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
            from vllm_bf16n.triton_spec import FI_WORKSPACE_BYTES, fi_plan
            ws = torch.zeros(FI_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
            fi = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
            fi_plan(fi, torch.tensor([0, nq], dtype=torch.int32), torch.tensor([L], dtype=torch.int32), bt, BS, HQ,
                    HKV, D)
            calls["fi"] = lambda out: fi.run(q, kv, out=out)
        if "fi-ragged" in kernels:  # the same FlashInfer kernel family on contiguous K / V: the paging cost
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
            from vllm_bf16n.triton_spec import FI_WORKSPACE_BYTES
            wsr = torch.zeros(FI_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
            fir = BatchPrefillWithRaggedKVCacheWrapper(wsr, "NHD")
            fir.plan(torch.tensor([0, nq], dtype=torch.int32), torch.tensor([0, L], dtype=torch.int32), HQ, HKV, D,
                     causal=True, sm_scale=SCALE, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            calls["fi-ragged"] = lambda out: fir.run(q, k_lin, v_lin, out=out)
        if any(k.startswith("fa2") for k in kernels):
            from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
            kc, vc = kv[:, 0], kv[:, 1]
            calls["fa2"] = lambda out: fa(q, kc, vc, max_seqlen_q=nq, cu_seqlens_q=cu_q, max_seqlen_k=L, seqused_k=used,
                                          softmax_scale=SCALE, causal=True, block_table=bt, out=out, fa_version=2)
            if "fa2-contig" in kernels:
                kl, vl = k_lin.contiguous(), v_lin.contiguous()
                cu_k = torch.tensor([0, L], device=dev, dtype=torch.int32)
                calls["fa2-contig"] = lambda out: fa(q, kl, vl, max_seqlen_q=nq, cu_seqlens_q=cu_q, max_seqlen_k=L,
                                                     cu_seqlens_k=cu_k, softmax_scale=SCALE, causal=True, out=out,
                                                     fa_version=2)
            # the same cache as 16-token pages for FA2: the cache viewed as [blocks * 2 * SUB, 16, HKV, D] is
            # contiguous; K sub-page j of block b is row b * 2 * SUB + j, V is SUB rows later (v view offset by SUB)
            SUB = BS // 16
            flat = kv.view(nb * 2 * SUB, 16, HKV, D)
            k16, v16 = flat[:-SUB], flat[SUB:]
            bt16 = (bt[0].long()[:, None] * 2 * SUB + torch.arange(SUB, device=dev)).reshape(1, -1).to(torch.int32)
            calls["fa2-p16"] = lambda out: fa(q, k16, v16, max_seqlen_q=nq, cu_seqlens_q=cu_q, max_seqlen_k=L,
                                              seqused_k=used, softmax_scale=SCALE, causal=True, block_table=bt16,
                                              out=out, fa_version=2)
            kg = torch.empty(nb * BS, HKV, D, device=dev, dtype=torch.bfloat16) if "fa2-gather" in kernels else None
            vg = torch.empty_like(kg) if kg is not None else None
            cu_kg = torch.tensor([0, L], device=dev, dtype=torch.int32)
            btl = bt[0].long()

            def gather_fa(out):  # copy the sequence's pages into one contiguous K / V, then FA2 on it
                torch.index_select(kc, 0, btl, out=kg.view(nb, BS, HKV, D))
                torch.index_select(vc, 0, btl, out=vg.view(nb, BS, HKV, D))
                return fa(q, kg, vg, max_seqlen_q=nq, cu_seqlens_q=cu_q, max_seqlen_k=L, cu_seqlens_k=cu_kg,
                          softmax_scale=SCALE, causal=True, out=out, fa_version=2)
            calls["fa2-gather"] = gather_fa
            if any(k.startswith("fa2-seg") for k in kernels):
                # cached prefix [0, prior) in segments of SEG_PAGES pages: gather a segment's pages into a small scratch,
                # FA2 non-causal with LSE; the chunk's own keys [prior, L) (forward's key / value in production) causal;
                # merge the partial outputs in FP32 by their LSE (fa2-seg-nomerge: the same FA2 calls, no merge)
                prior = L - nq
                sp = a.seg_pages
                ks = torch.empty(sp * BS, HKV, D, device=dev, dtype=torch.bfloat16)
                vs = torch.empty_like(ks)
                k_self, v_self = k_lin[prior:].contiguous(), v_lin[prior:].contiguous()
                cu_self = torch.tensor([0, nq], device=dev, dtype=torch.int32)
                segs = []
                for t0 in range(0, prior, sp * BS):
                    t1 = min(prior, t0 + sp * BS)
                    p0, p1 = t0 // BS, (t1 + BS - 1) // BS
                    segs.append((btl[p0:p1], p1 - p0, t0 - p0 * BS, t1 - t0,
                                 torch.tensor([0, t1 - t0], device=dev, dtype=torch.int32)))

                def seg_fa(out, merge=True):
                    parts = [fa(q, k_self, v_self, max_seqlen_q=nq, cu_seqlens_q=cu_q, max_seqlen_k=nq,
                                cu_seqlens_k=cu_self, softmax_scale=SCALE, causal=True, return_softmax_lse=True,
                                fa_version=2)]
                    for pages, n, off, length, cu in segs:
                        torch.index_select(kc, 0, pages, out=ks[:n * BS].view(n, BS, HKV, D))
                        torch.index_select(vc, 0, pages, out=vs[:n * BS].view(n, BS, HKV, D))
                        parts.append(fa(q, ks[off:off + length], vs[off:off + length], max_seqlen_q=nq,
                                        cu_seqlens_q=cu_q, max_seqlen_k=length, cu_seqlens_k=cu, softmax_scale=SCALE,
                                        causal=False, return_softmax_lse=True, fa_version=2))
                    if not merge:
                        return
                    lse = torch.stack([p[1] for p in parts])  # [segments, HQ, nq]
                    w = torch.exp(lse - torch.logsumexp(lse, 0))
                    acc = torch.zeros(nq, HQ, D, device=dev)
                    for (o, _), ws_ in zip(parts, w):
                        acc += o.float() * ws_.transpose(0, 1)[..., None]
                    out.copy_(acc)
                calls["fa2-seg"] = seg_fa
                calls["fa2-seg-nomerge"] = lambda out: seg_fa(out, merge=False)
            if "fa2" not in kernels:
                del calls["fa2"]
        if "fa2-route" in kernels:  # the production implementation (vllm_bf16n.triton_spec.fa2_prefill, T-101)
            from vllm_bf16n.triton_spec import Fa2Buffers, fa2_prefill, fa2_segments
            prior_r = L - nq
            segs_r = fa2_segments(prior_r, BS, a.seg_pages)
            cu_kr = torch.tensor([[0, t] for _, _, t in segs_r], dtype=torch.int32, device=dev) if segs_r else None
            bufs_r = Fa2Buffers(nq, HQ // HKV * a.kv_per_call, a.kv_per_call, D, a.seg_pages * BS, dev)
            k_own, v_own = k_lin[prior_r:].contiguous(), v_lin[prior_r:].contiguous()
            calls["fa2-route"] = lambda out: fa2_prefill(q, k_own, v_own, kv[:, 0], kv[:, 1], out, cu_q, bt[0], segs_r,
                                                         cu_kr, SCALE, bufs_r)
        if "triton" in kernels:
            from vllm.v1.attention.ops.triton_unified_attention import unified_attention
            kc, vc = kv[:, 0], kv[:, 1]
            calls["triton"] = lambda out: unified_attention(
                q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=nq, seqused_k=used, max_seqlen_k=L,
                softmax_scale=SCALE, causal=True, window_size=(-1, -1), block_table=bt, softcap=0, q_descale=None,
                k_descale=None, v_descale=None)
        fl = causal_flops(nq, L)
        outs, base = {}, None
        print(f"== context {L} ({L - nq} cached + {nq} queries), {fl / 1e12:.2f} TFLOP per layer", flush=True)
        for name in kernels:
            out = torch.empty_like(q)
            try:
                t = time_ms(lambda: calls[name](out), a.reps)
            except Exception as e:  # report and go on: one unsupported kernel must not hide the others
                print(f"  {name:10s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            err = (out[rows].float() - ref).abs().max().item()
            line = f"  {name:10s} {t:9.2f} ms  {fl / t / 1e9:6.2f} TFLOPS  ref err {err:.2e}"
            if base is None:
                base = (name, t, out.clone())
            else:
                d = (out.float() - base[2].float()).abs().max().item()
                line += f"  vs {base[0]}: {base[1] / t:5.2f}x, max |diff| {d:.2e}"
            print(line, flush=True)
        del kv, q, k_lin, v_lin, calls, outs, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
