"""T-057 microbench: split-KV decode attention for one Qwen3.6 attention layer (16 query heads, 2 KV heads, head_dim 256)
reading BF16 K/V raw or reconstructed in registers from an exponent-coded layout, against vLLM's Triton kernel.

Optimistic coded layout (an upper bound on what a GAAC-style format can gain): per token and KV head the sign|mantissa
byte of every value (256 B), three bit-planes of a 3-bit exponent code (96 B), the 2-bit group tags GAAC reads
(8 B, loaded but unused), and one per-channel exponent base; value = sign | (base[c] + code) << 7 | mantissa. That is
70 % of the raw bytes and the cheapest reconstruction such a format can have; the real one adds 4-bit and raw groups.
  python3 kv_gaac_attn.py [contexts...]   (default 65536 200000)
"""
import sys

import torch
import triton
import triton.language as tl

HQ, HKV, D, G = 16, 2, 256, 8
TILE = 16


@triton.jit
def _recon(sm_ptr, pl_ptr, base_ptr, toks, h, tmask, HKV: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr,
           EXPAND: tl.constexpr):
    d = tl.arange(0, D)
    sm = tl.load(sm_ptr + (toks[:, None] * HKV + h) * D + d[None, :], mask=tmask[:, None], other=0).to(tl.int32)
    code = tl.zeros(sm.shape, tl.int32)
    for p in tl.static_range(3):
        if EXPAND >= 1:  # one load per plane byte, bits spread in registers
            byte = tl.load(pl_ptr + ((toks[:, None] * HKV + h) * 3 + p) * (D // 8) + tl.arange(0, D // 8)[None, :],
                           mask=tmask[:, None], other=0).to(tl.int32)
            bit = (byte[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1
            code |= tl.reshape(bit, (TILE, D)) << p
        else:  # one byte load per value (the byte is shared by 8 channels)
            byte = tl.load(pl_ptr + ((toks[:, None] * HKV + h) * 3 + p) * (D // 8) + d[None, :] // 8,
                           mask=tmask[:, None], other=0).to(tl.int32)
            code |= ((byte >> (d[None, :] % 8)) & 1) << p
    if EXPAND == 2:  # control: the same bytes are loaded, the value is only the sign|mantissa byte in place
        bits = (sm << 8) | (code & 0)
    else:
        e = tl.load(base_ptr + h * D + d).to(tl.int32)[None, :] + code
        bits = ((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def attn_partial(q_ptr, k_ptr, v_ptr, ksm_ptr, kpl_ptr, ktag_ptr, kb_ptr, vsm_ptr, vpl_ptr, vtag_ptr, vb_ptr,
                 acc_ptr, m_ptr, l_ptr, L, NQ, tiles_per_seg, scale,
                 COMP: tl.constexpr, EXPAND: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                 TILE: tl.constexpr, BLOCK_M: tl.constexpr):
    seg = tl.program_id(0)
    h = tl.program_id(1)
    nseg = tl.num_programs(0)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < NQ * G
    qt = rows // G
    qh = h * G + rows % G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (qt[:, None] * HQ + qh[:, None]) * D + d[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -1e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    t0 = seg * tiles_per_seg * TILE
    for j in range(0, tiles_per_seg):
        toks = t0 + j * TILE + tl.arange(0, TILE)
        tmask = toks < L
        if COMP:
            k = _recon(ksm_ptr, kpl_ptr, kb_ptr, toks, h, tmask, HKV, D, TILE, EXPAND)
            tag = tl.load(ktag_ptr + (toks * HKV + h) * 8, mask=tmask, other=0)  # read as the real format would
            k = tl.where((tag == 255)[:, None], tl.zeros_like(k), k)
        else:
            k = tl.load(k_ptr + (toks[:, None] * HKV + h) * D + d[None, :], mask=tmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(tmask[None, :] & (toks[None, :] <= L - NQ + qt[:, None]), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if COMP:
            v = _recon(vsm_ptr, vpl_ptr, vb_ptr, toks, h, tmask, HKV, D, TILE, EXPAND)
            vtag = tl.load(vtag_ptr + (toks * HKV + h) * 8, mask=tmask, other=0)
            v = tl.where((vtag == 255)[:, None], tl.zeros_like(v), v)
        else:
            v = tl.load(v_ptr + (toks[:, None] * HKV + h) * D + d[None, :], mask=tmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (seg * HKV + h) * BLOCK_M + rows
    tl.store(acc_ptr + o[:, None] * D + d[None, :], acc)
    tl.store(m_ptr + o, m_i)
    tl.store(l_ptr + o, l_i)


def coded(L, dev):
    """Random coded K or V: sign|mantissa bytes, 3-bit codes as bit-planes, tags, bases; and the BF16 they encode."""
    sm = torch.randint(0, 256, (L, HKV, D), dtype=torch.uint8, device=dev)
    code = torch.randint(0, 8, (L, HKV, D), dtype=torch.int32, device=dev)
    base = torch.randint(118, 123, (HKV, D), dtype=torch.int32, device=dev)
    planes = torch.zeros(L, HKV, 3, D // 8, dtype=torch.int32, device=dev)
    for p in range(3):
        bit = ((code >> p) & 1).view(L, HKV, D // 8, 8)
        planes[:, :, p] = (bit << torch.arange(8, device=dev)).sum(-1)
    tags = torch.zeros(L, HKV, 8, dtype=torch.uint8, device=dev)
    s = sm.to(torch.int32)
    bits = ((s & 0x80) << 8) | ((base[None] + code) << 7) | (s & 0x7F)
    raw = bits.to(torch.int16).view(torch.bfloat16)
    return dict(sm=sm, planes=planes.to(torch.uint8), tags=tags, base=base.to(torch.uint8).contiguous(), raw=raw)


def block_m(nq):
    return max(16, triton.next_power_of_2(nq * G))


def run(q, K, V, L, nq, comp, nseg, num_warps, num_stages, expand=1):
    BLOCK_M = block_m(nq)
    tiles = triton.cdiv(L, TILE)
    tps = triton.cdiv(tiles, nseg)
    acc = torch.empty(nseg * HKV * BLOCK_M, D, device=q.device, dtype=torch.float32)
    m = torch.empty(nseg * HKV * BLOCK_M, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    attn_partial[(nseg, HKV)](q, K["raw"], V["raw"], K["sm"], K["planes"], K["tags"], K["base"], V["sm"], V["planes"],
                              V["tags"], V["base"], acc, m, l, L, nq, tps, D ** -0.5, COMP=comp, EXPAND=expand, HQ=HQ, HKV=HKV, D=D,
                              G=G, TILE=TILE, BLOCK_M=BLOCK_M, num_warps=num_warps, num_stages=num_stages)
    return acc, m, l


def combine(acc, m, l, nseg, nq):
    BLOCK_M = block_m(nq)
    acc, m, l = acc.view(nseg, HKV, BLOCK_M, D), m.view(nseg, HKV, BLOCK_M), l.view(nseg, HKV, BLOCK_M)
    mm = m.max(0).values
    w = torch.exp(m - mm)
    out = (acc * w[..., None]).sum(0) / (l * w).sum(0)[..., None]  # [HKV, BLOCK_M, D]
    rows = torch.arange(nq * G)
    return out[:, rows].view(HKV, nq, G, D).permute(1, 0, 2, 3).reshape(nq, HQ, D)


def reference(q, K, V, L, nq):
    k, v = K["raw"].float(), V["raw"].float()
    out = []
    for i in range(nq):
        n = L - nq + i + 1
        qi = q[i].float().view(HKV, G, D)
        s = torch.einsum("hgd,thd->hgt", qi, k[:n]) * D ** -0.5
        out.append(torch.einsum("hgt,thd->hgd", s.softmax(-1), v[:n]).reshape(HQ, D))
    return torch.stack(out)


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


def vllm_ms(q, K, V, L, nq):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    bs = 1056
    nb = (L + bs - 1) // bs
    kc = torch.zeros(nb * bs, HKV, D, dtype=torch.bfloat16, device=q.device)
    vc = torch.zeros_like(kc)
    kc[:L], vc[:L] = K["raw"], V["raw"]
    kc, vc = kc.view(nb, bs, HKV, D), vc.view(nb, bs, HKV, D)
    out = torch.empty_like(q)
    bt = torch.arange(nb, device=q.device, dtype=torch.int32)[None]
    cu = torch.tensor([0, nq], device=q.device, dtype=torch.int32)
    used = torch.tensor([L], device=q.device, dtype=torch.int32)
    so = torch.empty(64, HQ, 16, D, device=q.device)
    sm = torch.empty(64, HQ, 16, device=q.device)
    se = torch.empty(64, HQ, 16, device=q.device)

    def call():
        unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=used, max_seqlen_k=L,
                          softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                          q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
                          num_par_softmax_segments=16, softmax_segm_output=so, softmax_segm_max=sm,
                          softmax_segm_expsum=se)
    t = time_ms(call)
    del kc, vc
    return t, out


def main():
    lens = [int(x) for x in sys.argv[1:]] or [65536, 200000]
    dev = "cuda"
    torch.manual_seed(0)
    for L in lens:
        K, V = coded(L, dev), coded(L, dev)
        raw_bytes = 2 * L * HKV * D * 2
        coded_bytes = 2 * L * HKV * (D + 3 * D // 8 + 8)
        for nq in (1, 4):
            q = (torch.randn(nq, HQ, D, device=dev) * 0.5).to(torch.bfloat16)
            t_v, out_v = vllm_ms(q, K, V, L, nq)
            best = {}
            for comp, expand in ((False, 1), (True, 1), ("bytes", 2)):
                for nseg in (16, 32, 64):
                    for nw, ns in ((4, 2), (8, 2), (4, 3)):
                        t = time_ms(lambda: run(q, K, V, L, nq, bool(comp), nseg, nw, ns, expand))
                        if comp not in best or t < best[comp][0]:
                            best[comp] = (t, nseg, nw, ns)
            r = run(q, K, V, L, nq, False, *best[False][1:])
            same = all(torch.equal(a, b) for a, b in zip(r, run(q, K, V, L, nq, True, *best[False][1:])))
            o = combine(*r, best[False][1], nq)
            oc = combine(*run(q, K, V, L, nq, True, *best[False][1:]), best[False][1], nq)
            same = same or (oc - o).abs().max().item() < 1e-6
            err = (o - reference(q, K, V, L, nq)).abs().max().item() if L <= 70000 else float("nan")
            err_v = (out_v.float() - o).abs().max().item()
            tr, tc = best[False][0], best[True][0]
            print(f"L={L:6d} nq={nq}: vLLM {t_v:.3f} ms | raw {tr:.3f} ms {best[False][1:]} "
                  f"({raw_bytes / tr / 1e6:.0f} GB/s) | coded {tc:.3f} ms {best[True][1:]} "
                  f"({coded_bytes / tc / 1e6:.0f} GB/s of {coded_bytes / raw_bytes:.0%} bytes) | coded vs raw "
                  f"{100 * (1 - tc / tr):+.1f} %, vs vLLM {100 * (1 - tc / t_v):+.1f} % | bytes-only control "
                  f"{best['bytes'][0]:.3f} ms ({100 * (1 - best['bytes'][0] / tr):+.1f} %) | outputs equal {same}, "
                  f"|raw-ref32| {err:.1e}, |raw-vLLM| {err_v:.1e}", flush=True)
        del K, V
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
