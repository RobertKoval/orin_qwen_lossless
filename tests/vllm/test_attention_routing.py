"""vllm_bf16n.triton_spec: the split-KV Triton kernel for multi-token decode batches and the FlashInfer prefill plan,
against an FP32 causal-attention reference on a paged KV cache in vLLM's [num_blocks, 2, block, kv_heads, dim] layout.

Shapes are Qwen3.6's attention (16 query heads, 2 KV heads, head_dim 256). Context lengths are chosen around the
split-KV segment boundaries (16 segments x tiles): where the last segment holds fewer keys than there are query
tokens, some query rows see no key of that segment at all (fully masked partials).
"""
import pytest
import torch

HQ, HKV, D = 16, 2, 256


def paged(seq_lens, block, seed):
    """Random K/V for each sequence, scattered into shuffled blocks; returns cache, block table, dense K/V."""
    g = torch.Generator().manual_seed(seed)
    pages = [(s + block - 1) // block for s in seq_lens]
    nblk = sum(pages) + 3
    perm = torch.randperm(nblk, generator=g)
    cache = torch.zeros(nblk, 2, block, HKV, D, dtype=torch.bfloat16)
    bt = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32)
    dense, used = [], 0
    for i, s in enumerate(seq_lens):
        kv = (torch.randn(2, s, HKV, D, generator=g)).to(torch.bfloat16)
        dense.append(kv)
        for p in range(pages[i]):
            b = int(perm[used]); used += 1
            bt[i, p] = b
            n = min(block, s - p * block)
            cache[b, :, :n] = kv[:, p * block:p * block + n]
    return cache.cuda(), bt.cuda(), dense


def reference(q, dense, q_lens):
    """FP32 causal attention: query j of a sequence with s keys and L queries sees keys [0, s - L + j]."""
    out, o = [], 0
    for kv, L in zip(dense, q_lens):
        k, v = kv[0].float().cuda(), kv[1].float().cuda()           # [s, HKV, D]
        s = k.shape[0]
        qq = q[o:o + L].float()                                      # [L, HQ, D]
        k = k.repeat_interleave(HQ // HKV, dim=1)
        v = v.repeat_interleave(HQ // HKV, dim=1)
        sc = torch.einsum("lhd,shd->hls", qq, k) * D ** -0.5
        mask = torch.arange(s, device=q.device)[None, :] > (s - L + torch.arange(L, device=q.device))[:, None]
        sc = sc.masked_fill(mask[None], float("-inf"))
        out.append(torch.einsum("hls,shd->lhd", sc.softmax(-1), v))
        o += L
    return torch.cat(out)


def check(out, ref):
    # BF16 inputs/outputs, FP32 accumulation: |err| <= 2^-7 relative to the output scale plus BF16 rounding
    err = (out.float() - ref).abs().max().item()
    assert err <= 2e-2 * ref.abs().max().item() + 4e-3, err


@pytest.fixture(scope="module")
def triton_attn():
    from vllm.v1.attention.backends import triton_attn

    from vllm_bf16n import triton_spec

    triton_spec.install()
    return triton_attn


def run_triton(triton_attn, q, cache, bt, seq_lens, q_lens, segs=16):
    n_tok = q.shape[0]
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    rows = 64
    segm_max = torch.full((rows, HQ, segs), float("nan"), device="cuda")
    k, v = cache.unbind(1)
    triton_attn.unified_attention(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=max(q_lens),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), max_seqlen_k=max(seq_lens),
        softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0, q_descale=None,
        k_descale=None, v_descale=None, seq_threshold_3D=rows, num_par_softmax_segments=segs,
        softmax_segm_output=torch.empty(rows, HQ, segs, D, device="cuda"),
        softmax_segm_max=segm_max,
        softmax_segm_expsum=torch.empty(rows, HQ, segs, device="cuda"))
    used_split_kv = not torch.isnan(segm_max[:n_tok, :, 0]).any().item()  # segment partials were written
    return out[:n_tok], used_split_kv


# 1 query (stock 3D), 2..8 (patched: 3D, BLOCK_Q = next_pow2(L) tokens per program above 2), 9 (above SPEC_MAX_Q:
# stock 2D); contexts: short, a few keys past a segment boundary of the decode tile layout (16 keys per tile; 2 keys
# past one at 32 segments), and long; vLLM's 16 segments and the production count (triton_spec.SOFTMAX_SEGMENTS)
@pytest.mark.parametrize("segs", [16, 32])
@pytest.mark.parametrize("L", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("s", [9, 1043, 16 * 32 * 3 + 2, 20000])
def test_triton_multi_token_decode_matches_reference(triton_attn, L, s, segs):
    s = max(s, L)
    cache, bt, dense = paged([s], 1072, seed=L * 7 + s)
    q = torch.randn(L, HQ, D, generator=torch.Generator().manual_seed(s)).to(torch.bfloat16).cuda()
    out, split = run_triton(triton_attn, q, cache, bt, [s], [L], segs)
    check(out, reference(q, dense, [L]))
    assert split == (L <= 8)  # split-KV up to SPEC_MAX_Q query tokens, the stock 2D kernel above


@pytest.mark.parametrize("segs", [16, 32])
def test_triton_two_sequences_mixed_query_lengths(triton_attn, segs):
    # BLOCK_Q = 4 for the batch: the 2-token sequence fills half of its program's query rows
    cache, bt, dense = paged([3000, 517], 1072, seed=3)
    q = torch.randn(6, HQ, D, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16).cuda()
    out, split = run_triton(triton_attn, q, cache, bt, [3000, 517], [4, 2], segs)
    check(out, reference(q, dense, [4, 2]))
    assert split


@pytest.mark.parametrize("seq_lens, q_lens", [([1072], [1072]), ([5000], [4288]), ([7000], [6432]), ([2500, 1073], [300, 9])])
def test_fi_prefill_plan_matches_reference(seq_lens, q_lens):
    """Prefill chunks (with and without a cached prefix, a partial last page, two requests) through fi_plan + run."""
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    from vllm_bf16n.triton_spec import fi_plan

    cache, bt, dense = paged(seq_lens, 1072, seed=sum(seq_lens))
    q = torch.randn(sum(q_lens), HQ, D, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).cuda()
    w = BatchPrefillWithPagedKVCacheWrapper(torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"), "NHD")
    qo = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    fi_plan(w, qo, torch.tensor(seq_lens), bt, 1072, HQ, HKV, D)
    out = torch.empty_like(q)
    w.run(q, cache, out=out)
    check(out, reference(q, dense, q_lens))


# ---- FA2 route for long prefill chunks (triton_spec module docstring, 3; T-101)
BS = 1072


@pytest.mark.parametrize("seg_pages", [1, 2, 30])
@pytest.mark.parametrize("prior_in_segments", [0, 1, BS - 1, BS, BS + 1, "S-1", "S", "S+1", "2S", "2S+1", "3S-5"])
def test_fa2_segments_partition_the_prefix(prior_in_segments, seg_pages):
    """Segments cover [0, prior) exactly, in order, page-aligned, none longer than seg_pages pages."""
    from vllm_bf16n.triton_spec import fa2_segments

    S = seg_pages * BS
    named = {"S-1": S - 1, "S": S, "S+1": S + 1, "2S": 2 * S, "2S+1": 2 * S + 1, "3S-5": 3 * S - 5}
    prior = named.get(prior_in_segments, prior_in_segments)
    segs = fa2_segments(prior, BS, seg_pages)
    assert len(segs) == -(-prior // S)
    t = 0
    for p0, pages, tokens in segs:
        assert p0 * BS == t and 0 < tokens <= S and pages == -(-tokens // BS) <= seg_pages
        t += tokens
    assert t == prior


def test_fa2_segments_rejects_invalid_arguments():
    from vllm_bf16n.triton_spec import fa2_segments

    for args in ((-1, BS, 30), (10, 0, 30), (10, BS, 0)):
        with pytest.raises(ValueError):
            fa2_segments(*args)


def test_fa2_route_threshold():
    from vllm_bf16n.triton_spec import fa2_route

    t = 512
    assert not fa2_route(1, t - 1, t)
    assert fa2_route(1, t, t) and fa2_route(1, t + 1, t)
    assert not fa2_route(2, 4 * t, t)  # several requests keep FlashInfer


def _needles(cache, bt, dense, q, positions):
    """Make every query of the batch attend strongly to the keys at `positions` (score 10 against N(0, 1) for the
    rest) with a distinct value each, so a key lost or counted twice at a segment boundary moves the output by ~0.1+.
    All queries are one vector shared by every head."""
    qv = q[0, 0].float()
    k = (qv * (10.0 / (qv.dot(qv).item() * D ** -0.5))).to(torch.bfloat16).cpu()
    for j, pos in enumerate(positions):
        v = torch.zeros(D)
        v[j % D] = 4.0
        b, o = int(bt[0, pos // BS]), pos % BS
        for h in range(HKV):
            dense[0][0, pos, h] = k
            dense[0][1, pos, h] = v.to(torch.bfloat16)
        cache[b, 0, o] = k.cuda()
        cache[b, 1, o] = v.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("kv_per_call", [1, 2])
@pytest.mark.parametrize("q_len", [9, 600, 1000])
@pytest.mark.parametrize("prior", [0, 2 * BS - 1, 2 * BS, 2 * BS + 1, 4 * BS + 700])
def test_fa2_prefill_matches_reference(prior, q_len, kv_per_call):
    """The FA2 route at 2-page segments (S = 2144 tokens), one or both KV heads per FA2 call: no prefix, a prefix one key short of / exactly / one key
    past a segment, and three segments ending inside a page; needles at 0, around every segment boundary, at the last
    prefix key and at the chunk's first key."""
    from vllm_bf16n.triton_spec import Fa2Buffers, fa2_prefill, fa2_segments

    seg_pages, S = 2, 2 * BS
    s = prior + q_len
    cache, bt, dense = paged([s], BS, seed=prior + 31 * q_len)
    q = torch.randn(1, 1, D, generator=torch.Generator().manual_seed(q_len)).expand(q_len, HQ, D)
    q = q.to(torch.bfloat16).contiguous().cuda()
    marks = {0, prior}
    for b in range(S, prior + 1, S):
        marks |= {b - 1, b, b + 1}
    marks |= {prior - 1} if prior else set()
    _needles(cache, bt, dense, q, sorted(m for m in marks if 0 <= m < s))
    segs = fa2_segments(prior, BS, seg_pages)
    cu_k = torch.tensor([[0, t] for _, _, t in segs], dtype=torch.int32).cuda() if segs else None
    cu_q = torch.tensor([0, q_len], dtype=torch.int32).cuda()
    kv = dense[0][:, prior:].cuda()  # the chunk's own keys / values, as forward receives them
    out = torch.full_like(q, float("nan"))
    fa2_prefill(q, kv[0].contiguous(), kv[1].contiguous(), cache[:, 0], cache[:, 1], out, cu_q, bt[0], segs, cu_k,
                D ** -0.5, Fa2Buffers(q_len, HQ // HKV * kv_per_call, kv_per_call, D, seg_pages * BS, "cuda"))
    check(out, reference(q, dense, [q_len]))


def test_fa2_prefill_rejects_rows_beyond_the_scratch():
    from vllm_bf16n.triton_spec import Fa2Buffers, fa2_prefill

    q = torch.zeros(65, HQ, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(65, HKV, D, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(1, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 65], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        fa2_prefill(q, kv, kv, cache, cache, torch.empty_like(q), cu, torch.zeros(1, dtype=torch.int32, device="cuda"),
                    [], None, D ** -0.5, Fa2Buffers(64, HQ, HKV, D, BS, "cuda"))
