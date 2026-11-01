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


def run_triton(triton_attn, q, cache, bt, seq_lens, q_lens):
    n_tok = q.shape[0]
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    rows, segs = 64, 16
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


# 1 query (stock 3D), 2..8 (patched: 3D), 9 (above SPEC_MAX_Q: stock 2D); contexts: short, a few keys past a segment
# boundary of the decode tile layout, and long
@pytest.mark.parametrize("L", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("s", [9, 1043, 16 * 32 * 3 + 2, 20000])
def test_triton_multi_token_decode_matches_reference(triton_attn, L, s):
    s = max(s, L)
    cache, bt, dense = paged([s], 1072, seed=L * 7 + s)
    q = torch.randn(L, HQ, D, generator=torch.Generator().manual_seed(s)).to(torch.bfloat16).cuda()
    out, split = run_triton(triton_attn, q, cache, bt, [s], [L])
    check(out, reference(q, dense, [L]))
    assert split == (L <= 8)  # split-KV up to SPEC_MAX_Q query tokens, the stock 2D kernel above


def test_triton_two_sequences_mixed_query_lengths(triton_attn):
    cache, bt, dense = paged([3000, 517], 1072, seed=3)
    q = torch.randn(6, HQ, D, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16).cuda()
    out, split = run_triton(triton_attn, q, cache, bt, [3000, 517], [4, 2])
    check(out, reference(q, dense, [4, 2]))
    assert split


@pytest.mark.parametrize("seq_lens, q_lens", [([1072], [1072]), ([5000], [4288]), ([2500, 1073], [300, 9])])
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
