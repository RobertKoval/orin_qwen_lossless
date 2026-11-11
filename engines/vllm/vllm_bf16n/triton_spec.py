"""Attention routing for the TRITON_ATTN backend (M6): Triton for decode and speculative-decode batches (FULL CUDA
graphs, no host-side planning, no GPU->CPU syncs), FlashInfer's paged prefill kernel for prefill chunks.

1. Split-KV ("3D") Triton kernel for speculative-decode batches. vLLM 0.20's unified_attention uses its split-KV
   kernel only when every query length is 1. MTP verification (n + 1 query tokens) and the first draft step got the 2D
   kernel, whose grid for one sequence is (query blocks, KV heads) = (3, 2) programs at Qwen3.6's 8 query heads per KV
   head: one program walks the whole context (decode at 200K: 9 tok/s). The kernel itself is query-length agnostic:
   segment partials are indexed by query token, causal masking is per row inside each segment, fully masked rows are
   handled by softmax_step, and reduce_segments runs per query token. Batches of up to SPEC_MAX_Q query tokens per
   sequence now use it when the token count fits the preallocated segment buffers (rows = tokens). The choice depends
   only on shapes, so captured CUDA graphs replay the path they captured. (BF16N_TRITON_3D=0 disables.)
   vLLM sizes a program's query block from the heads alone (BLOCK_M = 16 rows = 2 tokens at 8 query heads per KV
   head), so a 4-token verification ran two programs per KV head and segment over the same K/V tiles. For these
   batches the launch takes BLOCK_Q = next_pow2(max query length) tokens per program (BLOCK_M = 32 for MTP 3), and
   every 3D launch splits the context into 32 segments instead of 16 (NUM_PAR_SOFTMAX_SEGMENTS, which sizes the
   builders' partial-softmax buffers): with BLOCK_Q alone 16 segments leave 32 programs on 16 SMs and the call is
   slower. T-063, bench/micro/spec_attn_blockq.py, one layer: 4 tokens at 200K 2.77 -> 2.49 ms, at 65K 1.01 -> 0.96;
   1 token at 200K 2.49 -> 2.45. Outputs move within an FP32 reordering of the segment combine (<= 3.1e-5).

2. Prefill chunks (more than SPEC_MAX_Q query tokens) through FlashInfer's BatchPrefillWithPagedKVCacheWrapper on the
   same KV cache: both backends use the [num_blocks, 2, block_size, kv_heads, head_dim] (NHD) layout, and the KV of
   the chunk is written before forward (TritonAttentionBackend.forward_includes_kv_cache_update is False). Triton's
   prefill kernel is 1.6-2.2x slower than FlashInfer's at 68K-200K contexts on Orin. Prefill steps run attention
   eagerly (outside CUDA graphs), so the per-step plan (host, with a seq_lens readback) costs nothing that matters.
   (BF16N_FI_PREFILL=0 disables.)

3. Long prefill chunks (one request, at least FA2_MIN_Q query tokens) through vLLM's FA2 instead (T-101). FlashInfer's
   paged prefill at head_dim 256 is tile-bound near 24.5 TFLOPS at every context length; FA2's 128 x 64 tile on 8 warps
   reaches 29.5 on contiguous K / V, but its paged path is 1.4x slower than its contiguous one, and on contiguous K / V
   it falls back to 24.6 once one layer's K + V exceed ~0.3 GB (161K tokens and up): every CTA walks the whole context,
   the CTAs drift apart, and the 64 KB GPU pages leave the TLB reach (bench/micro/tlbprobe.cu: L2 TLB 256-384 MB). So:
   the chunk's own keys (forward's key / value) causal, straight into the output; the cached prefix in FA2_SEG_PAGES-page
   segments gathered into a preallocated scratch and attended non-causal, each merged by its log-sum-exp into an FP32
   accumulator (each partial is FA2's BF16 output; the sum is FP32, rounded once). All heads go into each FA2 call: one KV head per call leaves half a wave
   of 16 SMs idle whenever the 128-row query blocks are odd (1152 queries: 1.0x instead of 1.13x).
   bench/micro/prefill_attn.py, one layer, against FlashInfer: 6432 queries 1.16x over 111K, 1.19x over 240K; 768-3200
   queries over 111K 1.12-1.15x; 384 1.07x (385, 513, 769, 1025: 1.07-1.12x). Below FA2_MIN_Q FlashInfer stays: it
   splits the KV across CTAs, FA2 varlen does not (128 queries 0.9x, 9 queries 0.12x). Batches of several requests
   keep FlashInfer. Scratch 224 MB (Fa2Buffers). Outputs move at BF16 rounding against FlashInfer. (BF16N_FA2_PREFILL=0 disables; BF16N_FA2_MIN_Q,
   BF16N_FA2_SEG_PAGES override the constants.)
"""
from __future__ import annotations

import os

from typing import NamedTuple

import torch
import triton
import triton.language as tl

SPEC_MAX_Q = 8
SOFTMAX_SEGMENTS = 32
FI_WORKSPACE_BYTES = 128 << 20
FA2_MIN_Q = 384  # FA2 1.07-1.08x FlashInfer at 384 queries over 111K / 240K, 1.01-1.04x at 256, 0.90-0.93x at 128
FA2_SEG_PAGES = 30  # 32160 tokens at production's 1072-token blocks (60 pages: ~1 % faster, +66 MB)
FA2_KV_PER_CALL = 0  # KV heads per FA2 call, 0 = all: 16 query heads x 128-row blocks fill whole waves of 16 SMs
_installed = False
_workspaces: dict = {}
_fa2_buffers: dict = {}


class _SpecBlockQ:
    """kernel_unified_attention with BLOCK_Q tokens per program while `block_q` is set (a spec-decode 3D launch):
    BLOCK_M = BLOCK_Q * queries per KV head, and the grid's q-block count recomputed as unified_attention does."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.block_q = 0

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        if not self.block_q:
            return launch
        bq = self.block_q

        def run(**kw):
            kw["BLOCK_Q"], kw["BLOCK_M"] = bq, bq * kw["num_queries_per_kv"]
            return self.kernel[(kw["query_ptr"].shape[0] // bq + kw["num_seqs"], *grid[1:])](**kw)

        return run

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention as tua

    if os.environ.get("BF16N_TRITON_3D", "1") != "0":
        triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = SOFTMAX_SEGMENTS  # read when the metadata builders are created
        orig_ua = triton_attn.unified_attention
        spec_kernel = _SpecBlockQ(tua.kernel_unified_attention)
        tua.kernel_unified_attention = spec_kernel

        def unified_attention(**kw):
            mq, segm, thr = kw["max_seqlen_q"], kw.get("softmax_segm_output"), kw.get("seq_threshold_3D")
            if not (1 < mq <= SPEC_MAX_Q and segm is not None and thr is not None
                    and kw["q"].shape[0] <= segm.shape[0] and kw["seqused_k"].shape[0] <= thr):
                return orig_ua(**kw)
            kw["max_seqlen_q"] = 1  # only selects the 3D launch; the kernel reads query lengths from cu_seqlens_q
            spec_kernel.block_q = triton.next_power_of_2(mq)
            try:
                return orig_ua(**kw)
            finally:
                spec_kernel.block_q = 0

        triton_attn.unified_attention = unified_attention

    if os.environ.get("BF16N_FI_PREFILL", "1") != "0":
        global FA2_MIN_Q, FA2_SEG_PAGES, FA2_KV_PER_CALL
        FA2_MIN_Q = int(os.environ.get("BF16N_FA2_MIN_Q", FA2_MIN_Q))
        FA2_SEG_PAGES = int(os.environ.get("BF16N_FA2_SEG_PAGES", FA2_SEG_PAGES))
        FA2_KV_PER_CALL = int(os.environ.get("BF16N_FA2_KV_PER_CALL", FA2_KV_PER_CALL))
        _install_fi_prefill(triton_attn, os.environ.get("BF16N_FA2_PREFILL", "1") != "0")
    _installed = True


def fi_plan(wrapper, qo_indptr_cpu, seq_lens_cpu, block_table, block_size, num_heads_q, num_heads_kv, head_dim):
    """Plan a FlashInfer paged prefill (causal, BF16) over vLLM's block table: request i has queries
    qo_indptr[i]:qo_indptr[i+1] at the end of its seq_lens[i] tokens, stored in blocks block_table[i, :pages]."""
    seq_lens = seq_lens_cpu.to(torch.int32)
    n = seq_lens.shape[0]
    pages = (seq_lens + block_size - 1) // block_size
    kv_indptr = torch.zeros(n + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(pages, 0)
    indices = torch.cat([block_table[i, :int(pages[i])] for i in range(n)]).to(torch.int32)
    last = seq_lens - (pages - 1) * block_size
    wrapper.plan(qo_indptr=qo_indptr_cpu.to(torch.int32), paged_kv_indptr=kv_indptr, paged_kv_indices=indices,
                 paged_kv_last_page_len=last, num_qo_heads=num_heads_q, num_kv_heads=num_heads_kv, head_dim_qk=head_dim,
                 page_size=block_size, causal=True, sm_scale=head_dim ** -0.5, q_data_type=torch.bfloat16,
                 kv_data_type=torch.bfloat16, o_data_type=torch.bfloat16)


def fa2_route(num_reqs: int, q_len: int, min_q: int) -> bool:
    """Whether a prefill batch (more than SPEC_MAX_Q query tokens) takes the FA2 route."""
    return num_reqs == 1 and q_len >= min_q


def fa2_segments(prior: int, block_size: int, seg_pages: int) -> list[tuple[int, int, int]]:
    """The cached prefix [0, prior) as consecutive segments (first page, pages, tokens) of at most seg_pages pages,
    each starting on a page boundary; the last one may end inside a page."""
    if prior < 0 or block_size < 1 or seg_pages < 1:
        raise ValueError(f"fa2_segments({prior}, {block_size}, {seg_pages})")
    step = seg_pages * block_size
    return [(t0 // block_size, -(-min(step, prior - t0) // block_size), min(step, prior - t0))
            for t0 in range(0, prior, step)]


class Fa2Buffers:
    """FA2-route scratch, preallocated at engine start and shared by every builder of a device (target and draft run
    on one stream), for `kv_heads` KV heads per FA2 call: their K and V segment, the matching query heads' BF16 partial
    and FP32 accumulator with its LSE. Production (6432 rows, 16 / 2 heads, head_dim 256, 30 x 1072-token segments):
    66 + 53 + 105 MB = 224 MB."""

    def __init__(self, rows: int, q_heads: int, kv_heads: int, head_dim: int, seg_tokens: int, device):
        bf = dict(dtype=torch.bfloat16, device=device)
        self.rows, self.q_heads, self.kv_heads, self.seg_tokens = rows, q_heads, kv_heads, seg_tokens
        self.k = torch.empty(seg_tokens, kv_heads, head_dim, **bf)
        self.v = torch.empty(seg_tokens, kv_heads, head_dim, **bf)
        self.part = torch.empty(rows, q_heads, head_dim, **bf)
        self.acc = torch.empty(rows, q_heads, head_dim, dtype=torch.float32, device=device)
        self.acc_lse = torch.empty(q_heads, rows, dtype=torch.float32, device=device)

    def fits(self, rows: int, q_heads: int, kv_heads: int, seg_tokens: int) -> bool:
        return (rows <= self.rows and q_heads == self.q_heads and kv_heads == self.kv_heads
                and seg_tokens <= self.seg_tokens)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.part, self.acc, self.acc_lse))


class Fa2Plan(NamedTuple):
    q_len: int
    segments: list  # fa2_segments(...)
    cu_k: torch.Tensor | None  # [segments, 2] int32 on the device: (0, tokens) per segment
    pages: torch.Tensor  # the request's block-table row
    cu_q: torch.Tensor  # (0, q_len) on the device
    bufs: Fa2Buffers


@triton.jit
def _lse_merge_kernel(a_ptr, a_lse_ptr, b_ptr, b_lse_ptr, o_ptr, o_lse_ptr, a_st, a_sh, b_st, b_sh, o_st, o_sh,
                      al_sh, bl_sh, ol_sh, D: tl.constexpr, WRITE_LSE: tl.constexpr):
    # one (token, head) row: o = (a e^la + b e^lb) / (e^la + e^lb) in FP32; o may alias a (read before write)
    t = tl.program_id(0)
    h = tl.program_id(1)
    la = tl.load(a_lse_ptr + h * al_sh + t)
    lb = tl.load(b_lse_ptr + h * bl_sh + t)
    m = tl.maximum(la, lb)
    wa = tl.exp(la - m)
    wb = tl.exp(lb - m)
    s = wa + wb
    offs = tl.arange(0, D)
    a = tl.load(a_ptr + t * a_st + h * a_sh + offs).to(tl.float32)
    b = tl.load(b_ptr + t * b_st + h * b_sh + offs).to(tl.float32)
    tl.store(o_ptr + t * o_st + h * o_sh + offs, ((a * wa + b * wb) / s).to(o_ptr.dtype.element_ty))
    if WRITE_LSE:
        tl.store(o_lse_ptr + h * ol_sh + t, m + tl.log(s))


def lse_merge(a, a_lse, b, b_lse, out, out_lse=None) -> None:
    """Merge two attention partials over disjoint key sets: a, b, out [n, heads, D] (out may be a); *_lse [heads, n]
    natural-log LSE of the scaled scores (FA2's softmax_lse layout); out_lse written when given (may be a_lse)."""
    n, h, d = out.shape
    for x in (a, b, out):
        assert x.shape == (n, h, d) and x.stride(-1) == 1, (x.shape, x.stride())
    ol = out_lse if out_lse is not None else a_lse
    _lse_merge_kernel[(n, h)](a, a_lse, b, b_lse, out, ol, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              out.stride(0), out.stride(1), a_lse.stride(0), b_lse.stride(0), ol.stride(0), D=d,
                              WRITE_LSE=out_lse is not None, num_warps=2)


def fa2_prefill(q, key, value, k_cache, v_cache, out, cu_q, pages, segments, cu_k, scale, bufs: Fa2Buffers) -> None:
    """One request's prefill chunk through FA2 (module docstring, 3). q / out [n, Hq, D], key / value [n, Hkv, D] the
    chunk's queries and its own keys / values; k_cache / v_cache [blocks, block, Hkv, D]; the cached prefix is
    `segments` (fa2_segments) over the pages listed in `pages`, cu_k[i] = (0, tokens of segment i)."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func as fa

    n, hq, d = q.shape
    hkv, bs = key.shape[1], k_cache.shape[1]
    kvc = bufs.kv_heads  # KV heads per FA2 call
    gq = hq // hkv * kvc  # query heads per call
    seg_tokens = max((np_ for _, np_, _ in segments), default=0) * bs
    if hkv % kvc or not bufs.fits(n, gq, kvc, seg_tokens):
        raise RuntimeError(f"FA2 prefill: {n} rows / {gq} query heads / {kvc} KV heads / {seg_tokens}-token segments "
                           f"do not fit {bufs.rows} / {bufs.q_heads} / {bufs.kv_heads} / {bufs.seg_tokens} (Hkv {hkv})")
    part, acc, acc_lse = bufs.part[:n], bufs.acc[:n], bufs.acc_lse[:, :n]
    for h in range(0, hkv, kvc):
        qh, oh = q[:, h // kvc * gq:(h // kvc + 1) * gq], out[:, h // kvc * gq:(h // kvc + 1) * gq]
        _, a_lse = fa(qh, key[:, h:h + kvc], value[:, h:h + kvc], max_seqlen_q=n, cu_seqlens_q=cu_q, max_seqlen_k=n,
                      cu_seqlens_k=cu_q, softmax_scale=scale, causal=True, return_softmax_lse=True, out=oh,
                      fa_version=2)
        a = oh
        kh, vh = k_cache[:, :, h:h + kvc], v_cache[:, :, h:h + kvc]
        for i, (p0, np_, nt) in enumerate(segments):
            idx = pages[p0:p0 + np_]
            torch.index_select(kh, 0, idx, out=bufs.k[:np_ * bs].view(np_, bs, kvc, d))
            torch.index_select(vh, 0, idx, out=bufs.v[:np_ * bs].view(np_, bs, kvc, d))
            _, b_lse = fa(qh, bufs.k[:nt], bufs.v[:nt], max_seqlen_q=n, cu_seqlens_q=cu_q, max_seqlen_k=nt,
                          cu_seqlens_k=cu_k[i], softmax_scale=scale, causal=False, return_softmax_lse=True, out=part,
                          fa_version=2)
            if i == len(segments) - 1:
                lse_merge(a, a_lse, part, b_lse, oh)
            else:
                lse_merge(a, a_lse, part, b_lse, acc, acc_lse)
                a, a_lse = acc, acc_lse


def _install_fi_prefill(triton_attn, fa2: bool) -> None:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    builder_cls, impl_cls = triton_attn.TritonAttentionMetadataBuilder, triton_attn.TritonAttentionImpl
    orig_init, orig_build, orig_forward = builder_cls.__init__, builder_cls.build, impl_cls.forward

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not fa2 or self.kv_cache_spec.dtype != torch.bfloat16:
            return
        rows = self.vllm_config.scheduler_config.max_num_batched_tokens
        kvc = self.num_heads_kv if FA2_KV_PER_CALL == 0 else FA2_KV_PER_CALL
        gq, seg_tokens = self.num_heads_q // self.num_heads_kv * kvc, FA2_SEG_PAGES * self.block_size
        old = _fa2_buffers.get(self.device)
        if old is None or not old.fits(rows, gq, kvc, seg_tokens):  # at engine start, before the KV cache
            b = _fa2_buffers[self.device] = Fa2Buffers(max(rows, old.rows if old else 0), gq, kvc, self.headdim,
                                                       max(seg_tokens, old.seg_tokens if old else 0), self.device)
            from vllm.logger import init_logger
            init_logger("vllm.bf16n").info(
                "bf16n: FA2 prefill route for one-request chunks of >= %d tokens, %d-page prefix segments, %d KV "
                "heads per call; scratch %.0f MB", FA2_MIN_Q, FA2_SEG_PAGES, kvc, b.nbytes() / 1e6)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        md.bf16n_fi = md.bf16n_fa2 = None
        if md.max_query_len <= SPEC_MAX_Q or md.use_cascade or self.kv_cache_spec.dtype != torch.bfloat16:
            return md
        cm = common_attn_metadata
        if fa2 and cm.num_reqs == 1:
            q_len = int(cm.query_start_loc_cpu[1] - cm.query_start_loc_cpu[0])
            if fa2_route(cm.num_reqs, q_len, FA2_MIN_Q):
                segs = fa2_segments(int(cm.seq_lens_cpu[0]) - q_len, self.block_size, FA2_SEG_PAGES)
                cu_k = (torch.tensor([[0, t] for _, _, t in segs], dtype=torch.int32).to(self.device, non_blocking=True)
                        if segs else None)
                md.bf16n_fa2 = Fa2Plan(q_len, segs, cu_k, cm.block_table_tensor[0], cm.query_start_loc[:2],
                                       _fa2_buffers[self.device])
                return md
        if getattr(self, "_bf16n_fi", None) is None:  # one device workspace for all builders (target and draft):
            # their plans and kernels are ordered on the same stream
            if self.device not in _workspaces:
                _workspaces[self.device] = torch.zeros(FI_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
            self._bf16n_fi = BatchPrefillWithPagedKVCacheWrapper(_workspaces[self.device], "NHD")
        n = cm.num_reqs
        fi_plan(self._bf16n_fi, cm.query_start_loc_cpu[:n + 1], cm.seq_lens[:n].to("cpu"), cm.block_table_tensor,
                self.block_size, self.num_heads_q, self.num_heads_kv, self.headdim)
        md.bf16n_fi = self._bf16n_fi
        return md

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        plan = getattr(attn_metadata, "bf16n_fa2", None) if attn_metadata is not None else None
        if (plan is not None and key is not None and value is not None
                and self.attn_type == triton_attn.AttentionType.DECODER and output_scale is None
                and output_block_scale is None and self.sliding_window == (-1, -1) and not self.logits_soft_cap
                and self.sinks is None and getattr(self, "alibi_slopes", None) is None
                and getattr(attn_metadata, "mm_prefix_range_tensor", None) is None):
            n = plan.q_len
            k_cache, v_cache = kv_cache.unbind(1)
            fa2_prefill(query[:n], key[:n], value[:n], k_cache, v_cache, output[:n], plan.cu_q, plan.pages,
                        plan.segments, plan.cu_k, self.scale, plan.bufs)
            return output
        fi = getattr(attn_metadata, "bf16n_fi", None) if attn_metadata is not None else None
        if (fi is None or self.attn_type != triton_attn.AttentionType.DECODER or output_scale is not None or output_block_scale is not None or self.sliding_window != (-1, -1)
                or self.logits_soft_cap or self.sinks is not None or abs(self.scale - fi._sm_scale) > 1e-12):
            return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                output_block_scale)
        n = attn_metadata.num_actual_tokens
        fi.run(query[:n], kv_cache, out=output[:n])
        return output

    builder_cls.__init__ = init
    builder_cls.build = build
    impl_cls.forward = forward
