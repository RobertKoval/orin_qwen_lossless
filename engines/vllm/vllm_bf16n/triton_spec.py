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

2. Prefill chunks (more than SPEC_MAX_Q query tokens) through FlashInfer's BatchPrefillWithPagedKVCacheWrapper on the
   same KV cache: both backends use the [num_blocks, 2, block_size, kv_heads, head_dim] (NHD) layout, and the KV of
   the chunk is written before forward (TritonAttentionBackend.forward_includes_kv_cache_update is False). Triton's
   prefill kernel is 1.6-2.2x slower than FlashInfer's at 68K-200K contexts on Orin. Prefill steps run attention
   eagerly (outside CUDA graphs), so the per-step plan (host, with a seq_lens readback) costs nothing that matters.
   (BF16N_FI_PREFILL=0 disables.)
"""
from __future__ import annotations

import os

import torch

SPEC_MAX_Q = 8
FI_WORKSPACE_BYTES = 128 << 20
_installed = False
_workspaces: dict = {}


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.v1.attention.backends import triton_attn

    if os.environ.get("BF16N_TRITON_3D", "1") != "0":
        orig_ua = triton_attn.unified_attention

        def unified_attention(**kw):
            mq, segm, thr = kw["max_seqlen_q"], kw.get("softmax_segm_output"), kw.get("seq_threshold_3D")
            if (1 < mq <= SPEC_MAX_Q and segm is not None and thr is not None
                    and kw["q"].shape[0] <= segm.shape[0] and kw["seqused_k"].shape[0] <= thr):
                kw["max_seqlen_q"] = 1  # only selects the 3D launch; the kernel reads query lengths from cu_seqlens_q
            return orig_ua(**kw)

        triton_attn.unified_attention = unified_attention

    if os.environ.get("BF16N_FI_PREFILL", "1") != "0":
        _install_fi_prefill(triton_attn)
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


def _install_fi_prefill(triton_attn) -> None:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    builder_cls, impl_cls = triton_attn.TritonAttentionMetadataBuilder, triton_attn.TritonAttentionImpl
    orig_build, orig_forward = builder_cls.build, impl_cls.forward

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        md.bf16n_fi = None
        if md.max_query_len <= SPEC_MAX_Q or md.use_cascade or self.kv_cache_spec.dtype != torch.bfloat16:
            return md
        if getattr(self, "_bf16n_fi", None) is None:  # one device workspace for all builders (target and draft):
            # their plans and kernels are ordered on the same stream
            if self.device not in _workspaces:
                _workspaces[self.device] = torch.zeros(FI_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
            self._bf16n_fi = BatchPrefillWithPagedKVCacheWrapper(_workspaces[self.device], "NHD")
        cm = common_attn_metadata
        n = cm.num_reqs
        fi_plan(self._bf16n_fi, cm.query_start_loc_cpu[:n + 1], cm.seq_lens[:n].to("cpu"), cm.block_table_tensor,
                self.block_size, self.num_heads_q, self.num_heads_kv, self.headdim)
        md.bf16n_fi = self._bf16n_fi
        return md

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        fi = getattr(attn_metadata, "bf16n_fi", None) if attn_metadata is not None else None
        if (fi is None or self.attn_type != triton_attn.AttentionType.DECODER or output_scale is not None or output_block_scale is not None or self.sliding_window != (-1, -1)
                or self.logits_soft_cap or self.sinks is not None or abs(self.scale - fi._sm_scale) > 1e-12):
            return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                output_block_scale)
        n = attn_metadata.num_actual_tokens
        fi.run(query[:n], kv_cache, out=output[:n])
        return output

    builder_cls.build = build
    impl_cls.forward = forward
