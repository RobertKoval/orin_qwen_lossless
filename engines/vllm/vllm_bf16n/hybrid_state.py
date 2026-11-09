"""Correct GDN (linear-attention) state handling in vLLM 0.20 for hybrid models served with MTP and prefix caching.

Three upstream defects make the target read a recurrent state that does not belong to the request (T-066, measured
11-09 with tools/serving/state_checks.py: garbage or NaN output, prefix-cache hits off by up to 91 nats):

1. Shape-aliased prefills. GPUModelRunner._is_uniform_decode classifies a batch by token counts only, so a prefill of
   exactly 1 + num_speculative_tokens tokens per request (a 4-token prompt with MTP 3, or a 4-token tail after a prefix
   hit) replays the FULL CUDA graph captured for spec decode. Its GDN kernels use the persistent capture-time state
   indices (the null block on the first request after a start: NaN, '!!!!!!!!'; later the previous request's slots).
   Upstream issue vllm-project/vllm#53051, fixes proposed in #53059 / #47123 (open). Fix: a batch is uniform decode
   only if every request is past its prompt (#53059's rule).
2. Fresh one-token prompts. The GDN metadata builder splits decodes from prefills by query length alone, so a new
   request whose first step is one token runs the decode kernel, which takes the slot's stale contents as the initial
   state. vLLM's Mamba2 builder already passes treat_short_extends_as_decodes=False for this; GDN does the same here.
3. EAGLE margin on Mamba groups. Qwen's KV cache groups carry no is_eagle_group flag, so the coordinator treats every
   group as an EAGLE group and lets the Mamba search extend one block past the full-attention hit (which EAGLE already
   shortened). The combined hit then counts a block whose attention KV was never computed. Upstream main gives Mamba
   groups no EAGLE margin ("draft models have no mamba layers"); this does the same by dropping Mamba groups from the
   coordinator's EAGLE set.

Installed by Qwen3_5MoeForConditionalGeneration.__init__ (model.py) for BF16N models; every patch fails loudly if the
vLLM symbol it replaces is missing.
"""
from __future__ import annotations

import functools

_installed = False


def is_uniform_decode(input_batch, max_num_scheduled_tokens: int, uniform_decode_query_len: int, num_tokens: int,
                      num_reqs: int, force_uniform_decode: bool | None = None) -> bool:
    """vLLM's shape rule plus: every request has computed its whole prompt (a prefill is never a decode batch)."""
    if force_uniform_decode is not None:
        return force_uniform_decode
    if max_num_scheduled_tokens != uniform_decode_query_len or num_tokens != max_num_scheduled_tokens * num_reqs:
        return False
    return bool((input_batch.num_computed_tokens_cpu[:num_reqs] >= input_batch.num_prompt_tokens[:num_reqs]).all())


def split_gdn(split, common_attn_metadata, *args, **kwargs):
    """split_decodes_and_prefills for GDN: a request still in its prompt is a prefill whatever its query length.
    Capture-time metadata without is_prefilling holds decodes only and keeps vLLM's rule."""
    if common_attn_metadata.is_prefilling is None:
        return split(common_attn_metadata, *args, **kwargs)
    return split(common_attn_metadata, *args, treat_short_extends_as_decodes=False, **kwargs)


def non_mamba_eagle_groups(attention_groups, eagle_indices) -> set[int]:
    from vllm.v1.kv_cache_interface import MambaSpec

    return {i for i in eagle_indices if not isinstance(attention_groups[i][0], MambaSpec)}


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.core import kv_cache_coordinator
    from vllm.v1.worker import gpu_model_runner

    runner = gpu_model_runner.GPUModelRunner
    assert isinstance(runner.__dict__["_is_uniform_decode"], staticmethod)

    def _is_uniform_decode(self, *args, **kwargs):
        return is_uniform_decode(self.input_batch, *args, **kwargs)

    runner._is_uniform_decode = _is_uniform_decode

    gdn_attn.split_decodes_and_prefills = functools.partial(split_gdn, gdn_attn.split_decodes_and_prefills)

    coord = kv_cache_coordinator.HybridKVCacheCoordinator
    orig_verify = coord.verify_and_split_kv_cache_groups

    def verify_and_split_kv_cache_groups(self) -> None:
        orig_verify(self)
        self.eagle_attn_group_indices = non_mamba_eagle_groups(self.attention_groups, self.eagle_attn_group_indices)

    coord.verify_and_split_kv_cache_groups = verify_and_split_kv_cache_groups
    _installed = True
