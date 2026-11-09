"""vllm_bf16n.hybrid_state: GDN state correctness for MTP + prefix caching on vLLM 0.20 (T-066). CPU only.

The coordinator case reproduces the production failure: with MTP (EAGLE-style last-block drop) a request whose first
block is cached in both groups got a 16-token "hit" with no attention block at all (vLLM 0.20), i.e. its prompt KV for
that block was taken as computed while nothing had computed it.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_bf16n import hybrid_state

B = 16  # block size


def batch(computed, prompt):
    return SimpleNamespace(num_computed_tokens_cpu=np.array(computed), num_prompt_tokens=np.array(prompt))


@pytest.mark.parametrize("computed,prompt,expected", [
    ([0], [4], False),      # a fresh 4-token prompt: the MTP-3 decode shape, but a prefill
    ([3], [4], False),      # one prompt token still to go
    ([4], [4], True),       # prompt done: a real verify step
    ([5], [4], True),
    ([9, 3], [8, 4], False),  # one request of the batch still in its prompt
    ([9, 6], [8, 4], True),
])
def test_uniform_decode_needs_every_prompt_done(computed, prompt, expected):
    n = len(computed)
    assert hybrid_state.is_uniform_decode(batch(computed, prompt), 4, 4, 4 * n, n) is expected


def test_uniform_decode_keeps_shape_rule_and_force():
    done = batch([10, 10], [4, 4])
    assert not hybrid_state.is_uniform_decode(done, 3, 4, 6, 2)      # shorter queries
    assert not hybrid_state.is_uniform_decode(done, 4, 4, 7, 2)      # ragged batch
    fresh = batch([0, 0], [4, 4])
    assert hybrid_state.is_uniform_decode(fresh, 4, 4, 8, 2, force_uniform_decode=True)   # capture keeps forcing
    assert not hybrid_state.is_uniform_decode(done, 4, 4, 8, 2, force_uniform_decode=False)


def meta(query_lens, prefilling):
    q = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    return SimpleNamespace(max_query_len=max(query_lens), num_reqs=len(query_lens), num_actual_tokens=int(q[-1]),
                           query_start_loc_cpu=q,
                           is_prefilling=None if prefilling is None else torch.tensor(prefilling))


@pytest.mark.parametrize("query_lens,prefilling,expected", [
    ([1], [True], (0, 1, 0, 1)),            # fresh one-token prompt: a prefill (zero initial state)
    ([1], [False], (1, 0, 1, 0)),           # one-token decode
    ([1, 1], [False, True], (1, 1, 1, 1)),  # decode first, then the fresh prompt (reordered batch)
    ([1, 5], [False, True], (1, 1, 1, 5)),
    ([1], None, (1, 0, 1, 0)),              # capture metadata without is_prefilling: vLLM's rule
])
def test_gdn_split_counts_fresh_prompts_as_prefills(query_lens, prefilling, expected):
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    assert hybrid_state.split_gdn(split_decodes_and_prefills, meta(query_lens, prefilling), decode_threshold=1) == expected


def manager(use_eagle):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

    full = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32)
    mamba = MambaSpec(block_size=B, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align",
                      num_speculative_blocks=3)
    cfg = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["gdn1"], mamba), KVCacheGroupSpec(["gdn2"], mamba), KVCacheGroupSpec(["attn"], full)])
    return KVCacheManager(cfg, max_model_len=4096, hash_block_size=B, enable_caching=True, use_eagle=use_eagle)


def request(rid, tokens):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request

    sp = SamplingParams(max_tokens=8)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid, prompt_token_ids=tokens, sampling_params=sp, pooling_params=None,
                   block_hasher=get_request_block_hasher(B, sha256))


def fill(m, rid, tokens, chunks):
    """Prefill `tokens` in the given chunk sizes (the scheduler's block-aligned split), then finish the request."""
    r = request(rid, tokens)
    blocks, n = m.get_computed_blocks(r)
    for c in chunks:
        assert m.allocate_slots(r, c, n, blocks) is not None
        r.num_computed_tokens += n + c
        blocks, n = None, 0
    m.free(r)


def hit(m, rid, tokens):
    blocks, n = m.get_computed_blocks(request(rid, tokens))
    return n, [len(g) for g in blocks.get_block_ids()]


@pytest.fixture(autouse=True)
def patched():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)  # one hash seed per test: every request of a test must hash blocks alike
    hybrid_state.install()


def test_hit_never_counts_tokens_without_attention_kv():
    m = manager(use_eagle=True)
    prompt = list(range(100, 100 + B))
    fill(m, "a", prompt, [B])                    # the prompt ends on a block boundary: both groups cache block 0
    n, per_group = hit(m, "b", prompt + list(range(8)))
    assert n == per_group[2] * B                 # attention covers every counted token
    assert n == 0                                # EAGLE drops attention's only block, so nothing is reusable


def test_aligned_hit_still_served():
    m = manager(use_eagle=True)
    prompt = list(range(200, 200 + 2 * B + 8))
    fill(m, "a", prompt, [B, B + 8])             # scheduler: force-cache the block before EAGLE's drop, then the rest
    n, per_group = hit(m, "b", prompt[:2 * B] + list(range(8)))
    assert n == B and per_group == [1, 1, 1]
