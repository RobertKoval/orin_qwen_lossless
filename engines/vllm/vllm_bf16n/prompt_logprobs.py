"""Bounded memory for prompt logprobs (T-045).

vLLM 0.20 computes the prompt logprobs of a prefill chunk in one go: logits for every token of the chunk, then an FP32
log_softmax over the 248320-token vocabulary, then top-k / gather / rank. For a 4288-token chunk that is ~2 GB of
BF16 logits plus ~4 GB of FP32 log-probabilities (and temporaries), allocated outside the activation budget vLLM
profiles; next to the 262144-token KV cache the engine was OOM-killed (M6: compare_llama's teacher-forced request).

All three steps are row-independent, so they are run here in slices of ROWS tokens: compute_logits -> log_softmax ->
gather per slice, results concatenated. Values are identical to the unsliced computation. The wrapper only changes
what GPUModelRunner._get_prompt_logprobs_dict's three calls (model.compute_logits, sampler.compute_logprobs,
sampler.gather_logprobs) do while it runs: compute_logits and compute_logprobs return a deferred object, and
gather_logprobs evaluates it slice by slice. Installed by the model override unless BF16N_PROMPT_LOGPROBS_ROWS=0.
"""
from __future__ import annotations

import os

import torch

ROWS = int(os.environ.get("BF16N_PROMPT_LOGPROBS_ROWS", "256"))
# slices never shorter than this (except a whole input that is): BF16N's lm_head uses the decode GEMV up to 8 rows
# and dequant + cuBLAS above, so a short tail slice would be computed by another kernel than the unsliced chunk
MIN_ROWS = 16
_installed = False


class _Deferred:
    """Hidden states whose logits / logprobs are computed slice by slice in gather_logprobs."""

    def __init__(self, hidden_states: torch.Tensor):
        self.hidden_states = hidden_states


def sliced_gather(compute_logits, compute_logprobs, gather_logprobs, hidden_states, num_logprobs, token_ids, rows):
    """gather_logprobs(compute_logprobs(compute_logits(hidden_states)), num_logprobs, token_ids), `rows` tokens at a
    time; returns the same LogprobsTensors fields (indices, logprobs, ranks)."""
    n = hidden_states.shape[0]
    rows = max(rows, MIN_ROWS)
    bounds = list(range(0, n, rows)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_ROWS:
        del bounds[-2]  # no small tail slice: the lm_head takes another kernel (GEMV) up to 8 rows
    parts = []
    for i, j in zip(bounds, bounds[1:]):
        lp = compute_logprobs(compute_logits(hidden_states[i:j]))
        parts.append(gather_logprobs(lp, num_logprobs, token_ids[i:j]))
        del lp
    first = parts[0]
    return type(first)(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                       torch.cat([p[2] for p in parts]))


def install() -> None:
    global _installed
    if _installed or ROWS <= 0:
        return
    from vllm.v1.worker import gpu_model_runner

    cls = gpu_model_runner.GPUModelRunner
    orig = cls._get_prompt_logprobs_dict

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        if not self.num_prompt_logprobs:
            return orig(self, hidden_states, num_scheduled_tokens)
        model, sampler = self.model, self.sampler
        real_logits, real_logprobs, real_gather = model.compute_logits, sampler.compute_logprobs, sampler.gather_logprobs

        def gather(lp, num_logprobs, token_ids):
            if isinstance(lp, _Deferred):
                return sliced_gather(real_logits, real_logprobs, real_gather, lp.hidden_states, num_logprobs,
                                     token_ids, ROWS)
            return real_gather(lp, num_logprobs, token_ids)

        model.compute_logits = _Deferred
        sampler.compute_logprobs = lambda x: x if isinstance(x, _Deferred) else real_logprobs(x)
        sampler.gather_logprobs = gather
        try:
            return orig(self, hidden_states, num_scheduled_tokens)
        finally:
            del model.compute_logits, sampler.compute_logprobs, sampler.gather_logprobs

    cls._get_prompt_logprobs_dict = _get_prompt_logprobs_dict
    _installed = True
