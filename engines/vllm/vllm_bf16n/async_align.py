"""Asynchronous accepted-token bookkeeping for MTP on hybrid (GDN) models with prefix caching (mamba_cache_mode="align").

vLLM 0.20's GPUModelRunner._update_states_after_model_execute reads the number of accepted draft tokens back to the CPU
(`.cpu()`) right after rejection sampling in align mode, and then copies GDN states into newly completed prefix-cache
blocks (mamba_utils.postprocess_mamba). That blocks the host until the target forward has finished, before the drafter
is even launched, so the drafter's and the next step's host work (~15 ms per step on Orin) runs with the GPU idle
(M6: 36 -> 41 tok/s without prefix caching, where vLLM already defers this read to the next step's input preparation).

This module does the same deferral for align mode: the accepted counts are copied to pinned memory asynchronously with
an event, and the CPU part (wait for the event, set num_accepted_tokens_cpu, the block-state copies) runs at the start
of the next execute_model, i.e. after the drafts are queued behind the target forward. Inputs of the copy decision
that the next step's state update changes (per-request computed tokens, state slot, scheduled counts) are captured at
the original call. Requests that finished or were preempted in between are copied too, like the stock
postprocess_mamba does in their last step: the scheduler has already registered the completed block in the prefix
cache, so skipping the copy would leave a cached block without its state. The copy runs before any kernel of the next
step, so a block that the scheduler has meanwhile handed to another request is overwritten by that request afterwards.
The GDN states it copies are not touched by the MTP drafter (full attention only), so moving the copies after the
drafts does not change any value.

Installed by model.py for BF16N models unless BF16N_ASYNC_ALIGN=0.
"""
from __future__ import annotations

import os

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed or os.environ.get("BF16N_ASYNC_ALIGN", "1") == "0":
        return
    from vllm.v1.worker import gpu_model_runner, mamba_utils

    cls = gpu_model_runner.GPUModelRunner
    orig_update = cls._update_states_after_model_execute
    orig_execute = cls.execute_model

    def _update_states_after_model_execute(self, output_token_ids, scheduler_output):
        if not (self.speculative_config and self.model_config.is_hybrid
                and self.cache_config.mamba_cache_mode == "align"):
            return orig_update(self, output_token_ids, scheduler_output)
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)
        buf = getattr(self, "_bf16n_accepted_cpu", None)
        if buf is None:
            buf = torch.empty(self.max_num_reqs, dtype=self.num_accepted_tokens.gpu.dtype, pin_memory=True)
            self._bf16n_accepted_cpu = buf
            self._bf16n_accepted_event = torch.cuda.Event()
        buf[:num_reqs].copy_(self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True)
        self._bf16n_accepted_event.record()
        spec = scheduler_output.scheduled_spec_decode_tokens
        snap = []
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            snap.append((i, req_id, req_state, req_state.num_computed_tokens, len(spec.get(req_id, [])),
                         scheduler_output.num_scheduled_tokens[req_id], self.mamba_state_idx[req_id]))
        self._bf16n_deferred = (num_reqs, snap)

    def _apply_deferred(self, scheduler_output) -> None:
        pending = getattr(self, "_bf16n_deferred", None)
        if pending is None:
            return
        self._bf16n_deferred = None
        num_reqs, snap = pending
        self._bf16n_accepted_event.synchronize()
        acc = self._bf16n_accepted_cpu[:num_reqs].tolist()
        nacc = self.input_batch.num_accepted_tokens_cpu
        copy_bufs = self._get_mamba_copy_bufs()
        copy_bufs.offset = 0
        block = copy_bufs.mamba_spec.block_size
        forward_context = self.compilation_config.static_forward_context
        copy_funcs = self.model.get_mamba_state_copy_func()
        # mamba_utils.postprocess_mamba with the captured inputs (same arithmetic)
        for i, req_id, req_state, num_computed, num_draft, num_sched, src_block_idx in snap:
            nacc[i] = acc[i]
            running = num_computed + num_sched - num_draft
            aligned = (running + acc[i] - 1) // block * block
            if aligned >= running:
                dest_block_idx = aligned // block - 1
                mamba_utils.collect_mamba_copy_meta(copy_bufs, self.kv_cache_config, copy_funcs,
                                                    copy_bufs.mamba_group_ids, src_block_idx, dest_block_idx,
                                                    aligned - running, req_state, forward_context)
                if src_block_idx == dest_block_idx:
                    nacc[i] = 1
        mamba_utils.do_mamba_copy_block(copy_bufs)

    def execute_model(self, scheduler_output, *args, **kwargs):
        _apply_deferred(self, scheduler_output)
        return orig_execute(self, scheduler_output, *args, **kwargs)

    cls._update_states_after_model_execute = _update_states_after_model_execute
    cls.execute_model = execute_model
    _installed = True
