# SPDX-License-Identifier: MIT
"""MTP drafter passes 2..N at their own batch size (T-084).

With FULL decode CUDA graphs and k draft tokens, vLLM rounds the capture sizes up to multiples of k + 1
(CompilationConfig.adjust_cudagraph_sizes_for_spec_decode, vLLM issue 28207: a FULL uniform-decode graph holds whole
requests). The drafter's dispatcher shares that list although the drafter runs PIECEWISE graphs only
(SpecDecodeBaseProposer.initialize_cudagraph_keys), and its passes 2..N feed one token per request: at k = 3 one or two
requests ran the draft layer on 4 rows (T-082 profile: 32 routed-expert pairs per pass for 8 or 16 real ones).

Here the drafter's dispatcher gets its own capture sizes: the shared ones plus the powers of two up to max_num_seqs. Its
graphs at the added sizes are captured right after its first captured graph, inside vLLM's capture context: a graph
first met while serving would be captured outside it and fail validate_cudagraph_capturing_enabled. The target model's
dispatcher, sizes and graphs are untouched, and so is the drafter's first pass (k + 1 rows per request).
"""
from __future__ import annotations

import copy

_installed = False


def drafter_capture_sizes(shared: list[int], max_num_seqs: int, max_size: int) -> list[int]:
    """The shared capture sizes plus every power of two up to min(max_num_seqs, max_size): the row counts of the
    drafter's passes 2..N (one token per running request)."""
    if max_num_seqs < 1 or max_size < 1:
        raise ValueError(f"drafter graphs: max_num_seqs {max_num_seqs}, max capture size {max_size}")
    bound = min(max_num_seqs, max_size)
    return sorted(set(shared) | {1 << i for i in range(bound.bit_length()) if 1 << i <= bound})


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.config import CUDAGraphMode
    from vllm.logger import init_logger
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    logger = init_logger("vllm.bf16n")
    init_keys, dummy_run = SpecDecodeBaseProposer.initialize_cudagraph_keys, SpecDecodeBaseProposer.dummy_run

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        d = self.cudagraph_dispatcher
        shared = list(d.compilation_config.cudagraph_capture_sizes or [])
        self._bf16n_added_sizes = []
        if shared and self.vllm_config.lora_config is None:  # LoRA keys depend on the shared sizes: left as vLLM has them
            own = copy.copy(d.compilation_config)
            own.cudagraph_capture_sizes = drafter_capture_sizes(
                shared, self.vllm_config.scheduler_config.max_num_seqs, own.max_cudagraph_capture_size)
            d.compilation_config = own
        init_keys(self, cudagraph_mode)
        if d.cudagraph_mode != CUDAGraphMode.NONE:
            self._bf16n_added_sizes = sorted(set(d.compilation_config.cudagraph_capture_sizes) - set(shared), reverse=True)
            logger.info("bf16n: MTP drafter CUDA graphs at %s rows (target model: %s)",
                        d.compilation_config.cudagraph_capture_sizes, shared)

    def dummy_run_(self, num_tokens: int, use_cudagraphs: bool = True, is_graph_capturing: bool = False,
                   slot_mappings=None) -> None:
        dummy_run(self, num_tokens, use_cudagraphs, is_graph_capturing, slot_mappings)
        added = getattr(self, "_bf16n_added_sizes", None)
        if not (is_graph_capturing and use_cudagraphs and added):
            return
        self._bf16n_added_sizes = []  # once, after the drafter's first (largest) captured graph
        warmups = self.vllm_config.compilation_config.cudagraph_num_of_warmups
        for n in added:  # largest first, as vLLM captures
            for _ in range(warmups):
                dummy_run(self, n, False, True, slot_mappings)
            dummy_run(self, n, True, True, slot_mappings)
            mode, desc = self.cudagraph_dispatcher.dispatch(n)
            if mode != CUDAGraphMode.PIECEWISE or desc.num_tokens != n:
                raise RuntimeError(f"bf16n: drafter graph for {n} rows dispatches to {mode.name} {desc}")
        logger.info("bf16n: captured the MTP drafter at %s rows", added)

    SpecDecodeBaseProposer.initialize_cudagraph_keys = initialize_cudagraph_keys
    SpecDecodeBaseProposer.dummy_run = dummy_run_
    _installed = True
