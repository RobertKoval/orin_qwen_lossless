"""Qwen3.5-MoE model override for BF16N (T-035), registered over Qwen3_5MoeForConditionalGeneration.

Inert unless the quant config is BF16N (`--quantization bf16n`): then it behaves exactly like the stock class. With
BF16N it
  * passes the quant config to embed_tokens / lm_head (the stock model constructs them without one), so both are
    stored in BF16N (embedding = exact row gather, lm_head = GEMV / dequant + cuBLAS);
  * runs the shared expert of each MoE block with an FP32 intermediate for up to 8 tokens: fused NIBBLE GLU-GEMV
    (FP32 out) -> down GEMV on FP32 input, as llama.cpp does (BF16 input would cost +3.7 % on that K = 512 GEMV, T-036).
"""
from __future__ import annotations

import contextlib
import types

import torch
from vllm.logger import init_logger
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp

logger = init_logger("vllm.bf16n")  # under vLLM's configured hierarchy (vllm_bf16n.* INFO was dropped)


def is_bf16n(quant_config) -> bool:
    from .config import BF16NConfig

    return isinstance(quant_config, BF16NConfig)


@contextlib.contextmanager
def quantized_vocab(quant_config):
    """While the stock model is constructed, VocabParallelEmbedding / ParallelLMHead in qwen3_5 get the quant config."""
    names = ("VocabParallelEmbedding", "ParallelLMHead")
    saved = {n: getattr(qwen3_5, n) for n in names}

    def with_quant(cls, default_prefix):
        def make(*args, **kwargs):
            kwargs.setdefault("quant_config", quant_config)
            if not kwargs.get("prefix"):
                kwargs["prefix"] = default_prefix
            return cls(*args, **kwargs)

        return make

    try:
        qwen3_5.VocabParallelEmbedding = with_quant(saved["VocabParallelEmbedding"], "embed_tokens")
        qwen3_5.ParallelLMHead = with_quant(saved["ParallelLMHead"], "lm_head")
        yield
    finally:
        for n, v in saved.items():
            setattr(qwen3_5, n, v)


def _shared_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
    from . import ops
    from .encode_layer import weight_of

    if x.shape[0] <= ops.GEMV_MAX_TOKENS and not ops.REFERENCE and hasattr(self.gate_up_proj, "bf16n_plane"):
        a = ops.mmv_glu(x.contiguous(), weight_of(self.gate_up_proj, "bf16n"), out_dtype=torch.float32)
        out = ops.mmv(a, weight_of(self.down_proj, "bf16n"), out_dtype=torch.bfloat16)
    else:  # the stock Qwen2MoeMLP path
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
    if self.expert_gate is not None:
        out = torch.sigmoid(self.expert_gate(x)[0]) * out
    return out


def fuse_shared_experts(model: torch.nn.Module) -> int:
    """Switch every BF16N shared expert (NIBBLE gate_up, encodable down) to the FP32-intermediate forward."""
    from .linear import BF16NLinearMethod
    from .policy import FMT_NIBBLE

    n = 0
    for name, m in model.named_modules():
        if name.rsplit(".", 1)[-1] != "shared_expert":
            continue
        gu, dn = getattr(m, "gate_up_proj", None), getattr(m, "down_proj", None)
        ok = all(isinstance(getattr(l, "quant_method", None), BF16NLinearMethod) for l in (gu, dn))
        if ok and gu.bf16n_choice is not None and gu.bf16n_choice.format == FMT_NIBBLE and dn.bf16n_choice is not None:
            m.forward = types.MethodType(_shared_expert_forward, m)
            n += 1
    return n


# Same class names as the stock ones: vLLM dispatches on model class names (e.g. the MTP proposer's multimodal
# handling lists "Qwen3_5MoeForConditionalGeneration"; M4 failed with the earlier "BF16NQwen3_5Moe..." name).
class Qwen3_5MoeForConditionalGeneration(qwen3_5.Qwen3_5MoeForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        if not is_bf16n(vllm_config.quant_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return
        with quantized_vocab(vllm_config.quant_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.bf16n_fused_shared_experts = fuse_shared_experts(self)
        from . import async_align, hybrid_state, prompt_logprobs, triton_spec

        # the scheduler's cache coordinator must see the patch: it lives in this process only with one worker
        assert vllm_config.parallel_config.world_size == 1, "vllm_bf16n patches the in-process scheduler (TP/PP = 1)"
        hybrid_state.install()  # GDN state correctness with MTP + prefix caching (hybrid_state.py, T-066)
        async_align.install()  # MTP + prefix caching without the per-step host sync (async_align.py)
        triton_spec.install()  # split-KV Triton attention for MTP verification batches (triton_spec.py)
        prompt_logprobs.install()  # prompt logprobs in bounded memory (prompt_logprobs.py, T-045)


class _FromTarget(torch.nn.Module):
    """Placeholder for the MTP draft's embed_tokens / lm_head: vLLM's proposer always replaces both with the target's
    ("Detected MTP model. Sharing target model ... weights"), so the draft never materialises its own 2 GB BF16 copies.
    Before sharing, a multimodal target makes the proposer probe embed_input_ids once (llm_base_proposer: "Even if the
    target model is multimodal ..."); answer it with zeros of the right shape so the draft keeps multimodal inputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        return torch.zeros(*input_ids.shape, self.hidden_size, dtype=torch.bfloat16, device=input_ids.device)


@contextlib.contextmanager
def shared_vocab_placeholders():
    names = ("VocabParallelEmbedding", "ParallelLMHead")
    saved = {n: getattr(qwen3_5_mtp, n) for n in names}
    try:
        for n in names:
            setattr(qwen3_5_mtp, n, lambda num_embeddings, embedding_dim, *a, **k: _FromTarget(embedding_dim))
        yield
    finally:
        for n, v in saved.items():
            setattr(qwen3_5_mtp, n, v)


class Qwen3_5MoeMTP(qwen3_5_mtp.Qwen3_5MoeMTP):
    def __init__(self, *, vllm_config, prefix: str = ""):
        bf16n = is_bf16n(vllm_config.quant_config)
        if not bf16n:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._bf16n = False
            return
        with shared_vocab_placeholders():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._bf16n = True
        self.bf16n_fused_shared_experts = fuse_shared_experts(self)

    def load_weights(self, weights):
        if getattr(self, "_bf16n", False):
            weights = ((n, w) for n, w in weights if "embed_tokens" not in n and "lm_head" not in n)
        return super().load_weights(weights)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        """Draft logits from the target's lm_head: with the draft-only k-bit copy (built at load when MTP is on), the
        top candidates by k-bit score get exact BF16N logits and the rest -inf (ops.draft_logits; the full head reads
        674 MB per draft token, the 2-bit copy 143 MB, the 4-bit one 270 MB). Target logits are untouched."""
        lm = self.lm_head
        head = getattr(lm, "bf16n_draft", None)
        if getattr(self, "_bf16n", False) and head is not None and hidden_states.shape[0] <= 4:
            from . import ops
            from .encode_layer import weight_of

            if not getattr(self, "_draft_head_logged", False):
                self._draft_head_logged = True
                logger.info("bf16n: MTP draft logits from the %d-bit candidate head (%d candidates)", head.bits,
                            head.candidates)
            return ops.draft_logits(hidden_states, weight_of(lm, "bf16n"), head, self.config.vocab_size)
        return super().compute_logits(hidden_states, spec_step_idx)
