"""BF16NEmbeddingMethod: vocab embedding (embedding() = exact row gather) and lm_head (apply() = GEMV up to 8 tokens,
dequant + cuBLAS above). The Qwen3.5 override (model.py) passes the quant config to both. Shapes the policy cannot
encode are handed to vLLM's UnquantizedEmbeddingMethod."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
from vllm.model_executor.utils import set_weight_attrs

from . import policy
from .encode_layer import attach, encode_matrix, weight_of


def _wants_draft_head(layer: torch.nn.Module) -> bool:
    """lm_head of a model served with MTP drafting, unless BF16N_DRAFT_HEAD=full: build the draft-only 4-bit copy
    (vllm_bf16n/model.py Qwen3_5MoeMTP.compute_logits; +262 MB for Qwen3.6's 248320 x 2048 head)."""
    import os

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if os.environ.get("BF16N_DRAFT_HEAD", "int4") != "int4" or not isinstance(layer, ParallelLMHead):
        return False
    spec = get_current_vllm_config().speculative_config
    return spec is not None and "mtp" in str(getattr(spec, "method", ""))


class BF16NEmbeddingMethod(QuantizeMethodBase):
    uses_meta_device: bool = True

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.delegate = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        rows = sum(output_partition_sizes)
        choice = policy.choose(rows, input_size_per_partition) if params_dtype == torch.bfloat16 else None
        layer.bf16n_choice = choice
        if choice is None:
            self.delegate = UnquantizedEmbeddingMethod()
            self.uses_meta_device = False
            return self.delegate.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size,
                                                output_size, params_dtype, **extra_weight_attrs)
        weight = torch.nn.Parameter(torch.empty(rows, input_size_per_partition, device="meta", dtype=params_dtype),
                                    requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.delegate is not None:
            return self.delegate.process_weights_after_loading(layer)
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            from .encode_layer import release_after_loading

            release_after_loading()
            return
        layer._already_called_process_weights_after_loading = True
        enc = encode_matrix(getattr(layer, "prefix", type(layer).__name__), layer.weight.data, layer.bf16n_choice)
        if _wants_draft_head(layer):
            from . import ops
            from .encode_layer import _weights_pool

            with torch.cuda.use_mem_pool(_weights_pool()):
                q, s = ops.int4_quantize(layer.weight.data.contiguous())
            layer.register_buffer("bf16n_draft_q", q, persistent=False)
            layer.register_buffer("bf16n_draft_s", s, persistent=False)
        del layer.weight
        attach(layer, "bf16n", enc)
        from .encode_layer import logger

        logger.info("bf16n: memory after %s: torch allocated %.2f GiB, reserved %.2f GiB",
                    getattr(layer, "prefix", "vocab"), torch.cuda.memory_allocated() / 2**30,
                    torch.cuda.memory_reserved() / 2**30)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.delegate is not None:
            return self.delegate.apply(layer, x, bias)
        from . import ops

        return ops.linear(x, weight_of(layer, "bf16n"), bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if self.delegate is not None:
            return self.delegate.embedding(layer, input_)
        from . import ops

        return ops.embedding(input_, weight_of(layer, "bf16n"))
