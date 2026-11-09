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


def _draft_head_bits(layer: torch.nn.Module) -> int | None:
    """Bits of the draft-only copy to build for the lm_head of a model served with MTP drafting (ops.draft_head_bits:
    BF16N_DRAFT_HEAD int2 by default, int4, or full = none; vllm_bf16n/model.py Qwen3_5MoeMTP.compute_logits;
    Qwen3.6's 248320 x 2048 head: +143 MB at 2 bits, +270 MB at 4)."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from . import ops

    if not isinstance(layer, ParallelLMHead):
        return None
    spec = get_current_vllm_config().speculative_config
    if spec is None or "mtp" not in str(getattr(spec, "method", "")):
        return None
    return ops.draft_head_bits()


class BF16NEmbeddingMethod(QuantizeMethodBase):
    uses_meta_device: bool = True

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.delegate = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        rows = sum(output_partition_sizes)
        # lm_head is multiplied (decode GEMV on the tensor cores in fragment order, T-080) and read by get_rows only for
        # the draft head's few candidate rows; the vocab embedding stays row-major: get_rows of every prompt token
        lm_head = isinstance(layer, ParallelLMHead)
        choice = (policy.choose(rows, input_size_per_partition, dense_frag_order=lm_head)
                  if params_dtype == torch.bfloat16 else None)
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
        bits = _draft_head_bits(layer)
        if bits is not None:
            from . import ops
            from .encode_layer import _weights_pool

            with torch.cuda.use_mem_pool(_weights_pool()):
                layer.bf16n_draft = ops.make_draft_head(layer.weight.data.contiguous(), bits)
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
