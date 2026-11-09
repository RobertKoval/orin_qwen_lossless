"""BF16NLinearMethod: BF16 weights loaded on the meta device and encoded layer by layer (vLLM's online-quantization
path, `uses_meta_device`), applied with torch.ops.bf16n.linear (GEMV up to 8 tokens, dequant + cuBLAS above).
Shapes the policy cannot encode are handed to vLLM's UnquantizedLinearMethod unchanged."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
from vllm.model_executor.parameter import ModelWeightParameter

from . import policy
from .encode_layer import attach, encode_matrix, weight_of


class BF16NLinearMethod(LinearMethodBase):
    uses_meta_device: bool = True

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.delegate = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        rows = sum(output_partition_sizes)
        choice = (policy.choose(rows, input_size_per_partition, dense_frag_order=True)
                  if params_dtype == torch.bfloat16 else None)
        layer.bf16n_choice = choice
        if choice is None:  # stays BF16: the stock method, real parameters, no layerwise processing
            self.delegate = UnquantizedLinearMethod()
            self.uses_meta_device = False
            return self.delegate.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size,
                                                output_size, params_dtype, **extra_weight_attrs)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = rows
        layer.orig_dtype = params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(rows, input_size_per_partition, device="meta", dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
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
        del layer.weight
        attach(layer, "bf16n", enc)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.delegate is not None:
            return self.delegate.apply(layer, x, bias)
        from . import ops

        return ops.linear(x, weight_of(layer, "bf16n"), bias)
