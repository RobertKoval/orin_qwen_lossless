"""BF16NMoEMethod: routed experts of a FusedMoE layer in BF16N. w13 (gate; up per expert) and w2 are created on the
meta device, loaded through vLLM's own expert loaders, then split and encoded layer by layer. apply() receives vLLM's
topk_weights / topk_ids and runs ops.moe_experts (expert GEMVs up to 8 tokens, the batched kernel above)."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
from vllm.model_executor.utils import set_weight_attrs

from . import policy
from .encode_layer import attach, encode_matrix, weight_of


class BF16NMoEMethod(FusedMoEMethodBase):
    uses_meta_device: bool = True

    def __init__(self, quant_config, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        if not self.moe.is_act_and_mul or self.moe.has_bias:
            raise NotImplementedError("bf16n MoE supports gated (act-and-mul) experts without bias")
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        N, H = intermediate_size_per_partition, hidden_size
        layer.bf16n_gate_choice = policy.choose(N, H, num_experts)
        layer.bf16n_down_choice = policy.choose(H, N, num_experts)
        if layer.bf16n_gate_choice is None or layer.bf16n_down_choice is None:
            raise NotImplementedError(f"bf16n MoE: expert shapes {N}x{H} / {H}x{N} are not encodable")
        w13 = torch.nn.Parameter(torch.empty(num_experts, 2 * N, H, device="meta", dtype=params_dtype),
                                 requires_grad=False)
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)
        w2 = torch.nn.Parameter(torch.empty(num_experts, H, N, device="meta", dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            from .encode_layer import release_after_loading

            release_after_loading()
            return
        layer._already_called_process_weights_after_loading = True
        name = getattr(layer, "layer_name", type(layer).__name__)
        w13 = layer.w13_weight.data
        N = w13.shape[1] // 2
        attach(layer, "bf16n_gate", encode_matrix(f"{name}.gate", w13[:, :N], layer.bf16n_gate_choice))
        attach(layer, "bf16n_up", encode_matrix(f"{name}.up", w13[:, N:], layer.bf16n_gate_choice))
        attach(layer, "bf16n_down", encode_matrix(f"{name}.down", layer.w2_weight.data, layer.bf16n_down_choice))
        del layer.w13_weight
        del layer.w2_weight

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts_input: torch.Tensor | None) -> torch.Tensor:
        from . import ops

        return ops.moe_experts(x, weight_of(layer, "bf16n_gate"), weight_of(layer, "bf16n_up"),
                               weight_of(layer, "bf16n_down"), topk_weights, topk_ids, layer.num_experts)
