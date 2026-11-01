"""`--quantization bf16n`: lossless BF16N weights for a plain BF16 checkpoint (no quantization_config needed).

Linear layers and FusedMoE experts get BF16N methods; embeddings / lm_head get BF16NEmbeddingMethod when a model passes
the quant config to them (the Qwen3.5 override in model.py does). Everything else stays unquantized.
Heavy vLLM modules are imported lazily so that registering the plugin is cheap in every vLLM process.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase


@register_quantization_config("bf16n")
class BF16NConfig(QuantizationConfig):
    def __init__(self):
        super().__init__()
        # The per-matrix metadata (format, R, aux offsets) are Python ints that Dynamo bakes into the compiled graph,
        # and so is BF16N_REFERENCE. vLLM's compile-cache key does not cover the quant config (vllm/config/vllm.py
        # compute_hash: "should be captured by model_config.quantization"), so a cached graph from another encoding
        # policy silently uses stale metadata (M2: garbage output after an R policy change). Compile per start
        # instead (~4 min), unless the launcher keyed vLLM's cache directory by vllm_bf16n.cache_key(). Set in the
        # frontend before the engine core is spawned, which inherits it.
        from . import cache_key

        key = cache_key()
        keyed = os.environ.get("BF16N_CACHE_KEY") == key and os.environ.get("VLLM_CACHE_ROOT", "").endswith(key)
        if not keyed:
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    def get_name(self) -> str:
        return "bf16n"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 87  # the extension is built for sm_87 (Orin) only

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []  # plain BF16 checkpoints: vLLM then constructs BF16NConfig() without a file

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BF16NConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        if prefix.startswith("visual.") or ".visual." in prefix:
            # the vision tower stays stock BF16 (M5 decision); LinearBase asserts a method, so give it the stock one
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            from .linear import BF16NLinearMethod

            return BF16NLinearMethod(self)
        if isinstance(layer, FusedMoE):
            from .moe import BF16NMoEMethod

            return BF16NMoEMethod(self, layer.moe_config)
        if isinstance(layer, VocabParallelEmbedding):  # ParallelLMHead is a subclass
            from .embedding import BF16NEmbeddingMethod

            return BF16NEmbeddingMethod(self)
        return None
