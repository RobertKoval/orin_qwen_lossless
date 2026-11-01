# SPDX-License-Identifier: MIT
"""BF16N for vLLM: lossless compressed BF16 weights (torch.ops.bf16n) and the `--quantization bf16n` plugin.

Importing the package is cheap and touches neither CUDA nor the extension. `register()` (the vllm.general_plugins
entry point, called in every vLLM process) registers the quantization config and the Qwen3.5-MoE override, both inert
unless `--quantization bf16n` is used, and applies BF16N_CUDA_SCHED only if that variable is set (vllm_bf16n.sched).
"""

import hashlib
import os
import pathlib

ARCH = "Qwen3_5MoeForConditionalGeneration"
_PKG = pathlib.Path(__file__).resolve().parent


# encoding-defining sources, hashed in this order; values: path in the repo (package at engines/vllm/vllm_bf16n)
_KEY_FILES = {
    "format.h": "core/include/bf16n/format.h",
    "encode.cpp": "core/src/encode.cpp",
    "policy.h": "core/include/bf16n/policy.h",
    "policy.cpp": "core/src/policy.cpp",
}


def _key_source(name: str) -> bytes:
    """Bytes of an encoding-defining source: the copy shipped in the package (written by setup.py at build time, i.e.
    the sources _C was built from), else the repo checkout around the package. Missing both is an error: a key that
    ignores the encoder lets a compile cache of another encoding be reused (M2: garbage output)."""
    shipped, tree = _PKG / "_keysrc" / name, _PKG.parents[2] / _KEY_FILES[name]
    for p in (shipped, tree):
        if p.is_file():
            return p.read_bytes()
    raise RuntimeError(f"vllm_bf16n.cache_key: {name} is neither {shipped} nor {tree}")


def cache_key() -> str:
    """Key of everything that decides the per-matrix metadata baked into compiled graphs: the format policy, the
    encoder / format definition, and BF16N_REFERENCE. Launchers put vLLM's caches under a directory named by it
    (VLLM_CACHE_ROOT=.../bf16n-<key>, BF16N_CACHE_KEY=<key>); otherwise the compile cache is disabled (config.py)."""
    h = hashlib.sha256()
    h.update((_PKG / "policy.py").read_bytes())
    for name in _KEY_FILES:
        h.update(_key_source(name))
    h.update(os.environ.get("BF16N_REFERENCE", "0").encode())
    return h.hexdigest()[:16]


def tuned_config_dir() -> str:
    """Orin-tuned Triton fused-MoE configs shipped with the package; launchers pass it as VLLM_TUNED_CONFIG_FOLDER."""
    return str(_PKG / "configs" / "fused_moe")


_registered = False


def register() -> None:
    """vllm.general_plugins entry point (idempotent)."""
    global _registered
    from . import config  # noqa: F401  (the decorator registers BF16NConfig)

    if not _registered:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model(ARCH, "vllm_bf16n.model:Qwen3_5MoeForConditionalGeneration")
        ModelRegistry.register_model("Qwen3_5MoeMTP", "vllm_bf16n.model:Qwen3_5MoeMTP")
        _registered = True
    from . import sched

    msg = sched.apply_from_env()
    if msg:
        from vllm.logger import init_logger

        init_logger("vllm.bf16n").info(msg)
