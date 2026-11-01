# vLLM BF16N plugin

An out-of-tree quantization plugin for original BF16 Qwen3.6-35B-A3B checkpoints. The name `--quantization bf16n` selects **lossless weight storage**, not reduced-precision weights. Encoding, policy and CUDA operations live in the shared [`core/`](../../core/). [Measured results](RESULTS.md).

**Input is the original BF16 safetensors checkpoint directory—not GGUF.** GGUF conversion is specific to llama.cpp benchmarks/backend use.

The current shared-core candidate is **not the earlier deployed serving baseline**. Numerical checks do not erase its measured MTP-off performance loss.

## Compatibility boundary

Historical measurements used SM87, CUDA12.6, PyTorch2.11.0, vLLM0.20.0+cu126, Triton3.5.0, FlashInfer0.6.8 and a Transformers4.57.6-labelled installation. See [version labels](../../versions.env).

**These labels are not a reproducible dependency lock.** The measured aarch64 environment had local dependency patches, including model/configuration support. The complete external patch set is not supplied. A clean stock installation bearing those version strings is not claimed to work or reproduce the measurements.

Before loading, verify that the environment provides `vllm.model_executor.models.qwen3_5`, `qwen3_5_mtp`, the `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeMTP` classes and configuration/tokenizer support for the pinned Qwen3.6 checkpoint. This repository's overrides depend on those base definitions; they do not replace the entire upstream implementation. Missing compatible definitions/patch sources are a fresh-environment reproduction dependency, not something this build command repairs.

## Build

Use a separate environment with the compatible dependencies already installed. From the repository root:

```bash
WHEEL_DIR="$(mktemp -d)"
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=1 python3 -m pip wheel . \
  --no-build-isolation --no-deps --wheel-dir "$WHEEL_DIR"
python3 -m pip install --no-deps "$WHEEL_DIR"/vllm_bf16n-*.whl
```

The extension targets SM87 with CUDA `-O3 -use_fast_math`. Identical source does not guarantee identical emitted instructions across adapters or build environments. No prebuilt dependency environment is bundled.

## Text-only serving example

```bash
export CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export BF16N_MOE_DEQUANT_SPLIT=4
export BF16N_CUDA_SCHED=yield
export VLLM_TUNED_CONFIG_FOLDER="$(python3 -c \
  'import pathlib,vllm_bf16n; print(pathlib.Path(vllm_bf16n.__file__).parent/"configs"/"fused_moe")')"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --quantization bf16n --dtype bfloat16 \
  --kv-cache-dtype auto --attention-backend TRITON_ATTN --language-model-only \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2112 \
  --enable-chunked-prefill --kv-cache-memory-bytes 1073741824 \
  --gpu-memory-utilization 0.66 \
  --host 127.0.0.1 --port 8000
```

This is an MTP-off, small-context example, **not** the historical full262144-token/vision/MTP configuration. Model weights, KV, recurrent state, temporary workspaces, JIT compilation and host memory all contribute to footprint. Neither a small KV budget nor one free-memory counter establishes safe admission.

Use BF16 model weights and `--kv-cache-dtype auto`. The measured Triton implementation rejects literal `bfloat16` as a cache-dtype argument; `auto` resolves to BF16 for this model. Verify actual tensor dtype when performing cache experiments.

### Compile-cache identity

Encoding policy and tensor metadata affect compiled graphs. Never reuse a cache from another encoding configuration. The plugin disables compile-cache reuse unless the environment provides its current encoding key and a matching cache-root suffix. To enable keyed reuse before starting vLLM:

```bash
export BF16N_CACHE_KEY="$(python3 -c 'import vllm_bf16n; print(vllm_bf16n.cache_key())')"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/vllm/bf16n-${BF16N_CACHE_KEY}"
mkdir -p "$VLLM_CACHE_ROOT"
```

Cache identity is necessary for correctness; it does not guarantee a cache hit, eliminate dependency drift or prove performance equivalence.

## Representation and execution

- Linear/MoE weights are encoded layer by layer while loading, checked bit-exact, then released in their uncompressed form. Small matrices use NIBBLE; larger matrices use HUFPAIR, with chunk-major layouts for appropriate expert projections.
- The Qwen model overrides pass the quantization configuration to embeddings and the target LM head. Vision, normalization, routing and other unsupported tensors remain ordinary BF16.
- Shared experts use BF16 input → fused GLU with FP32 intermediate → FP32-input down projection → BF16 output for small batches. Routed experts have separate small-batch and batched paths. Weight fidelity does not prescribe every intermediate dtype.
- Large prefill uses a combination of custom batched expert kernels and exact dequantization plus upstream matrix multiplication. The chosen path depends on batch shape and tuning data.
- MTP is optional. Target verification, the draft, candidate-head approximations and sampler penalties are distinct mechanisms. An approximate draft is not evidence that target weights changed; a sampler intervention is not lossless compression.

## Tests and measurements

After building on compatible hardware:

```bash
python3 -m pytest tests/core tests/vllm -q
python3 bench/serving/bench_openai.py --help
python3 bench/serving/bench_prefill.py --help
```

Serving clients require a separately started server. Choose their supported arguments explicitly. Historical prefill/long-context inputs are not shipped as the same corpus: public-client workloads require new measurements, and must not inherit historical throughput numbers by name or token count alone.

Offline KV reference tools are experimental CPU analysis, not a plugin KV backend. [Their measured scope](../RESULTS.md#experimental-bf16-kv-reference) excludes resident allocation and throughput claims.
