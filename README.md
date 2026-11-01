# orin_qwen_lossless

**Lossless BF16 weight storage for Qwen3.6-35B-A3B on Jetson AGX Orin 64GB.**

One C++/CUDA core serves two adapters: a llama.cpp patch series and an out-of-tree vLLM plugin. BF16N stores sign/mantissa bytes directly and compresses exponents. Weights are encoded while loading and checked bit-for-bit against the original BF16 checkpoint. Decode kernels reconstruct weights during multiplication rather than retaining a full decompressed model.

This is not lower-precision weight quantization. Exact weight reconstruction does **not** imply identical floating-point reductions, logits or sampled output across engines and configurations.

## Status and measured scope

Historical engine-specific builds achieved roughly one-third smaller weight storage, about31–33 tokens/s without a draft, and about40–41 tokens/s with the model's MTP draft on this board. **Those serving baselines are not measurements of a deployed shared-core release.**

The shared-core vLLM release is now deployed. A final matched warm MTP-on comparison measured **40.88 → 41.21 tokens/s (+0.80%)**, with all four texts/token counts matching. This single pair is not a universal speedup claim. The earlier bounded MTP-off comparison still showed about1.8–3.5% lower decode interval rates; that tradeoff is not erased by the MTP-on result. See [results](docs/RESULTS.md), [vLLM measurements](docs/vllm/RESULTS.md) and [llama.cpp measurements](docs/llamacpp/RESULTS.md).

Historical prefill and long-context figures are workload-specific. The public prompt corpus differs from the historical corpus; new public-client results require a fresh baseline, not an assertion of exact historical reproduction.

## Repository

| Path | Purpose |
|---|---|
| `core/` | Shared format, encoder, policy and CUDA kernels |
| `engines/llamacpp/` | Pinned llama.cpp patch series and build scripts |
| `engines/vllm/` | Python plugin, SM87 extension configuration and tuning data |
| `bench/` | Serving clients and numerical comparisons |
| `tests/` | Host contracts and explicit GPU tests |
| `tools/analysis/` | Offline analysis and experimental reference tools; not serving backends |

Format details: [BF16N format](docs/FORMAT.md). Dependency version labels and model revision: [versions.env](versions.env).

## Build and run

The CUDA kernels target **SM87**. Historical measurements used CUDA12.6, GCC11 and L4T R36.5. Use an isolated environment and sufficient build/load memory. Commands below are examples, not a turnkey installer or a guarantee that every configuration fits.

### llama.cpp

Build the pinned upstream commit plus the supplied patches:

```bash
MAX_JOBS=1 engines/llamacpp/build.sh
L=engines/llamacpp/src

# CHECKPOINT is a local directory containing the original BF16 model.
export CHECKPOINT=/path/to/Qwen3.6-35B-A3B
python3 "$L/convert_hf_to_gguf.py" "$CHECKPOINT" \
  --outtype bf16 --outfile model-bf16.gguf

LLAMA_BF16N=hybrid "$L/build/bin/llama-server" \
  -m model-bf16.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8000
```

Use the ordinary BF16 GGUF, not an already lossy converted model. The patch series adds runtime BF16N conversion and verification. Vision requires a separately converted projector; MTP is optional and is not enabled by this example.

### vLLM

**vLLM reads the original BF16 safetensors checkpoint directory, not GGUF.** The GGUF conversion above belongs only to the llama.cpp backend/benchmarks.

First provide a compatible CUDA/aarch64 PyTorch and vLLM environment. **Historical version labels alone are insufficient:** the measured installation contained local dependency/model-definition patches whose complete sources are not distributed here. Confirm Qwen3.6 configuration and the Qwen3.5-MoE model/MTP classes are available before building or loading. See [vLLM setup and limitations](docs/vllm/README.md).

Inside that isolated environment:

```bash
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=1 python3 -m pip install . \
  --no-build-isolation --no-deps

export CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export BF16N_MOE_DEQUANT_SPLIT=4
python3 -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --quantization bf16n --dtype bfloat16 \
  --kv-cache-dtype auto --language-model-only --attention-backend TRITON_ATTN \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2112 \
  --enable-chunked-prefill --kv-cache-memory-bytes 1073741824 \
  --gpu-memory-utilization 0.66 \
  --host 127.0.0.1 --port 8000
```

`auto` selects the cache dtype from the BF16 model; literal `bfloat16` is rejected by the measured Triton cache path. The explicit1GiB cache budget is a small-context example, not the historical full-window recipe or a universal memory safety bound. Uncompressed KV, recurrent state, workspaces and compilation memory remain additional costs.

## Verification

```bash
# No CUDA compiler/device required.
cmake -S core -B build/core-host -DBF16N_CUDA=OFF
cmake --build build/core-host
ctest --test-dir build/core-host --output-on-failure

# After building the extension, on a suitable CUDA device:
python3 -m pytest tests/core tests/vllm -q
```

GPU tests and server examples perform real allocations; they are not passive checks. Follow engine-specific dtype and sampling settings when interpreting output parity. MTP acceptance depends on the draft and verification path; an approximate candidate head is distinct from lossless target weights. Sampling interventions must be disclosed separately.

## Experimental KV reference

An offline, GAAC-inspired BF16 KV reference measured **8,388,608 raw bytes →6,224,256 accounted bytes** (25.80% smaller), including slot padding and metadata, with exact reconstruction of all4,194,304 uint16 values. This was English8K data from only two attention layers. It has no per-group offset table and demonstrates neither resident-vLLM memory savings nor throughput gains. Raw captures are not distributed. [Scope and accounting](docs/RESULTS.md#experimental-bf16-kv-reference).

## License and credits

MIT: [LICENSE](LICENSE). Third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md).
Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm) and
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Related lossless-compression work includes DFloat11, ZipNN and ZipServ. The experimental KV reference is inspired by Guo et al., [GAAC](https://arxiv.org/abs/2610.38210); it is not the authors' implementation or an official reproduction.
