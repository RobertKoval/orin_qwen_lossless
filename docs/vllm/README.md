# vLLM engine: BF16N as an out-of-tree vLLM plugin

`vllm-bf16n` serves Qwen3.6-35B-A3B from the **original BF16 safetensors** with BF16N weights: every matrix is
encoded layer by layer while loading, checked bit-exact against its source, and multiplied by the same BF16N kernels
llama.cpp uses ([../FORMAT.md](../FORMAT.md), sources in `core/`). Measured results: [RESULTS.md](RESULTS.md).
Pinned versions: [`versions.env`](../../versions.env).

* Package `vllm_bf16n` (`engines/vllm/vllm_bf16n/`), torch extension `vllm_bf16n._C` built from `core/` (op namespace
  `torch.ops.bf16n`), entry point `vllm.general_plugins: bf16n`.
* Inert unless vLLM runs with `--quantization bf16n`: `register()` only registers the quantization config and a model
  override over the same architecture names, so other vLLM deployments sharing the venv are unaffected.

## Compatibility boundary

Measurements used SM87, CUDA 12.6, PyTorch 2.11.0, vLLM 0.20.0+cu126, Triton 3.5.0, FlashInfer 0.6.8 and Transformers
4.57.6 ([version labels](../../versions.env)). **These labels are not a reproducible dependency lock:** the board's
environment carries local vLLM changes (model and configuration support, speculative decoding, rejection sampling,
attention, startup hooks) whose sources are not supplied here, and a clean stock installation with those version
strings is not claimed to work or to reproduce the measurements. Before loading, check that the environment provides
`vllm.model_executor.models.qwen3_5` and `qwen3_5_mtp` with the `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeMTP`
classes and configuration and tokenizer support for the pinned Qwen3.6 checkpoint: the plugin's overrides build on
those definitions.

## Build and install

On the Orin (CUDA 12.6, gcc-11, torch 2.11, vLLM 0.20), from the repo root. Develop in a **separate** venv; never
replace a serving installation with an editable install.

```bash
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=1 pip install -e . --no-build-isolation --no-deps --no-index   # development
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4 pip install . --no-build-isolation --no-deps                 # serving
```

The kernels are compiled with `-O3 -use_fast_math -lineinfo` for SM87, like ggml-cuda, so they compile to the same
instructions as in llama.cpp. Tests (GPU, on the Orin; memory capped at 40 % per process by `tests/conftest.py`):
`make test-core test-vllm`.

Start: the server command in the [README](../../README.md#vllm) is the production recipe (settings below). The
board's launcher takes the plugin and the compile-cache key from the **installed** package
and refuses a source checkout unless `VBF16N_ALLOW_SOURCE_PACKAGE=1`.

## Design

### vLLM 0.20 extension points used

* **Quantization config.** `@register_quantization_config("bf16n")` registers `BF16NConfig`. Its
  `get_config_filenames()` returns `[]`, so vLLM instantiates it without an HF `quantization_config` and
  `--quantization bf16n` works on the plain BF16 checkpoint.
* **Streaming load without a full BF16 model.** The linear and MoE methods set `uses_meta_device = True` and call
  vLLM's online-processing hook (the mechanism of `quantization/online/moe_base.py`). vLLM then buffers each layer's
  checkpoint slices, materialises the layer, replays the loads through the stock loaders (so vLLM's own q/k/v and
  gate/up merging is used), calls `process_weights_after_loading` (encode + bitwise check + drop the BF16 params) and
  moves on. Peak memory is one layer of BF16 plus the buffered slices.
* **Fused MoE.** The whole MoE runs inside the opaque custom op `vllm::moe_forward`, so Dynamo never traces it; it is
  still captured in CUDA graphs. vLLM computes `topk_ids` / `topk_weights`; the plugin's `apply` does the rest.
* **Embedding and lm_head.** Qwen3.5 creates `embed_tokens` and `lm_head` without a quant config, so a thin subclass of
  `Qwen3_5MoeForConditionalGeneration` (and of `Qwen3_5MoeMTP`) passes it in. The overrides keep the stock class names:
  vLLM's MTP proposer dispatches on them, and the draft shares the target's embedding and head.
* **Linear ops are traced** by Dynamo (`-O3`), so they are real `torch.library` ops with fake implementations that
  never mutate inputs.

### Module → format (policy.py, tested in `tests/core/test_policy.py`)

| vLLM module | rows × K | format |
|---|---|---|
| experts w13 (gate, up) | 256 × 512 × 2048 each | HUFPAIR R = 8, chunk-major |
| experts w2 (down) | 256 × 2048 × 512 | HUFPAIR R = 16 |
| `self_attn.qkv_proj` (merged q, k, v) | 9216 × 2048 | HUFPAIR R = 2 |
| `self_attn.o_proj`, `linear_attn.out_proj`, MTP `fc` | 2048 × 4096 | HUFPAIR R = 4 |
| `linear_attn.in_proj_qkvz` | 12288 × 2048 | HUFPAIR R = 8 |
| `linear_attn.in_proj_ba`, shared expert `gate_up_proj` / `down_proj` | 64 × 2048, 1024 × 2048, 2048 × 512 | NIBBLE |
| `lm_head`, `embed_tokens` | 248320 × 2048 | HUFPAIR R = 16 |
| router `gate`, `shared_expert_gate`, conv1d, norms, `A_log`, `dt_bias`, vision tower | — | BF16 (not encoded) |

The rules are llama.cpp's (`llama_bf16n_format_for` + the unit sizes of `ggml_backend_cuda_bf16n_convert`), with one
addition: vLLM merges projections into shapes (qkvz 12288 rows, qkv 9216) where llama.cpp's "largest R with ≥ 512
units" leaves a half-empty second wave of the decode GEMV, so the dense R comes from a wave model (`policy.dense_R`;
identical to llama.cpp for llama.cpp's shapes).

### Ops (`torch.ops.bf16n`)

* Weights are passed explicitly: each encoded matrix lives on its layer as `bf16n_plane` / `bf16n_aux` uint8 tensors
  allocated through torch (vLLM's memory accounting sees them); format, K, R, rows and offsets are Python ints parsed
  once from the 256-byte aux header. A view (one expert) is `row0` plus a row count.
* `encode` (CPU, multithreaded), `dequant`, `dequant_into` (straight into vLLM's interleaved w13 layout), `mmv`
  (dense or with expert ids, BF16 or F32 in and out), `mmv_glu`, `linear` (GEMV up to 8 tokens, dequant + cuBLAS
  above, one op for Dynamo), `get_rows`, `moe_mm_ids` (routing helper + batched MMA kernel, fused GLU), `moe_combine`
  (deterministic slot sum), `swiglu`, `int4_quantize` / `int4_gemv` (the 4-bit draft head, `ops.draft_logits`),
  `init`.
* Dispatch thresholds are llama.cpp's: ≤ 8 tokens → GEMV kernels; more → batched kernels. Every MoE prefill chunk runs
  the batched BF16N kernels (`moe_mm_ids`: gate/up with fused GLU, then down), with no BF16 workspace (T-023: +7–10 %
  prefill over the former dequant + Triton `fused_experts` path for chunks of ≥ 4096 tokens). The fragment-order
  layout for down experts (`FLAG_FRAG_ORDER`, halves the batched down kernel) exists and is tested, but the policy does
  not choose it while its 4-token decode GEMV is slower than row-major.
* CUDA-graph safety: no `cudaMalloc` (workspaces from the caching allocator), no host sync, no `getenv` in hot paths,
  kernel attributes set once in `init()` before capture.
* Compile cache: BF16N metadata are baked into compiled graphs and vLLM's cache key does not see them, so launchers
  put vLLM's caches under `bf16n-<vllm_bf16n.cache_key()>` (policy + encoder sources + `BF16N_REFERENCE`) and export
  `BF16N_CACHE_KEY`; without it the plugin disables the compile cache.

### Serving path (M6)

* Attention `TRITON_ATTN` with routing (`triton_spec.py`): MTP verification and the first draft step use Triton's
  split-KV kernel, prefill chunks run FlashInfer's paged prefill on the same KV cache.
* Align-mode prefix caching with asynchronous accepted-token bookkeeping (`async_align.py`).
* GDN state correctness on vLLM 0.20 (`hybrid_state.py`, T-066), three upstream defects that made the target read a
  recurrent state not its own: a prefill of exactly 1 + 3 tokens was dispatched to the FULL spec-decode CUDA graph
  with stale state indices (vllm-project/vllm#53051; NaN on the first such request after a start); a fresh one-token
  prompt ran the GDN decode kernel on a stale slot; and the hybrid cache coordinator let Mamba groups extend a prefix
  hit one block past the full-attention hit that the MTP last-block drop had shortened, so the hit counted attention
  KV that was never computed. `tests/serving/state_checks.py` checks a running server: short prompts independent of
  the previous request, and prefix-cache hits equal to a no-cache run.
* MTP with 3 draft tokens; a **lossy, draft-only** 4-bit copy of lm_head selects 32 candidates. Their BF16N weight
  rows are reconstructed exactly and re-scored with FP32 arithmetic; other draft logits become `-inf`. This is not
  the full draft head: it can omit the best token, and its FP32 scores need not select the same token as the ordinary
  BF16-output head even when both tokens are candidates. M6 measured equal acceptance and outputs on four prompts,
  not universal equivalence. The target head still uses the losslessly stored BF16 weights.
* Prompt logprobs are computed in 256-token slices (`prompt_logprobs.py`, bitwise equal to the one-shot result).

### Fidelity boundaries

**Lossless** describes reconstruction of target BF16 weights, not bitwise equality of every arithmetic path or
same-seed generated sequence. Different reductions, kernels and rounding can change logits. The 4-bit candidate
selector is an additional approximate *draft* representation; `BF16N_DRAFT_HEAD=full` disables it.

Target sampling is the unmodified model distribution: the launcher keeps the host venv's MTP loop breaker (a
rejection-sampler patch of another engine that penalises detected repetition cycles) off with
`VLLM_MTP_LOOP_BREAKER_ENABLED=0`. A request that makes the model repeat itself repeats until `max_tokens`; clients
that want to discourage repetition set the OpenAI sampling parameters (`presence_penalty`, `frequency_penalty`)
themselves. For an evaluation with the full draft head, use `BF16N_DRAFT_HEAD=full` (or install with `--fidelity`).

An approximate draft need not bias the final target distribution if rejection uses its actual proposal distribution:
accept with `min(1, p/q)` and recover from normalized `(p-q)+`. Deterministic drafting has a point-mass proposal;
missing draft probabilities are then appropriate, not probabilities from a hypothetical full head. This mathematical
argument assumes correct implementation and unmodified target sampling policy; it is not established by matching a
few outputs. The host's optional stochastic-draft plumbing is locally patched and must be fingerprinted separately
from its upstream version label; see the deployment provenance procedure.

## Settings

Production launcher (`deploy/orin/run_qwen3_a3b_vllm_bf16n.sh`; the launcher's own names, so a shared runner's
variables cannot leak in):

| variable | default | meaning |
|---|---|---|
| `VBF16N_CONTEXT_LEN` | 262144 | max model length |
| `VBF16N_KV_BYTES` | 6442450944 | `--kv-cache-memory-bytes` |
| `VBF16N_MTP` | 3 | MTP draft tokens, 0 = off |
| `VBF16N_VISION` | 1 | vision tower on (BF16) |
| `VBF16N_MAX_PIXELS` | 4194304 | longest-edge pixel budget per image |
| `VBF16N_BATCH` | 4288 | `--max-num-batched-tokens` |
| `VBF16N_ATTN` | TRITON_ATTN | attention backend |
| `VBF16N_PREFIX_CACHE` | 1 | 0 = no prefix caching (GDN cache mode `none`) |
| `VBF16N_UTIL` | 0.55 | `--gpu-memory-utilization` (only satisfies vLLM's start check on unified memory) |
| `VBF16N_EXTRA`, `VBF16N_NSYS` | — | more vLLM args; nsys profile prefix |
| `BF16N_API_KEY` | — | `--api-key` |

Plugin knobs (read at import; changing one that shapes compiled graphs needs a fresh compile cache, T-049):

| variable | default | meaning |
|---|---|---|
| `BF16N_CUDA_SCHED` | unset (launchers: `yield`) | CUDA context scheduling flags, set before CUDA init in every vLLM process (host waits must not slow kernels) |
| `BF16N_REFERENCE` | 0 | exact dequant + stock kernels everywhere (needs `--enforce-eager`); part of the cache key |
| `BF16N_DRAFT_HEAD` | int4 | `full` = draft scores with the exact head |
| `BF16N_DRAFT_CANDIDATES` | 32 | candidates re-scored exactly by the 4-bit draft head |
| `BF16N_TRITON_3D`, `BF16N_FI_PREFILL`, `BF16N_ASYNC_ALIGN` | 1 | the M6 routing / bookkeeping, 0 = stock |
| `BF16N_PROMPT_LOGPROBS_ROWS` | 256 | prompt-logprob slice |
| `BF16N_CACHE_KEY` | set by launchers | must match `cache_key()`, or the compile cache is disabled |

## Environment on the Orin

* One shared venv (version labels in `versions.env`) serves this and the board's other vLLM models. It carries the
  local vLLM changes listed under [Compatibility boundary](#compatibility-boundary); the deployed behaviour depends on
  that exact stack.
* Unified memory: on integrated GPUs vLLM checks `MemAvailable ≥ total × gpu_memory_utilization` at start, and
  `MemAvailable` excludes the nvmap page pool and page cache. Drop caches before a start and keep the utilization low;
  the KV cache is sized by `--kv-cache-memory-bytes`.
* FlashInfer on SM87 has no TRTLLM attention and reports single-token decode only, so multi-token (MTP) batches fall
  back to piecewise graphs; Triton attention captures FULL graphs. GDN layers split graphs (`vllm::linear_attention`).
* Development servers use port 8010; production uses 8000.
