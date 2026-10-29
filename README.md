<h1 align="center">orin_qwen_lossless</h1>

<p align="center">
  <b>A 35B mixture-of-experts model in full BF16 precision on a 64 GB Jetson:<br>
  bit-exact weights, 33 % less memory, ≈ 30 tokens/s.</b>
</p>

<p align="center">
  <img alt="license MIT" src="https://img.shields.io/badge/license-MIT-2a78d6">
  <img alt="llama.cpp 338fce6" src="https://img.shields.io/badge/llama.cpp-338fce6-52514e">
  <img alt="Jetson AGX Orin 64GB" src="https://img.shields.io/badge/Jetson-AGX%20Orin%2064GB-1baf7a">
  <img alt="Qwen3.6-35B-A3B" src="https://img.shields.io/badge/model-Qwen3.6--35B--A3B-6250d6">
  <img alt="lossless" src="https://img.shields.io/badge/weights-bit--exact%20BF16-eb6834">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/decode-dark.svg">
    <img alt="Decode speed after a 2048-token context: PyTorch runtime 5.0, BF16N v1 with fused GLU 23.6, pair Huffman 28.4, Huffman for dense tensors 29.3, warm start and nothing on the CPU 30.5, current base with pinned clocks 31.2 tokens per second" src="docs/img/decode-light.svg" width="760">
  </picture>
</p>

**Qwen3.6-35B-A3B** in BF16 needs 64.6 GiB, more than the 61.4 GiB of unified memory on an **NVIDIA Jetson AGX Orin
64GB**. The usual way out is quantisation, which changes the weights. This project keeps every weight bit-identical and
still makes the model fit. It entropy-codes only the part of a BF16 number that is compressible (the exponent) and
decodes it inside llama.cpp's CUDA matrix–vector kernels, so a decode step streams 33 % fewer bytes from DRAM.

## Highlights

* **Lossless.** All 34.66 B weights round-trip bit-exactly, verified on the GPU for every tensor when the model loads.
  Greedy generations follow llama.cpp's stock BF16 path up to near-ties that any change of summation order can flip
  (see [Limitations](#limitations)).
* **Fits with room to spare.** 46.5 GB resident instead of 69.3 GB (ratio 0.671): the model's full 262,144-token window
  and its vision projector fit beside the weights.
* **Fast.** ≈ 31 tok/s single-stream decode (≈ 37 with the checkpoint's own multi-token-prediction layer as a
  speculative draft) and 830–1480 tok/s prefill on a 60 W board. Per matrix, the fused
  decode+GEMV beats raw BF16: the 248320×2048 output projection takes 4.3 ms vs 5.7 ms for cuBLAS BF16.
* **GPU only.** Embedding, weights and every model op run on the GPU; MoE experts are picked on the GPU, so the decode
  step stays one CUDA graph.
* **Drop-in.** One patch on upstream llama.cpp, an ordinary BF16 GGUF and `LLAMA_BF16N=hybrid`. Runs as
  `llama-server` with the OpenAI-compatible API, text and images, and under `llama-bench`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/memory-dark.svg">
    <img alt="Resident weights: BF16 64.6 GiB does not fit the 61.4 GiB unified memory; BF16N 43.3 GiB" src="docs/img/memory-light.svg" width="760">
  </picture>
</p>

## How it works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/format-dark.svg">
    <img alt="A BF16 weight is split into a raw sign and mantissa byte and an exponent that is Huffman-coded in pairs; the fused GEMV kernel decodes both in registers" src="docs/img/format-light.svg" width="760">
  </picture>
</p>

A BF16 weight is `sign · 8-bit exponent · 7-bit mantissa`. Sign and mantissa are essentially random (7.86 of 8 bits of
entropy), the exponent is not (≈ 2.6 bits). So:

* **Sign and mantissa** stay raw, one byte per weight, laid out so a warp reads them with fully coalesced 16-byte loads.
* **Exponents** are coded in **pairs** of neighbouring weights with a length-limited Huffman code per tensor
  (≈ 2.7 bit/weight). One lookup in a 4096-entry table yields both exponents already in BF16 bit positions, so a single
  `lop3` with the sign|mantissa bytes produces two weights.
* Each warp owns a few rows and each lane decodes its own bitstream, so decoding runs in parallel with no escapes, no
  synchronisation and no intermediate BF16 copy. Small matrices, where entropy-coded units would starve the GPU, use a
  fixed-width 4-bit exponent code with a per-row table instead.
* llama.cpp gets a new type `GGML_TYPE_BF16N`. Tensors are converted while an ordinary BF16 GGUF loads, checked
  bit-for-bit, and dispatched to the fused kernels for `MUL_MAT`, `MUL_MAT_ID` (MoE) and `GET_ROWS`.
* Prefill runs every MoE matrix as one persistent kernel: routing on the GPU, each expert's weights decoded once per tile
  of routed tokens into shared memory and multiplied on the BF16 tensor cores, with the next tile's decode started while
  the current one finishes. Gate, up and SwiGLU of the experts are one kernel; the lane streams reach it through small
  per-lane rings in shared memory filled by `cp.async`, so no warp waits on its own refill loads. Dense matrices in
  prefill use exact
  dequantisation plus the stock cuBLAS BF16 path. The gated delta net runs upstream llama.cpp's chunked tensor-core
  kernel (3.3 vs 15.8 ms per layer for 2048 tokens against the token-by-token recurrence), reading the recurrent state in
  place instead of gathering a copy.

The full specification is in [docs/FORMAT.md](docs/FORMAT.md). Why the kernels look the way they do (the INT32 datapath on
GA10B, L1 behaviour, what did not work) is in [docs/RESULTS.md](docs/RESULTS.md) and [docs/ORIN_NOTES.md](docs/ORIN_NOTES.md).

## Results

Jetson AGX Orin Developer Kit 64GB, MAXN with GPU and CPU clocks pinned at maximum, CUDA 12.6, L4T R36.5,
`llama-bench -ngl 99 -fa 1`, single sequence.

| | |
|---|---|
| resident weights | **46.5 GB** of 69.3 GB (ratio 0.671), 472/472 tensors bit-exact |
| decode | **31.2 tok/s** at context 0 and at 2048 (±0.3 % between processes) |
| prefill (8K prompt) | **832 tok/s** with the default `-ub 512`, **1323** with `-ub 2048`, **1481** with `-b 8192 -ub 4096` (2.4–2.7× the dequant + cuBLAS path) |
| previous best on this board, PyTorch runtime with a fused lossless format | 5.0 tok/s |
| BF16 without compression | does not fit |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/prefill-dark.svg">
    <img alt="Prefill of an 8192-token prompt in tokens/s, dequant + cuBLAS / batched MoE kernel / + chunked gated delta net and faster norms / + fused SwiGLU and lane-stream rings / rebased on upstream 338fce6: -ub 512 306 / 665 / 800 / 811 / 832, -ub 2048 530 / 943 / 1249 / 1279 / 1323, -ub 4096 613 / 1018 / 1403 / 1429 / 1481" src="docs/img/prefill-light.svg" width="760">
  </picture>
</p>

### Long context

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/context-dark.svg">
    <img alt="Decode speed vs. context: 30.6 tokens/s at 2K, 30.0 at 8K, 27.7 at 32K, 25.1 at 64K, 20.9 at 128K, 16.6 at 224K" src="docs/img/context-light.svg" width="760">
  </picture>
</p>

The model's full 262,144-token window fits next to the compressed weights. Decode slows down linearly with context
because the 10 full-attention layers read the whole F16 KV cache (20 KiB per token) at every step; the 30 Gated DeltaNet
layers keep a fixed-size state. The step time fits `32.3 ms + 0.118 µs × context`, i.e. attention streams the cache at
≈ 174 GB/s, close to the 180 GB/s the board reaches with plain reads.

| context | 2K | 8K | 32K | 64K | 128K | 224K |
|---|---:|---:|---:|---:|---:|---:|
| decode, tok/s | 30.6 | 30.0 | 27.7 | 25.1 | 20.9 | 16.6 |

2K–128K: `llama-bench -p 0 -n 32 -d <context> -ub 2048`, pinned clocks, desktop running. 224K: the server above with a
224,303-token prompt, which it prefilled at 648 tok/s on average (5.8 min) and answered correctly, with ≥ 3.8 GB free
throughout.

Correctness, methodology and the full history of what was tried: [docs/RESULTS.md](docs/RESULTS.md).

## Quick start (on the Orin)

```bash
git clone https://github.com/RobertKoval/orin_qwen_lossless && cd orin_qwen_lossless
scripts/build_llama.sh        # clones llama.cpp 338fce6, applies the patch, builds for SM87

# BF16 GGUFs of Qwen/Qwen3.6-35B-A3B: the language model (with its multi-token-prediction layer, 71 GB) and the
# vision projector
python3 llama.cpp/convert_hf_to_gguf.py Qwen3.6-35B-A3B --outtype bf16 --outfile qwen3.6-35b-a3b-bf16.gguf
python3 llama.cpp/convert_hf_to_gguf.py Qwen3.6-35B-A3B --outtype bf16 --mmproj --outfile mmproj-bf16.gguf

# OpenAI-compatible server (/v1/chat/completions, images included) with the full 262,144-token window and the MTP
# layer as a speculative draft (+18.5 % decode; the full model verifies every token, so outputs stay its own)
LLAMA_BF16N=hybrid GGML_CUDA_VMM_POOL_GB=4 LLAMA_MTP_SWA=32768 ./llama.cpp/build/bin/llama-server \
  -m qwen3.6-35b-a3b-bf16.gguf --mmproj mmproj-bf16.gguf -c 262144 -np 1 -ngl 99 -fa on --fit off \
  -b 2048 -ub 2048 -cram 0 --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048 \
  --alias qwen3-a3b-30b --host 0.0.0.0 --port 8000
```

Loading reads the 71 GB GGUF and encodes it (≈ 3 min on the Orin's NVMe and CPU). With the full window, the projector
and the MTP draft the server process holds 56.4 GB after a 4K image and about 3 GB of the 61.4 GiB stay free (desktop
running, the kernel keeps 1 GB in reserve), so the server gets one slot (`-np 1`).
What the flags are for:

* `--fit off`: llama.cpp's automatic fitting does not see the encoded exponent streams (≈ 11 GB) and would shrink
  the context; set `-c` yourself (KV costs 20 KiB per token, 5 GiB for the full window).
* `-ub 2048`: larger micro-batches lift prefill from ≈ 830 to ≈ 1320 tok/s at the price of a larger compute buffer.
* `-cram 0`: no prompt cache in RAM, which the unified memory cannot spare.
* `GGML_CUDA_VMM_POOL_GB=4`: the projector's CUDA context needs its own pool; the default 32 GB address reservation
  per pool fails next to the model.
* Draft: the MTP layer is part of the GGUF and runs on the model's own weights (embedding and output matrix shared,
  the layer itself BF16N like the rest). One draft token: the dense matrices verify 2 tokens for ≈ 1.1× the cost
  of one, but each extra token brings its own experts, so 2–3 draft tokens are slower (34.1 / 34.4 vs 36.9 tok/s);
  F16 draft KV, because a quantized one makes flash attention keep F16 copies of the whole window.
  `LLAMA_MTP_SWA=32768` lets the draft attend to its last 32768 positions only. Its KV and attention mask then stay
  small, so it takes the prompt in full 2048-token micro-batches: the draft costs 3–6 % of prefill instead of 14–24 %
  with 64-token ones, the process stays at the same size, and beyond 32K the cheaper draft step outweighs the lower
  acceptance (224K: 22.6 vs 21.9 tok/s with a full-window draft).
* The attention mask of the model itself is 1 GiB at `-ub 2048`. On Orin the GPU reads it, like every graph input,
  straight from host memory instead of a device copy (−1.06 GB); upstream switches this off because llama.cpp
  overwrote inputs the GPU was still reading (#15034), which the patch prevents by synchronizing before it writes
  them. `-ub 1024` would save the same again but costs 13–17 % of prefill speed.

Close desktop applications for best speed: on Orin the compositor and remote desktop share the GPU and DRAM.

| variable | effect |
|---|---|
| `LLAMA_BF16N=hybrid` | enable (recommended): pair-Huffman for experts and large tensors, 4-bit codes for small ones |
| `LLAMA_BF16N=huffman` / `nibble` | one format for everything (experiments) |
| `LLAMA_BF16N_DENSE_MIN=<n>` | size threshold for entropy-coding dense tensors (default 8388608 weights) |
| `LLAMA_BF16N_INPUT_CPU=1` | keep the embedding on the CPU as upstream does |
| `GGML_BF16N_REFERENCE=1` | route every product through exact dequant + stock cuBLAS (slow; for verification) |
| `GGML_BF16N_NO_MMA=1` | prefill MoE through per-expert dequant + cuBLAS instead of the batched kernel |
| `GGML_CUDA_NO_GDN_ROWS=1` | disable the in-place recurrent-state read |
| `GGML_CUDA_VMM_POOL_GB=<n>` | virtual address space each CUDA memory pool reserves (default 32); `4` lets the image projector's context start next to the model |
| `GGML_BF16N_NO_MULTI_COL=1` | decode a dense matrix once per column again (the GEMV before the multi-column kernel) |
| `GGML_CUDA_NO_INTEGRATED=1` | copy graph inputs to the device as upstream does, instead of reading them from host memory in place |
| `LLAMA_MTP_SWA=<n>` | the MTP draft attends to its last `n` positions (sliding window); its KV holds `n` + micro-batch cells |
| `GGML_BF16N_HP_GLU=1` | fused gate/up/SWIGLU for pair-Huffman experts in decode (measured slower on Orin) |
| `GGML_BF16N_NO_GLU=1` | no gate/up/SWIGLU fusion at all (decode GEMV and prefill MoE kernel) |

## Reproduce

```bash
MODEL=qwen3.6-35b-a3b-bf16.gguf scripts/bench.sh                      # llama-bench decode + prefill
MODEL=... scripts/run_prompts.sh fast LLAMA_BF16N=hybrid                # greedy outputs + top-5 logprobs
MODEL=... scripts/run_prompts.sh ref  LLAMA_BF16N=hybrid GGML_BF16N_REFERENCE=1
python3 scripts/compare_outputs.py out/prompts-fast.json out/prompts-ref.json
python3 tools/make_charts.py                                            # regenerate the figures
# where a prefill forward goes: nsys of llama-bench, then the per-kernel split of the last forward
LLAMA_BF16N=hybrid nsys profile -t cuda -s none -o prof llama-bench -m $MODEL -ngl 99 -fa 1 -p 2048 -n 0 -b 2048 -ub 2048 -r 3
nsys export --type sqlite -o prof.sqlite prof.nsys-rep && python3 tools/fwd_breakdown.py prof.sqlite
```

Kernel microbenchmarks on real weights live in [`bench/`](bench): `bwprobe.cu` (DRAM read bandwidth), `nibble_gemv.cu` and
`hufpair_gemv.cu` (format + kernel; each checks its output bit-for-bit against a raw-BF16 kernel with the same summation
order), `locate_tensor.py`, `exponent_coverage.py`, and `kv_entropy.py` (lossless compressibility of a saved
llama-server KV cache). Build with `nvcc -O3 -arch=sm_87 -std=c++17 -o X bench/X.cu -lcublas`.

`moe_prefill.cpp` runs one BF16N `MUL_MAT_ID` of a real expert tensor (or `MUL_MAT` of a 2-D one) through the patched
ggml CUDA backend (any token count, uniform or skewed routing), reports the median of individually synchronised
repetitions after a 2 s GPU warm-up, prints a hash of the output (equal hashes = bitwise equal results) and checks
sampled (or all) output columns against a CPU reference:

```bash
g++ -O2 -std=c++17 -I llama.cpp/ggml/include bench/moe_prefill.cpp -o moe_prefill \
    -L llama.cpp/build/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,$PWD/llama.cpp/build/bin
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_gate_exps.weight 2048    # GGML_BF16N_NO_MMA=1: previous path
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_down_exps.weight 4096 30 1 32768   # 30 reps, skewed, every column
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_gate_up_exps.weight 2048 # gate + up + SWIGLU, fused vs unfused bitwise
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.attn_qkv.weight 1 300        # a 2-D tensor: MUL_MAT, the decode GEMV at 1 token
```

`gdn_check.cpp` runs one gated delta net layer (model-like inputs, decay from the layer's own `ssm_a` and `dt_bias`)
through the same backend and compares outputs and final state with a double-precision recurrence:

```bash
g++ -O2 -std=c++17 -I llama.cpp/ggml/include bench/gdn_check.cpp -o gdn_check \
    -L llama.cpp/build/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,$PWD/llama.cpp/build/bin
./gdn_check qwen3.6-35b-a3b-bf16.gguf 0 2048        # layer 0, 2048 tokens; [reps] [1 = random initial state]
```

## FAQ

**Is this quantisation?** No. Every BF16 bit pattern is reproduced exactly. In decode the products are FP32 FMAs of the
exact weights; in prefill, as in llama.cpp's own BF16 path, the activations are rounded to BF16 for the tensor cores and
accumulated in FP32. Only the order of the floating-point summation differs from llama.cpp's own BF16 kernels, as it does
between any two GEMM implementations. The chunked gated delta net multiplies a two-part BF16 split of its FP32
operands (≈ 1e-5 relative error against FP64, vs ≈ 1e-6 for the FP32 recurrence); its effect on the next-token
distributions is the same size as changing the micro-batch size ([details](docs/RESULTS.md#prefill-chunked-gated-delta-net-upstream-kernel-ported)).

**Why not FP8 or 4-bit?** They are smaller and faster, and they change the model. This is for when you want the
reference model itself: evaluation baselines, quantisation research, regulated settings, or just not wondering.

**Other models or GPUs?** The format is model-agnostic (any BF16 matrix whose rows are a multiple of 512 and at most 4096
wide); the policy and the kernels were tuned on Qwen3.6-35B-A3B and SM87. Other CUDA GPUs should work but are untested.
Discrete GPUs have more compute per byte of bandwidth, which should make the decode even cheaper there.

**How was it built?** By measuring. Microbenchmarks on real weights, SASS inspection, ablations and oracles (loads
replaced by register values to bound a gain), Nsight Compute where it decided between hypotheses (it needs root on
Jetson), and alternating A/B runs for every end-to-end change. [docs/RESULTS.md](docs/RESULTS.md) lists
what worked, what did not, and by how much.

## Limitations

* Tested on one configuration: Qwen3.6-35B-A3B, Jetson AGX Orin 64GB (SM87), CUDA 12.6, llama.cpp `338fce6`.
* The compressed exponent buffers are allocated outside ggml's buffers, so llama.cpp's automatic memory fitting does not
  see them (≈ 11 GB); run with `--fit off` and set the context size explicitly.
* Not compressed: the vision projector (0.9 GB BF16) and the KV cache (F16; a lossless format would save ≈ 11 %, see
  [docs/RESULTS.md](docs/RESULTS.md#kv-cache-compressibility-measured-dropped)).
* The multi-token-prediction draft pays only with one draft token (+18.5 %,
  [docs/RESULTS.md](docs/RESULTS.md#speculative-decoding-with-the-mtp-head)): every extra verified token reads its own
  8 experts.
* Dense matrices in prefill are still dequantised to BF16 per micro-batch.
* Outputs are deterministic for a given build and settings, but not bitwise equal across prefill kernels or micro-batch
  sizes: summation order moves next-token distributions by a mean KLD ≈ 0.002 (top-1 agreement ≈ 98.4 % on wikitext-2,
  perplexity unchanged). The weights are bit-exact either way; see [docs/RESULTS.md](docs/RESULTS.md).
* The chunk-major expert layout that the prefill kernel needs costs 0.5 % of decode speed (gate/up GEMV 82 vs 80 µs).
* The in-place recurrent-state read is only enabled for single-sequence batches (and not for the last recurrent layer,
  whose row index is freed before its kernel runs).

## Roadmap

Decode runs its matrix–vector work at the DRAM limit of its access pattern (≈ 161–170 GB/s); what is left is ≈ 3 ms of
fixed cost over 231 kernel launches per token, ≈ 3.4 ms in ≈ 1000 small kernels and ≈ 1.3 ms in latency-bound small
matrices, so the practical ceiling with this format is ≈ 40 tok/s and a persistent per-layer decode kernel is the way
there. Prefill is limited by the latency of the MoE kernel's activation loads, which needs a different kernel shape
(the expert weights decoded straight into tensor-core fragments).
For serving, a matrix–vector kernel that decodes each dense weight once for several tokens made the
multi-token-prediction draft worth +18.5 %; more draft tokens need cheaper experts per extra token.
The next research stages, with expected gains and how to measure them, are in [docs/ROADMAP.md](docs/ROADMAP.md).

## License and credits

MIT, same as llama.cpp ([LICENSE](LICENSE)). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml and
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Related ideas: exponent entropy coding for lossless BF16
(DFloat11, ZipNN) and decoding inside the GEMM (ZipServ).
