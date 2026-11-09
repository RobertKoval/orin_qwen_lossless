# Results: headline numbers

Jetson AGX Orin Developer Kit 64GB, MAXN with pinned clocks, CUDA 12.6, L4T R36.5, Qwen3.6-35B-A3B, single stream.
Comparisons are BF16 only: llama.cpp-BF16N vs vLLM-BF16N, plain decode vs the model's MTP draft, and BF16N kernels vs
the stock BF16 kernels per op. Uncompressed BF16 does not fit the board in either engine (64.6 GiB > 61.4 GiB).
Benchmark inputs are frozen (`bench/prompts/`). Details and methodology per engine:
[llamacpp/RESULTS.md](llamacpp/RESULTS.md) (the research history of the format and kernels) and
[vllm/RESULTS.md](vllm/RESULTS.md) (milestones M1–M6 of the vLLM plugin).

| metric | engine | value | how |
|---|---|---|---|
| weights resident | llama.cpp | 46.5 GB of 69.3 GB (ratio 0.671), 472/472 tensors bit-exact | load log |
| weights resident | vLLM | 43.5 GiB text model + 1.0 MTP layer (+ 0.9 BF16 vision), every tensor bit-exact | "Model loading took 45.46 GiB" |
| decode, no draft | llama.cpp | 31.2 tok/s (tg64, ±0.3 %); 28.6 same client | `llama-bench`; `bench_openai.py` |
| decode, no draft | vLLM | 32.7 tok/s (M2, FlashInfer); 31.1 (M6 config) | `bench_openai.py` |
| decode, MTP draft | llama.cpp | 40.2 tok/s (1 draft token, 4 × 400 tokens); 35.5 same client | `llama-server`; `bench_openai.py` |
| decode, MTP draft | vLLM | **42.3 tok/s** (3 draft tokens, without logprobs; 2-bit draft head, 2026-11-09); 41.9 with logprobs | `bench_openai.py` |
| prefill 8k | llama.cpp | 832 / 1323 / 1481 tok/s (ub 512 / 2048 / 4096) | `llama-bench` |
| prefill 8k / 23k | vLLM | 1903 / 1744 tok/s in the production config (6432-token chunks; 1762 / 1696 with 4288, 1516 / 1488 before the fragment-order down experts, 1439 / 1367 before the batched MoE prefill) | `bench_prefill.py` |
| prefill 23k | llama.cpp | 1192 tok/s | `llama-server`, same client |
| long-context decode | llama.cpp | 27.1 @ 64K, 22.6 @ 224K (MTP); 25.1 / 16.6 without | `llama-server` / `llama-bench` |
| long-context prefill | llama.cpp | 648 tok/s at 224K (585 with the MTP draft) | `llama-server` |
| long-context prefill | vLLM | 1269 tok/s at 72K, 895 at 197K, 724 at 246K (production config, 2026-11-09) | `long_ctx_test.py` |
| long-context decode | vLLM | 31.9 @ 72K, 27.1 @ 197K (MTP 3, 2026-11-09 haystack, draft acceptance 0.57 / 0.60; step 85.0 / 103.6 ms); 23.3 @ 246K | `long_ctx_test.py` |
| free memory, full window + vision | llama.cpp | ≈ 3 GB (process 56.4–56.7 GB after a 4K image) | |
| free memory, full window + vision | vLLM | ≥ 2.7 GiB after a 4K image and a 248K request | MemFree + Cached |
| agreement between engines | both | top-1 99.5 %, mean KL ≈ 2e-3 (teacher-forced on llama.cpp's tokens) | `compare_llama.py` |
| per op: lm_head 248320 × 2048 | both | 4.3 ms fused decode+GEMV vs 5.7 ms cuBLAS BF16 | nsys |
| per op: prefill MoE | llama.cpp | batched kernel 2.4–2.7× over dequant + cuBLAS | `moe_prefill.cpp` |
| per op: prefill MoE | vLLM | 22 vs 29.5 ms per layer at 2048 tokens vs dequant + Triton `fused_experts`; every chunk batched since 11-09 | nsys |

The vLLM numbers are from the production configuration (full 262,144-token window, vision, MTP with 3 draft tokens,
prefix caching) unless stated otherwise. The shared-core release (one `core/` for both engines, deployed 2026-11-01)
measured 40.88 → 41.21 tok/s against the earlier vLLM-only plugin with identical texts on the four prompts; without the
draft it is 1.8–3.5 % slower than that plugin in one bounded comparison. The board's vLLM 0.20 carries local patches
whose sources are not distributed here ([vllm/README.md](vllm/README.md#compatibility-boundary)).

## Experimental BF16 KV reference

A CPU reference codec inspired by [GAAC](https://arxiv.org/abs/2610.38210) (per-channel exponent windows, sign and
mantissa kept as they are) was run on **real post-RoPE BF16 K and BF16 V** of the model. It is an independent
reference format, not the authors' code or an official reproduction.

English calibration used 4096 input tokens; held-out evaluation used 8192. Only attention layers 3 and 39 were sampled
(both KV heads, all 256 channels), with 2048 held-out rows per 128-channel partition. Calibration and evaluation texts
were distinct; raw snapshots are not distributed.

| Held-out sample | Raw bytes | Accounted reference capacity |
|---|---:|---:|
| Layer 3 K | 2,097,152 | 1,555,520 |
| Layer 3 V | 2,097,152 | 1,556,608 |
| Layer 39 K | 2,097,152 | 1,556,608 |
| Layer 39 V | 2,097,152 | 1,555,520 |
| **Total** | **8,388,608** | **6,224,256** |

**25.80 % smaller (1.3477× raw / capacity); all 4,194,304 values reconstructed exactly across 2048 blocks.** The
accounting includes 6,036,512 used bytes plus 187,744 bytes of compressed-slot slack, 192,640 bytes of metadata
(49,152 headers, 8,192 block-table entries, 4,352 calibration tables, 130,944 group tags) and two raw-fallback blocks
(8,240 bytes with headers). The 3032-byte compressed slot (with its 24-byte header) was not tuned on the evaluation
data. The sequential CPU format has no per-group offset table; a random-access GPU layout would add costs not measured
here.

**All ten layers at long positions (2026-11-04).** The same codec and slot on real KV of all ten attention layers
(both KV heads, 256-row samples at positions 0, 65,280 and 204,544 of a 204,800-token text), calibrated once on a
65,536-token English text:

| Evaluation text | Smaller | K | V | positions ≤ 64K | position ~200K | raw-fallback blocks |
|---|---:|---:|---:|---:|---:|---:|
| English (prose) | 25.1 % | 24.4 % | 25.7 % | 25.9 % | 23.9 % | 2.5 % |
| Code (C/C++ sources) | 25.4 % | 25.0 % | 25.7 % | 25.9 % | 24.7 % | 1.5 % |
| Ukrainian (Wikipedia) | 25.1 % | 24.5 % | 25.7 % | 25.9 % | 23.9 % | 2.4 % |

Every block round-trips exactly. The weakest layer is K of layer 3 (21–23 %); the fixed 3032-byte slot caps the saving
at 26.0 %, so most blocks sit at the cap. The saving holds across text types and context positions.

**Attention reading compressed KV (2026-11-04, measured, not adopted).** A split-KV decode kernel for one attention
layer (`bench/micro/kv_gaac_attn.py`) read either raw BF16 or an optimistic coded layout: the sign|mantissa byte plus a
3-bit exponent code in bit planes and a per-channel base, 70 % of the bytes and the cheapest reconstruction such a
format can have. A control read the same bytes with no reconstruction. Milliseconds per call for 1 / 4 query tokens:

| Context | vLLM's kernel | raw BF16 | coded | bytes only (free reconstruction) |
|---|---:|---:|---:|---:|
| 64K | 0.905 / 0.994 | 0.881 / 0.890 | 1.379 / 1.639 | 0.637 / 0.849 |
| 200K | 2.446 / 2.736 | 2.419 / 2.429 | 3.937 / 5.160 | 1.712 / 2.297 |

Reconstruction is exact (outputs within 7e-6 of the raw kernel's). On this GPU it is arithmetic-bound: the coded
kernel is 1.6–2.1× slower than raw. Even free reconstruction helps 4-token verification calls by only 5 %, because
they are already close to compute-bound, so the whole decode step would gain about 2 % at 200K. Lossless KV
compression therefore does not pay as a speed measure on the Orin, and reading it in attention costs decode time;
it remains an option for capacity (about 25 % more tokens in the same cache), not adopted.
