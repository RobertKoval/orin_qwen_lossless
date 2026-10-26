# Roadmap: next research stages

Where the time goes today, and what is left on the table. Numbers are per decode step (≈ 33 ms at ≈ 30 tok/s) on the
AGX Orin 64GB unless stated otherwise; "gain" estimates come from the measurements in [RESULTS.md](RESULTS.md) and are
hypotheses until measured with alternating A/B runs.

## Ceiling

A decode step streams ≈ 4.0 GB of compressed weights; at the 180 GB/s the board can actually read, that is **≈ 22 ms
(≈ 45 tok/s)** for this format. At short context, today's ≈ 33 ms splits roughly into (long context adds KV-cache
traffic on top, see Stage 2):

| part | ms | gap to its own floor |
|---|---:|---|
| pair-Huffman GEMV (experts, lm_head, projections) | ≈ 26 | runs at ≈ 75 % of bandwidth: per-launch ramp-up and tails over ≈ 230 launches |
| ≈ 1000 tiny kernels (norms, adds, router, α/β, conv state, GLU) | ≈ 3.3 | almost entirely fixed launch/start cost, 1.5–8 µs each |
| small NIBBLE matrices (shared expert, k/v, α/β, router) | ≈ 2.5 | latency-bound, 1 row per warp |
| gated delta net, attention, host/launch gap | ≈ 1.5 | close to their floors |

Single items below ~1 % are not worth doing on their own; they only pay off bundled.

## Stage 1 — prefill (cheap, large)

Prefill is ≈ 265–305 tok/s with the default micro-batch, so a 128K-token prompt takes minutes. The MoE prefill path
dequantises (almost) every expert to BF16 per micro-batch and runs one cuBLAS GEMM per expert with a host sync per layer.

1. **Micro-batch size — measured.** pp8192 at `-ub 512 / 1024 / 2048 / 4096`: 265 / 352 / 434 / 502 tok/s (× 1.9).
   Still open: memory headroom of `-ub 4096` at long contexts, `-ub 8192`, and whether llama-server should pick the
   micro-batch automatically for BF16N models.
2. **Grouped expert GEMM with on-the-fly decode** on BF16 tensor cores (MMQ-style tiles, no BF16 copy, no host sync,
   CUDA-graph capturable). Expected ≈ 1000+ tok/s (compute-bound, not bandwidth-bound). Effort: large.
3. **Chunked dequant cache**: keep decoded hot experts across micro-batches of one prompt. Effort: medium.

## Stage 2 — long context (KV-cache traffic)

Measured: `t ≈ 33.5 ms + 0.138 µs × context` (see [RESULTS.md](RESULTS.md#long-context)). The 10 full-attention layers
read 20 KiB of F16 KV cache per token of context at ≈ 149 GB/s, so from ≈ 64K tokens on the cache traffic dominates and
at 256K decode is 14.4 tok/s.

1. **Q8_0 KV cache** (`-ctk q8_0 -ctv q8_0`): ≈ 0.53× the traffic. Predicted 256K decode ≈ 19 tok/s (52.7 ms). This
   quantises activations, not weights, so it breaks bit-exactness of the outputs; measure it and its effect on logits as an
   explicit trade-off, never as a default.
2. **Lossless KV compression — measured on a real 32K cache** ([RESULTS.md](RESULTS.md#kv-cache-compressibility-measured-not-implemented)):
   only the F16 exponent is redundant, ≈ −15 % with per-channel Huffman tables (static tables from the first 4K tokens
   work), −17 % entropy bound. That clears the 10 % bar, but predicts only +8 % decode at 256K (+3 % at 64K) and 0.8 GiB
   less memory at 262K. Cost: decoding inside the flash-attention kernel, block-wise encoding of the append-only cache.
   Risk: at 256K a step must decode 2.7 G values at ≥ 88 G values/s to hide under the 30 ms compressed read, alongside
   the attention math; the weight GEMV reaches ≈ 125 G values/s with the INT datapath to itself. Below prefill and MTP in priority; reconsider if memory at 256K becomes
   the constraint.
3. **Flash-attention efficiency**: at 83 % of the read bandwidth, ≤ 10 % at 256K is left in the kernel itself.

## Stage 3 — fewer, fatter kernels in the decode step (+10–20 %, medium–large)

Graph-level fusions in ggml-cuda for the patterns this model repeats 30–40 times per token:

* `rms_norm × weight` folded into the prologue of every BF16N GEMV that consumes it (the GEMV stages x anyway).
* router GEMV + shared-expert gate (same input) in one launch; α and β projections of the gated delta net in one launch.
* conv-state gather + concat + `ssm_conv` + state write-back as one kernel (≈ 0.5 ms).
* gate and up expert GEMVs in one launch without doubling per-SM shared memory (two LUTs per block measured slower).

Measure each with the node-level nsys breakdown (`docs/RESULTS.md`) and alternating llama-bench runs.

## Stage 4 — GEMV efficiency (+3–8 %, medium)

* Balance per-launch tails: smaller units (`R`) with dynamic work distribution so the slowest warp does not set the
  kernel time; measure the warp finish-time spread with the `DIAG` mode of `bench/hufpair_gemv.cu`.
* Benchmarks must rotate several copies of the data: the expert-down working set (5.6 MB) is close to the 4 MB L2 and
  makes single-copy microbenchmarks optimistic.
* Small dense matrices: split K across warps, or entropy-code them with smaller units now that launch dimensions adapt.

## Stage 5 — compression = speed (≈ 1–2 %, small–medium)

The step is bandwidth-bound, so every saved bit is time. Code the exponent jointly with the top mantissa bit (measured
mantissa+sign entropy given the exponent: 7.86 of 8 bits), or use context from the neighbouring pair. Must stay a single
table lookup per two weights.

## Stage 6 — more tokens per weight pass (× 1.5–2, out of scope so far)

Multi-token prediction (the checkpoint ships an MTP head) or speculative decoding: one pass over the weights verifies
several tokens. Needs a multi-column GEMV that decodes each weight once and multiplies it with all columns (today every
column re-decodes). This is the only lever beyond ≈ 45 tok/s on this hardware.

## Stage 7 — platform

* Desktop off: measured, +0.5 % with no windows open (RESULTS.md); one open application window cost 13 %. Nothing to gain
  beyond running headless.
* DRAM above 3199 MHz: researched only (see [ORIN_NOTES.md](ORIN_NOTES.md)); ≤ 6 % with a real risk of silent corruption.
  Not planned without a multi-day memory-integrity test.

## Stage 8 — portability

* Other models with BF16 weights and rows that are multiples of 512 (format and policy are model-agnostic).
* Other CUDA GPUs (more compute per byte should make decoding cheaper); account the exponent buffers in llama.cpp's
  memory fitting.
* Upstreaming: a self-contained `GGML_TYPE_BF16N` with GGUF serialisation would remove the load-time encode (≈ 3–4 min).
