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

## Stage 1 — prefill

Done: the batched MoE kernel ([RESULTS.md](RESULTS.md#prefill-batched-moe-kernel)), a tiled transpose for the
conv-state concat and a persistent MoE kernel (one block per SM, LUT loaded once, the next work item's decode started
during the current one's last step, units interleaved per warp for down at small token counts) took prefill from
306 / 530 / 613 to 665 / 943 / 1018 tok/s (pp8192 at `-ub 512 / 2048 / 4096`, pinned clocks) for −0.3 % decode, and
upstream's chunked gated delta net (ported; 3.3 instead of 15.8 ms per layer for 2048 tokens) took it to
760 / 1153 / 1283. The "12 % GPU idle" of the first profile was cold start; steady state is ≈ 2–3 %. A 2048-token forward
at `-ub 2048` (1.83 s) now spends 48 % in the MoE kernel, 10.4 % in cuBLAS for the dense matrices, 6.9 % in F32→BF16
conversions, 6.8 % in `rms_norm`, 5.5 % in the gated delta net, 3.0 % in SwiGLU, 1.5 % in the conv-state concat.

1. **MoE kernel** (48 %): decode and products now overlap; at ≥ 2048 tokens down is bound by its tensor-core phase
   (products alone 6.5 ms at 2048 tokens, ≈ 5.3 TFLOPS), and its single-chain variant (IL1, ≥ 1537 tokens) runs ≈ 11 %
   slower than a schedule the compiler produced once (9.5 vs 10.6 ms at 2048 tokens) but that no clean source form has
   reproduced yet. Open: the board's measured BF16 `mma` peak, `ldmatrix` and larger token tiles for down.
2. **Conversions** (6.9 %): the MoE kernel and the dense cuBLAS path each convert F32 activations to BF16; the norm that
   produces them could write BF16 directly.
3. **Dense matrices** (10.4 % + 1.6 % dequant): run them through the same decode-in-shared-memory kernel instead of
   dequantising to BF16 per micro-batch. By a parameter-count estimate (≈ 5.8 TFLOP in 190 ms) cuBLAS already runs at
   ≈ 30 TFLOPS, so the gain is mostly the dequant.
4. **Micro-batch size**: `-ub 2048` is worth 1.52× over 512 and `-ub 4096` another 11 % at 8K; still open: memory headroom
   of large micro-batches at long contexts and whether llama-server should pick it for BF16N models.
5. **GEMV/batched threshold** (now 8 tokens): the first batched kernel cost 1.2 ms at 9 tokens vs 0.6 ms for the GEMV
   at 8; the crossover is probably at 24–32 tokens (speculative decoding and parallel requests sit there).

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
