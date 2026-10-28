# Roadmap: next research stages

Where the time goes today, and what is left on the table. Numbers are per decode step (≈ 32 ms at ≈ 31 tok/s) on the
AGX Orin 64GB unless stated otherwise; "gain" estimates come from the measurements in [RESULTS.md](RESULTS.md) and are
hypotheses until measured with alternating A/B runs.

## Ceiling

A decode step streams ≈ 4.0 GB of compressed weights; at the 180 GB/s a plain stream reads, that is **≈ 22 ms
(≈ 45 tok/s)** for this format. The GEMVs' own access pattern (a 512-byte plane row and 32 lane streams per warp) reads
at ≈ 161–170 GB/s, so the practical ceiling with every overhead removed is **≈ 40 tok/s**. At short context, today's
≈ 32 ms splits roughly into (long context adds KV-cache traffic on top, see Stage 2):

| part | ms | gap to its own floor |
|---|---:|---|
| pair-Huffman GEMV (experts, lm_head, projections) | ≈ 26 | work phase at the pattern's DRAM limit; ≈ 3 ms is fixed per launch (cold prologue + tail, ≈ 13 µs) over ≈ 230 launches |
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
760 / 1153 / 1283, and upstream's newer norm and conversion kernels (fused RMS_NORM + SCALE, float4 `rms_norm`, four
elements per thread in the F32→BF16 conversion) to 800 / 1249 / 1403, and lane-stream rings plus a fused gate / up /
SwiGLU MoE kernel ([RESULTS.md](RESULTS.md#prefill-moe-kernel-latency-lane-stream-rings-fused-swiglu)) to
811 / 1279 / 1429. The "12 % GPU idle" of the first profile was cold start; steady state is ≈ 2–3 %. Before the last
step a 2048-token forward at `-ub 2048` (≈ 1.65 s) spent 53 % in the MoE kernel, 11 % in cuBLAS for the dense
matrices, 6 % in the gated delta net, 4.3 % in F32→BF16 conversions, 3.3 % in SwiGLU (now fused), 2.8 % in `rms_norm`,
2.2 % each in the weighted sum of the experts and the broadcast adds, 2.1 % in `ssm_conv`.

1. **MoE kernel** (≈ 50 %): latency-bound, not compute-bound. Nsight Compute and an oracle show that hiding the
   activation loads would cut a layer by 22–25 % at 2048 tokens, but this kernel shape has no registers (gate: 128 at
   512 threads) or shared memory (the 64-row slab) left to do it. Planned after the decode work: a layout of the expert
   tensors in tensor-core fragment order, so each lane decodes exactly its `mma` A-fragment elements and multiplies them
   at once — no slab, no barrier between decode and products, the activations staged once per block by `cp.async` and
   shared by all warps (as Marlin does for 4-bit weights). Estimate ≈ 22 → ≈ 14 ms per layer at 2048 tokens; the expert
   GEMV of the decode path must move to the same layout without losing speed, so it is planned together with decode.
2. **Conversions** (≈ 4 %, at bandwidth): every BF16N matmul converts its F32 input again (≈ 300 conversions per
   forward for ≈ 80 distinct tensors). Converting each tensor once is worth ≈ 2 % and needs a BF16 buffer that lives
   across ops and stays valid under CUDA graph replay.
3. **Dense matrices** (10.4 % + 1.6 % dequant): run them through the same decode-in-shared-memory kernel instead of
   dequantising to BF16 per micro-batch. By a parameter-count estimate (≈ 5.8 TFLOP in 190 ms) cuBLAS already runs at
   ≈ 30 TFLOPS, so the gain is mostly the dequant.
4. **Micro-batch size**: `-ub 2048` is worth 1.52× over 512 and `-ub 4096` another 11 % at 8K; still open: memory headroom
   of large micro-batches at long contexts and whether llama-server should pick it for BF16N models.
5. **GEMV/batched threshold** (now 8 tokens): the first batched kernel cost 1.2 ms at 9 tokens vs 0.6 ms for the GEMV
   at 8; the crossover is probably at 24–32 tokens (speculative decoding and parallel requests sit there).

## Stage 2 — long context (KV-cache memory and traffic)

Measured on the current build: `t ≈ 32.3 ms + 0.118 µs × context` (see [RESULTS.md](RESULTS.md#long-context)). The 10
full-attention layers read 20 KiB of F16 KV cache per token of context at ≈ 174 GB/s (97 % of plain-read bandwidth), so
from ≈ 64K tokens on the cache traffic dominates; 16.6 tok/s at 224K, ≈ 15.8 predicted at 256K. The KV cache stays
lossless (user decision 10-28: compress it the way the weights are compressed, no Q8_0).

1. **Lossless KV compression — measured on a real 32K cache** ([RESULTS.md](RESULTS.md#kv-cache-compressibility-measured-not-implemented)):
   only the F16 exponent is redundant, ≈ −15 % with per-channel Huffman tables (static tables from the first 4K tokens
   work), −17 % entropy bound: 0.8 GiB less memory at 262K and ≈ +8 % decode at 256K (+3 % at 64K). Memory is now the
   constraint: the MTP draft (Stage 6) needs ≈ 1.5–2 GB that the full window with an image does not have. Cost: decoding
   inside the flash-attention kernel, block-wise encoding of the append-only cache. Risk: at 256K a step must decode
   2.7 G values at ≥ 88 G values/s to hide under the compressed read, alongside the attention math; the weight GEMV
   reaches ≈ 125 G values/s with the INT datapath to itself.
2. **Flash-attention efficiency**: at 97 % of the plain-read bandwidth, nothing is left in the kernel itself.

## Stage 3 — fewer, fatter kernels in the decode step (+10–20 %, medium–large)

The main item is a persistent decode kernel per layer (or per block of it): one launch runs the layer's GEMVs and small
ops as stages with grid-wide barriers, and the next stage's independent prologue (exponent table, unit offsets, first
stream words and plane rows) is issued before the barrier, so it overlaps the current stage's tail instead of following
it. Estimate: ≈ 2 ms of the per-launch cost plus ≈ 2 ms of small kernels, ≈ +12–15 % (31 → ≈ 35 tok/s). The GEMV
stages should take the expert layout as a parameter, since the prefill plan (Stage 1) wants a fragment-order layout.

Smaller graph-level fusions for the patterns this model repeats 30–40 times per token (each ≈ 1 %, worth it only
bundled or inside the persistent kernel):

* `rms_norm × weight` folded into the prologue of every BF16N GEMV that consumes it (the GEMV stages x anyway).
* router GEMV + shared-expert gate (same input) in one launch; α and β projections of the gated delta net in one launch.
* conv-state gather + concat + `ssm_conv` + state write-back as one kernel (≈ 0.5 ms).
* gate and up expert GEMVs in one launch without doubling per-SM shared memory (two LUTs per block measured slower).

Measure each with the node-level nsys breakdown (`docs/RESULTS.md`) and alternating llama-bench runs.

## Stage 4 — GEMV efficiency (+1–3 %, medium)

The work phase is already at the DRAM limit of the access pattern; `cp.async` rings that remove the long-scoreboard
stalls did not make it faster ([RESULTS.md](RESULTS.md#what-did-not-help-measured)). What is left here is the tail:

* Balance per-launch tails: smaller units (`R`) with dynamic work distribution so the slowest warp does not set the
  kernel time; measure the warp finish-time spread with the `DIAG` mode of `bench/hufpair_gemv.cu`.
* Benchmarks must rotate several copies of the data: the expert-down working set (5.6 MB) is close to the 4 MB L2 and
  makes single-copy microbenchmarks optimistic.
* Small dense matrices: split K across warps, or entropy-code them with smaller units now that launch dimensions adapt.

## Stage 5 — compression = speed (≈ 1–2 %, small–medium)

The step is bandwidth-bound, so every saved bit is time. Code the exponent jointly with the top mantissa bit (measured
mantissa+sign entropy given the exponent: 7.86 of 8 bits), or use context from the neighbouring pair. Must stay a single
table lookup per two weights.

## Stage 6 — more tokens per weight pass (+35–50 %, medium–large)

Measured (RESULTS.md, MTP section): the checkpoint's MTP head as a draft accepts 85 % of single drafts and gives
+7–9 % with one draft token; with three it would average 2.87 tokens per step, but verifying 4 tokens costs ≈ 3 decode
steps because every column re-decodes the weights (dense 2.6× one column at 3–4 columns) and each token brings its own
experts (3.7× at 4 tokens). A multi-column GEMV that decodes each weight once and multiplies it with all columns (and
decodes a shared expert once for the tokens routed to it) leaves the experts as the floor: ≈ 42–46 tok/s estimated.
Memory comes first: the draft adds ≈ 1.5–2 GB, which the full window with an image does not have today (share the
model's embedding and output matrices with the draft, compress the KV cache). This is the only lever beyond ≈ 45 tok/s
on this hardware.

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
