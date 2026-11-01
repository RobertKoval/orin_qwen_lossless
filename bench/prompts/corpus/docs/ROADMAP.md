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

1. **MoE kernel** (≈ 50 %): latency-bound, not compute-bound. A fragment-order layout of the expert tensors (each lane
   decodes exactly its `mma` A-fragment elements and multiplies them at once, activations staged once per block) is
   prototyped ([RESULTS.md](RESULTS.md#prefill-fragment-order-expert-layout-prototype)): down 9.48 → 3.96 ms per layer
   at 2048 tokens, 4.84 → 3.00 ms at 512; gate/up (K = 2048) not better yet. **Next step (planned for the week of
   11-02): the fragment order for the down experts** — encoder at load, the batched kernel and the decode GEMV in
   `bf16n.cu`, then pp8192 / tg64 / MTP and KLD. Estimate +13–16 % prefill for ≈ −0.4 % decode (≈ −1.3 % with the draft).
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

1. **Lossless KV compression — dropped (user decision 10-29)** ([RESULTS.md](RESULTS.md#kv-cache-compressibility-measured-dropped)):
   only the F16 exponent is redundant. Per-channel Huffman reaches 0.848 of F16 but needs variable-size rows; flash
   attention wants fixed rows, and a 3-bit per-channel exponent code with an escape area gives ≈ 0.89 plus a spill
   path for the rare rows that overflow. That is ≈ −0.55 GiB at 262K, ≈ +5 % decode at 224K and ≈ +1 % up to 32K, for
   decoding inside the flash-attention tile loader and an encoder in SET_ROWS. The multi-column GEMV (Stage 6) comes
   first. Memory is no longer the constraint: the MTP draft fits next to the full window and an image.
2. **Flash-attention efficiency**: at 97 % of the plain-read bandwidth, nothing is left in the kernel itself.

## Stage 3 — fewer, fatter kernels in the decode step (measured: ≤ 4–6 %, parked)

Measured on 10-29 ([RESULTS.md](RESULTS.md#decode-what-a-persistent-kernel-could-still-win-measured)): launches cost
nothing in the CUDA graph (0.1 µs gaps), concurrent graph branches gain nothing, an L2-warm start saves 1–5 µs per GEMV.
What is left is the ramp and tail inside mid-size GEMVs (≈ 2–2.5 ms per step), recoverable only by a persistent kernel
that decodes the next matrix's first units before its input exists: ≤ 4–6 %, several days, uncertain. The small-kernel
fusions below are ≤ 1 % each (≈ 3 ms per step in ≈ 1000 kernels of a dozen kinds); the 2-token conv-state concat that
cost 2.7 ms per MTP verification step is fixed.

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

## Stage 6 — more tokens per weight pass (dense part, cheaper draft head, 2-token concat done: +29 % with the MTP draft)

The checkpoint's MTP head accepts 85 % of single drafts. The multi-column GEMV (RESULTS.md, MTP section) decodes each
dense row once for up to 4 tokens: verifying 2 tokens costs ≈ 1.1× one on the dense matrices, so one draft token gives
36.9 tok/s (+18.5 %, was +6.6 %); a draft head over the first 65536 token ids instead of all 248,320 adds 5 % more
(38.8 tok/s, +25 %). Two or three draft tokens are still slower (36.7 / 37.8) because every verified token reads its
own 8 experts. What is left here:

* Experts shared by the verified tokens: measured and dropped. Consecutive tokens share ≈ 2.9 of 8 experts, but the
  per-column GEMV already reads a shared expert from DRAM once (L2); explicit grouping only saves its duplicate
  decoding (≈ 2 %), and every grouped kernel tried cost more (RESULTS.md).
* The 4-bit NIBBLE GEMVs still decode per column: ≈ +1.5 % at best, below the bar.
* The MTP layer still runs its full 256-expert MoE per draft token; with the trimmed head that is most of the
  remaining draft cost.

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
