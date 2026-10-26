# Results

Hardware: Jetson AGX Orin Developer Kit 64GB (P3701-0005, GA10B SM87 @ 1300.5 MHz, LPDDR5 3199 MHz = 204.8 GB/s,
SK hynix DRAM), L4T R36.5, CUDA 12.6, `nvpmodel` MAXN, `jetson_clocks` on. Model: `Qwen/Qwen3.6-35B-A3B`, BF16 GGUF
converted with llama.cpp. llama.cpp `b29c606` + `patches/llama.cpp-b29c606-bf16n.patch`. Single sequence, greedy.

## Memory

| | bytes | ratio |
|---|---:|---:|
| BF16 text weights (all 472 2-D/3-D tensors incl. embedding, lm_head, routers) | 69.32 GB | 1.0000 |
| BF16N resident (planes + aux) | **46.52 GB** | **0.6711** |
| of which HUFPAIR (34.49 B weights) | | 0.6707 |
| of which NIBBLE (0.17 B weights) | | 0.7588 |

Every tensor passed the bit-exact GPU round-trip check at load (472/472). The uncompressed BF16 model (64.6 GiB) does not
fit into the 61 GiB of unified memory at all.

## Throughput

`llama-bench -ngl 99 -fa 1 -t 2`, each row a separate process (each process re-encodes and re-verifies all weights).

| configuration | tg64 @ 0 | tg64 @ 2048 | notes |
|---|---:|---:|---|
| previous PyTorch/HF runtime, fused TCA-format GEMV | 4.97 | — | 201 ms/token |
| BF16N v1: nibble dense + single-symbol Huffman experts | 24.65 | — | |
| + fused gate/up/SWIGLU, activations through L1 | 24.77 | 23.55 | |
| + exponent-pair Huffman experts (HUFPAIR) | 28.80 | 28.40 | |
| + HUFPAIR for dense ≥ 8M weights, router F32→BF16N | 29.31 | 29.31 | |
| + GDN state read in place, no gate/up fusion for HUFPAIR | 29.0–29.4 | 29.2–29.6 | 3 runs |
| + embedding on GPU (nothing on the CPU) | 28.70 | 28.72 | busy desktop, see below |
| + warm-start prologue in the pair-Huffman GEMV (final) | **30.3** | **30.3** | mean of 6 processes, range 28.9–31.0 |

Final A/B in alternating processes (three quiet rounds): without the warm start 30.36 / 30.42 / 30.42, with it
30.91 / 30.83 / 30.91 tok/s (+1.5 % in every pair). The same final patch on a clean upstream tree measured 29.76 / 29.84 in a
single noisier run.

Prefill with the first published patch (pp512 / pp2048): 307 / 305–311 tok/s (exact dequantisation + stock BF16 cuBLAS
GEMM per expert). The batched MoE kernel below replaced that path.

The patch was also verified on a clean upstream checkout (`git archive b29c606` + `patch -p1`, built with
`scripts/build_llama.sh` flags): it applies without fuzz, builds, passes all 472 round-trip checks and decodes at
28.6–29.6 tok/s depending on the run.

Run-to-run spread between processes was ±2–3 % until the page cache was dropped and memory compacted before each run
(`echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory`); with that it is ±0.3 %. Earlier comparisons
above therefore use alternating runs of the two variants in separate processes.

Desktop on vs off (gdm stopped), final patch on clean upstream, caches dropped and memory compacted before each run,
three processes each:

| | tg64 @ 0 | tg64 @ 2048 | bwprobe |
|---|---|---|---|
| desktop running | 29.71 / 29.87 / 29.77 | 30.49 / 30.53 / 30.44 | 179.2–179.7 GB/s |
| desktop stopped | 29.91 / 29.83 / 29.86 | 30.69 / 30.70 / 30.66 | 179.7–179.8 GB/s |

With no application windows open the desktop costs ≈ 0.5 %; the 13 % loss below came from one open window. Desktop activity matters a lot on Orin because the compositor and remote
desktop share the GPU and DRAM: with the Ubuntu *Software Updater* window open, `bwprobe` medians dropped to 156–168 GB/s
(from 180) and decode to **25.0 tok/s**; closing that single window restored 180 GB/s and 28.7 tok/s.

## Long context

Model geometry (from the GGUF): 40 layers, every 4th is full attention (10 layers, 2 KV heads × 256), the rest are Gated
DeltaNet with a fixed-size state. The F16 KV cache therefore costs 10 × 2 × 2 × 256 × 2 B = **20 KiB per token**,
5.0 GiB for the native 262,144-token window.

`llama-bench -ngl 99 -fa 1 -p 0 -n 32 -d <depth>`, final patch on clean upstream, desktop running, caches dropped and
memory compacted before the process. Depths 32K–256K are one repetition from a single sweep; 2K and 8K are `-r 3` from a
second process (the sweep's first test right after loading read 26.4 tok/s at depth 0, a second process measured
28.8 ± 1.7 there with one slow repetition, so the first test after load is not representative; the dedicated three-process
runs above give 29.8).

| depth | tok/s | ms / token | KV read per token |
|---:|---:|---:|---:|
| 2048 | 30.39 ± 0.43 | 32.9 | 0.04 GB |
| 8192 | 29.69 ± 0.42 (sweep: 29.28) | 33.7 | 0.17 GB |
| 32768 | 26.30 | 38.0 | 0.67 GB |
| 65536 | 23.37 | 42.8 | 1.34 GB |
| 131072 | 19.30 | 51.8 | 2.68 GB |
| 262144 | 14.42 | 69.3 | 5.37 GB |

A least-squares line through the sweep gives `t ≈ 33.5 ms + 0.138 µs × depth`, i.e. 20 KiB per token read at
≈ 149 GB/s (83 % of the measured 180 GB/s read bandwidth); flash attention over the KV cache is near its floor, and at 256K
the cache traffic is larger than the weight traffic. At 262,144 tokens the run fit, but the kernel's `MemAvailable` fell
below 100 MiB (desktop running, `-ub 512`).

Prefill of an 8192-token prompt against the micro-batch size with the first published patch (`-p 8192 -n 0 -b 8192
-ub <n>`, one process per pair of values; MoE prefill dequantised the used experts once per micro-batch):

| `-ub` | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| pp8192, tok/s | 265.4 | 351.8 | 433.5 | 501.8 |

## Prefill: batched MoE kernel

**Before.** nsys of pp2048 with the published patch: at `-ub 2048` the GPU was idle 45 % of the time while the host looped
over 256 experts for each of the 120 MoE matrices (ids copied to the host, two stream syncs per matrix, a dequant, a
conversion and a cuBLAS GEMM per expert); `k_hp_dequant` took 20 % of the GPU time, experts with ≤ 8 tokens went through
the GEMV (36 % at `-ub 512`).

**Now.** One launch per `MUL_MAT_ID`: `ggml_cuda_launch_mm_ids_helper` sorts the tokens by expert on the GPU, the
activations are converted to BF16 once, and `k_hp_mm_ids` decodes each expert's weights once per block into a shared slab
and multiplies them on the tensor cores with all tokens routed to that expert (`mma.m16n8k16`, FP32 accumulation). This
needed chunk-major lane streams for the gate/up experts (a K-slab of many rows must be decodable at once); the format,
size and the bit-exact round-trip are otherwise unchanged ([FORMAT.md](FORMAT.md)).

Per matrix, real `blk.20` weights, 8 of 256 experts per token drawn uniformly, GPU warmed for 2 s
([`bench/moe_prefill.cpp`](../bench/moe_prefill.cpp); ms per `MUL_MAT_ID`):

| tokens | gate (K 2048, N 512): old → new | down (K 512, N 2048): old → new |
|---:|---:|---:|
| 512 | 23.6 → **6.3** | 16.5 → **10.1** |
| 2048 | 35.5 → **9.2** | 33.5 → **13.4** |
| 4096 | 53.8 → **15.1** | 47.7 → **23.6** |

Every result is checked against a CPU reference (exact BF16 weights, BF16-rounded activations, double accumulation):
max |error| / Σ|w·x| ≤ 4.1e-7 for 9 to 4096 tokens, also with skewed routing; the GEMV path (≤ 8 tokens, F32
activations) ≤ 1.1e-7.

End to end, alternating processes, caches dropped before each (`llama-bench -p 2048,8192 -n 0 -b 8192`, tok/s, two runs
each):

| | published patch | batched MoE kernel |
|---|---:|---:|
| pp2048, `-ub 512` | 271.7 / 272.3 | **507.4 / 506.2** |
| pp2048, `-ub 2048` | 429.6 / 437.9 | **767.2 / 765.8** |
| pp8192, `-ub 512` | 264.6 / 266.5 | **495.5 / 495.4** |
| pp8192, `-ub 2048` | 435.9 / 435.7 | **768.5 / 766.7** |
| pp8192, `-ub 4096` | 505.2 / 504.8 | **811.0 / 809.5** |

Decode is unchanged: tg64 @ 2048 30.32 / 30.30 tok/s with chunk-major gate/up vs 30.38 / 30.37 with the row-major format
(same build, `GGML_BF16N_NO_CM`, alternating). Getting there took three attempts: the first chunk-major GEMV, with 8 rows
unrolled in its loop body, cost 3.2–3.8 % of decode speed; nsys in real decode showed 97 vs 84 µs per gate/up call while
an isolated microbenchmark (the same 8 experts every repetition, partly L2-resident) showed the opposite, and the SASS
had the same instruction count per weight but a loop body twice as large. A 4-row body with rotating accumulators fits
the instruction cache and restores parity. Adding prefetch registers instead pushed it past the 64-register cap (spills,
−8 %).

Model-level check (`scripts/run_prompts.sh`, now with a ~1500-token prompt, greedy, against `GGML_BF16N_REFERENCE=1`):
the batched path is text-identical on the three short prompts and first differs at token 64 of the long one; the
published prefill path differs on one short prompt (token 47) and at the same token 64 of the long one. Median top-1
|Δlogprob| 3.9e-5 vs 3.6e-5: both are summation-order differences at near-ties.

Where pp2048 (`-ub 2048`, one forward ≈ 2.7 s) goes now: `k_hp_mm_ids` 46 %, `gated_delta_net_cuda` 18 % (16.8 ms per
layer), `concat_non_cont` of the conv state 8 %, cuBLAS GEMMs of the dense matrices 7 %, `rms_norm` + conversions 9 %,
GPU idle 12 %.

Measurement note: the GPU clock governor (`nvhost_podgov`, 306–1300 MHz) ramps up only under sustained load, so kernel
microbenchmarks need a warm-up; without one, two runs of the same kernel differed by 1.5×.

## KV-cache compressibility (measured, not implemented)

32,768 tokens of wikitext-2 (test split) prefilled through `llama-server` (`-fa on -ub 2048`, 402 tok/s), the slot saved
with `POST /slots/0?action=save` and analysed with [`bench/kv_entropy.py`](../bench/kv_entropy.py): 10 attention layers ×
K and V, each 32768 × 512 F16 values. Entropies are plug-in estimates (≥ 32K samples per channel); means over the 10 layers:

| field (bits in F16) | K | V |
|---|---:|---:|
| sign (1): one table / per channel | 1.000 / 0.944 | 1.000 / 0.925 |
| exponent (5): one table per tensor | 2.695 | 2.700 |
| exponent: pairs of neighbouring channels, per tensor | 2.689 | 2.698 |
| exponent: one table per channel | 2.499 | 2.513 |
| exponent: per channel, given the previous token | 2.383 | 2.457 |
| top 3 mantissa bits (3), given exponent and channel | 2.910 | 2.872 |
| mantissa (10), per channel | 9.948 | 9.899 |

| scheme, bits per value (ratio) | K | V |
|---|---:|---:|
| exponent Huffman per channel, real code lengths | 13.54 (0.846) | 13.58 (0.849) |
| same, tables built on the first 4096 tokens only | 13.56 (0.847) | 13.53 (0.846) |
| entropy bound: sign \| channel + exponent \| channel, previous token + mantissa \| channel | 13.28 (0.830) | 13.28 (0.830) |
| `lzma -6` on byte planes (layer 4), for reference | 0.859 | 0.856 |

* Only the exponent is redundant (≈ 2.5 of 5 bits); mantissa and sign are close to random. Lossless F16 KV compression
  therefore tops out at ≈ −17 %, ≈ −15 % with a per-channel Huffman code, against −33 % for the BF16 weights (same ≈ 2.6
  bits of exponent information, but out of 8).
* Per-channel tables are worth ≈ 0.2 bit, the previous token another ≈ 0.1; neighbouring-channel pairs are worth nothing.
  Rotated (RoPE) dimensions of K are slightly more compressible (2.2–2.45 bits) than the pass-through ones (2.54).
* Tables built on the first 4096 tokens cost only 0.02–0.06 bit on the rest, so static per-channel tables fit an
  append-only cache.
* Outlier: V of the last attention layer (layer 39) needs only 1.93 exponent bits and 0.68 sign bits per channel.
* With the long-context fit above, KV traffic × 0.85 predicts 15.6 instead of 14.4 tok/s at 256K (+8 %), ≈ +3 % at 64K,
  and 0.8 GiB less memory at 262K. A lossy Q8_0 cache would predict ≈ 19 tok/s at 256K.
* Single text domain (English Wikipedia prose); code or other languages may differ.

## Where a decode step goes (nsys node-level trace; tracing adds ≈ 10 %, captured with the busy desktop)

| part | ms / token |
|---|---:|
| HUFPAIR matrix–vector (experts, lm_head, qkv, gates, out projections) | ≈ 31.6 (traced) |
| NIBBLE matrix–vector + fused shared-expert gate/up | ≈ 3.0 |
| gated delta net, conv, norms, attention, routing, small ops | ≈ 3.9 |
| idle between steps (graph launch ≈ 0.3 ms + host work) | ≈ 0.5 untraced |

Per-shape kernel throughput (microbenchmarks in `bench/`, real weights, working sets ≫ L2):

| shape | raw BF16 kernel | cuBLAS | NIBBLE | HUFPAIR |
|---|---:|---:|---:|---:|
| lm_head 248320×2048 | 5.70 ms | 5.69 ms | 4.38 ms | **4.29 ms** |
| qkv 8192×2048 | 0.200 ms | 0.201 ms | 0.155 ms | **0.151 ms** |
| 8 experts gate+up 8192×2048 | 0.200 ms | 0.199 ms | 0.158 ms | 0.141 ms (119 Gw/s) |
| 8 experts down 16384×512 | 0.112 ms | — | 0.093 ms | 0.064 ms (132 Gw/s) |

`bench/bwprobe.cu`: best plain streaming read is 180 GB/s (88 % of 204.8 GB/s); HUFPAIR kernels reach 150–175 GB/s of
compressed bytes.

## Correctness

* Weights: bit-exact round trip for every tensor (see above).
* Arithmetic: products of the exact BF16 weights with FP32 activations, FP32 accumulation; only the summation order differs
  from llama.cpp's BF16 kernels. Against an FP64 reference the per-row error is 1e-8…4e-8 of Σ|w·x| (cuBLAS BF16 GEMV: ~4e-7).
* End to end, `scripts/run_prompts.sh` (3 prompts × 96 greedy tokens):
  * BF16N kernels vs `GGML_BF16N_REFERENCE=1` (exact dequant + stock cuBLAS BF16 path): identical text, no top-1
    divergence in 288 tokens, top-1 |Δlogprob| median 1.2e-5 (p99 0.07; the reference rounds activations to BF16).
  * GDN state in place vs gathered copy, embedding on GPU vs CPU: bitwise-identical text and logprobs.

## What did not help (measured)

* Replicated per-bank LUTs (bank-conflict-free): slower — they take L1 the lane streams need.
* Warp-interleaved exponent streams with ballot/shuffle refill: 67 vs 84 Gweights/s.
* Fusing gate+up+SWIGLU for HUFPAIR: −0.6 tok/s (two 16 KiB LUTs per block).
* HUFPAIR for every dense tensor (incl. 32–512-row matrices): 25.1 vs 28.7 tok/s.
* Host threads (`-t 1/2/8`), polling, embedding placement: no effect on decode.
* Batched MoE kernel on the row-major format: a warp can only produce one row at a time, so a block needs 32 lockstep
  steps per unit and re-reads the activations once per row; 20 vs 12 ms for the final kernel (gate, 2048 tokens; both
  measured before the clock warm-up was added, so ±1.5×).
* A 128-row shared slab: leaves 28 KB of L1 for the lane streams (64 KB needed); decode per warp was ≈ 4.6× slower than
  with a 64-row slab (same caveat).
* Prefetching activations and planes more deeply in the chunk-major GEMV: register spills at the 64-register cap, −8 %
  decode.
