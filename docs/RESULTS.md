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

Prefill (pp512 / pp2048): **307 / 305–311 tok/s** (exact dequantisation + stock BF16 cuBLAS GEMM), i.e. ≈ 6.7 s for a
2048-token prompt.

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

Prefill of an 8192-token prompt against the micro-batch size (`-p 8192 -n 0 -b 8192 -ub <n>`, one process per pair of
values; MoE prefill dequantises the used experts once per micro-batch):

| `-ub` | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| pp8192, tok/s | 265.4 | 351.8 | 433.5 | 501.8 |

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
