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
| + embedding on GPU (nothing on the CPU) | **28.70** | **28.72** | measured with a busy desktop, see below |

Prefill (pp512 / pp2048): **307 / 305 tok/s** (exact dequantisation + stock BF16 cuBLAS GEMM), i.e. ≈ 6.7 s for a
2048-token prompt.

Run-to-run spread between processes is about ±2 %. Desktop activity matters a lot on Orin because the compositor and remote
desktop share the GPU and DRAM: with the Ubuntu *Software Updater* window open, `bwprobe` medians dropped to 156–168 GB/s
(from 180) and decode to **25.0 tok/s**; closing that single window restored 180 GB/s and 28.7 tok/s. A run with the GUI
stopped (`systemctl isolate multi-user.target`) is still to be done.

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
