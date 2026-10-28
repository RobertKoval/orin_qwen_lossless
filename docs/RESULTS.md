# Results

Hardware: Jetson AGX Orin Developer Kit 64GB (P3701-0005, GA10B SM87 @ 1300.5 MHz, LPDDR5 3199 MHz = 204.8 GB/s,
SK hynix DRAM), L4T R36.5, CUDA 12.6, `nvpmodel` MAXN. Model: `Qwen/Qwen3.6-35B-A3B`, BF16 GGUF
converted with llama.cpp. llama.cpp `338fce6` + `patches/llama.cpp-338fce6-bf16n.patch` (sections before
[the rebase](#rebase-onto-upstream-llamacpp-338fce6) were measured on `b29c606` with the patch of that time). Single
sequence, greedy.

Clocks: until 2026-10-26 the board ran with dynamic clocks (GPU devfreq `nvhost_podgov` 306–1300.5 MHz, CPU `schedutil`
from 729 MHz), not with `jetson_clocks` as this file said earlier. Since a reboot that day the GPU is pinned at
1300.5 MHz and the CPU at ≥ 2.2 GHz, and the measurement scripts print both before every run. Sections marked
**pinned clocks** were measured that way, as were Long context, Serving and the MTP section; everything else
(Throughput, the first prefill table) with dynamic clocks. Pinning made decode ≈ 1–3 % faster and the host-bound published prefill path 16–22 % faster, so
ratios between variants are only compared within one clock state.

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

Current build (upstream `338fce6` + patch), pinned clocks, desktop running, caches dropped and memory compacted before
the process: `llama-bench -ngl 99 -fa 1 -t 2 -b 2048 -ub 2048 -p 0 -n 32 -d <depth> -r 1`, one process for the whole
sweep. At depth 262,144 llama-bench was OOM-killed twice with `-ub 2048` (its test needs 262,176 cells, and it
allocates more than the server); the 224K point is the production server instead (`-c 262144 -ub 2048`, projector
loaded): a 224,303-token prompt (llama.cpp sources) prefilled in 348 s (648 tok/s on average), 21 tokens generated at
16.61 tok/s, correct answer, zone free memory ≥ 3.85 GB throughout.

| depth | tok/s | ms / token | previous base (`b29c606`, dynamic clocks) |
|---:|---:|---:|---:|
| 2048 | 30.64 | 32.6 | 30.39 |
| 8192 | 30.04 | 33.3 | 29.69 |
| 32768 | 27.66 | 36.2 | 26.30 |
| 65536 | 25.10 | 39.8 | 23.37 |
| 131072 | 20.90 | 47.8 | 19.30 |
| 224303 (server) | 16.61 | 60.2 | — |
| 262144 | — | — | 14.42 |

A least-squares line through 2K–128K gives `t ≈ 32.3 ms + 0.118 µs × depth` (previous base: 33.5 ms + 0.138 µs), i.e. 20 KiB
per token read at ≈ 174 GB/s (97 % of the measured 180 GB/s read bandwidth, 85 % of the 204.8 GB/s peak); the line
predicts 15.8 tok/s at 262,144 and 17.0 at 224K (measured 16.6 over only 21 tokens). The gain at depth (+5 % at 32K,
+8 % at 128K) is upstream's flash-attention work picked up by the rebase; the previous base also ran with dynamic clocks.

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

**Now.** One persistent launch per `MUL_MAT_ID`, one block per SM. `ggml_cuda_launch_mm_ids_helper` sorts the
tokens by expert on the GPU and the activations are converted to BF16 once; each block of `k_hp_mm_ids` builds its own
work list (an expert × a tile of the tokens routed to it × a group of units) from the routing bounds in shared memory,
loads the exponent LUT once, decodes the weights once per token tile into a shared slab and multiplies them on the
tensor cores (`mma.m16n8k16`, FP32 accumulation). The next item's lane streams and plane rows start during the current
item's last step, so decode and products overlap. While the experts have few tokens (≤ 48 on average) down interleaves
two or four units per warp to give each SM more independent decode chains. This needed chunk-major lane streams for the
gate/up experts (a K-slab of many rows must be decodable at once); the format, size and the bit-exact round-trip are
otherwise unchanged ([FORMAT.md](FORMAT.md)).

Per matrix, real `blk.20` weights, 8 of 256 experts per token drawn uniformly, pinned clocks; median of 30
individually synchronised repetitions, page cache dropped before each run
([`bench/moe_prefill.cpp`](../bench/moe_prefill.cpp); ms per `MUL_MAT_ID`; dequant + cuBLAS = `GGML_BF16N_NO_MMA=1`,
the published path):

| tokens | gate (K 2048, N 512): dequant + cuBLAS / 5fd8a05 / this patch | down (K 512, N 2048): dequant + cuBLAS / 5fd8a05 / this patch |
|---:|---:|---:|
| 512 | 20.8 / 6.0 / **4.2** | 14.9 / 10.0 / **4.9** |
| 2048 | 26.6 / 8.8 / **6.5** | 20.5 / 13.1 / **10.6** |
| 4096 | 37.2 / 14.3 / **10.8** | 28.3 / 22.9 / **18.0** |

An earlier version of this table was measured with dynamic clocks and averaged the repetitions; it overstated the
dequant + cuBLAS path (gate 23.6 / 35.5 / 53.8 ms).

Every result is checked against a CPU reference (exact BF16 weights, BF16-rounded activations, double accumulation):
max |error| / Σ|w·x| ≤ 7e-7 for 9 to 4096 tokens, uniform and skewed routing, on 64 random output columns per run and
on every column (`check_cols` = 8 × tokens) at 100, 768, 769, 1537 and 4096 tokens, both sides of the down interleave
thresholds; the GEMV path (≤ 8 tokens, F32 activations) ≤ 1.1e-7.

End to end at pinned clocks, alternating processes with caches dropped before each (`tg64 -r 3`, then
`-p 8192 -n 0 -b 8192 -r 2`), tok/s. The first column is the published patch, the second the first batched kernel
(commit 65baf68 plus the conv-state concat below, 5fd8a05), the third the persistent kernel (66f5be6); the 5fd8a05 numbers
reproduce within 0.3 % across the two sessions, which ran published / 5fd8a05 and 5fd8a05 / this patch:

| | published patch | 5fd8a05 | 66f5be6 | vs published |
|---|---:|---:|---:|---:|
| pp8192, `-ub 512` | 306.5 / 306.0 | 514.6 / 513.3 | **664.2 / 665.1** | 2.17× |
| pp8192, `-ub 2048` | 526.5 / 532.7 | 819.2 / 816.4 | **942.0 / 944.9** | 1.78× |
| pp8192, `-ub 4096` | 613.1 / 612.5 | 900.1 / 900.1 | **1017.3 / 1018.3** | 1.66× |
| tg64 @ 0 | 30.64 / 30.67 | 30.50 / 30.51 | 30.57 / 30.57 | −0.3 % |
| tg64 @ 2048 | 30.70 / 30.66 | 30.52 / 30.56 | 30.59 / 30.63 | −0.3 % |

With dynamic clocks the first batched kernel had measured 1.6–1.9× over the published patch; that was inflated, because
the published path's host loop over experts ran slower with dynamic clocks.

Model-level check (`scripts/run_prompts.sh`: three short prompts and one of ~1500 tokens, greedy, against
`GGML_BF16N_REFERENCE=1`): text-identical on three of the four, including the long one; the fourth first differs at
token 47, where the published prefill path differs too. Median top-1 |Δlogprob| 6.9e-5 (p99 0.10, max 0.20 over 381
tokens; the first batched kernel 3.9e-5, the published path 3.6e-5). The outputs are not bitwise those of 5fd8a05: the
token tiles and the K-split summation order changed. The every-column check above rules out a fault hidden in columns
the 64-column sample skips.

### Conv-state concat

Each Gated DeltaNet layer prepends its conv state to the micro-batch: `concat(conv_state [3, C], transpose(qkv) [T, C])`
along dim 0, i.e. a transpose of a T × C matrix. The generic `concat_non_cont` read the transposed source with a stride
of C·4 bytes (one 4-byte read per 32-byte sector): 7.6 ms per layer at T = 2048, 8 % of the forward.
`concat_dim0_transposed` stages 32 × 32 tiles through shared memory: **0.80 ms**. Same build with a temporary switch,
alternating, pp2048: `-ub 512` 511.8 / 512.8 → 529.8 / 530.8 (+3.5 %), `-ub 2048` 779.2 / 783.6 → 838.6 / 838.6
(+7.3 %); outputs bitwise identical. The tiled path is taken only when both dims of the transposed source are ≥ 32:
with one token (decode) the tile is almost empty and it cost 0.85 % of decode speed. `test-backend-ops` has cases for
transposed sources (partial tiles, 31/32/33 in both dims, one token; F32/F16/BF16/I32).

### GPU idle

The earlier profile's "12 % idle" was cold start inside the one forward nsys traced per kernel: lazy loading of CUDA
modules, the first cuBLAS call (208 ms), growth of the memory pool. Later forwards run as CUDA graphs (the MoE matrices
no longer sync with the host), so the 3-repetition profile traces only the first one. Steady state is ≈ 2 % idle.

Where pp2048 (`-ub 2048`, persistent MoE kernel 66f5be6, one forward 2.20 s under nsys; it was 2.56 s with the first batched kernel)
goes ([`tools/fwd_breakdown.py`](../tools/fwd_breakdown.py) on the nsys export, last forward): `k_hp_mm_ids` 40 % (874 ms, was 1237), `gated_delta_net_cuda` 22 % (15.9 ms per layer, unchanged), cuBLAS GEMMs
of the dense matrices 8.7 %, F32→BF16 conversions 5.8 %, `rms_norm` 5.6 %, SwiGLU 2.5 %, the weighted sum of the
experts 1.7 %, broadcast adds 1.7 %, `ssm_conv` 1.6 %, dense dequant 1.4 %, conv-state concat 1.2 %, idle 2.4 %.

### Cost for decode

Decode pays 0.5 % for the chunk-major gate/up layout. Node-level nsys of `tg64 -r 3` on both trees (193 tokens): the
total GPU time differs by +0.47 %, all of it in the gate/up GEMV, 82.1 µs per call chunk-major vs 79.8 µs row-major
(+2.9 %); every other kernel matches within ±1.4 ms over the run. Keeping both layouts would double the expert memory,
so the trade stays: 1.66–2.17× prefill for −0.5 % decode.

Getting there took four attempts. The first chunk-major GEMV, with 8 rows unrolled in its loop body, cost 3.2–3.8 % of
decode speed; nsys in real decode showed 97 vs 84 µs per gate/up call while an isolated microbenchmark (the same 8
experts every repetition, partly L2-resident) showed the opposite, and the SASS had the same instruction count per
weight but a loop body twice as large. A 4-row body with rotating accumulators fits the instruction cache (measured
with dynamic clocks as parity, 30.32 / 30.30 vs 30.38 / 30.37 with the row-major format; pinned clocks resolve the
remaining 0.5 %). Prefetch registers pushed it past the 64-register cap (spills, −8 %); a plane prefetch ring that
continues across chunk boundaries (64 registers, no spills) did not help.

Measurement note: with dynamic clocks the GPU governor (`nvhost_podgov`, 306–1300 MHz) ramps up only under sustained
load, so kernel microbenchmarks need a warm-up; without one, two runs of the same kernel differed by 1.5×.
`bench/moe_prefill.cpp` warms the GPU for 2 s.

## Prefill: chunked gated delta net (upstream kernel, ported)

In `b29c606` the CUDA gated delta net is a token-by-token recurrence: each warp owns one column of the 128 × 128 state
and does two warp reductions per token, 15.9 ms per layer for 2048 tokens, 22 % of the forward. Upstream llama.cpp has
since merged two replacements, both ported here: #30087 (four state columns per warp, same algorithm, different FP32
summation order) and #29353 (chunked kernel: chunks of 16 tokens, the products on the tensor cores with every FP32
operand split into a high and a low BF16 part and three `mma` per product, FP32 state). The port keeps this patch's
in-place state read (the chunked kernel reads the state rows through the same ids), takes the two swizzled `ldmatrix`
helpers it needs into `mma.cuh`, and adds Orin (cc 8.7) to the chunked kernel's architecture list.
`test-backend-ops -o GATED_DELTA_NET`: 43/43.

One layer against a sequential recurrence in double precision ([`bench/gdn_check.cpp`](../bench/gdn_check.cpp): 32 value
heads, 16 key heads, L2-normalised q and k, β = sigmoid, decay from layer 0's `ssm_a` and `dt_bias`; median of 10,
pinned clocks):

| kernel | ms, 512 tokens | ms, 2048 tokens | relative RMS error, tokens 1024–2047 | final state |
|---|---:|---:|---:|---:|
| one column per warp (`b29c606`) | 3.97 | 15.78 | 1.43e-6 | 1.41e-6 |
| four columns per warp (#30087) | 1.88 | 7.41 | 1.43e-6 | 1.41e-6 |
| chunked (#29353) | **0.87** | **3.30** | 8.3e-6 | 7.0e-6 |

The chunked kernel is ≈ 6× less accurate than the FP32 recurrence, ≈ 200× below one BF16 rounding (≤ 3.9e-3), which
every matmul input in the model takes anyway. Of its launch shapes, upstream's choice for 32 heads (one block of 256
threads per head) is also the fastest on Orin: 3.33 ms at 2048 tokens vs 4.66–7.28 ms for the value-split variants.

End to end, alternating 66f5be6 / this patch, two rounds (pp: `-b 8192 -r 2`, `-ub 4096`: base / new / new / base),
caches dropped before each process, desktop stopped, pinned clocks, tok/s:

| | 66f5be6 | this patch | change |
|---|---:|---:|---:|
| pp2048, `-ub 512` | 683.4 / 684.1 | **781.3 / 780.2** | +14.2 % |
| pp8192, `-ub 512` | 670.4 / 668.4 | **760.0 / 760.4** | +13.6 % |
| pp2048, `-ub 2048` | 966.2 / 968.2 | **1175.4 / 1174.8** | +21.5 % |
| pp8192, `-ub 2048` | 954.1 / 953.8 | **1153.7 / 1152.3** | +20.9 % |
| pp8192, `-ub 4096` | 1024.6 / 1024.8 | **1286.4 / 1280.1** | +25.2 % |
| tg64 @ 0 / @ 2048 | 30.79 / 30.82, 30.81 / 30.85 | 30.81 / 30.87, 30.76 / 30.79 | ±0.1 % |

The four-columns-per-warp kernel alone gives +13 % at `-ub 2048` and +9 % at `-ub 512`. Decode uses the recurrent
kernel either way. A 2048-token forward at `-ub 2048` (nsys, last eager forward) now takes 1.83 s: `k_hp_mm_ids` 48 %,
cuBLAS GEMMs of the dense matrices 10.4 %, F32→BF16 conversions 6.9 %, `rms_norm` 6.8 %, the chunked gated delta net
5.5 % (3.3 ms per layer), SwiGLU 3.0 %, weighted sum of the experts 2.0 %, broadcast adds 2.0 %, `ssm_conv` 1.9 %, dense
dequant 1.6 %, conv-state concat 1.5 %, idle 3.2 %.

**Is the output still the same?** Not bitwise: any change of FP32 summation order moves the next-token distributions,
and this one does too. `llama-perplexity --kl-divergence` on wikitext-2 (test), 8 × 2048 tokens, against 66f5be6 at
`-ub 2048`:

| change against the base run | PPL | mean KLD | same top-1 |
|---|---:|---:|---:|
| none (second run of the same build) | 4.7470 | 0 | 100 % |
| same build, `-ub 512` | 4.7551 | 0.0021 | 98.40 % |
| same build, `GGML_BF16N_REFERENCE=1` (dequant + cuBLAS) | 4.7451 | 0.0019 | 98.25 % |
| four columns per warp | 4.7467 | 0.0021 | 98.44 % |
| chunked kernel (this patch) | 4.7465 | 0.0024 | 98.30 % |
| this patch at `-ub 512` vs 66f5be6 at `-ub 512` (state carried over four micro-batches) | 4.7489 vs 4.7551 | 0.0026 | 98.36 % |

A different micro-batch size, or this patch's own exact-dequantisation reference path, moves the outputs as much as the
new kernel: ≈ 1.6 % of positions are near-ties between the two best tokens, and any rounding difference upstream of them
can flip the choice. In greedy generation the text stays identical up to the first such tie and then continues differently (when the
batched MoE kernel changed the summation order, two of four `scripts/run_prompts.sh` prompts first differed where the
two best tokens had 0.476 vs 0.457 and 0.200 vs 0.199). Perplexity does not move (±0.005).
The weights stay bit-exact, and a given build with given settings is deterministic.

## Prefill: norms and conversions (upstream kernels, ported)

After the chunked gated delta net, conversions and norms were 14 % of a 2048-token forward and ran at 53–103 GB/s.
By launch shape (nsys, pp2048 `-ub 2048`, ebf3f70): 349 F32→BF16 conversions of 2048 × 2048 activations at 245 µs
(≈ 103 GB/s) — every BF16N matmul converts its input again, so the same normed activation is converted 8–9 times per
layer; the gated delta net's q/k L2 norm as `rms_norm` + `scale` over 128-wide rows (631 + 267 µs, 53 GB/s); the hidden
norms (2048 rows of 2048, 1024-thread blocks, 501 µs, 67 GB/s); the gated output norm (32 × 2048 rows of 128, 1276 µs).

Upstream has rewritten all of these since `b29c606`; ported unchanged: #29155 (contiguous conversions four elements per
thread), #29393 (RMS_NORM + SCALE fused into one kernel), #28175 (norm grid limits) and #29720 (float4 `rms_norm`, row
kept in registers). `test-backend-ops` on CUDA: RMS_NORM 53/53, NORM 51/51, L2_NORM 21/21, ROPE 450/450, CPY 249/249,
SCALE 4/4. Afterwards a conversion of 2048 × 2048 takes 136 µs (≈ 185 GB/s) and every norm shape runs at 150–178 GB/s:
conversions 126 → 70 ms, norms + scale 140 → 48 ms per forward.

Alternating ebf3f70 / this patch, two rounds (`-ub 4096`: base / new / new / base), same conditions as above, tok/s:

| | ebf3f70 | this patch | change |
|---|---:|---:|---:|
| pp2048, `-ub 512` | 780.6 / 779.7 | **821.2 / 820.3** | +5.2 % |
| pp8192, `-ub 512` | 759.6 / 759.8 | **800.1 / 799.7** | +5.3 % |
| pp2048, `-ub 2048` | 1177.7 / 1177.8 | **1271.6 / 1261.1** | +7.5 % |
| pp8192, `-ub 2048` | 1155.6 / 1155.4 | **1249.6 / 1248.6** | +8.1 % |
| pp8192, `-ub 4096` | 1277.3 / 1282.4 | **1404.2 / 1401.9** | +9.6 % |
| tg64 @ 0 / @ 2048 | 30.89 / 30.92, 30.85 / 30.86 | 31.08 / 31.11, 31.12 / 31.13 | +0.7 % |

KL against ebf3f70 (`-ub 2048`, as above): PPL 4.7513 vs 4.7465, mean KLD 0.0025, same top-1 98.19 % — the same
summation-order noise as in the table above. Converting each activation only once would save another ≈ 36 ms (≈ 2 %);
not done.

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
* With the long-context fit above (current build), KV traffic × 0.85 predicts 17.1 instead of 15.8 tok/s at 256K (+8 %),
  ≈ +3 % at 64K, and 0.8 GiB less memory at 262K. A lossy Q8_0 cache would predict ≈ 20.5 tok/s at 256K.
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

## Prefill: MoE kernel latency (lane-stream rings, fused SwiGLU)

At 9077d0c the MoE path was ≈ 61 % of a 2048-token forward at `-ub 2048`: gate + up 475 ms, down 400 ms, SwiGLU 51 ms,
weighted sum 37 ms, conversions 33 ms, routing helper 17 ms per forward. Nsight Compute (`moe_prefill`, layer 20,
2048 tokens) shows a latency-bound kernel: gate issues an instruction in 25 % of cycles and keeps the tensor cores 15 %
busy; 43 % of stall samples wait on global loads, 19 % on barriers. Two instructions take most of it: the first `mma`
of each group waiting for its activation fragment (20 % of all samples; the load is issued one group ahead and mostly
hits L2) and the moves of the decoder's word queue (≈ 20 %).

The queue looks four words deep but behaves like one. Lanes refill at different pair positions while the scoreboard is
per warp register, so moving `wq[k]` waits for the newest load into `wq[k]` by any lane. A deeper or batched register
queue keeps that property (and spilled: 2.6× slower). Software prefetch does not help either: `prefetch.global.L1`
(CCTL.E.PF1) halves the `mma` waits but crowds the MIO queue, `prefetch.global.L2` doubles the decoder's load waits
(gate +22–24 %).

An oracle bounds the prize: replacing loads by register values (garbage output) cuts gate / down at 2048 tokens by
17 / 23 % (lane-stream words), 10 / 12 % (sign|mantissa planes), 3 / 23 % (activations) and 45 / 49 % for all three.
The effects add up super-linearly; the loads queue behind each other.

What the patch does, all bitwise equal to the previous kernel (output hash in `moe_prefill`, CPU reference, and logits):

* **Lane-stream rings.** Each lane reads its stream through a ring of two 16-byte blocks in shared memory filled by
  `cp.async`; a group is committed at each of the four refill points of a row and a refill point lets four groups pend.
  Nothing waits on a register any more; the decoder's long-scoreboard stalls disappear. Rings are used where a warp
  decodes one or two streams (gate/up, down above 768 tokens). Down with four interleaved streams per warp (up to 768
  tokens) keeps the register queue: there its independent chains hide the latency, and the rings were 13 % slower.
* **Fused gate / up / SwiGLU.** One work item takes the same rows of gate and up; in each 32-row group of the slab the
  first 16-row `mma` tile holds gate rows and the second the same up rows, so a thread holds both values of its outputs
  and writes `silu(gate) · up`. This removes the SwiGLU pass, one F32→BF16 conversion and one routing helper per layer.
  Fused vs unfused GPU ops: 0 of 8.4 M values differ at 2048 tokens (also at 9, 100, 512, skewed and uniform).
* Tile offsets in shared memory are 16-bit, so gate's shared memory (LUT, slab, rings, bounds) still fits the 100 KB
  carve-out. The L1 matters: the activations of two warps that share tokens meet there. With 66 KB of dummy shared
  memory (L1 92 → 28 KB) the old gate kernel ran 30 % slower.

Per matrix (`moe_prefill`, layer 20, ms, median of 20; previous = the 9077d0c library):

| | tokens | previous | this patch |
|---|---:|---:|---:|
| gate + up + SwiGLU (fused vs separate ops) | 512 | 8.44 (skewed 8.72) | **8.08** (8.36) |
| | 2048 | 13.40 (14.09) | **12.45** (13.21) |
| down | 512 | 4.83 (5.34) | 4.82 (5.32) |
| | 1024 | 6.71 (7.38) | **6.40** (7.34) |
| | 2048 | 10.41 (11.49) | **9.56** (10.81) |

End to end, alternating 9077d0c / this patch, two rounds, pinned clocks, desktop stopped, caches dropped per process,
tok/s:

| | 9077d0c | this patch | change |
|---|---:|---:|---:|
| pp8192, `-ub 512` | 800.3 / 798.4 | **812.0 / 810.9** | +1.5 % |
| pp8192, `-ub 2048` | 1248.5 / 1247.3 | **1280.5 / 1278.1** | +2.5 % |
| pp8192, `-ub 4096` | 1399.1 / 1389.3 | **1431.3 / 1426.7** | +2.5 % |
| tg64 @ 0 / @ 2048 | 31.13 / 31.16, 31.12 / 31.12 | 31.17 / 31.21, 31.09 / 31.14 | unchanged |

Logits against 9077d0c (wikitext-2, 8 × 2048, `-ub 512` and `-ub 2048`): mean KLD 0.000000, maximum 5.7e-5 (the
precision of the stored logits, the same as 9077d0c against itself), same top-1 100 %.

What is left is the activation side: with the rings in place, the first `mma` of a group waiting for its activations
is 23 % (gate) / 17 % (down) of the samples, the barrier after the products 13 % / 8 %. Removing the activation loads
altogether (oracle) would cut a layer by 22–25 % at 2048 tokens and 10–13 % at 512. Hiding them inside this kernel
runs into its own budget: gate is at 128 registers for 512 threads and the slab leaves no shared memory for an
activation tile. A different kernel shape is sketched in [ROADMAP.md](ROADMAP.md).

## Rebase onto upstream llama.cpp 338fce6

The patch moved from `b29c606` to upstream `338fce6` (978 commits later). Upstream already contains every kernel the
previous patch had ported (chunked gated delta net, fused RMS_NORM + SCALE, float4 norms, vectorised conversions, norm
grid limits), so the patch is now only this project's own code: 2988 lines instead of 4768. Three conflicts, all small.

One regression showed up and is fixed: upstream's `build_rs` now gathers all recurrent states with one GET_ROWS and
takes two views of it, the state of the current sequence and an empty view of "extra" states. The in-place state read
of the gated delta net required the GET_ROWS to have a single consumer, stopped firing, and decode lost 1.5 % to 30
state copies per token (620 µs); zero-element consumers are no longer counted.

Checks: `test-backend-ops` CONCAT 275/275, GATED_DELTA_NET 43/43; `run_prompts.sh` against the previous build: two of
four greedy prompts text-identical, the other two diverge at the known near-ties (token 47 of prompt 1, where the exact
dequant + cuBLAS reference diverges too, and token 36 of the long prompt), median top-1 |Δlogprob| 2e-5. KL against
the previous build (wikitext-2, 8 × 2048, `-ub 512`): PPL 4.7464 vs 4.7481, mean KLD 0.0029, same top-1 98.30 % —
upstream's own kernel changes, the summation-order band seen before.

Alternating previous build / rebased, two rounds, pinned clocks, desktop stopped, tok/s:

| | previous (`b29c606` base) | rebased (`338fce6`) | change |
|---|---:|---:|---:|
| pp8192, `-ub 512` | 810.5 / 810.9 | **830.8 / 833.3** | +2.6 % |
| pp8192, `-ub 2048` | 1282.1 / 1277.9 | **1323.5 / 1323.0** | +3.4 % |
| pp8192, `-ub 4096` | 1429.0 / 1428.6 | **1480.8 / 1480.7** | +3.6 % |
| tg64 @ 0 | 31.07 / 31.12 | **31.23 / 31.24** | +0.4 % |
| tg64 @ 2048 | 31.13 / 31.16 | 31.14 / 31.17 | unchanged |

## Serving: full window and images

`llama-server` with the full 262,144-token window, one slot, `-ub 2048` and the BF16 vision projector
(`--mmproj`) leaves about 4 GB of the 61.4 GiB free with the desktop running. Two bugs only showed up there:

* **The projector did not start.** Every ggml CUDA context reserves 32 GB of virtual address space for its memory pool
  up front; the projector has its own context, and a second reservation next to the model fails
  (`cuMemAddressReserve`), although the pool maps only tens of MB. `GGML_CUDA_VMM_POOL_GB` sets the reservation; the
  server runs with 4.
* **Image prompts crashed with an illegal memory access in the gated delta net.** The in-place state read passes the
  kernel the row index of the GET_ROWS it replaces. That index is a graph input, and ggml's allocator frees it after its
  last reader, which for the last recurrent layer is that layer's own GET_ROWS: the memory could be reused before the
  kernel ran (a debug print read 1058905918, the bits of a float). Text prompts survived only by allocation luck. The
  in-place read is now enabled only for layers whose index is still read by a later node; the last layer gathers
  normally (2 MB per token).

Checks after the fix: a 1024×576 image (598 prompt tokens) and a 3840×2160 one (4102 tokens) are described correctly,
and the answer is text-identical to a run with the in-place read disabled. `llama-bench` pp2048 / tg64, pinned clocks,
desktop running: 1347.5 / 30.64 tok/s with the fix, 1352.4 / 30.38 with `GGML_CUDA_NO_GDN_ROWS=1`.

## Speculative decoding with the MTP head

The checkpoint ships one multi-token-prediction layer (`mtp.*`, a full-attention block with its own 256-expert MoE).
Upstream `338fce6` runs it as a draft (`--spec-type draft-mtp`) from an MTP-only GGUF (`convert_hf_to_gguf.py --mtp`,
which also copies the embedding, final norm and output matrix: 3.7 GB in BF16). The target model verifies every drafted
token, so the draft's precision only moves the acceptance rate, never the output; the draft was quantized with
`llama-quantize` (Q4_K_M 1192 MiB, Q8_0 1888 MiB). BF16 drafts go through the BF16N loader like the main model.

`llama-server -c 32768 -ub 2048 -md <draft> --spec-type draft-mtp -ngld 99 --spec-draft-n-max N`, greedy, four prompts
(code, Ukrainian, arithmetic, English) × 400 tokens, pinned clocks, desktop running:

| draft | `n_max` | tok/s | accepted drafts |
|---|---:|---:|---:|
| none | — | 30.92 | — |
| Q4_K_M | 1 | **33.20** (+7.4 %) | 0.850 |
| Q4_K_M | 2 | 26.31 | 0.738 |
| Q4_K_M | 3 | 27.93 | 0.639 (2.87 tokens per step) |
| Q8_0 | 1 | 32.20 | 0.853 |
| BF16 | 1 | 32.48 | 0.853 |

Long prompts (16,690 tokens of llama.cpp source, 20,634 tokens of Ukrainian notes): 28.91 tok/s without a draft,
31.58 (+9 %) with the Q4_K_M draft (acceptance 0.849; BF16 draft 0.855). Quantizing the draft does not lower acceptance,
at short or long context: the MTP layer reads the target's exact hidden state, so its own rounding does not accumulate.
Greedy text against the run without a draft: three prompts identical, one diverges at character 1583 of 1812, inside the
summation-order band (verification runs 2–4 tokens through other kernels).

More than one draft token loses today because verifying 3–4 tokens costs ≈ 3 decode steps. Measured per GEMV on real
tensors (single-tensor harness): a dense matrix costs 1.34× one column for 2 columns (the second column hits L2) but
2.6× for 3–4 (the grid splits the SMs per column); expert matrices scale with the distinct experts the tokens route to
(1.95/3.3/3.7× for 2/3/4 tokens). A kernel that decodes each weight once for all columns would cut the dense part to
≈ 1.15× and leave the experts as the floor: ≈ 53–59 ms per 4-token verification plus ≈ 10 ms of drafting for 2.87 tokens,
≈ 42–46 tok/s (+36–48 %).

Memory. The first attempt (full window, projector, Q4_K_M draft with q8_0 draft KV, defaults otherwise) left 1.23 GB
free in the Normal zone after loading (the kernel's min watermark is 1.04 GB) and a 3840×2160 image got the server
OOM-killed. `-lv 4` shows where the draft's ≈ 4.2 GB went:

| draft context, full window | `-ub 2048`, KV q8_0 | `--spec-draft-n-ubatch 64`, KV q8_0 | `--spec-draft-n-ubatch 64`, KV F16 |
|---|---:|---:|---:|
| weights | 1190 MiB | 1190 MiB | 1190 MiB |
| KV cache (1 layer) | 272 MiB | 272 MiB | 512 MiB |
| CUDA compute buffer | 1728 MiB | 550 MiB | **37 MiB** |
| host compute buffer | 1040 MiB | 33 MiB | 33 MiB |
| Normal zone free after load (no draft: 5.27 GB) | OOM at the projector warm-up | 3.20 GB | **3.53 GB** |

* The host buffer is the attention mask, window × micro-batch in F16 (262,144 × 2048 × 2 B = 1 GiB); a 64-token draft
  micro-batch shrinks it to 32 MiB. Prefill with the draft runs at 1105 vs 1185 tok/s on a 16.7K-token prompt.
* A quantized draft KV makes flash attention keep F16 copies of K and V for the whole window (2 × 256 MiB), so the
  F16 draft cache is smaller in total.
* The main context has the same mask: 1056 MiB in the host buffer at `-ub 2048` and very likely a device copy of the
  same size inside its 1440 MiB compute buffer. On a unified-memory board the copy is redundant, but upstream disables
  the CUDA backend's integrated-GPU path (reading host buffers directly) because of corrupted outputs (llama.cpp
  #15034), so it is not used here; `-ub 1024` would halve both at the cost of prefill speed.

With the last column the production server (full window, projector, Q4_K_M draft, one draft token) answers a
3840×2160 image at 33.85 tok/s and text at 33.4–34.0 tok/s, and 2.7 GB stay free after the image.

## Correctness

* Weights: bit-exact round trip for every tensor (see above).
* Arithmetic: products of the exact BF16 weights with FP32 activations, FP32 accumulation; only the summation order differs
  from llama.cpp's BF16 kernels. Against an FP64 reference the per-row error is 1e-8…4e-8 of Σ|w·x| (cuBLAS BF16 GEMV: ~4e-7).
* End to end, `scripts/run_prompts.sh` (3 prompts × 96 greedy tokens):
  * BF16N kernels vs `GGML_BF16N_REFERENCE=1` (exact dequant + stock cuBLAS BF16 path): identical text, no top-1
    divergence in 288 tokens, top-1 |Δlogprob| median 1.2e-5 (p99 0.07; the reference rounds activations to BF16).
  * GDN state in place vs gathered copy, embedding on GPU vs CPU: bitwise-identical text and logprobs.
* Prefill summation order (batched MoE kernel, chunked gated delta net, micro-batch size) moves next-token
  distributions by a mean KLD ≈ 0.002 (top-1 agreement ≈ 98.3–98.4 % on wikitext-2) with unchanged perplexity; see
  [the gated delta net section](#prefill-chunked-gated-delta-net-upstream-kernel-ported).

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
* A plane prefetch ring in the chunk-major GEMV that runs on across chunk boundaries (64 registers, no spills):
  tg64 30.46 vs 30.54 for the ring that restarts per chunk, within noise.
* The tiled conv-state concat for one token (decode): −0.85 % decode; it is now used only from 32 × 32 up.
* MoE prefill kernel: software prefetch of activations (`prefetch.global.L1`) or of the next item (`prefetch.global.L2`):
  gate +22–24 %; a batched register queue for the lane streams: spills, 2.6× slower; activation fragments 3–4 groups
  ahead in registers: 13–16 % slower even at depth 2 with a generic loop (same semantics as the hand-written
  two-buffer loop; not understood); plane rows staged in shared memory by `cp.async`: frees 16 registers but takes L1
  from the activations, 1 % slower than plane rows in registers.
* Decode GEMVs with the lane streams and the planes through `cp.async` rings in shared memory (as in the MoE prefill
  kernel): Nsight Compute had shown 60–68 % of the stall samples on long scoreboard, but the GEMV work phase already
  runs at the DRAM limit of its access pattern (165–170 GB/s inside a launch, 161 GB/s for the whole lm_head); the
  stalls are the normal state of a memory-bound kernel. Two versions, bitwise equal: the first cut the long-scoreboard
  share to 21 % with 69 % more instructions at the same speed, the leaner second one (predicated refill, one wait per
  step) was slower. An oracle that removes loads removes their bandwidth as well; it bounds nothing for a memory-bound
  kernel.
