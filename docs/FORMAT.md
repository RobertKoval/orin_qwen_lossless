# BF16N: lossless resident BF16 weights

BF16N stores BF16 weights so that every original bit pattern is reconstructed exactly, while the GEMV kernels decode
them in registers on the fly and the batched (prefill) MoE kernel decodes them into shared memory once per block. It is a
runtime representation: tensors are converted while the engine loads an ordinary BF16 checkpoint (llama.cpp: a BF16
GGUF; vLLM: the original safetensors) and are never written to disk. One implementation serves both engines: the
constants and header are `core/include/bf16n/format.h`, the encoder `core/src/encode.cpp`, the decoders
`core/include/bf16n/kernels.cuh` (see [Engine mapping](#engine-mapping)).

A BF16 weight is `sign(1) | exponent(8) | mantissa(7)`. Sign and mantissa are close to incompressible (measured entropy of
mantissa+sign given the exponent: 7.86 of 8 bits), so they are stored raw. Only the exponent (entropy ≈ 2.57 bit per tensor)
is entropy-coded.

## Sign|mantissa plane

`blck_size = 512`, `type_size = 512`: one byte per weight, `byte = sign << 7 | mantissa`.

Each row of `K` weights (`K % 512 == 0`, `K ≤ 4096`) is split into 512-weight chunks. Inside a chunk the layout is lane-major
so that a warp loads it with one coalesced 16-byte load per lane: lane `l ∈ [0,32)` owns the 16 positions

    p(l, s) = 128·j + 4·l + t,   s = 4·j + t,   j, t ∈ [0,4)

stored at byte `16·l + s`. The same `(l, s)` ownership is used by every exponent format and by the activation reads
(`float4` at `128·j + 4·l`, bank-conflict free), so row-major units sum the products in the same order in every format.

Rows are stored one after another (`row · K`), except inside a chunk-major HUFPAIR unit (below), whose `R` rows are stored
chunk by chunk: chunk `c` of row `r` at byte `(c · R + r) · 512` of the unit. Units never straddle an expert, so expert views
keep their offsets.

## Exponent formats (device "aux" buffer per tensor)

Each encoded matrix has a separate aux buffer next to its plane (how an engine keeps the two together: [Engine
mapping](#engine-mapping)). It starts with a 256-byte header (`magic "B16N"`, format, `K`, `R`, rows, array offsets).

### 1 — NIBBLE (fixed width, ≈ 12.1 bit/weight)

* 4-bit code per weight, lane `l` stores its 16 codes in 8 bytes; slot `s` sits at bits `4·(s & 7)` of word `s >> 3`.
* Per-row LUT of 16 exponent bytes: entry 0 is exponent 0, entries 1..15 are the 15 most frequent exponents `< 128` of the
  row. Decoding is two `prmt` byte lookups per 4 weights (`prmt(L0,L1,codes) | prmt(L2,L3,codes ^ 0x8888)`).
* Weights whose exponent is not in the LUT (1e-5 … 1e-4 of the weights) are escapes: plane byte 0 and code 0 decode to `+0`, and a
  per-row escape list (`u32 = col | sm_byte << 12 | exponent << 20`, row offsets `u32[rows+1]`) adds the exact product.
* Used for small dense matrices where entropy-coded units would starve the GPU of parallelism.

### 3 — HUFPAIR (entropy coded, ≈ 10.7 bit/weight)

* Symbols are exponent **pairs** of adjacent weights (`t, t+1`), coded with one length-limited canonical Huffman code per
  tensor (package-merge, max length 12). Adjacent exponents are correlated, and one table lookup yields two weights.
* LUT: 4096 × u32, entry `= e_odd << 23 | e_even << 7 | len` — the two BF16 exponent fields in place, so
  `lop3(sign_mantissa_pair, entry, 0x807F807F)` produces a `bf16x2` directly.
* Rows are grouped into units of `R` rows (`R ∈ {1,2,4,8,16}`, `R` divides the rows of one expert). One warp decodes one
  unit; every lane has its own MSB-first bitstream with the symbols of its slots in order rows → chunks → pairs.
  Per unit: `u32` stream base + `u16[32]` lane offsets.
* **Chunk-major units** (header `flags` bit 0) order the streams chunks → rows → pairs instead, and store the plane chunk-major
  (see above). Used for 3-D expert tensors with `K > 512` (gate and up), so the batched kernel can decode a 512-column slab of
  many rows at a time; for `K = 512` (down) the two orders coincide. Same size; the GEMV adds per-chunk partial sums.
* **Fragment-order units** (header `flags` bit 1, `R = 16`, vLLM only) hold 16-row bands in `mma.m16n8k16` A-fragment
  order: per 32-column tile pair a lane's 16 plane bytes and 8 exponent pairs are exactly its A registers of two 16×16
  tiles. A unit is a 512-column slab of a band (`K / 512` units per band; for `K = 512` the band), so a decode can be
  split along K. Used for the down experts (batched prefill on the tensor cores) and for vLLM's dense linear layers:
  the decode GEMV multiplies a band's fragments with up to 8 BF16 token columns at once (8 tokens cost about one), and
  a K-split band's slab partial sums are added in slab order, so a column's result does not depend on what is batched
  with it. Same code and LUT as row-major units; per-slab stream offsets add ≈ 0.4 % of the matrix bytes.
* Chosen whenever the tensor has ≤ 4096 distinct exponent pairs (true for every tensor of Qwen3.6-35B-A3B); otherwise the
  single-symbol variant 2 (`HUFFMAN`, max code length 10, 1024-entry LUT) is used.

## Conversion policy

The same rules in both engines (llama.cpp `LLAMA_BF16N=hybrid`, the default; vLLM `vllm_bf16n/policy.py`):

| tensor | format |
|---|---|
| routed experts (`*_exps`) | HUFPAIR, `R = 16` if ≥ 2048 rows per expert else 8; chunk-major units when `K > 512` |
| dense with ≥ 8M weights (lm_head, embedding, qkv, gates, out projections) | HUFPAIR, largest `R` with ≥ 512 units; vLLM linear layers (not lm_head / embedding, which `get_rows` reads): 16-row fragment-order bands |
| smaller dense (k/v, shared expert, α/β) | NIBBLE |
| F32 router `ffn_gate_inp` | NIBBLE, after checking every value is exactly BF16 |

llama.cpp: `LLAMA_BF16N=huffman` / `nibble` force one family; `LLAMA_BF16N_DENSE_MIN` changes the 8M threshold.
vLLM merges projections into shapes llama.cpp does not have (linear-attention qkvz 12288 rows, attention qkv 9216 rows),
where "largest `R` with ≥ 512 units" leaves a half-empty second wave of the decode GEMV; there the dense `R` comes from
a wave model (`policy.dense_R`: qkvz `R = 8`, qkv `R = 2`), which gives llama.cpp's `R` for llama.cpp's shapes. vLLM
does not encode the router (its layer is created without a quantization config).

## Decoder details that matter on Orin (SM87)

* The sign|mantissa plane is streamed with `ld.global.nc.L1::no_allocate` so it does not evict the per-lane exponent streams,
  which live in L1.
* 1024-thread blocks (≈ one per SM) so a single LUT copy shares the SM's L1/shared memory.
* Bit buffer is one 64-bit register pair; the refill check runs every 2 pairs (4 weights), inserts a prefetched word with a
  multiply and keeps a 4-word prefetch queue.
* INT32 and IMAD share one datapath on GA10B; the decoder is designed to minimise non-FFMA instructions per weight.
* The chunk-major GEMV keeps a 4-row loop body (accumulators rotated by 4): the fully unrolled 8-row body did not fit the
  instruction cache next to 31 other warps and cost 3–4 % of decode speed.
* Batched MUL_MAT_ID (more than 8 tokens): routing via `ggml_cuda_launch_mm_ids_helper` on the GPU (vLLM: its port, `core/src/mmid_helper.cu`), activations converted to
  BF16 once, one kernel per matrix. A block owns 16 units of one expert (8 for down) and up to 128 of its tokens; per step
  every warp decodes a 512-column chunk of a slice of its unit's rows into a 64-row shared slab (small enough to leave L1 to
  the lane streams), and the warps of that slice multiply it on the tensor cores (`mma.m16n8k16` BF16, FP32 accumulation;
  the k order inside each 32-column group is permuted identically for weights and activations). Warps without tokens
  split K instead.

## Verification at load

After encoding, every tensor is decoded on the GPU and compared with its source bit for bit; any mismatch aborts
loading. llama.cpp decodes in 64 MiB chunks and logs `BF16N round-trip mismatch` on failure; vLLM checks each matrix
as its layer is processed. The llama.cpp log line per tensor is:

    bf16n: token_embd.weight                hufpair K= 2048 rows= 248320 R=16 ratio=0.6685 roundtrip=OK
    bf16n: blk.0.ffn_gate_exps.weight       hufpair K= 2048 rows= 131072 R= 8 ratio=0.6352 roundtrip=OK
    bf16n: blk.0.attn_qkv.weight            hufpair K= 2048 rows=   8192 R=16 ratio=0.6742 roundtrip=OK

## Engine mapping

| | llama.cpp | vLLM |
|---|---|---|
| type | `GGML_TYPE_BF16N` (ggml type id 43); `blck_size = type_size = 512`, the tensor data is the plane | `bf16n_plane` / `bf16n_aux` uint8 tensors registered on the layer, header fields parsed once into Python ints |
| aux buffer | separate `cudaMalloc` registered against the tensor's data range; views (one expert of a 3-D tensor) find it by address | passed explicitly to every op; a view is `row0` + a row count |
| conversion | at load, `ggml_backend_cuda_bf16n_convert` (`LLAMA_BF16N`) | at load, layer by layer (`--quantization bf16n`, vLLM's meta-device loading) |
| kernels | `ggml-cuda/bf16n.cu` in the patch series (`engines/llamacpp/patches`), its own copy of the kernels (same decode and summation order, different templates and SASS) until T-034 makes it a thin glue over `core/` | `torch.ops.bf16n` (`core/torch/torch_bindings.cpp` over `core/`) |
| activations | F32 in, F32 out (GEMV); BF16 for the tensor cores (batched) | BF16 or F32 in and out; FP32 accumulation, same summation order |
