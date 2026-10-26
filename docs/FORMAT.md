# BF16N: lossless resident BF16 weights

`GGML_TYPE_BF16N` (ggml type id 43) stores BF16 weights so that every original bit pattern is reconstructed exactly, while the
GEMV kernels decode them in registers on the fly. It is a runtime representation: tensors are converted while llama.cpp loads
an ordinary BF16 GGUF and are never written to a GGUF file.

A BF16 weight is `sign(1) | exponent(8) | mantissa(7)`. Sign and mantissa are close to incompressible (measured entropy of
mantissa+sign given the exponent: 7.86 of 8 bits), so they are stored raw. Only the exponent (entropy ≈ 2.57 bit per tensor)
is entropy-coded.

## Sign|mantissa plane (the ggml tensor itself)

`blck_size = 512`, `type_size = 512`: one byte per weight, `byte = sign << 7 | mantissa`.

Each row of `K` weights (`K % 512 == 0`, `K ≤ 4096`) is split into 512-weight chunks. Inside a chunk the layout is lane-major
so that a warp loads it with one coalesced 16-byte load per lane: lane `l ∈ [0,32)` owns the 16 positions

    p(l, s) = 128·j + 4·l + t,   s = 4·j + t,   j, t ∈ [0,4)

stored at byte `16·l + s`. The same `(l, s)` ownership is used by every exponent format and by the activation reads
(`float4` at `128·j + 4·l`, bank-conflict free), so all formats sum the products in the same order.

## Exponent formats (device "aux" buffer per tensor)

The aux buffer is a separate `cudaMalloc` registered against the tensor's data range; views (e.g. one expert of a 3-D expert
tensor) find it by address. It starts with a 256-byte header (`magic "B16N"`, format, `K`, `R`, rows, array offsets).

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
* Chosen whenever the tensor has ≤ 4096 distinct exponent pairs (true for every tensor of Qwen3.6-35B-A3B); otherwise the
  single-symbol variant 2 (`HUFFMAN`, max code length 10, 1024-entry LUT) is used.

## Conversion policy (`LLAMA_BF16N=hybrid`, default)

| tensor | format |
|---|---|
| routed experts (`*_exps`) | HUFPAIR, `R = 16` if ≥ 2048 rows per expert else 8 |
| dense with ≥ 8M weights (lm_head, embedding, qkv, gates, out projections) | HUFPAIR, largest `R` with ≥ 512 units |
| smaller dense (k/v, shared expert, α/β) | NIBBLE |
| F32 router `ffn_gate_inp` | NIBBLE, after checking every value is exactly BF16 |

`LLAMA_BF16N=huffman` / `nibble` force one family; `LLAMA_BF16N_DENSE_MIN` changes the 8M threshold.

## Decoder details that matter on Orin (SM87)

* The sign|mantissa plane is streamed with `ld.global.nc.L1::no_allocate` so it does not evict the per-lane exponent streams,
  which live in L1.
* 1024-thread blocks (≈ one per SM) so a single LUT copy shares the SM's L1/shared memory.
* Bit buffer is one 64-bit register pair; the refill check runs every 2 pairs (4 weights), inserts a prefetched word with a
  multiply and keeps a 4-word prefetch queue.
* INT32 and IMAD share one datapath on GA10B; the decoder is designed to minimise non-FFMA instructions per weight.

## Verification at load

After encoding, every tensor is decoded on the GPU in 64 MiB chunks and compared with the source with `memcmp`. Any mismatch
aborts loading (`BF16N round-trip mismatch`). The log line per tensor is:

    bf16n: token_embd.weight                hufpair K= 2048 rows= 248320 R=16 ratio=0.6685 roundtrip=OK
    bf16n: blk.0.ffn_gate_exps.weight       hufpair K= 2048 rows= 131072 R= 8 ratio=0.6352 roundtrip=OK
    bf16n: blk.0.attn_qkv.weight            hufpair K= 2048 rows=   8192 R=16 ratio=0.6742 roundtrip=OK
