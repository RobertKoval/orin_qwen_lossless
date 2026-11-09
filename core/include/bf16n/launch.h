// SPDX-License-Identifier: MIT
// Host-side launchers of the BF16N kernels (launch.cu). No ggml, no allocation, no host synchronisation: every
// launcher only enqueues kernels on the given stream, so all of them can be captured into CUDA graphs.
#pragma once

#include "bf16n/format.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace bf16n {

// device view of one encoded matrix (plane + aux), the metadata parsed once from the aux header
struct weight {
    const uint8_t * plane = nullptr;
    const uint8_t * aux   = nullptr;
    int     format = 0;
    int64_t K = 0, rows = 0;
    int     R = 0;
    bool    cm = false; // HUFPAIR chunk-major units
    bool    fo = false; // HUFPAIR fragment-order bands (FLAG_FRAG_ORDER: K = 512, R = 16)
    uint64_t off_a = 0, off_b = 0, off_c = 0, off_d = 0;
};

// y[col] = W[rbase : rbase + nrows] x[col] for ncols columns.
// Dense: column c = token c, rbase = row0.
// With ids (MoE): column c = (token c / n_used, slot c % n_used); expert e = ids[token, slot]; rbase = row0 + e *
// rows_per_matrix; x of a slot = x_tok * token + x_slot * (slot % x_per_slot); y at y_tok * token + y_slot * slot.
// Signed strides in bytes. ids hold int32, or int64 < 2^31 (read as the low word).
// Callers own valid backing ranges and the selected kernel's vector alignment; raw launchers do not
// comprehensively validate these obligations. The Torch frontend additionally requires 16-byte input bases/row strides.
struct mmv_args {
    const void * x = nullptr; bool x_bf16 = false;
    void * y = nullptr;       bool y_bf16 = false;
    int ncols = 0;
    int64_t x_tok = 0, y_tok = 0;
    const void * ids = nullptr; int64_t ids_s0 = 0, ids_s1 = 0;
    int n_used = 1, x_per_slot = 1;
    int64_t x_slot = 0, y_slot = 0;
    int64_t rows_per_matrix = 0, row0 = 0, nrows = 0;
};

// once per process, before any launch (and before graph capture): device attributes and kernel shared-memory limits
void init();
int sm_count();

// multi_col=false selects the original single-column HP kernel even for dense batches; no effect on chunk-major,
// NIBBLE, HUFFMAN, or expert-id launches. Defaults to the existing vLLM multi-column behaviour.
void mmv(const weight & w, const mmv_args & a, cudaStream_t stream, bool multi_col = true);
// GLU for NIBBLE matrices whose rows_per_matrix rows are [gate (nrows); up (nrows)] (vLLM's merged gate_up layout):
// y = silu(gate x) * up x
void mmv_glu(const weight & w, const mmv_args & a, cudaStream_t stream);
// Separate gate/up matrices with the same K, format, row count and row layout (including expert stride), and matching
// R for Huffman formats. Requires 0 <= row0 <= rows and 0 <= nrows <= rows - row0; nrows=0 enqueues nothing.
// Supports NIBBLE, row-major HUFPAIR (R=1,2,4,8,16), and HUFFMAN (R=8,16), matching llama's fused path.
// Chunk-major fusion was not supported by llama: use mmv(gate), mmv(up), swiglu instead.
// With BF16 output, rounds only the final GLU result, not the intermediate gate/up products.
void mmv_glu(const weight & gate, const weight & up, const mmv_args & a, cudaStream_t stream);
// exact BF16 bits of rows [row0, row0 + nrows) into out (row-major, nrows x K)
void dequant(const weight & w, int64_t row0, int64_t nrows, uint16_t * out, cudaStream_t stream);
// dst[i] = W[row0 + ids[i]] (one row per id), FP32 or BF16, dst row stride in bytes.
// Requires n >= 0 and 0 <= row0 < rows (row0 == rows allowed only when n == 0). IDs must index valid rows;
// checking their device values is the caller's responsibility, without host synchronisation here.
// NIBBLE and row-major HUFPAIR only; single-symbol HUFFMAN and chunk-major weights fail explicitly.
void get_rows(const weight & w, const void * ids, int64_t ids_s0, int64_t n, void * dst, bool dst_bf16, int64_t dst_s1,
              cudaStream_t stream, int64_t row0 = 0);

// batched MUL_MAT_ID (prefill): routing (ids_src1 / ids_dst / bounds, sizes n_tok * n_used, n_tok * n_used,
// n_experts + 1) is computed on the GPU by mm_ids_route; x: BF16 rows of K (x_row elements apart, row index from
// ids_src1); dst[ids_dst[j] * dst_col + row]. up != nullptr: fused gate/up + SWIGLU (dst = silu(w x) * up x).
bool mm_ids_supported(const weight & w, const weight * up);
void mm_ids_route(const void * ids, bool ids_i64, int64_t ids_row_stride, int32_t * ids_src1, int32_t * ids_dst,
                  int32_t * bounds, int n_experts, int n_tok, int n_used, int nchannels_y, int sis1, cudaStream_t stream);
void mm_ids(const weight & w, const weight * up, const void * x_bf16, int64_t x_row, const int32_t * ids_src1,
            const int32_t * ids_dst, const int32_t * bounds, void * dst, bool dst_bf16, int64_t dst_col, int n_experts,
            int n_tok, int n_used, cudaStream_t stream);

// y[t, n] = sum_k w[t, k] * ys[t, k, n] in slot order (FP32), written as FP32 or BF16; w FP32 or BF16
void moe_combine(const float * ys, const void * w, bool w_bf16, void * y, bool y_bf16, int n_tok, int n_used, int64_t n,
                 cudaStream_t stream);
// y = silu(g) * u elementwise (FP32), the activation between MoE gate/up and down GEMVs
void swiglu(const float * g, const float * u, float * y, int64_t n, cudaStream_t stream);

} // namespace bf16n
