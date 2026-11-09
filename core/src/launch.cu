// SPDX-License-Identifier: MIT
// Launchers for the BF16N kernels: template dispatch and launch geometry from the llama.cpp patch (launch_mmv,
// launch_mmv_glu, launch_dequant, launch_mm_ids, hp_launch_dims, mmv_grid), without ggml types and environment toggles.
// Kernel attributes are set once in init(); launchers only enqueue work.
#include "bf16n/kernels.cuh"
#include "bf16n/launch.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>

namespace bf16n {

void launch_mm_ids_helper_i32(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);
void launch_mm_ids_helper_i64(const int64_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);
void mm_ids_helper_init(int smem_optin);

namespace {

using namespace kern;
using bf16 = __nv_bfloat16;

int g_sms = 0;
int g_smem_optin = 0;
std::once_flag g_once;

nib_ptrs nibp(const weight & w) {
    return {w.plane, w.aux + w.off_a, (const uint4 *) (w.aux + w.off_b), (const uint32_t *) (w.aux + w.off_c),
            (const uint32_t *) (w.aux + w.off_d)};
}
// the up half of a merged [gate; up] NIBBLE matrix: every per-row array shifted by `rows` rows (escape offsets are
// absolute indices into the escape list, so shifting esc_off is enough)
nib_ptrs nibp_shift(const weight & w, int64_t rows) {
    nib_ptrs p = nibp(w);
    p.sm += rows * w.K;
    p.nib += rows * (w.K / 2);
    p.lut_row += rows;
    p.esc_off += rows;
    return p;
}
huf_ptrs hufp(const weight & w) {
    return {w.plane, (const uint32_t *) (w.aux + w.off_a), (const uint32_t *) (w.aux + w.off_b),
            (const uint32_t *) (w.aux + w.off_c)};
}
hp_ptrs hpp(const weight & w) {
    return {w.plane, (const uint32_t *) (w.aux + w.off_a), (const uint32_t *) (w.aux + w.off_b),
            (const uint16_t *) (w.aux + w.off_c), (const uint32_t *) (w.aux + w.off_d)};
}

// pair kernels: ~one 1024-thread block per SM when there is enough work; smaller blocks spread few units over all SMs
void hp_launch_dims(const weight & e, int64_t nrows, int ncols, dim3 & grid, int & nt) {
    const int64_t units = nrows / e.R;
    const int64_t sms = g_sms;
    const int64_t per_col = std::max<int64_t>(1, sms / std::max(1, ncols));
    const int64_t warps_per_block = std::clamp<int64_t>((units + per_col - 1) / per_col, 2, HP_NT / 32);
    nt = (int) warps_per_block * 32;
    grid = dim3((unsigned) std::max<int64_t>(1, std::min<int64_t>((units + warps_per_block - 1) / warps_per_block, per_col)), ncols);
}

dim3 mmv_grid(const weight & e, int64_t nrows, int ncols) {
    const int64_t work = e.format == FMT_NIBBLE ? nrows : nrows / e.R; // rows or units, one warp each
    const int64_t cap = std::max<int64_t>(1, (int64_t) g_sms * 4 / std::min(ncols, 4));
    return dim3((unsigned) std::max<int64_t>(1, std::min<int64_t>((work + 7) / 8, cap)), ncols);
}

cols_t make_cols(const mmv_args & a) {
    cols_t p = {};
    p.x = (const char *) a.x;
    p.dst = (char *) a.y;
    p.x_tok = a.x_tok; p.dst_tok = a.y_tok;
    p.ids = (const char *) a.ids;
    p.ids_s0 = a.ids_s0; p.ids_s1 = a.ids_s1;
    p.n_used = a.ids ? a.n_used : 1;
    p.ne11 = a.x_per_slot;
    p.x_slot = a.x_slot; p.dst_slot = a.y_slot;
    p.rows_per_matrix = a.rows_per_matrix;
    p.row0 = a.row0;
    p.nrows = a.nrows;
    p.dst_bf16 = a.y_bf16;
    return p;
}

size_t mc_shm(int C, int NC) { return (size_t) NC * C * 512 * sizeof(float) + (4u << HP_LP); }

constexpr int MM_MAX_EXPERTS_INIT = 512;
constexpr int FO_NTOK = 96, FO_NW = 16; // fragment-order batched kernel: tokens per item, warps (T-023 prototype)
size_t mm_shm(bool glu, int NW, int IL, int B, int n_experts) {
    return ((glu ? 8u : 4u) << HP_LP) + 64 * HPM_LD * 2 + (size_t) NW * IL * 32 * 16 * B +
           (size_t) (n_experts + 1) * (sizeof(int) + sizeof(uint16_t));
}

template <typename TX>
void launch_mmv_t(const weight & e, const cols_t & p, int ncols, cudaStream_t stream, bool multi_col) {
    const int64_t K = e.K;
    const dim3 grid = mmv_grid(e, p.nrows, ncols);
    if (e.format == FMT_HUFPAIR) {
        dim3 g2; int nt;
        hp_launch_dims(e, p.nrows, ncols, g2, nt);
        const size_t shm = K * sizeof(float) + (4u << HP_LP);
        if (e.fo) {
            k_fo_mmv<TX><<<g2, nt, fo_mmv_shm(), stream>>>(hpp(e), p);
        } else if (e.cm) {
#define CM_CASE(CC, RR) case CC * 100 + RR: k_hp_mmv_cm<TX, CC, RR><<<g2, nt, shm, stream>>>(hpp(e), p); break;
            switch (K / 512 * 100 + e.R) {
                CM_CASE(2, 4) CM_CASE(2, 8) CM_CASE(2, 16) CM_CASE(4, 4) CM_CASE(4, 8) CM_CASE(4, 16)
                CM_CASE(8, 4) CM_CASE(8, 8) CM_CASE(8, 16)
                default: throw std::invalid_argument("bf16n: unsupported chunk-major K/R");
            }
#undef CM_CASE
        } else if (!p.ids && ncols > 1 && multi_col) {
            // columns in groups of up to 4, each group decodes the matrix once
            hp_launch_dims(e, p.nrows, 1, g2, nt);
            for (int c0 = 0; c0 < ncols; c0 += 4) {
                const int nc = std::min(4, ncols - c0);
                cols_t q = p;
                q.x += c0 * p.x_tok;
                q.dst += c0 * p.dst_tok;
                const size_t shm_mc = mc_shm((int) (K / 512), nc);
                switch (K / 512 * 10 + nc) {
#define MC_CASE(CC, NC) case CC * 10 + NC: \
    if (p.dst_bf16) { k_hp_mmv_mc<TX, CC, NC, true><<<g2, nt, shm_mc, stream>>>(hpp(e), e.R, q); } \
    else { k_hp_mmv_mc<TX, CC, NC, false><<<g2, nt, shm_mc, stream>>>(hpp(e), e.R, q); } \
    break;
                    MC_CASE(1, 1) MC_CASE(1, 2) MC_CASE(1, 3) MC_CASE(1, 4) MC_CASE(2, 1) MC_CASE(2, 2) MC_CASE(2, 3) MC_CASE(2, 4)
                    MC_CASE(4, 1) MC_CASE(4, 2) MC_CASE(4, 3) MC_CASE(4, 4) MC_CASE(8, 1) MC_CASE(8, 2) MC_CASE(8, 3) MC_CASE(8, 4)
#undef MC_CASE
                    default: throw std::invalid_argument("bf16n: unsupported K");
                }
            }
        } else {
#define HP_CASE(CC) case CC: k_hp_mmv<TX, CC><<<g2, nt, shm, stream>>>(hpp(e), e.R, p); break;
            switch (K / 512) { HP_CASE(1) HP_CASE(2) HP_CASE(4) HP_CASE(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef HP_CASE
        }
    } else if (e.format == FMT_NIBBLE) {
#define NIB_CASE(CC) case CC: k_nibble_mmv<TX, CC><<<grid, 256, 0, stream>>>(nibp(e), p); break;
        switch (K / 512) { NIB_CASE(1) NIB_CASE(2) NIB_CASE(4) NIB_CASE(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef NIB_CASE
    } else {
        const size_t shm = K * sizeof(float) + (4u << HUF_LMAX);
#define HUF_CASE(CC) case CC: k_huffman_mmv<TX, CC><<<grid, 256, shm, stream>>>(hufp(e), e.R, p); break;
        switch (K / 512) { HUF_CASE(1) HUF_CASE(2) HUF_CASE(4) HUF_CASE(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef HUF_CASE
    }
    BF16N_CUDA_CHECK(cudaGetLastError());
}

template <typename TX>
void launch_mmv_glu_t(const weight & g, const cols_t & p, int ncols, cudaStream_t stream) {
    const dim3 grid = mmv_grid(g, p.nrows, ncols);
    const nib_ptrs gp = nibp(g), up = nibp_shift(g, p.nrows);
#define NIB_CASE(CC) case CC: \
    if (p.dst_bf16) { k_nibble_mmv_glu<TX, CC, __nv_bfloat16><<<grid, 256, 0, stream>>>(gp, up, p); } \
    else { k_nibble_mmv_glu<TX, CC, float><<<grid, 256, 0, stream>>>(gp, up, p); } \
    break;
    switch (g.K / 512) { NIB_CASE(1) NIB_CASE(2) NIB_CASE(4) NIB_CASE(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef NIB_CASE
    BF16N_CUDA_CHECK(cudaGetLastError());
}

template <typename TX>
void launch_mmv_glu_t(const weight & g, const weight & u, const cols_t & p, int ncols, cudaStream_t stream) {
    const int64_t K = g.K;
    const dim3 grid = mmv_grid(g, p.nrows, ncols);
    if (g.format == FMT_HUFPAIR) {
        dim3 g2; int nt;
        hp_launch_dims(g, p.nrows, ncols, g2, nt);
        const size_t shm = K * sizeof(float) + (8u << HP_LP);
#define HP_CASE(CC, RR) case CC * 100 + RR: k_hp_mmv_glu<TX, CC, RR><<<g2, nt, shm, stream>>>(hpp(g), hpp(u), p); break;
#define HP_CASES(RR) HP_CASE(1, RR) HP_CASE(2, RR) HP_CASE(4, RR) HP_CASE(8, RR)
        switch ((K / 512) * 100 + g.R) {
            HP_CASES(1) HP_CASES(2) HP_CASES(4) HP_CASES(8) HP_CASES(16)
            default: throw std::invalid_argument("bf16n: unsupported K/R for glu");
        }
#undef HP_CASES
#undef HP_CASE
    } else if (g.format == FMT_NIBBLE) {
#define NIB_CASE(CC) case CC: \
    if (p.dst_bf16) { k_nibble_mmv_glu<TX, CC, __nv_bfloat16><<<grid, 256, 0, stream>>>(nibp(g), nibp(u), p); } \
    else { k_nibble_mmv_glu<TX, CC, float><<<grid, 256, 0, stream>>>(nibp(g), nibp(u), p); } \
    break;
        switch (K / 512) { NIB_CASE(1) NIB_CASE(2) NIB_CASE(4) NIB_CASE(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef NIB_CASE
    } else {
        const size_t shm = K * sizeof(float) + (8u << HUF_LMAX);
#define HUF_CASE(CC, RR) case CC * 100 + RR: k_huffman_mmv_glu<TX, CC, RR><<<grid, 256, shm, stream>>>(hufp(g), hufp(u), p); break;
        switch ((K / 512) * 100 + g.R) {
            HUF_CASE(1, 8) HUF_CASE(2, 8) HUF_CASE(4, 8) HUF_CASE(8, 8)
            HUF_CASE(1, 16) HUF_CASE(2, 16) HUF_CASE(4, 16) HUF_CASE(8, 16)
            default: throw std::invalid_argument("bf16n: unsupported K/R for glu");
        }
#undef HUF_CASE
    }
    BF16N_CUDA_CHECK(cudaGetLastError());
}

void check_weight(const weight & w) {
    BF16N_REQUIRE(w.K % 512 == 0 && w.K >= 512 && w.K <= K_MAX, "K must be a multiple of 512 in [512, 4096]");
    BF16N_REQUIRE(w.format == FMT_NIBBLE || w.format == FMT_HUFFMAN || w.format == FMT_HUFPAIR, "unknown format");
    BF16N_REQUIRE(w.format == FMT_NIBBLE || (w.R > 0 && w.rows % w.R == 0), "rows must be a multiple of R");
    BF16N_REQUIRE(!w.cm || w.format == FMT_HUFPAIR, "chunk-major units exist only for HUFPAIR");
    BF16N_REQUIRE(!w.fo || (w.format == FMT_HUFPAIR && !w.cm && w.K == FO_K && w.R == 16),
                  "fragment order exists only for HUFPAIR with K = 512 and R = 16, without chunk-major units");
}

void ensure_init() { std::call_once(g_once, [] {
    int dev;
    BF16N_CUDA_CHECK(cudaGetDevice(&dev));
    BF16N_CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
    BF16N_CUDA_CHECK(cudaDeviceGetAttribute(&g_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    // every kernel instance that may need more than the default 48 KiB of dynamic shared memory, at the size the
    // llama.cpp launchers would set (multi-column GEMV: exact; batched MoE: up to MM_MAX_EXPERTS_INIT experts)
#define MC_ATTR(TX, CC, NC) \
    BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_hp_mmv_mc<TX, CC, NC, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mc_shm(CC, NC))); \
    BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_hp_mmv_mc<TX, CC, NC, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mc_shm(CC, NC)));
#define MC_ATTRS(TX) MC_ATTR(TX, 1, 1) MC_ATTR(TX, 1, 2) MC_ATTR(TX, 1, 3) MC_ATTR(TX, 1, 4) MC_ATTR(TX, 2, 1) MC_ATTR(TX, 2, 2) \
    MC_ATTR(TX, 2, 3) MC_ATTR(TX, 2, 4) MC_ATTR(TX, 4, 1) MC_ATTR(TX, 4, 2) MC_ATTR(TX, 4, 3) MC_ATTR(TX, 4, 4) MC_ATTR(TX, 8, 1) \
    MC_ATTR(TX, 8, 2) MC_ATTR(TX, 8, 3) MC_ATTR(TX, 8, 4)
    MC_ATTRS(float) MC_ATTRS(bf16)
#undef MC_ATTRS
#undef MC_ATTR
    BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_fo_mm_ids<FO_NTOK, FO_NW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int) std::min<size_t>(g_smem_optin, fo_mm_shm(FO_NTOK, MM_MAX_EXPERTS_INIT))));
#define MM_ATTR(CC, RR, NW, IL, WT, B, G) BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_hp_mm_ids<CC, RR, NW, IL, WT, B, G>, \
    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) std::min<size_t>(g_smem_optin, mm_shm(G, NW, IL, B, MM_MAX_EXPERTS_INIT))));
    MM_ATTR(4, 8, 16, 1, 2, 2, true) MM_ATTR(4, 8, 16, 1, 2, 2, false) MM_ATTR(1, 16, 8, 1, 4, 2, false)
    MM_ATTR(1, 16, 8, 2, 2, 2, false) MM_ATTR(1, 16, 8, 4, 1, 0, false)
#undef MM_ATTR
    mm_ids_helper_init(g_smem_optin);
}); }

} // namespace

void init() { ensure_init(); }

int sm_count() {
    ensure_init();
    return g_sms;
}

void mmv(const weight & w, const mmv_args & a, cudaStream_t stream, bool multi_col) {
    ensure_init();
    check_weight(w);
    BF16N_REQUIRE(a.ncols > 0 && a.ncols <= 65535, "ncols out of range");
    BF16N_REQUIRE(w.format == FMT_NIBBLE || (a.nrows % w.R == 0 && a.row0 % w.R == 0 && a.rows_per_matrix % w.R == 0),
                  "rows of a product must be whole units");
    const cols_t p = make_cols(a);
    if (a.x_bf16) {
        launch_mmv_t<bf16>(w, p, a.ncols, stream, multi_col);
    } else {
        launch_mmv_t<float>(w, p, a.ncols, stream, multi_col);
    }
}

void mmv_glu(const weight & w, const mmv_args & a, cudaStream_t stream) {
    ensure_init();
    check_weight(w);
    BF16N_REQUIRE(w.format == FMT_NIBBLE, "mmv_glu supports NIBBLE matrices only");
    BF16N_REQUIRE(a.rows_per_matrix == 2 * a.nrows, "mmv_glu: rows per matrix must be [gate; up] of nrows each");
    BF16N_REQUIRE(a.ncols > 0 && a.ncols <= 65535, "ncols out of range");
    const cols_t p = make_cols(a);
    if (a.x_bf16) {
        launch_mmv_glu_t<bf16>(w, p, a.ncols, stream);
    } else {
        launch_mmv_glu_t<float>(w, p, a.ncols, stream);
    }
}

void mmv_glu(const weight & gate, const weight & up, const mmv_args & a, cudaStream_t stream) {
    ensure_init();
    check_weight(gate);
    check_weight(up);
    BF16N_REQUIRE(gate.format == up.format && gate.K == up.K, "mmv_glu: gate/up format and K must match");
    BF16N_REQUIRE(gate.rows == up.rows, "mmv_glu: gate/up row counts must match");
    BF16N_REQUIRE(a.row0 >= 0 && a.nrows >= 0 && a.row0 <= gate.rows && a.nrows <= gate.rows - a.row0,
                  "mmv_glu: product rows out of range");
    BF16N_REQUIRE(!gate.cm && !up.cm, "mmv_glu: chunk-major weights require separate mmv launches");
    BF16N_REQUIRE(!gate.fo && !up.fo, "mmv_glu: fragment-order weights require separate mmv launches");
    BF16N_REQUIRE(gate.format == FMT_NIBBLE || gate.R == up.R, "mmv_glu: gate/up R must match");
    BF16N_REQUIRE(a.ncols > 0 && a.ncols <= 65535, "ncols out of range");
    BF16N_REQUIRE(gate.format == FMT_NIBBLE ||
                  (a.nrows % gate.R == 0 && a.row0 % gate.R == 0 && a.rows_per_matrix % gate.R == 0),
                  "rows of a product must be whole units");
    if (a.nrows == 0) {
        return;
    }
    const cols_t p = make_cols(a);
    if (a.x_bf16) {
        launch_mmv_glu_t<bf16>(gate, up, p, a.ncols, stream);
    } else {
        launch_mmv_glu_t<float>(gate, up, p, a.ncols, stream);
    }
}

void dequant(const weight & e, int64_t row0, int64_t nrows, uint16_t * out, cudaStream_t stream) {
    ensure_init();
    check_weight(e);
    if (nrows == 0) {
        return;
    }
    const int64_t work = e.format == FMT_NIBBLE ? nrows : nrows / e.R;
    const int gx = (int) std::max<int64_t>(1, std::min<int64_t>((work + 7) / 8, (int64_t) g_sms * 16));
    if (e.format != FMT_NIBBLE) {
        BF16N_REQUIRE(nrows % e.R == 0 && row0 % e.R == 0, "dequant rows must be whole units");
    }
    const nib_ptrs np = nibp(e);
#define DQ(CC)                                                                                                         \
    case CC:                                                                                                           \
        if (e.format == FMT_NIBBLE) {                                                                                  \
            k_nibble_dequant<CC><<<gx, 256, 0, stream>>>(np.sm, np.nib, np.lut_row, np.esc_off, np.esc, row0, nrows, out); \
        } else if (e.fo) {                                                                                             \
            k_fo_dequant<<<gx, 256, 0, stream>>>(hpp(e), row0, nrows, out);                                        \
        } else if (e.format == FMT_HUFPAIR && e.cm) {                                                                  \
            switch (e.R) {                                                                                             \
                case 4: k_hp_dequant_cm<CC, 4><<<gx, 256, 0, stream>>>(hpp(e), row0, nrows, out); break;              \
                case 8: k_hp_dequant_cm<CC, 8><<<gx, 256, 0, stream>>>(hpp(e), row0, nrows, out); break;              \
                case 16: k_hp_dequant_cm<CC, 16><<<gx, 256, 0, stream>>>(hpp(e), row0, nrows, out); break;            \
                default: throw std::invalid_argument("bf16n: unsupported chunk-major R");                             \
            }                                                                                                          \
        } else if (e.format == FMT_HUFPAIR) {                                                                          \
            k_hp_dequant<CC><<<gx, 256, 0, stream>>>(hpp(e), e.R, row0, nrows, out);                                    \
        } else {                                                                                                       \
            const huf_ptrs hq = hufp(e);                                                                               \
            k_huffman_dequant<CC><<<gx, 256, 0, stream>>>(hq.sm, hq.lut, hq.lane_off, hq.stream, e.R, row0, nrows, out); \
        }                                                                                                              \
        break;
    switch (e.K / 512) { DQ(1) DQ(2) DQ(4) DQ(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef DQ
    BF16N_CUDA_CHECK(cudaGetLastError());
}

void get_rows(const weight & e, const void * ids, int64_t ids_s0, int64_t n, void * dst, bool dst_bf16, int64_t dst_s1,
              cudaStream_t stream, int64_t row0) {
    ensure_init();
    check_weight(e);
    BF16N_REQUIRE(n >= 0, "get_rows: n must be nonnegative");
    BF16N_REQUIRE(row0 >= 0 && (n == 0 ? row0 <= e.rows : row0 < e.rows), "get_rows: row0 out of range");
    BF16N_REQUIRE(!e.cm, "get_rows of a chunk-major (expert) matrix");
    BF16N_REQUIRE(!e.fo, "get_rows of a fragment-order (expert) matrix");
    BF16N_REQUIRE(e.format != FMT_HUFFMAN, "get_rows is not implemented for the single-symbol Huffman format");
    if (n == 0) {
        return;
    }
    const dim3 grid((unsigned) ((n + 7) / 8));
    const char * ip = (const char *) ids;
    char * dp = (char *) dst;
#define GR(CC)                                                                                                         \
    case CC:                                                                                                           \
        if (e.format == FMT_NIBBLE) {                                                                                  \
            if (dst_bf16) { k_nibble_get_rows<CC, true><<<grid, 256, 0, stream>>>(nibp(e), ip, ids_s0, n, row0, dp, dst_s1); } \
            else { k_nibble_get_rows<CC, false><<<grid, 256, 0, stream>>>(nibp(e), ip, ids_s0, n, row0, dp, dst_s1); }     \
        } else {                                                                                                       \
            if (dst_bf16) { k_hp_get_rows<CC, true><<<grid, 256, 0, stream>>>(hpp(e), e.R, ip, ids_s0, n, row0, dp, dst_s1); } \
            else { k_hp_get_rows<CC, false><<<grid, 256, 0, stream>>>(hpp(e), e.R, ip, ids_s0, n, row0, dp, dst_s1); }     \
        }                                                                                                              \
        break;
    switch (e.K / 512) { GR(1) GR(2) GR(4) GR(8) default: throw std::invalid_argument("bf16n: unsupported K"); }
#undef GR
    BF16N_CUDA_CHECK(cudaGetLastError());
}

// (chunks per row, rows per unit) pairs with a batched kernel instance: slab of 64 rows
bool mm_ids_supported(const weight & w, const weight * up) {
    const int C = (int) (w.K / 512);
    const bool shape = w.format == FMT_HUFPAIR && (w.cm || w.K == 512) && ((C == 4 && w.R == 8) || (C == 1 && w.R == 16));
    if (!up) {
        return shape; // K = 512, R = 16: k_hp_mm_ids row-major, k_fo_mm_ids in fragment order
    }
    return shape && w.K == 2048 && w.R == 8 && up->format == FMT_HUFPAIR && up->cm && up->K == w.K && up->R == w.R &&
           up->rows == w.rows;
}

void mm_ids_route(const void * ids, bool ids_i64, int64_t ids_row_stride, int32_t * ids_src1, int32_t * ids_dst,
                  int32_t * bounds, int n_experts, int n_tok, int n_used, int nchannels_y, int sis1, cudaStream_t stream) {
    ensure_init();
    BF16N_REQUIRE(n_tok > 0 && n_tok < (1 << 22) && n_used > 0 && n_used < (1 << 10), "routing sizes out of range");
    BF16N_REQUIRE((size_t) n_tok * 4 <= (size_t) g_smem_optin, "too many tokens for the routing helper");
    BF16N_REQUIRE((int64_t) n_tok * ids_row_stride < INT32_MAX, "routing ids too large for 32-bit indexing");
    if (ids_i64) {
        launch_mm_ids_helper_i64((const int64_t *) ids, ids_src1, ids_dst, bounds, n_experts, n_tok, n_used, nchannels_y,
                                 (int) ids_row_stride, sis1, stream);
    } else {
        launch_mm_ids_helper_i32((const int32_t *) ids, ids_src1, ids_dst, bounds, n_experts, n_tok, n_used, nchannels_y,
                                 (int) ids_row_stride, sis1, stream);
    }
}

void mm_ids(const weight & e, const weight * u, const void * x_bf16, int64_t x_row, const int32_t * ids_src1,
            const int32_t * ids_dst, const int32_t * bounds, void * dst, bool dst_bf16, int64_t dst_col, int n_experts,
            int n_tok, int n_used, cudaStream_t stream) {
    ensure_init();
    check_weight(e);
    BF16N_REQUIRE(mm_ids_supported(e, u), "no batched kernel for this matrix");
    BF16N_REQUIRE(e.rows % n_experts == 0, "rows must split evenly into experts");
    hpm_args a = {};
    a.x = (const bf16 *) x_bf16; a.x_row = x_row;
    a.ids_src1 = ids_src1; a.ids_dst = ids_dst; a.bounds = bounds;
    a.dst = (char *) dst; a.dst_col = dst_col; a.dst_bf16 = dst_bf16;
    a.rows = e.rows / n_experts; a.n_experts = n_experts;
    // Units interleaved per warp (independent decode chains) for down's 8 warps while the experts have few tokens and
    // decode dominates; one unit per warp, and so more tokens per item, once the tensor cores have enough work. Gate/up
    // (16 warps) is fastest with one at every token count. Measured with engines/llamacpp/bench/moe_prefill.cpp on Orin (T-006).
    const int64_t routed = (int64_t) n_tok * n_used; // / n_experts = average tokens per expert
    const int il = e.K != 512 ? 1 : routed <= 24 * (int64_t) n_experts ? 4 : routed <= 48 * (int64_t) n_experts ? 2 : 1;
    BF16N_REQUIRE(routed / 8 + n_experts < 65536, "token tiles exceed the kernel's 16-bit tile offsets");
#define MM(CC, RR, NW, IL, WT, B, G)                                                                                   \
    {                                                                                                                  \
        const size_t shm = mm_shm(G, NW, IL, B, n_experts);                                                           \
        if (n_experts > MM_MAX_EXPERTS_INIT) {                                                                         \
            BF16N_REQUIRE(shm <= (size_t) g_smem_optin, "too many experts for the batched kernel's shared memory");   \
            BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_hp_mm_ids<CC, RR, NW, IL, WT, B, G>,                               \
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));            \
        }                                                                                                              \
        k_hp_mm_ids<CC, RR, NW, IL, WT, B, G><<<g_sms, NW * 32, shm, stream>>>(hpp(e), hpp(G ? *u : e), a);           \
    }
    if (e.fo) {
        const size_t shm = fo_mm_shm(FO_NTOK, n_experts);
        BF16N_REQUIRE(shm <= (size_t) g_smem_optin, "too many experts for the fragment-order kernel's shared memory");
        if (n_experts > MM_MAX_EXPERTS_INIT) {
            BF16N_CUDA_CHECK(cudaFuncSetAttribute(k_fo_mm_ids<FO_NTOK, FO_NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
        }
        k_fo_mm_ids<FO_NTOK, FO_NW><<<g_sms, FO_NW * 32, shm, stream>>>(hpp(e), a);
        BF16N_CUDA_CHECK(cudaGetLastError());
        return;
    }
    // lane-stream rings (RB = 2) where a warp decodes one or two streams, the register queue for down's four (T-019)
    if (u) {
        MM(4, 8, 16, 1, 2, 2, true)
    } else {
        switch ((int) (e.K / 512) * 1000 + e.R * 10 + il) {
            case 4081: MM(4, 8, 16, 1, 2, 2, false) break;
            case 1161: MM(1, 16, 8, 1, 4, 2, false) break;
            case 1162: MM(1, 16, 8, 2, 2, 2, false) break;
            case 1164: MM(1, 16, 8, 4, 1, 0, false) break;
            default: throw std::invalid_argument("bf16n: no batched kernel for this K/R");
        }
    }
#undef MM
    BF16N_CUDA_CHECK(cudaGetLastError());
}

namespace {

__global__ void k_moe_combine(const float * __restrict__ ys, const char * __restrict__ w, bool w_bf16, char * __restrict__ y,
                              bool y_bf16, int n_used, int64_t n) {
    const int t = blockIdx.y;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < n_used; k++) {
            const int64_t s = (int64_t) t * n_used + k;
            const float wk = w_bf16 ? to_f(((const bf16 *) w)[s]) : ((const float *) w)[s];
            acc = fmaf(wk, ys[s * n + i], acc);
        }
        const ydst out = {y + (int64_t) t * n * (y_bf16 ? 2 : 4), y_bf16};
        out.set(i, acc);
    }
}

__global__ void k_swiglu(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        y[i] = silu_single(g[i]) * u[i];
    }
}

} // namespace

void moe_combine(const float * ys, const void * w, bool w_bf16, void * y, bool y_bf16, int n_tok, int n_used, int64_t n,
                 cudaStream_t stream) {
    ensure_init();
    if (n_tok == 0 || n == 0) {
        return;
    }
    BF16N_REQUIRE(n_tok <= 65535, "moe_combine: too many tokens for one launch");
    const dim3 grid((unsigned) std::min<int64_t>((n + 255) / 256, 64), (unsigned) n_tok);
    k_moe_combine<<<grid, 256, 0, stream>>>(ys, (const char *) w, w_bf16, (char *) y, y_bf16, n_used, n);
    BF16N_CUDA_CHECK(cudaGetLastError());
}

void swiglu(const float * g, const float * u, float * y, int64_t n, cudaStream_t stream) {
    ensure_init();
    if (n == 0) {
        return;
    }
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, (int64_t) g_sms * 8);
    k_swiglu<<<blocks, 256, 0, stream>>>(g, u, y, n);
    BF16N_CUDA_CHECK(cudaGetLastError());
}

} // namespace bf16n

#include "bf16n/draft_head.cuh"
