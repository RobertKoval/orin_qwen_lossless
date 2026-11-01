// Fragment-order (FO) layout for BF16N expert matrices, standalone prototype (T-023).
//
// The pair-Huffman format codes the exponents of adjacent weights (2c, 2c+1) together. An mma.m16n8k16 A fragment is
// four bf16x2 registers, each such a pair: lane (g = lane / 4, t = lane % 4) of a 16x16 tile holds rows g, g + 8 and
// columns 2t, 2t + 1 (a0, a1) and 2t + 8, 2t + 9 (a2, a3). FO stores each 16-row band of an expert so that a lane's
// sign|mantissa bytes (8 per tile, 16 per tile pair) and its exponent-pair codes (4 per tile) come in exactly that
// order: the lane decodes its A fragments straight into registers and multiplies them, without a shared-memory slab.
// A band's lane streams are cut into S units along K (S = 2 for K = 2048), so the decode GEMV has as many units as the
// current layout; the batched kernel just continues with the next unit's streams.
//
//   fo_moe <model.gguf> <blk.N.ffn_down_exps.weight | blk.N.ffn_gate_up_exps.weight> <tokens> [reps=20] [check_cols=64]
//
// Checks: FO decode == the BF16 source bit for bit; every kernel against a CPU double reference.
#include "gguf.h"
#include "ggml.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int HP_LP = 12; // pair code length limit
constexpr int HP_PF = 4;  // per-lane stream prefetch depth (words)
#ifndef FO_PF
#define FO_PF 2 // GEMV plane chunks in flight
#endif

template <typename F>
static void parallel_for(int64_t n, F && f) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int k = 0; k < nt; k++) {
        th.emplace_back([&, k] { for (int64_t i = k; i < n; i += nt) { f(i); } });
    }
    for (auto & t : th) { t.join(); }
}

// package-merge length-limited Huffman
static std::vector<int> limited_lengths(const std::vector<uint64_t> & f, int L) {
    std::vector<int> syms;
    for (int i = 0; i < (int) f.size(); i++) { if (f[i]) { syms.push_back(i); } }
    std::vector<int> len(f.size(), 0);
    const int n = syms.size();
    if (n == 1) { len[syms[0]] = 1; return len; }
    std::sort(syms.begin(), syms.end(), [&](int a, int b) { return f[a] < f[b]; });
    struct It { uint64_t w; int leaf; int a; };
    std::vector<std::vector<It>> lv(L);
    std::vector<It> leaves;
    for (int s : syms) { leaves.push_back({f[s], s, -1}); }
    lv[0] = leaves;
    for (int d = 1; d < L; d++) {
        std::vector<It> pk;
        auto & c = lv[d - 1];
        for (size_t i = 0; i + 1 < c.size(); i += 2) { pk.push_back({c[i].w + c[i + 1].w, -1, (int) i}); }
        std::vector<It> mg;
        size_t i = 0, j = 0;
        while (i < pk.size() || j < leaves.size()) {
            if (j < leaves.size() && (i >= pk.size() || leaves[j].w <= pk[i].w)) { mg.push_back(leaves[j++]); } else { mg.push_back(pk[i++]); }
        }
        lv[d] = mg;
    }
    std::function<void(int, int)> walk = [&](int d, int idx) {
        const It & it = lv[d][idx];
        if (it.leaf >= 0) { len[it.leaf]++; return; }
        walk(d - 1, it.a); walk(d - 1, it.a + 1);
    };
    for (int i = 0; i < 2 * n - 2; i++) { walk(L - 1, i); }
    return len;
}

// ---------------------------------------------------------------- FO encoder

// a lane's four pairs of a 16x16 tile, in A-register order: (row, column) of the first weight of each pair
static inline void fo_pair(int lane, int q, int & r, int & c) {
    const int g = lane >> 2, t = lane & 3;
    r = g + 8 * (q & 1);
    c = 2 * t + 8 * (q >> 1);
}

struct FO {
    int64_t rows = 0, K = 0, bands = 0, S = 1;
    std::vector<uint8_t> plane;     // rows*K: band, tile pair, lane: 16 bytes (tile 2p, then 2p+1; a0..a3, low byte first)
    std::vector<uint32_t> lut;      // 1 << HP_LP entries: (e_hi << 23) | (e_lo << 7) | length
    std::vector<uint32_t> ubase;    // per unit (band * S + s): first word of its lane streams
    std::vector<uint16_t> lrel;     // per unit and lane: offset of the lane's stream from ubase
    std::vector<uint32_t> stream;
};

static FO encode_fo(const uint16_t * w, int64_t rows, int64_t K, int S) {
    FO fo;
    fo.rows = rows; fo.K = K; fo.bands = rows / 16; fo.S = S;
    auto ex = [&](int64_t i) { return (w[i] >> 7) & 0xFF; };
    auto sm = [&](int64_t i) { return (uint8_t) (((w[i] >> 8) & 0x80) | (w[i] & 0x7F)); };
    std::vector<uint64_t> freq(65536, 0);
    for (int64_t i = 0; i < rows * K; i += 2) { freq[ex(i) | ex(i + 1) << 8]++; }
    int nsym = 0;
    for (auto f : freq) { nsym += f != 0; }
    if (nsym > (1 << HP_LP)) { fprintf(stderr, "too many pair symbols: %d\n", nsym); exit(1); }
    const std::vector<int> len = limited_lengths(freq, HP_LP);
    std::vector<uint32_t> code(65536, 0);
    {
        uint32_t c = 0;
        for (int l = 1; l <= HP_LP; l++) {
            for (int s = 0; s < 65536; s++) { if (len[s] == l) { code[s] = c++; } }
            c <<= 1;
        }
    }
    fo.lut.assign(1 << HP_LP, 0);
    for (int s = 0; s < 65536; s++) {
        if (len[s]) {
            const uint32_t lo = code[s] << (HP_LP - len[s]), hi = (code[s] + 1) << (HP_LP - len[s]);
            for (uint32_t i = lo; i < hi; i++) { fo.lut[i] = ((uint32_t) (s >> 8) << 23) | ((uint32_t) (s & 0xFF) << 7) | (uint32_t) len[s]; }
        }
    }
    const int64_t kt_n = K / 16, units = fo.bands * S;
    fo.plane.resize(rows * K);
    std::vector<std::vector<uint32_t>> us(units);
    std::vector<uint32_t> rel32(units * 32);
    parallel_for(fo.bands, [&](int64_t b) {
        uint8_t * pb = fo.plane.data() + b * 16 * K;
        for (int64_t p = 0; p < kt_n / 2; p++) {
            for (int l = 0; l < 32; l++) {
                uint8_t * o = pb + p * 512 + l * 16;
                for (int kk = 0; kk < 2; kk++) {
                    for (int q = 0; q < 4; q++) {
                        int r, c;
                        fo_pair(l, q, r, c);
                        const int64_t i = (b * 16 + r) * K + (2 * p + kk) * 16 + c;
                        o[8 * kk + 2 * q] = sm(i);
                        o[8 * kk + 2 * q + 1] = sm(i + 1);
                    }
                }
            }
        }
        for (int s = 0; s < S; s++) {
            const int64_t u = b * S + s;
            auto & out = us[u];
            for (int l = 0; l < 32; l++) {
                rel32[u * 32 + l] = (uint32_t) out.size();
                uint64_t acc = 0;
                int nb = 0;
                for (int64_t kt = s * kt_n / S; kt < (s + 1) * kt_n / S; kt++) {
                    for (int q = 0; q < 4; q++) {
                        int r, c;
                        fo_pair(l, q, r, c);
                        const int64_t i = (b * 16 + r) * K + kt * 16 + c;
                        const int sym = ex(i) | ex(i + 1) << 8;
                        acc = (acc << len[sym]) | code[sym];
                        nb += len[sym];
                        if (nb >= 32) {
                            out.push_back((uint32_t) (acc >> (nb - 32)));
                            nb -= 32;
                            acc &= nb ? ((1ull << nb) - 1) : 0;
                        }
                    }
                }
                if (nb) { out.push_back((uint32_t) (acc << (32 - nb))); }
            }
        }
    });
    fo.ubase.resize(units);
    fo.lrel.resize(units * 32);
    uint64_t nw = 0;
    for (int64_t u = 0; u < units; u++) {
        fo.ubase[u] = (uint32_t) nw;
        for (int l = 0; l < 32; l++) {
            if (rel32[u * 32 + l] > 0xFFFF) { fprintf(stderr, "lane offset overflow\n"); exit(1); }
            fo.lrel[u * 32 + l] = (uint16_t) rel32[u * 32 + l];
        }
        nw += us[u].size();
    }
    fo.stream.reserve(nw + 32);
    for (auto & v : us) { fo.stream.insert(fo.stream.end(), v.begin(), v.end()); }
    fo.stream.resize(nw + 32, 0); // lookahead of the decoder's prefetch
    return fo;
}

// ---------------------------------------------------------------- device side (decoder as in bf16n.cu)

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
__device__ __forceinline__ uint32_t sel_bits(uint32_t a, uint32_t b) { // (a & 0x807F807F) | (b & ~0x807F807F)
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, 0x807F807F, 0xE4;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint4 ld_plane(const uint4 * p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t shl64w(uint64_t b, uint32_t n) {
    uint64_t d;
    asm("{.reg .b32 l, h; mov.b64 {l, h}, %1; shf.l.wrap.b32 h, l, h, %2; shf.l.wrap.b32 l, 0, l, %2; mov.b64 %0, {l, h};}" : "=l"(d) : "l"(b), "r"(n));
    return d;
}

struct fo_dev { const uint8_t * plane; const uint32_t * lut; const uint32_t * ubase; const uint16_t * lrel; const uint32_t * stream; };

// one lane's decoder over a unit's stream (the HP_UNIT_* / HP_PAIRS logic of bf16n.cu)
struct fo_dec {
    const uint32_t * lst;
    uint32_t idx, wq[HP_PF], v, pa, pb;
    uint64_t buf;
    __device__ __forceinline__ void begin(const fo_dev & f, int64_t gu, int lane) {
        lst = f.stream;
        idx = __ldg(f.ubase + gu) + __ldg(f.lrel + gu * 32 + lane);
        buf = (uint64_t) __ldg(lst + idx) << 32;
#pragma unroll
        for (int i = 0; i < HP_PF; i++) { wq[i] = __ldg(lst + idx + 1 + i); }
        idx += 1 + HP_PF;
        v = 32; pa = 0; pb = 0;
    }
    __device__ __forceinline__ void pairs(uint32_t (&ex)[8], const uint32_t * lut) {
#pragma unroll
        for (int q = 0; q < 8; q++) {
            if (q % 2 == 0) {
                v = v - pa - pb;
                const uint32_t m = __funnelshift_r(0u, 1u, v);
                if (!(v & 32u)) {
                    buf += (uint64_t) wq[0] * m;
#pragma unroll
                    for (int i = 0; i + 1 < HP_PF; i++) { wq[i] = wq[i + 1]; }
                    wq[HP_PF - 1] = __ldg(lst + idx);
                    idx++;
                }
                v |= 32u;
            }
            const uint32_t ent = lut[(uint32_t) (buf >> 32) >> (32 - HP_LP)];
            buf = shl64w(buf, ent);
            ex[q] = ent;
            if (q % 2 == 0) { pa = ent; } else { pb = ent; }
        }
    }
};

// the A registers of tile pair (plane word pl, exponent entries ex[8]): a[0..3] for the even tile, a[4..7] for the odd one
__device__ __forceinline__ void fo_a(const uint4 & pl, const uint32_t (&ex)[8], uint32_t (&a)[8]) {
    a[0] = sel_bits(prmt(pl.x, 0, 0x9180u), ex[0]); a[1] = sel_bits(prmt(pl.x, 0, 0xB3A2u), ex[1]);
    a[2] = sel_bits(prmt(pl.y, 0, 0x9180u), ex[2]); a[3] = sel_bits(prmt(pl.y, 0, 0xB3A2u), ex[3]);
    a[4] = sel_bits(prmt(pl.z, 0, 0x9180u), ex[4]); a[5] = sel_bits(prmt(pl.z, 0, 0xB3A2u), ex[5]);
    a[6] = sel_bits(prmt(pl.w, 0, 0x9180u), ex[6]); a[7] = sel_bits(prmt(pl.w, 0, 0xB3A2u), ex[7]);
}

// FO -> dense BF16 (row-major), one warp per band: the bitwise round trip
__global__ void k_fo_decode(fo_dev f, int64_t K, int64_t bands, int S, uint16_t * out) {
    __shared__ uint32_t lut[1 << HP_LP];
    for (int i = threadIdx.x; i < (1 << HP_LP); i += blockDim.x) { lut[i] = f.lut[i]; }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t b = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (b >= bands) { return; }
    const int g = lane >> 2, t = lane & 3;
    const int64_t np = K / 32;
    fo_dec d;
    for (int64_t p = 0; p < np; p++) {
        if (p % (np / S) == 0) { d.begin(f, b * S + p / (np / S), lane); }
        const uint4 pl = ld_plane((const uint4 *) (f.plane + b * 16 * K + p * 512) + lane);
        uint32_t ex[8], a[8];
        d.pairs(ex, lut);
        fo_a(pl, ex, a);
        for (int kk = 0; kk < 2; kk++) {
            for (int q = 0; q < 4; q++) {
                const int r = g + 8 * (q & 1), c = 2 * t + 8 * (q >> 1);
                const int64_t i = (b * 16 + r) * K + (2 * p + kk) * 16 + c;
                out[i] = (uint16_t) (a[4 * kk + q] & 0xFFFF);
                out[i + 1] = (uint16_t) (a[4 * kk + q] >> 16);
            }
        }
    }
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t * a, const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void * src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
}

struct fo_item { int e, col0, ncol; };

// stage columns [0, npad) x [k0, k0 + KS) of an item's BF16 activations into xs (row stride KS + 8 elements)
template <int KS>
__device__ __forceinline__ void stage_x(uint32_t xs_s, __nv_bfloat16 * xs, const fo_item & w, int npad, int k0, int K,
        const int * col_x, const __nv_bfloat16 * xb) {
    for (int i = threadIdx.x; i < npad * (KS / 8); i += blockDim.x) {
        const int r = i / (KS / 8), c8 = i % (KS / 8);
        if (r < w.ncol) {
            cp_async16(xs_s + (r * (KS + 8) + c8 * 8) * 2, xb + (int64_t) col_x[w.col0 + r] * K + k0 + c8 * 8);
        } else {
            *(uint4 *) (xs + r * (KS + 8) + c8 * 8) = make_uint4(0, 0, 0, 0);
        }
    }
}

// a tile pair times all token pairs of the staged activations
template <int NTOK, int KS>
__device__ __forceinline__ void fo_mma_pair(float (&acc)[NTOK / 8][4], const uint32_t (&a)[8], uint32_t lrow, int kp, int npairs) {
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        const uint32_t kofs = (uint32_t) ((2 * kp + kk) * 16 * 2);
#pragma unroll
        for (int q = 0; q < NTOK / 16; q++) {
            if (q < npairs) {
                uint32_t b[4];
                ldmatrix_x4(b, lrow + q * 16 * (KS + 8) * 2 + kofs);
                mma_bf16(acc[2 * q], a + 4 * kk, b[0], b[1]);
                mma_bf16(acc[2 * q + 1], a + 4 * kk, b[2], b[3]);
            }
        }
    }
}

// batched MUL_MAT_ID, K = 512 (down): a block takes a work item (expert, up to NTOK of its columns), stages the columns'
// BF16 activations once (cp.async, rows padded by 16 bytes so ldmatrix is conflict-free), and its warps take the
// expert's 16-row bands: per tile pair a lane decodes its A fragments and multiplies them with every token pair
template <int K, int NTOK, int NW>
__global__ void __launch_bounds__(NW * 32, 1) k_fo_mm(fo_dev f, const fo_item * items, int n_items, const int * col_x,
        const int * col_y, const __nv_bfloat16 * xb, float * y, int rows_pe) {
    extern __shared__ __align__(16) uint8_t shm[];
    uint32_t * lut = (uint32_t *) shm;
    __nv_bfloat16 * xs = (__nv_bfloat16 *) (shm + (4 << HP_LP));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    for (int i = threadIdx.x; i < (1 << HP_LP); i += blockDim.x) { lut[i] = f.lut[i]; }
    const int bands_pe = rows_pe / 16;
    const uint32_t xs_s = (uint32_t) __cvta_generic_to_shared(xs);
    // ldmatrix row address of this lane: token 16q + (lane / 16) * 8 + lane % 8, column 8 * ((lane / 8) % 2)
    const uint32_t lrow = xs_s + (((lane >> 4) * 8 + (lane & 7)) * (K + 8) + ((lane >> 3) & 1) * 8) * 2;
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const fo_item w = items[it];
        __syncthreads(); // the previous item's readers of xs are done
        const int npad = (w.ncol + 15) & ~15;
        stage_x<K>(xs_s, xs, w, npad, 0, K, col_x, xb);
        asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
        __syncthreads();
        for (int band = warp; band < bands_pe; band += NW) {
            const int64_t gu = (int64_t) w.e * bands_pe + band;
            fo_dec d;
            d.begin(f, gu, lane);
            float acc[NTOK / 8][4];
#pragma unroll
            for (int n = 0; n < NTOK / 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
            const uint4 * pp = (const uint4 *) (f.plane + gu * 16 * K) + lane;
            uint4 nxt = ld_plane(pp);
#pragma unroll 1
            for (int p = 0; p < K / 32; p++) {
                const uint4 pl = nxt;
                if (p + 1 < K / 32) { nxt = ld_plane(pp + (p + 1) * 32); }
                uint32_t ex[8], a[8];
                d.pairs(ex, lut);
                fo_a(pl, ex, a);
                fo_mma_pair<NTOK, K>(acc, a, lrow, p, npad / 16);
            }
            const int row = band * 16 + g;
#pragma unroll
            for (int n = 0; n < NTOK / 8; n++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int col = n * 8 + 2 * t + h;
                    if (col < w.ncol) {
                        float * yo = y + (int64_t) col_y[w.col0 + col] * rows_pe;
                        yo[row] = acc[n][h];
                        yo[row + 8] = acc[n][2 + h];
                    }
                }
            }
        }
    }
}

// batched gate + up + SwiGLU, K = 2048: as k_fo_mm, but a warp decodes the same band of gate and up and multiplies both
// with the same activation fragments; the activations come in KS-column slices, double-buffered by cp.async; a band's
// streams are S units along K
template <int K, int NTOK, int NW, int S, int KS>
__global__ void __launch_bounds__(NW * 32, 1) k_fo_mm_gu(fo_dev fg, fo_dev fu, const fo_item * items, int n_items,
        const int * col_x, const int * col_y, const __nv_bfloat16 * xb, float * y, int rows_pe) {
    extern __shared__ __align__(16) uint8_t shm[];
    uint32_t * lutg = (uint32_t *) shm;
    uint32_t * lutu = lutg + (1 << HP_LP);
    __nv_bfloat16 * xs = (__nv_bfloat16 *) (shm + (8 << HP_LP));
    constexpr int XSL = NTOK * (KS + 8); // elements per slice buffer
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    for (int i = threadIdx.x; i < (1 << HP_LP); i += blockDim.x) { lutg[i] = fg.lut[i]; lutu[i] = fu.lut[i]; }
    const int bands_pe = rows_pe / 16;
    const uint32_t xs_s = (uint32_t) __cvta_generic_to_shared(xs);
    const uint32_t lrow0 = xs_s + (((lane >> 4) * 8 + (lane & 7)) * (KS + 8) + ((lane >> 3) & 1) * 8) * 2;
    constexpr int NSL = K / KS, PPS = KS / 32, PPU = K / 32 / S; // slices, tile pairs per slice, tile pairs per unit
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const fo_item w = items[it];
        const int npad = (w.ncol + 15) & ~15;
        for (int bg = 0; bg < bands_pe; bg += NW) {
            __syncthreads(); // xs is free
            stage_x<KS>(xs_s, xs, w, npad, 0, K, col_x, xb);
            asm volatile("cp.async.commit_group;" ::: "memory");
            const int band = bg + warp;
            const int64_t gb = (int64_t) w.e * bands_pe + band;
            fo_dec dg, du;
            float ag[NTOK / 8][4], au[NTOK / 8][4];
#pragma unroll
            for (int n = 0; n < NTOK / 8; n++) {
#pragma unroll
                for (int j = 0; j < 4; j++) { ag[n][j] = 0.f; au[n][j] = 0.f; }
            }
            const uint4 * ppg = (const uint4 *) (fg.plane + gb * 16 * K) + lane;
            const uint4 * ppu = (const uint4 *) (fu.plane + gb * 16 * K) + lane;
            uint4 nxg = make_uint4(0, 0, 0, 0), nxu = nxg;
            if (band < bands_pe) { nxg = ld_plane(ppg); nxu = ld_plane(ppu); }
#pragma unroll 1
            for (int sl = 0; sl < NSL; sl++) {
                if (sl + 1 < NSL) {
                    stage_x<KS>(xs_s + ((sl + 1) & 1) * XSL * 2, xs + ((sl + 1) & 1) * XSL, w, npad, (sl + 1) * KS, K, col_x, xb);
                }
                asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
                __syncthreads(); // slice sl is in
                const uint32_t lrow = lrow0 + (sl & 1) * XSL * 2;
#pragma unroll 1
                for (int pp = 0; pp < PPS; pp++) {
                    const int p = sl * PPS + pp;
                    if (p % PPU == 0 && band < bands_pe) {
                        dg.begin(fg, gb * S + p / PPU, lane);
                        du.begin(fu, gb * S + p / PPU, lane);
                    }
                    if (band < bands_pe) {
                        const uint4 plg = nxg, plu = nxu;
                        if (p + 1 < K / 32) { nxg = ld_plane(ppg + (p + 1) * 32); nxu = ld_plane(ppu + (p + 1) * 32); }
                        uint32_t ex[8], a[8];
                        dg.pairs(ex, lutg);
                        fo_a(plg, ex, a);
                        fo_mma_pair<NTOK, KS>(ag, a, lrow, pp, npad / 16);
                        du.pairs(ex, lutu);
                        fo_a(plu, ex, a);
                        fo_mma_pair<NTOK, KS>(au, a, lrow, pp, npad / 16);
                    }
                }
                __syncthreads(); // everyone is done with slice sl before it is overwritten
            }
            if (band < bands_pe) {
                const int row = band * 16 + g;
#pragma unroll
                for (int n = 0; n < NTOK / 8; n++) {
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int col = n * 8 + 2 * t + h;
                        if (col < w.ncol) {
                            float * yo = y + (int64_t) col_y[w.col0 + col] * rows_pe;
                            const float g0 = ag[n][h], g8 = ag[n][2 + h];
                            yo[row] = g0 / (1.0f + expf(-g0)) * au[n][h];
                            yo[row + 8] = g8 / (1.0f + expf(-g8)) * au[n][2 + h];
                        }
                    }
                }
            }
        }
    }
}


#define HP_UNIT_RESET(F, GU)                                                                         \
    idx = __ldg((F).ubase + (GU)) + __ldg((F).lrel + (GU) * 32 + lane);                               \
    buf = (uint64_t) __ldg(lst + idx) << 32;                                                         \
    _Pragma("unroll") for (int i_ = 0; i_ < HP_PF; i_++) { wq[i_] = __ldg(lst + idx + 1 + i_); }    \
    idx += 1 + HP_PF;                                                                                \
    v = 32; pa = 0; pb = 0;

#define HP_PAIRS(EX)                                                                                 \
    _Pragma("unroll") for (int q_ = 0; q_ < 8; q_++) {                                               \
        if (q_ % 2 == 0) {                                                                           \
            v = v - pa - pb;                                                                         \
            const uint32_t m_ = __funnelshift_r(0u, 1u, v);                                          \
            if (!(v & 32u)) {                                                                        \
                buf += (uint64_t) wq[0] * m_;                                                        \
                _Pragma("unroll") for (int i_ = 0; i_ + 1 < HP_PF; i_++) { wq[i_] = wq[i_ + 1]; }    \
                wq[HP_PF - 1] = __ldg(lst + idx);                                                    \
                idx++;                                                                               \
            }                                                                                        \
            v |= 32u;                                                                                \
        }                                                                                            \
        const uint32_t ent_ = lut[(uint32_t) (buf >> 32) >> (32 - HP_LP)];                           \
        buf = shl64w(buf, ent_);                                                                     \
        (EX)[q_] = ent_;                                                                             \
        if (q_ % 2 == 0) { pa = ent_; } else { pb = ent_; }                                          \
    }

// decode GEMV for a few columns (one or two tokens' experts): a warp takes a (column, band, K part) unit, a lane
// accumulates its rows g and g + 8 over its columns of each tile and the four lanes of a row group sum up at the end;
// with S > 1 the parts are added into a zeroed output (two addends: the order does not change the sum)
template <int K, int S>
__global__ void __launch_bounds__(1024, 1) k_fo_mmv(fo_dev f, const int * col_e, const int * col_x, const int * col_y,
        int ncols, int nx, const float * x, float * y, int rows_pe) {
    extern __shared__ __align__(16) uint8_t shm[];
    uint32_t * lut = (uint32_t *) shm;
    float * xs = (float *) (shm + (4 << HP_LP));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int bands_pe = rows_pe / 16;
    const int units = ncols * bands_pe * S;
    constexpr int NP = K / 32 / S, PF = NP < FO_PF ? NP : FO_PF;
    for (int i = threadIdx.x; i < (1 << HP_LP); i += blockDim.x) { lut[i] = f.lut[i]; }
    // activations in lane order: per column, tile pair p and t, the 8 values a lane with that t multiplies
    // (columns 2t, 2t + 1, 2t + 8, 2t + 9 of tiles 2p and 2p + 1), broadcast to the lanes of a t
    for (int i = threadIdx.x; i < nx * K; i += blockDim.x) {
        const int c = i / K, dd = i % K, j = dd & 7, tt = (dd >> 3) & 3, pp_ = dd >> 5;
        const int src = 32 * pp_ + 16 * (j >> 2) + 2 * tt + (j & 1) + 8 * ((j >> 1) & 1);
        xs[i] = x[(int64_t) c * K + src];
    }
    __syncthreads();
    // blocks per column as the per-column GEMV (a block streams one expert); more blocks than columns split its bands
    const int bpc = gridDim.x >= ncols ? gridDim.x / ncols : 1;
    const int upc = bands_pe * S;
    const uint32_t * lst = f.stream;
    uint32_t idx = 0, wq[HP_PF], v = 32, pa = 0, pb = 0;
    uint64_t buf = 0;
    for (int cc = gridDim.x >= ncols ? blockIdx.x / bpc : blockIdx.x; cc < ncols; cc += gridDim.x >= ncols ? ncols : gridDim.x) {
    for (int uc = (gridDim.x >= ncols ? blockIdx.x % bpc : 0) * wpb + warp; uc < upc; uc += bpc * wpb) {
        const int c = cc, band = uc / S, s = uc % S;
        const int64_t gb = (int64_t) col_e[c] * bands_pe + band;
        HP_UNIT_RESET(f, gb * S + s)
        const float4 * xc = (const float4 *) (xs + col_x[c] * K) + s * NP * 8 + 2 * t;
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s8[4] = {0.f, 0.f, 0.f, 0.f};
        const uint4 * pp = (const uint4 *) (f.plane + gb * 16 * K) + s * NP * 32 + lane;
        uint4 ring[PF];
#pragma unroll
        for (int i = 0; i < PF; i++) { ring[i] = ld_plane(pp + i * 32); }
#pragma unroll
        for (int p = 0; p < NP; p++) {
            const uint4 pl = ring[p % PF];
            if (p + PF < NP) { ring[p % PF] = ld_plane(pp + (p + PF) * 32); }
            uint32_t ex[8], a[8];
            HP_PAIRS(ex)
            fo_a(pl, ex, a);
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const float4 xv = xc[p * 8 + kk];
                const uint32_t * A = a + 4 * kk;
                s0[0] = fmaf(__uint_as_float(A[0] << 16), xv.x, s0[0]);
                s0[1] = fmaf(__uint_as_float(A[0] & 0xFFFF0000u), xv.y, s0[1]);
                s0[2] = fmaf(__uint_as_float(A[2] << 16), xv.z, s0[2]);
                s0[3] = fmaf(__uint_as_float(A[2] & 0xFFFF0000u), xv.w, s0[3]);
                s8[0] = fmaf(__uint_as_float(A[1] << 16), xv.x, s8[0]);
                s8[1] = fmaf(__uint_as_float(A[1] & 0xFFFF0000u), xv.y, s8[1]);
                s8[2] = fmaf(__uint_as_float(A[3] << 16), xv.z, s8[2]);
                s8[3] = fmaf(__uint_as_float(A[3] & 0xFFFF0000u), xv.w, s8[3]);
            }
        }
        float r0 = (s0[0] + s0[1]) + (s0[2] + s0[3]), r8 = (s8[0] + s8[1]) + (s8[2] + s8[3]);
        r0 += __shfl_xor_sync(0xffffffffu, r0, 1); r8 += __shfl_xor_sync(0xffffffffu, r8, 1);
        r0 += __shfl_xor_sync(0xffffffffu, r0, 2); r8 += __shfl_xor_sync(0xffffffffu, r8, 2);
        if (t == 0) {
            float * yo = y + (int64_t) col_y[c] * rows_pe + band * 16 + g;
            if (S == 1) {
                yo[0] = r0;
                yo[8] = r8;
            } else {
                atomicAdd(yo, r0);
                atomicAdd(yo + 8, r8);
            }
        }
    }
    }
}

// ---------------------------------------------------------------- host

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16; return (uint16_t) u; }
static double bf2d(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

static std::vector<uint16_t> read_bf16(gguf_context * g, ggml_context * meta, const char * path, const std::string & name,
        int64_t & K, int64_t & N, int64_t & E) {
    const int64_t tid = gguf_find_tensor(g, name.c_str());
    const ggml_tensor * tm = tid < 0 ? nullptr : ggml_get_tensor(meta, name.c_str());
    if (!tm || tm->type != GGML_TYPE_BF16 || ggml_n_dims(tm) != 3) { fprintf(stderr, "%s: need a 3-D BF16 expert tensor\n", name.c_str()); exit(1); }
    K = tm->ne[0]; N = tm->ne[1]; E = tm->ne[2];
    std::vector<uint16_t> w(K * N * E);
    const int fd = open(path, O_RDONLY);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    size_t done = 0, total = w.size() * 2;
    while (done < total) {
        const ssize_t r = pread(fd, (char *) w.data() + done, total - done, off + done);
        if (r <= 0) { perror("pread"); exit(1); }
        done += r;
    }
    close(fd);
    return w;
}

template <typename T> static T * dcopy(const std::vector<T> & v) {
    T * d;
    CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static fo_dev upload(const FO & fo) {
    return {dcopy(fo.plane), dcopy(fo.lut), dcopy(fo.ubase), dcopy(fo.lrel), dcopy(fo.stream)};
}

static bool round_trip(const FO & fo, const fo_dev & fd, const std::vector<uint16_t> & w) {
    uint16_t * d_out;
    CK(cudaMalloc(&d_out, w.size() * 2));
    k_fo_decode<<<(fo.bands + 7) / 8, 256>>>(fd, fo.K, fo.bands, (int) fo.S, d_out);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> back(w.size());
    CK(cudaMemcpy(back.data(), d_out, w.size() * 2, cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return memcmp(back.data(), w.data(), w.size() * 2) == 0;
}

// median kernel time over reps of `inner` back-to-back launches, after a 2 s warm-up (GPU clock governor)
template <typename L>
static float time_ms(L && launch, int reps, int inner) {
    for (const auto t0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2);) {
        launch(); CK(cudaDeviceSynchronize());
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(e0);
        for (int i = 0; i < inner; i++) { launch(); }
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float m; cudaEventElapsedTime(&m, e0, e1); ms.push_back(m / inner);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tensor tokens [reps] [check_cols]\n", argv[0]); return 1; }
    const char * path = argv[1];
    const std::string name = argv[2];
    const int T = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 20;
    const int check_cols = argc > 5 ? atoi(argv[5]) : 64;
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    const size_t gu_at = name.find("gate_up");
    const bool glu = gu_at != std::string::npos;
    const bool down = name.find("down") != std::string::npos;
    int64_t K, N, E;
    std::vector<uint16_t> w = read_bf16(g, meta, path, glu ? std::string(name).replace(gu_at, 7, "gate") : name, K, N, E);
    std::vector<uint16_t> w2;
    if (glu) { int64_t K2, N2, E2; w2 = read_bf16(g, meta, path, std::string(name).replace(gu_at, 7, "up"), K2, N2, E2); }
    const int S = K > 512 ? 2 : 1;
    const auto t0 = std::chrono::steady_clock::now();
    FO fo = encode_fo(w.data(), N * E, K, S), fo2;
    if (glu) { fo2 = encode_fo(w2.data(), N * E, K, S); }
    const double enc_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double bytes = fo.plane.size() + fo.stream.size() * 4.0 + fo.lrel.size() * 2.0 + fo.ubase.size() * 4.0;
    printf("%s: K=%lld N=%lld E=%lld S=%d, FO encode %.1f s, ratio %.4f\n", name.c_str(), (long long) K, (long long) N, (long long) E, S,
           enc_s, bytes / (w.size() * 2.0));
    const fo_dev fd = upload(fo), fd2 = glu ? upload(fo2) : fo_dev{};
    if (!round_trip(fo, fd, w) || (glu && !round_trip(fo2, fd2, w2))) { printf("round trip: MISMATCH\n"); return 1; }
    printf("round trip: bitwise OK\n");

    // routing (uniform, seed 1234): T tokens x 8 distinct experts; down takes one input row per (token, slot)
    const int n_used = 8;
    std::mt19937_64 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t nx = down ? (int64_t) T * n_used : T;
    std::vector<uint16_t> xh(nx * K);
    for (auto & v : xh) { v = f2bf(nd(rng)); }
    std::vector<int> ids(T * n_used);
    std::uniform_int_distribution<int> ue(0, (int) E - 1);
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < n_used; s++) {
            int e;
            do { e = ue(rng); } while (std::find(&ids[t * n_used], &ids[t * n_used] + s, e) != &ids[t * n_used] + s);
            ids[t * n_used + s] = e;
        }
    }
    std::vector<std::vector<int>> per_e(E);
    for (int t = 0; t < T; t++) { for (int s = 0; s < n_used; s++) { per_e[ids[t * n_used + s]].push_back(t * n_used + s); } }
    std::vector<int> cx, cy, e_col0(E);
    int maxe = 0, used = 0;
    for (int e = 0; e < E; e++) {
        e_col0[e] = cx.size();
        maxe = std::max(maxe, (int) per_e[e].size());
        used += !per_e[e].empty();
        for (int c : per_e[e]) { cx.push_back(down ? c : c / n_used); cy.push_back(c); }
    }
    printf("tokens %d: columns per expert max %d, mean %.1f, experts used %d\n", T, maxe, (double) T * n_used / E, used);
    const int64_t ny = (int64_t) T * n_used * N;
    float * d_y, * d_y2;
    CK(cudaMalloc(&d_y, ny * 4));
    CK(cudaMalloc(&d_y2, ny * 4));
    std::vector<float> y(ny), y2(ny);

    // CPU double reference of a column of one matrix
    auto ref_col = [&](const std::vector<uint16_t> & wm, int c, std::vector<double> & out, std::vector<double> & sabs) {
        const int e = ids[c];
        const uint16_t * xr = &xh[(int64_t) (down ? c : c / n_used) * K];
        out.assign(N, 0); sabs.assign(N, 0);
        for (int64_t r = 0; r < N; r++) {
            const uint16_t * wr = &wm[((int64_t) e * N + r) * K];
            double s = 0, sa = 0;
            for (int64_t k = 0; k < K; k++) { const double p = bf2d(wr[k]) * bf2d(xr[k]); s += p; sa += std::fabs(p); }
            out[r] = s; sabs[r] = sa;
        }
    };
    auto check = [&](const char * tag, bool swiglu) {
        std::mt19937 r2(7);
        int bad = 0; double maxrel = 0;
        std::vector<double> rg, ag, ru, au;
        for (int k = 0; k < check_cols; k++) {
            const int c = r2() % (T * n_used);
            ref_col(w, c, rg, ag);
            if (swiglu) { ref_col(w2, c, ru, au); }
            for (int64_t r = 0; r < N; r++) {
                double ref, scale;
                if (swiglu) {
                    const double sg = 1.0 / (1.0 + std::exp(-rg[r]));
                    ref = rg[r] * sg * ru[r];
                    scale = ag[r] * std::fabs(ru[r]) + std::fabs(rg[r] * sg) * au[r] + 1e-30;
                } else {
                    ref = rg[r]; scale = ag[r] + 1e-30;
                }
                const double err = std::fabs(y[(int64_t) c * N + r] - ref) / scale;
                maxrel = std::max(maxrel, err);
                bad += err > 1e-5;
            }
        }
        printf("  %s check %d columns x %lld rows: max |err| / scale %.3g, %d outside 1e-5 -> %s\n", tag, check_cols, (long long) N,
               maxrel, bad, bad ? "FAIL" : "OK");
    };

    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (T > 8) {
        int * d_cx = dcopy(cx), * d_cy = dcopy(cy);
        __nv_bfloat16 * d_x = (__nv_bfloat16 *) dcopy(xh);
        auto items_for = [&](int ntok) {
            std::vector<fo_item> items;
            for (int e = 0; e < E; e++) {
                for (int c = 0; c < (int) per_e[e].size(); c += ntok) {
                    items.push_back({e, e_col0[e] + c, std::min(ntok, (int) per_e[e].size() - c)});
                }
            }
            std::stable_sort(items.begin(), items.end(), [](const fo_item & a, const fo_item & b) { return a.ncol > b.ncol; });
            return items;
        };
        auto run = [&](auto kern, int ntok, int nw, size_t shm, const char * tag, bool swiglu) {
            const std::vector<fo_item> items = items_for(ntok);
            fo_item * d_items = dcopy(items);
            CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
            const float ms = time_ms([&] {
                if (swiglu) {
                    kern<<<sms, nw * 32, shm>>>(fd, fd2, d_items, (int) items.size(), d_cx, d_cy, d_x, d_y, (int) N);
                } else {
                    kern<<<sms, nw * 32, shm>>>(fd, fd, d_items, (int) items.size(), d_cx, d_cy, d_x, d_y, (int) N);
                }
            }, reps, 1);
            printf("%s NTOK %3d: %zu items, %.3f ms (median of %d)\n", tag, ntok, items.size(), ms, reps);
            CK(cudaMemcpy(y.data(), d_y, ny * 4, cudaMemcpyDeviceToHost));
            check(tag, swiglu);
            cudaFree(d_items);
        };
        if (glu) {
            constexpr int KS = 256;
            run(k_fo_mm_gu<2048, 64, 16, 2, KS>, 64, 16, (8 << HP_LP) + 2 * 64 * (KS + 8) * 2, "fo gate+up+swiglu", true);
            run(k_fo_mm_gu<2048, 48, 16, 2, KS>, 48, 16, (8 << HP_LP) + 2 * 48 * (KS + 8) * 2, "fo gate+up+swiglu", true);
        } else {
            for (int ntok : {64, 96}) {
                const std::vector<fo_item> items = items_for(ntok);
                fo_item * d_items = dcopy(items);
                const size_t shm = (4 << HP_LP) + (size_t) ntok * (K + 8) * 2;
                float ms;
                if (ntok == 64) {
                    CK(cudaFuncSetAttribute(k_fo_mm<512, 64, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
                    ms = time_ms([&] { k_fo_mm<512, 64, 16><<<sms, 512, shm>>>(fd, d_items, (int) items.size(), d_cx, d_cy, d_x, d_y, (int) N); }, reps, 1);
                } else {
                    CK(cudaFuncSetAttribute(k_fo_mm<512, 96, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
                    ms = time_ms([&] { k_fo_mm<512, 96, 16><<<sms, 512, shm>>>(fd, d_items, (int) items.size(), d_cx, d_cy, d_x, d_y, (int) N); }, reps, 1);
                }
                printf("fo down NTOK %3d: %zu items, %.3f ms (median of %d)\n", ntok, items.size(), ms, reps);
                CK(cudaMemcpy(y.data(), d_y, ny * 4, cudaMemcpyDeviceToHost));
                check("fo down", false);
                cudaFree(d_items);
            }
        }
        return 0;
    }

    // GEMV: F32 activations (the decode path multiplies F32), columns = (token, slot), gate and up as separate launches
    const int ncols = T * n_used;
    std::vector<float> xf(xh.size());
    for (size_t i = 0; i < xh.size(); i++) { xf[i] = (float) bf2d(xh[i]); }
    std::vector<int> ce(ncols), cxx(ncols), cyy(ncols);
    for (int c = 0; c < ncols; c++) { ce[c] = ids[c]; cxx[c] = down ? c : c / n_used; cyy[c] = c; }
    int * d_ce = dcopy(ce), * d_cxx = dcopy(cxx), * d_cyy = dcopy(cyy);
    float * d_xf = dcopy(xf);
    const int units = ncols * (int) (N / 16) * S;
    const int per_col = std::max(1, sms / ncols);
    const int wpb = std::max(2, std::min(32, ((int) (N / 16) * S + per_col - 1) / per_col));
    const int grid = per_col * ncols;
    const size_t shm = (4 << HP_LP) + (size_t) nx * K * 4;
    auto gemv = [&](const fo_dev & f, float * dy) {
        if (S == 1) {
            k_fo_mmv<512, 1><<<grid, wpb * 32, shm>>>(f, d_ce, d_cxx, d_cyy, ncols, (int) nx, d_xf, dy, (int) N);
        } else {
            CK(cudaMemsetAsync(dy, 0, ny * 4));
            k_fo_mmv<2048, 2><<<grid, wpb * 32, shm>>>(f, d_ce, d_cxx, d_cyy, ncols, (int) nx, d_xf, dy, (int) N);
        }
    };
    CK(cudaFuncSetAttribute(k_fo_mmv<512, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
    CK(cudaFuncSetAttribute(k_fo_mmv<2048, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm));
    const float ms = time_ms([&] { gemv(fd, d_y); }, reps, 100);
    printf("fo GEMV %d columns, %d units, %d x %d warps: %.1f us per call (median of %d x 100 back to back)\n", ncols, units, grid, wpb,
           ms * 1000, reps);
    CK(cudaMemcpy(y.data(), d_y, ny * 4, cudaMemcpyDeviceToHost));
    check("fo GEMV", false);
    return 0;
}
