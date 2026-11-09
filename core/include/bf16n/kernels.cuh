// SPDX-License-Identifier: MIT
// BF16N device code: GEMV / GLU-GEMV (up to a few columns, optional expert ids), exact dequant, row gather, and the
// batched MUL_MAT_ID kernel (prefill). Extracted from the llama.cpp patch (ggml-cuda/bf16n.cu, device section).
// Changes against the llama.cpp kernels, none of which touch the decode or the summation order:
//   * the activations are read through load_x4 / to_f, templated on TX = float or __nv_bfloat16: BF16 inputs are widened
//     to FP32 exactly where the F32 kernels staged or loaded x, products and sums stay FP32;
//   * outputs store FP32 or BF16 (round to nearest even): ydst uses a runtime flag; MC specializes the output type.
// Weight rows: plane at row * K (chunk-major units: chunk c of unit row r at (c * R + r) * 512 within the unit).
#pragma once

#include "bf16n/compat.cuh"
#include "bf16n/format.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace bf16n {
namespace kern {

// ---------------------------------------------------------------- activation / output access

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __uint_as_float((uint32_t) __bfloat16_as_ushort(v) << 16); }

// elements 4*i .. 4*i+3 of x as FP32 (x 16-byte aligned for float, 8-byte aligned for bf16)
__device__ __forceinline__ float4 load_x4(const float * x, int64_t i) { return __ldg((const float4 *) x + i); }
__device__ __forceinline__ float4 load_x4(const __nv_bfloat16 * x, int64_t i) {
    const uint2 v = __ldg((const uint2 *) x + i);
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u), __uint_as_float(v.y << 16),
                       __uint_as_float(v.y & 0xFFFF0000u));
}
__device__ __forceinline__ float load_x1(const float * x, int64_t i) { return __ldg(x + i); }
__device__ __forceinline__ float load_x1(const __nv_bfloat16 * x, int64_t i) {
    return __uint_as_float((uint32_t) __ldg((const unsigned short *) x + i) << 16);
}

// elements i .. i+3 of activations staged in shared memory in their input type (BF16 inputs stay BF16: half the bytes)
__device__ __forceinline__ float4 xs_load4(const float * xs, int i) { return *(const float4 *) (xs + i); }
__device__ __forceinline__ float4 xs_load4(const __nv_bfloat16 * xs, int i) {
    const uint2 v = *(const uint2 *) (xs + i);
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u), __uint_as_float(v.y << 16),
                       __uint_as_float(v.y & 0xFFFF0000u));
}

// one output vector (column) in FP32 or BF16
struct ydst {
    char * p;
    bool   bf16;
    __device__ __forceinline__ void init(bool value) { bf16 = value; }
    __device__ __forceinline__ void point(char * value) { p = value; }
    __device__ __forceinline__ void set(int64_t i, float v) const {
        if (bf16) {
            ((__nv_bfloat16 *) p)[i] = __float2bfloat16_rn(v);
        } else {
            ((float *) p)[i] = v;
        }
    }
};

// A typed pointer carries the store representation without a live runtime discriminator.
// cols_t remains unchanged: this specializes device setup/store, not the launch ABI.
template <typename TY>
struct typed_ydst {
    static_assert(std::is_same<TY, float>::value || std::is_same<TY, __nv_bfloat16>::value,
                  "BF16N output must be FP32 or BF16");
    TY * p;
    __device__ __forceinline__ void init(bool) {}
    __device__ __forceinline__ void point(char * value) { p = reinterpret_cast<TY *>(value); }
    __device__ __forceinline__ void set(int64_t i, float v) const {
        if constexpr (std::is_same<TY, float>::value) {
            p[i] = v;
        } else {
            p[i] = __float2bfloat16_rn(v);
        }
    }
};

// ---------------------------------------------------------------- decode primitives

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

// column = one output vector: (token) for a dense product, (token, expert slot) with expert ids
// ids are read as the low 32 bits of each element (int32, or little-endian int64 holding values < 2^31)
struct cols_t {
    const char * x; char * dst;
    int64_t x_tok, x_slot, dst_tok, dst_slot; // byte strides
    const char * ids; int64_t ids_s0, ids_s1; int n_used, ne11;
    int64_t rows_per_matrix, row0, nrows;
    bool dst_bf16;
};

template <typename TX, typename YD>
__device__ __forceinline__ void col_setup(const cols_t & p, int col, const TX *& x, YD & dst, int64_t & rbase) {
    dst.init(p.dst_bf16);
    if (p.ids) {
        const int t = col / p.n_used, i = col % p.n_used;
        const int e = *(const int32_t *) (p.ids + t * p.ids_s1 + i * p.ids_s0);
        x = (const TX *) (p.x + t * p.x_tok + (i % p.ne11) * p.x_slot);
        dst.point(p.dst + t * p.dst_tok + i * p.dst_slot);
        rbase = p.row0 + (int64_t) e * p.rows_per_matrix;
    } else {
        x = (const TX *) (p.x + col * p.x_tok);
        dst.point(p.dst + col * p.dst_tok);
        rbase = p.row0;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

struct nib_ptrs { const uint8_t * sm; const uint8_t * nib; const uint4 * lut_row; const uint32_t * esc_off; const uint32_t * esc; };
struct huf_ptrs { const uint8_t * sm; const uint32_t * lut; const uint32_t * lane_off; const uint32_t * stream; };

// one row of the nibble format held in registers: loads are issued before any of them is used
template <int C>
struct nib_row {
    uint4 smv[C];
    uint2 nbv[C];
    uint4 L;
};

template <int C>
__device__ __forceinline__ void nib_row_load(nib_row<C> & d, const nib_ptrs & w, int64_t gr, int lane) {
    constexpr int K = C * 512;
    const uint8_t * sp = w.sm + gr * K;
    const uint8_t * np = w.nib + gr * (K / 2);
#pragma unroll
    for (int c = 0; c < C; c++) {
        d.smv[c] = __ldg((const uint4 *) (sp + c * 512) + lane);
        d.nbv[c] = __ldg((const uint2 *) (np + c * 256) + lane);
    }
    d.L = __ldg(w.lut_row + gr);
}

// dot(row gr, x) for the nibble format; x read through L1 (no shared staging); result in every lane.
// XR: this lane's 16 activations of a K = 512 row come widened in registers (xr), loaded once per warp instead of once
// per row: for BF16 input the per-row widening cost 7 % of the tiny K = 512 kernel (shared-expert down, M1 bench)
template <typename TX, int C, bool XR = false>
__device__ __forceinline__ float nib_row_dot(const nib_row<C> & d, const nib_ptrs & w, int64_t gr, const TX * __restrict__ x, int lane,
                                             const float4 * xr = nullptr) {
    const uint4 * smv = d.smv;
    const uint2 * nbv = d.nbv;
    const uint4 L = d.L;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int c = 0; c < C; c++) {
        const uint32_t smw[4] = {smv[c].x, smv[c].y, smv[c].z, smv[c].w};
        const uint32_t nw[2] = {nbv[c].x, nbv[c].y};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float4 xv = XR ? xr[c * 4 + j] : load_x4(x + c * 512 + 128 * j + 4 * lane, 0);
            const uint32_t sel = (j & 1) ? (nw[j >> 1] >> 16) : nw[j >> 1];
            const uint32_t E4 = prmt(L.x, L.y, sel) | prmt(L.z, L.w, sel ^ 0x8888u);
            const uint32_t w0 = (prmt(smw[j], 0, 0x9180u) & 0x807F807Fu) | (prmt(E4, 0, 0x4140u) << 7);
            const uint32_t w1 = (prmt(smw[j], 0, 0xB3A2u) & 0x807F807Fu) | (prmt(E4, 0, 0x4342u) << 7);
            a0 = fmaf(__uint_as_float(w0 << 16), xv.x, a0);
            a1 = fmaf(__uint_as_float(w0 & 0xFFFF0000u), xv.y, a1);
            a2 = fmaf(__uint_as_float(w1 << 16), xv.z, a2);
            a3 = fmaf(__uint_as_float(w1 & 0xFFFF0000u), xv.w, a3);
        }
    }
    float acc = (a0 + a1) + (a2 + a3);
    const uint32_t e_beg = w.esc_off[gr], e_end = w.esc_off[gr + 1];
    for (uint32_t i = e_beg + lane; i < e_end; i += 32) {
        const uint32_t rec = w.esc[i];
        const uint32_t smb = (rec >> 12) & 0xFF, ex = rec >> 20;
        const uint32_t bits = ((smb & 0x80) << 8) | (ex << 7) | (smb & 0x7F);
        acc = fmaf(__uint_as_float(bits << 16), load_x1(x, rec & 0xFFF), acc);
    }
    return warp_sum(acc);
}

struct huf_state {
    const uint32_t * sp;
    uint32_t hi, lo, nw0, nw1, acc, acc_chk;
    int nbits;
};

__device__ __forceinline__ void huf_init(huf_state & st, const huf_ptrs & w, int64_t gu, int lane) {
    st.sp = w.stream + w.lane_off[gu * 32 + lane];
    st.hi = __ldg(st.sp); st.lo = __ldg(st.sp + 1);
    st.nw0 = __ldg(st.sp + 2); st.nw1 = __ldg(st.sp + 3);
    st.sp += 4;
    st.nbits = 64; st.acc = 0; st.acc_chk = 0;
}

// decode the next row of the warp's unit and return dot(row, x); x and lut in shared memory
template <int C>
__device__ __forceinline__ float huf_row_dot(huf_state & st, const uint8_t * __restrict__ sm, const uint32_t * lut,
        int64_t gr, const float * xs, int lane) {
    constexpr int K = C * 512;
    const uint8_t * rp = sm + gr * K;
    uint4 smv[C];
#pragma unroll
    for (int c = 0; c < C; c++) {
        smv[c] = __ldg((const uint4 *) (rp + c * 512) + lane);
    }
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int c = 0; c < C; c++) {
        const uint32_t smw[4] = {smv[c].x, smv[c].y, smv[c].z, smv[c].w};
        uint32_t ex[16];
#pragma unroll
        for (int s = 0; s < 16; s++) {
            if (s % HUF_GROUP == 0) {
                st.nbits -= (int) ((st.acc - st.acc_chk) & 127u);
                st.acc_chk = st.acc;
                if (st.nbits <= 32) {
                    const uint32_t sh = 32 - st.nbits;
                    st.hi |= __funnelshift_lc(st.nw0, 0u, sh);
                    st.lo |= __funnelshift_lc(0u, st.nw0, sh);
                    st.nbits += 32;
                    st.nw0 = st.nw1;
                    st.nw1 = __ldg(st.sp);
                    st.sp++;
                }
            }
            const uint32_t ent = lut[__umulhi(st.hi, 1u << HUF_LMAX)];
            st.hi = __funnelshift_l(st.lo, st.hi, ent);
            st.lo = __funnelshift_l(0u, st.lo, ent);
            st.acc += ent;
            ex[s] = ent;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float4 xv = *(const float4 *) (xs + c * 512 + 128 * j + 4 * lane);
            const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), prmt(ex[4 * j], ex[4 * j + 1], 0x5410u));
            const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), prmt(ex[4 * j + 2], ex[4 * j + 3], 0x5410u));
            a0 = fmaf(__uint_as_float(w0 << 16), xv.x, a0);
            a1 = fmaf(__uint_as_float(w0 & 0xFFFF0000u), xv.y, a1);
            a2 = fmaf(__uint_as_float(w1 << 16), xv.z, a2);
            a3 = fmaf(__uint_as_float(w1 & 0xFFFF0000u), xv.w, a3);
        }
    }
    return warp_sum((a0 + a1) + (a2 + a3));
}

template <typename TX, int C>
__global__ void __launch_bounds__(256) k_nibble_mmv(const nib_ptrs w, cols_t p) {
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr bool XR = C == 1 && sizeof(TX) == 2;
    float4 xr[4];
    if (XR) {
#pragma unroll
        for (int j = 0; j < 4; j++) { xr[j] = load_x4(xg, (128 * j + 4 * lane) / 4); }
    }
    for (int64_t r = blockIdx.x * 8 + warp; r < p.nrows; r += gridDim.x * 8) {
        nib_row<C> d;
        nib_row_load<C>(d, w, rbase + r, lane);
        const float v = nib_row_dot<TX, C, XR>(d, w, rbase + r, xg, lane, xr);
        if (lane == 0) {
            out.set(r, v);
        }
    }
}

// out = up(x) * silu(gate(x)), bitwise the same as MUL_MAT, MUL_MAT, GLU(SWIGLU)
template <typename TX, int C, typename TY>
__global__ void __launch_bounds__(256) k_nibble_mmv_glu(const nib_ptrs g, const nib_ptrs u, cols_t p) {
    const TX * xg; typed_ydst<TY> out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr bool XR = C == 1 && sizeof(TX) == 2;
    float4 xr[4];
    if (XR) {
#pragma unroll
        for (int j = 0; j < 4; j++) { xr[j] = load_x4(xg, (128 * j + 4 * lane) / 4); }
    }
    for (int64_t r = blockIdx.x * 8 + warp; r < p.nrows; r += gridDim.x * 8) {
        nib_row<C> dg, du;
        nib_row_load<C>(dg, g, rbase + r, lane); // both rows in flight before either is decoded
        nib_row_load<C>(du, u, rbase + r, lane);
        const float gv = nib_row_dot<TX, C, XR>(dg, g, rbase + r, xg, lane, xr);
        const float uv = nib_row_dot<TX, C, XR>(du, u, rbase + r, xg, lane, xr);
        if (lane == 0) {
            out.set(r, uv * silu_single(gv));
        }
    }
}

template <typename TX, int C>
__device__ __forceinline__ void stage(float * xs, uint32_t * lut, const TX * xg, const uint32_t * lut_g) {
    for (int i = threadIdx.x; i < C * 512; i += blockDim.x) {
        xs[i] = to_f(xg[i]);
    }
    for (int i = threadIdx.x; i < (1 << HUF_LMAX); i += blockDim.x) {
        lut[i] = lut_g[i];
    }
}

template <typename TX, int C>
__global__ void __launch_bounds__(256) k_huffman_mmv(const huf_ptrs w, int R, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ float shm[];
    float * xs = shm;
    uint32_t * lut = (uint32_t *) (shm + K);
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    stage<TX, C>(xs, lut, xg, w.lut);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t units = p.nrows / R;
    for (int64_t u = blockIdx.x * 8 + warp; u < units; u += gridDim.x * 8) {
        const int64_t gu = rbase / R + u;
        huf_state st;
        huf_init(st, w, gu, lane);
        for (int rr = 0; rr < R; rr++) {
            const float v = huf_row_dot<C>(st, w.sm, lut, gu * R + rr, xs, lane);
            if (lane == 0) {
                out.set(u * R + rr, v);
            }
        }
    }
}

// Separate gate/up matrices; staging and row reduction order match the original llama kernel.
template <typename TX, int C, int R>
__global__ void __launch_bounds__(256) k_huffman_mmv_glu(const huf_ptrs g, const huf_ptrs w_up, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ float shm[];
    float * xs = shm;
    uint32_t * lut_g = (uint32_t *) (shm + K);
    uint32_t * lut_u = lut_g + (1 << HUF_LMAX);
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    stage<TX, C>(xs, lut_g, xg, g.lut);
    for (int i = threadIdx.x; i < (1 << HUF_LMAX); i += blockDim.x) {
        lut_u[i] = w_up.lut[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t units = p.nrows / R;
    for (int64_t u = blockIdx.x * 8 + warp; u < units; u += gridDim.x * 8) {
        const int64_t gu = rbase / R + u;
        float gv[R];
        {
            huf_state st;
            huf_init(st, g, gu, lane);
#pragma unroll
            for (int rr = 0; rr < R; rr++) {
                gv[rr] = huf_row_dot<C>(st, g.sm, lut_g, gu * R + rr, xs, lane);
            }
        }
        huf_state st;
        huf_init(st, w_up, gu, lane);
#pragma unroll
        for (int rr = 0; rr < R; rr++) {
            const float uv = huf_row_dot<C>(st, w_up.sm, lut_u, gu * R + rr, xs, lane);
            if (lane == 0) {
                out.set(u * R + rr, uv * silu_single(gv[rr]));
            }
        }
    }
}

// exact reconstruction to BF16 (row-major, contiguous) for the batched cuBLAS path
template <int C>
__global__ void __launch_bounds__(256) k_nibble_dequant(const uint8_t * __restrict__ sm, const uint8_t * __restrict__ nib,
        const uint4 * __restrict__ lut_row, const uint32_t * __restrict__ esc_off, const uint32_t * __restrict__ esc,
        int64_t row0, int64_t nrows, uint16_t * __restrict__ out) {
    constexpr int K = C * 512;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int64_t r = blockIdx.x * 8 + warp; r < nrows; r += gridDim.x * 8) {
        const int64_t gr = row0 + r;
        const uint4 L = __ldg(lut_row + gr);
        uint16_t * o = out + r * K;
        for (int c = 0; c < C; c++) {
            const uint4 smv = __ldg((const uint4 *) (sm + gr * K + c * 512) + lane);
            const uint2 nbv = __ldg((const uint2 *) (nib + gr * (K / 2) + c * 256) + lane);
            const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
            const uint32_t nw[2] = {nbv.x, nbv.y};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t sel = (j & 1) ? (nw[j >> 1] >> 16) : nw[j >> 1];
                const uint32_t E4 = prmt(L.x, L.y, sel) | prmt(L.z, L.w, sel ^ 0x8888u);
                const uint32_t w0 = (prmt(smw[j], 0, 0x9180u) & 0x807F807Fu) | (prmt(E4, 0, 0x4140u) << 7);
                const uint32_t w1 = (prmt(smw[j], 0, 0xB3A2u) & 0x807F807Fu) | (prmt(E4, 0, 0x4342u) << 7);
                *(uint2 *) (o + c * 512 + 128 * j + 4 * lane) = make_uint2(w0, w1);
            }
        }
        __syncwarp();
        for (uint32_t i = esc_off[gr] + lane; i < esc_off[gr + 1]; i += 32) {
            const uint32_t rec = esc[i];
            const uint32_t smb = (rec >> 12) & 0xFF, ex = rec >> 20;
            o[rec & 0xFFF] = (uint16_t) (((smb & 0x80) << 8) | (ex << 7) | (smb & 0x7F));
        }
    }
}

template <int C>
__global__ void __launch_bounds__(256) k_huffman_dequant(const uint8_t * __restrict__ sm, const uint32_t * __restrict__ lut_g,
        const uint32_t * __restrict__ lane_off, const uint32_t * __restrict__ stream, int R, int64_t row0, int64_t nrows,
        uint16_t * __restrict__ out) {
    constexpr int K = C * 512;
    __shared__ uint32_t lut[1 << HUF_LMAX];
    for (int i = threadIdx.x; i < (1 << HUF_LMAX); i += blockDim.x) {
        lut[i] = lut_g[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int64_t u = blockIdx.x * 8 + warp; u < nrows / R; u += gridDim.x * 8) {
        const int64_t gu = row0 / R + u;
        const uint32_t * sp = stream + lane_off[gu * 32 + lane];
        uint32_t hi = sp[0], lo = sp[1], nw0 = sp[2], nw1 = sp[3];
        sp += 4;
        int nbits = 64;
        uint32_t acc = 0, acc_chk = 0;
        for (int rr = 0; rr < R; rr++) {
            const int64_t gr = gu * R + rr;
            uint16_t * o = out + (gr - row0) * K;
            for (int c = 0; c < C; c++) {
                const uint4 smv = __ldg((const uint4 *) (sm + gr * K + c * 512) + lane);
                const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
                uint32_t ex[16];
#pragma unroll
                for (int s = 0; s < 16; s++) {
                    if (s % HUF_GROUP == 0) {
                        nbits -= (int) ((acc - acc_chk) & 127u);
                        acc_chk = acc;
                        if (nbits <= 32) {
                            const uint32_t sh = 32 - nbits;
                            hi |= __funnelshift_lc(nw0, 0u, sh);
                            lo |= __funnelshift_lc(0u, nw0, sh);
                            nbits += 32; nw0 = nw1; nw1 = *sp++;
                        }
                    }
                    const uint32_t ent = lut[__umulhi(hi, 1u << HUF_LMAX)];
                    hi = __funnelshift_l(lo, hi, ent);
                    lo = __funnelshift_l(0u, lo, ent);
                    acc += ent;
                    ex[s] = ent;
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), prmt(ex[4 * j], ex[4 * j + 1], 0x5410u));
                    const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), prmt(ex[4 * j + 2], ex[4 * j + 3], 0x5410u));
                    *(uint2 *) (o + c * 512 + 128 * j + 4 * lane) = make_uint2(w0, w1);
                }
            }
        }
    }
}

struct hp_ptrs { const uint8_t * sm; const uint32_t * lut; const uint32_t * ubase; const uint16_t * lrel; const uint32_t * stream; };

__device__ __forceinline__ uint4 ld_plane(const uint4 * p) { // streamed once: keep it out of L1 (the lane streams live there)
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint64_t shl64w(uint64_t b, uint32_t n) { // {hi:lo} <<= (n & 31)
    uint64_t d;
    asm("{.reg .b32 l, h; mov.b64 {l, h}, %1; shf.l.wrap.b32 h, l, h, %2; shf.l.wrap.b32 l, 0, l, %2; mov.b64 %0, {l, h};}" : "=l"(d) : "l"(b), "r"(n));
    return d;
}

// per-warp decoder state for one unit (bit buffer MSB-first; v = valid bits, exact in its low 7 bits)
#define HP_UNIT_BEGIN(W, GU) HP_UNIT_BEGIN_AT(W, __ldg((W).ubase + (GU)) + __ldg((W).lrel + (GU) * 32 + lane))

// HP_UNIT_BEGIN at a lane stream position already known (IDX = ubase[unit] + lrel[unit * 32 + lane])
#define HP_UNIT_BEGIN_AT(W, IDX)                                                                     \
    const uint32_t * lst = (W).stream;                                                               \
    uint32_t idx = (IDX);                                                                            \
    uint64_t buf = (uint64_t) __ldg(lst + idx) << 32;                                                \
    uint32_t wq[HP_PF];                                                                              \
    _Pragma("unroll") for (int i_ = 0; i_ < HP_PF; i_++) { wq[i_] = __ldg(lst + idx + 1 + i_); }    \
    idx += 1 + HP_PF;                                                                                \
    uint32_t v = 32, pa = 0, pb = 0;

// re-point the decoder state declared by HP_UNIT_BEGIN at another unit
#define HP_UNIT_RESET(W, GU)                                                                         \
    idx = __ldg((W).ubase + (GU)) + __ldg((W).lrel + (GU) * 32 + lane);                               \
    buf = (uint64_t) __ldg(lst + idx) << 32;                                                         \
    _Pragma("unroll") for (int i_ = 0; i_ < HP_PF; i_++) { wq[i_] = __ldg(lst + idx + 1 + i_); }    \
    idx += 1 + HP_PF;                                                                                \
    v = 32; pa = 0; pb = 0;

// decode one row (C chunks) and leave dot(row, x) summed over the warp in ACC
#define HP_ROW(SM, GR, ACC)                                                                          \
    {                                                                                                \
        const uint8_t * rp_ = (SM) + (GR) * K;                                                       \
        uint4 smv_[C];                                                                               \
        _Pragma("unroll") for (int c_ = 0; c_ < C; c_++) { smv_[c_] = ld_plane((const uint4 *) (rp_ + c_ * 512) + lane); } \
        float a0_ = 0.f, a1_ = 0.f, a2_ = 0.f, a3_ = 0.f;                                            \
        _Pragma("unroll") for (int c_ = 0; c_ < C; c_++) {                                           \
            const uint32_t smw_[4] = {smv_[c_].x, smv_[c_].y, smv_[c_].z, smv_[c_].w};               \
            uint32_t ex_[8];                                                                         \
            _Pragma("unroll") for (int q_ = 0; q_ < 8; q_++) {                                       \
                if (q_ % 2 == 0) {                                                                   \
                    v = v - pa - pb;                                                                 \
                    const uint32_t m_ = __funnelshift_r(0u, 1u, v);                                  \
                    if (!(v & 32u)) {                                                                \
                        buf += (uint64_t) wq[0] * m_;                                                \
                        _Pragma("unroll") for (int i_ = 0; i_ + 1 < HP_PF; i_++) { wq[i_] = wq[i_ + 1]; } \
                        wq[HP_PF - 1] = __ldg(lst + idx);                                            \
                        idx++;                                                                       \
                    }                                                                                \
                    v |= 32u;                                                                        \
                }                                                                                    \
                const uint32_t ent_ = lut[(uint32_t) (buf >> 32) >> (32 - HP_LP)];                   \
                buf = shl64w(buf, ent_);                                                             \
                ex_[q_] = ent_;                                                                      \
                if (q_ % 2 == 0) { pa = ent_; } else { pb = ent_; }                                  \
            }                                                                                        \
            _Pragma("unroll") for (int j_ = 0; j_ < 4; j_++) {                                       \
                const float4 xv_ = XREG_C1 ? xr[j_] : xs_load4(xs, c_ * 512 + 128 * j_ + 4 * lane);          \
                const uint32_t w0_ = sel_bits(prmt(smw_[j_], 0, 0x9180u), ex_[2 * j_]);              \
                const uint32_t w1_ = sel_bits(prmt(smw_[j_], 0, 0xB3A2u), ex_[2 * j_ + 1]);          \
                a0_ = fmaf(__uint_as_float(w0_ << 16), xv_.x, a0_);                                  \
                a1_ = fmaf(__uint_as_float(w0_ & 0xFFFF0000u), xv_.y, a1_);                          \
                a2_ = fmaf(__uint_as_float(w1_ << 16), xv_.z, a2_);                                  \
                a3_ = fmaf(__uint_as_float(w1_ & 0xFFFF0000u), xv_.w, a3_);                          \
            }                                                                                        \
        }                                                                                            \
        ACC = warp_sum((a0_ + a1_) + (a2_ + a3_));                                                   \
    }

#define HP_STAGE(XG, LUT_SRC, LUT_DST)                                                               \
    for (int i_ = threadIdx.x; i_ < K; i_ += blockDim.x) { xs[i_] = (XG)[i_]; } /* in the input type */ \
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { (LUT_DST)[i_] = (LUT_SRC)[i_]; }

#define HP_XREG                                                                                      \
    constexpr bool XREG_C1 = C == 1; /* K=512: this lane's 16 activations stay in registers */      \
    float4 xr[4];                                                                                    \
    if (XREG_C1) { _Pragma("unroll") for (int j_ = 0; j_ < 4; j_++) { xr[j_] = xs_load4(xs, 128 * j_ + 4 * lane); } }

template <typename TX, int C>
__global__ void __launch_bounds__(HP_NT) k_hp_mmv(const hp_ptrs w, int R, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ __align__(16) unsigned char hp_shm_raw[];
    TX * xs = (TX *) hp_shm_raw; // activations in the input type (T-076: BF16 halves the shared bytes and loads)
    uint32_t * lut = (uint32_t *) (hp_shm_raw + K * sizeof(TX));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int t = threadIdx.x;
    // warm start: issue every independent load of the prologue (LUT, ids -> x, the first unit's stream words) before
    // the first use, so their latencies overlap instead of stalling the in-order issue one after another
    const bool one_pass = blockDim.x * 4 >= (1 << HP_LP) && blockDim.x * 4 >= K;
    uint4 l4 = make_uint4(0, 0, 0, 0);
    if (one_pass && t < (1 << HP_LP) / 4) {
        l4 = __ldg((const uint4 *) w.lut + t);
    }
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    using XV = std::conditional_t<sizeof(TX) == 2, uint2, float4>; // 4 activations in the input type
    XV x4{};
    if (one_pass && t < K / 4) {
        x4 = __ldg((const XV *) xg + t);
    }
    const int64_t units = p.nrows / R;
    const int64_t u0 = blockIdx.x * wpb + warp;
    HP_UNIT_BEGIN(w, rbase / R + (u0 < units ? u0 : 0))
    if (one_pass) {
        if (t < (1 << HP_LP) / 4) {
            ((uint4 *) lut)[t] = l4;
        }
        if (t < K / 4) {
            ((XV *) xs)[t] = x4;
        }
    } else {
        HP_STAGE(xg, w.lut, lut)
    }
    __syncthreads();
    HP_XREG
    for (int64_t u = u0; u < units; u += gridDim.x * wpb) {
        const int64_t gu = rbase / R + u;
        if (u != u0) {
            HP_UNIT_RESET(w, gu)
        }
        for (int rr = 0; rr < R; rr++) {
            float acc;
            HP_ROW(w.sm, gu * R + rr, acc)
            if (lane == 0) {
                out.set(u * R + rr, acc);
            }
        }
    }
}

// decode the 8 exponent pairs of one 512-column chunk of one row into EX (this lane's 16 weights)
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

// Y = W X for NC columns of one dense matrix (speculative verification, small batches): every row is decoded once and
// multiplied by all columns. Each column sums in the order of k_hp_mmv, so its result is bitwise the single-column one.
// Output typing is compile-time: the FP32 path must not retain BF16 stores/aliasing or their register pressure.
template <typename TX, int C, int NC, bool Y_BF16>
__global__ void __launch_bounds__(HP_NT) k_hp_mmv_mc(const hp_ptrs w, int R, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ __align__(16) unsigned char mc_shm_raw[];
    // NC x K activations in the input type: staged as FP32, 4 BF16 columns took 32 KiB (K = 2048) to 64 KiB (K = 4096)
    // of the L1 that holds the lane streams and doubled the shared-memory wavefronts; 4 columns ran at 1.6x the time of
    // one, limited by the LSU pipe (ncu, T-076). BF16 values are the same numbers, so every sum is bitwise unchanged.
    TX * xs = (TX *) mc_shm_raw;
    uint32_t * lut = (uint32_t *) (mc_shm_raw + (size_t) NC * K * sizeof(TX));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int64_t units = p.nrows / R;
    const int64_t gbase = p.row0 / R;
    const int64_t u0 = blockIdx.x * wpb + warp;
    HP_UNIT_BEGIN(w, gbase + (u0 < units ? u0 : 0))
    for (int i = threadIdx.x; i < NC * K / 4; i += blockDim.x) { // 4 elements: 16 bytes (FP32) or 8 bytes (BF16)
        const int n = i / (K / 4);
        const TX * xg = (const TX *) (p.x + n * p.x_tok);
        if constexpr (sizeof(TX) == 2) {
            ((uint2 *) xs)[i] = __ldg((const uint2 *) xg + (i - n * (K / 4)));
        } else {
            ((float4 *) xs)[i] = __ldg((const float4 *) xg + (i - n * (K / 4)));
        }
    }
    for (int i = threadIdx.x; i < (1 << HP_LP) / 4; i += blockDim.x) { ((uint4 *) lut)[i] = __ldg((const uint4 *) w.lut + i); }
    __syncthreads();
    for (int64_t u = u0; u < units; u += gridDim.x * wpb) {
        const int64_t gu = gbase + u;
        if (u != u0) {
            HP_UNIT_RESET(w, gu)
        }
        for (int rr = 0; rr < R; rr++) {
            const uint8_t * rp = w.sm + (gu * R + rr) * K;
            float a[NC][4];
#pragma unroll
            for (int n = 0; n < NC; n++) { a[n][0] = a[n][1] = a[n][2] = a[n][3] = 0.f; }
            constexpr int PF = C <= 4 ? C : 2; // plane chunks in flight (all of a row up to K = 2048; more spills)
            uint4 ring[PF];
#pragma unroll
            for (int c = 0; c < PF; c++) { ring[c] = ld_plane((const uint4 *) (rp + c * 512) + lane); }
#pragma unroll
            for (int c = 0; c < C; c++) {
                const uint4 cur = ring[c % PF];
                if (c + PF < C) { ring[c % PF] = ld_plane((const uint4 *) (rp + (c + PF) * 512) + lane); }
                const uint32_t smw[4] = {cur.x, cur.y, cur.z, cur.w};
                uint32_t ex[8];
                HP_PAIRS(ex)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                    const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                    const float f0 = __uint_as_float(w0 << 16), f1 = __uint_as_float(w0 & 0xFFFF0000u);
                    const float f2 = __uint_as_float(w1 << 16), f3 = __uint_as_float(w1 & 0xFFFF0000u);
#pragma unroll
                    for (int n = 0; n < NC; n++) {
                        const float4 xv = xs_load4(xs, n * K + c * 512 + 128 * j + 4 * lane);
                        a[n][0] = fmaf(f0, xv.x, a[n][0]);
                        a[n][1] = fmaf(f1, xv.y, a[n][1]);
                        a[n][2] = fmaf(f2, xv.z, a[n][2]);
                        a[n][3] = fmaf(f3, xv.w, a[n][3]);
                    }
                }
            }
#pragma unroll
            for (int n = 0; n < NC; n++) {
                const float sum = warp_sum((a[n][0] + a[n][1]) + (a[n][2] + a[n][3]));
                if (lane == 0) {
                    if constexpr (Y_BF16) {
                        ((__nv_bfloat16 *) (p.dst + n * p.dst_tok))[u * R + rr] = __float2bfloat16_rn(sum);
                    } else {
                        ((float *) (p.dst + n * p.dst_tok))[u * R + rr] = sum;
                    }
                }
            }
        }
    }
}

// Row-major only, as in llama: chunk-major weights use separate MMVs followed by SWIGLU.
template <typename TX, int C, int R>
__global__ void __launch_bounds__(HP_NT) k_hp_mmv_glu(const hp_ptrs g, const hp_ptrs wu, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ __align__(16) unsigned char hp_shm_raw[];
    TX * xs = (TX *) hp_shm_raw; // activations in the input type
    uint32_t * lut_g = (uint32_t *) (hp_shm_raw + K * sizeof(TX));
    uint32_t * lut_u = lut_g + (1 << HP_LP);
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    HP_STAGE(xg, g.lut, lut_g)
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { lut_u[i_] = wu.lut[i_]; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    HP_XREG
    const int64_t units = p.nrows / R;
    const int wpb = blockDim.x >> 5;
    for (int64_t u = blockIdx.x * wpb + warp; u < units; u += gridDim.x * wpb) {
        const int64_t gu = rbase / R + u;
        float gv[R];
        {
            const uint32_t * lut = lut_g;
            HP_UNIT_BEGIN(g, gu)
            _Pragma("unroll") for (int rr = 0; rr < R; rr++) {
                HP_ROW(g.sm, gu * R + rr, gv[rr])
            }
        }
        {
            const uint32_t * lut = lut_u;
            HP_UNIT_BEGIN(wu, gu)
            _Pragma("unroll") for (int rr = 0; rr < R; rr++) {
                float uv;
                HP_ROW(wu.sm, gu * R + rr, uv)
                if (lane == 0) {
                    out.set(u * R + rr, uv * silu_single(gv[rr]));
                }
            }
        }
    }
}

template <int C>
__global__ void __launch_bounds__(256) k_hp_dequant(const hp_ptrs w, int R, int64_t row0, int64_t nrows, uint16_t * __restrict__ out) {
    constexpr int K = C * 512;
    __shared__ uint32_t lut[1 << HP_LP];
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { lut[i_] = w.lut[i_]; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int64_t u = blockIdx.x * 8 + warp; u < nrows / R; u += gridDim.x * 8) {
        const int64_t gu = row0 / R + u;
        HP_UNIT_BEGIN(w, gu)
        for (int rr = 0; rr < R; rr++) {
            const int64_t gr = gu * R + rr;
            uint16_t * o = out + (gr - row0) * K;
            for (int c = 0; c < C; c++) {
                const uint4 smv = ld_plane((const uint4 *) (w.sm + gr * K + c * 512) + lane);
                const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
                uint32_t ex[8];
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
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                    const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                    *(uint2 *) (o + c * 512 + 128 * j + 4 * lane) = make_uint2(w0, w1);
                }
            }
        }
    }
}


// 4 consecutive weights (two bf16x2 words) of a gathered row, as FP32 or as the exact BF16 bits
template <bool BF>
__device__ __forceinline__ void store_w4(char * o, int64_t i, uint32_t w0, uint32_t w1) {
    if (BF) {
        *(uint2 *) (o + 2 * i) = make_uint2(w0, w1);
    } else {
        *(float4 *) (o + 4 * i) = make_float4(
            __uint_as_float(w0 << 16), __uint_as_float(w0 & 0xFFFF0000u), __uint_as_float(w1 << 16), __uint_as_float(w1 & 0xFFFF0000u));
    }
}

// GET_ROWS (token embedding): dst[i, :] = W[ids[i], :] as FP32 or BF16 (BF); one warp per requested row
template <int C, bool BF>
__global__ void __launch_bounds__(256) k_nibble_get_rows(const nib_ptrs w, const char * __restrict__ ids, int64_t ids_s0,
        int64_t n, int64_t row0, char * __restrict__ dst, int64_t dst_s1) {
    constexpr int K = C * 512;
    const int lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x * (int64_t) (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= n) {
        return;
    }
    const int64_t gr = row0 + *(const int32_t *) (ids + i * ids_s0);
    char * o = dst + i * dst_s1;
    const uint4 L = __ldg(w.lut_row + gr);
    for (int c = 0; c < C; c++) {
        const uint4 smv = __ldg((const uint4 *) (w.sm + gr * K + c * 512) + lane);
        const uint2 nbv = __ldg((const uint2 *) (w.nib + gr * (K / 2) + c * 256) + lane);
        const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
        const uint32_t nw[2] = {nbv.x, nbv.y};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t sel = (j & 1) ? (nw[j >> 1] >> 16) : nw[j >> 1];
            const uint32_t E4 = prmt(L.x, L.y, sel) | prmt(L.z, L.w, sel ^ 0x8888u);
            const uint32_t w0 = (prmt(smw[j], 0, 0x9180u) & 0x807F807Fu) | (prmt(E4, 0, 0x4140u) << 7);
            const uint32_t w1 = (prmt(smw[j], 0, 0xB3A2u) & 0x807F807Fu) | (prmt(E4, 0, 0x4342u) << 7);
            store_w4<BF>(o, c * 512 + 128 * j + 4 * lane, w0, w1);
        }
    }
    __syncwarp();
    for (uint32_t e = w.esc_off[gr] + lane; e < w.esc_off[gr + 1]; e += 32) {
        const uint32_t rec = w.esc[e];
        const uint32_t smb = (rec >> 12) & 0xFF, ex = rec >> 20;
        const uint32_t bits = ((smb & 0x80) << 8) | (ex << 7) | (smb & 0x7F);
        if (BF) {
            ((uint16_t *) o)[rec & 0xFFF] = (uint16_t) bits;
        } else {
            ((float *) o)[rec & 0xFFF] = __uint_as_float(bits << 16);
        }
    }
}

// the warp decodes its row's unit from the start and writes only the requested row
template <int C, bool BF>
__global__ void __launch_bounds__(256) k_hp_get_rows(const hp_ptrs w, int R, const char * __restrict__ ids, int64_t ids_s0,
        int64_t n, int64_t row0, char * __restrict__ dst, int64_t dst_s1) {
    constexpr int K = C * 512;
    __shared__ uint32_t lut[1 << HP_LP];
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { lut[i_] = w.lut[i_]; }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x * (int64_t) (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= n) {
        return;
    }
    const int64_t gr = row0 + *(const int32_t *) (ids + i * ids_s0);
    const int64_t gu = gr / R;
    char * o = dst + i * dst_s1;
    HP_UNIT_BEGIN(w, gu)
    for (int64_t r = gu * R; r <= gr; r++) {
        for (int c = 0; c < C; c++) {
            const uint4 smv = ld_plane((const uint4 *) (w.sm + r * K + c * 512) + lane);
            const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
            uint32_t ex[8];
#pragma unroll
            for (int q = 0; q < 8; q++) {
                if (q % 2 == 0) {
                    v = v - pa - pb;
                    const uint32_t m = __funnelshift_r(0u, 1u, v);
                    if (!(v & 32u)) {
                        buf += (uint64_t) wq[0] * m;
#pragma unroll
                        for (int k = 0; k + 1 < HP_PF; k++) { wq[k] = wq[k + 1]; }
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
            if (r == gr) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                    const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                    store_w4<BF>(o, c * 512 + 128 * j + 4 * lane, w0, w1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- chunk-major units (expert matrices with K > 512)

// y = W x with chunk-major units: x chunk c is read once per unit and reused for its R rows
template <typename TX, int C, int R>
__global__ void __launch_bounds__(HP_NT) k_hp_mmv_cm(const hp_ptrs w, cols_t p) {
    constexpr int K = C * 512;
    extern __shared__ __align__(16) unsigned char hp_shm_raw[];
    TX * xs = (TX *) hp_shm_raw; // activations in the input type (T-076: BF16 halves the shared bytes and loads)
    uint32_t * lut = (uint32_t *) (hp_shm_raw + K * sizeof(TX));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int t = threadIdx.x;
    // warm start (as k_hp_mmv): every independent prologue load is in flight before the first use
    const bool one_pass = blockDim.x * 4 >= (1 << HP_LP) && blockDim.x * 4 >= K;
    uint4 l4 = make_uint4(0, 0, 0, 0);
    if (one_pass && t < (1 << HP_LP) / 4) {
        l4 = __ldg((const uint4 *) w.lut + t);
    }
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    using XV = std::conditional_t<sizeof(TX) == 2, uint2, float4>; // 4 activations in the input type
    XV x4{};
    if (one_pass && t < K / 4) {
        x4 = __ldg((const XV *) xg + t);
    }
    const int64_t units = p.nrows / R;
    const int64_t u0 = blockIdx.x * wpb + warp;
    HP_UNIT_BEGIN(w, rbase / R + (u0 < units ? u0 : 0))
    if (one_pass) {
        if (t < (1 << HP_LP) / 4) {
            ((uint4 *) lut)[t] = l4;
        }
        if (t < K / 4) {
            ((XV *) xs)[t] = x4;
        }
    } else {
        HP_STAGE(xg, w.lut, lut)
    }
    __syncthreads();
    for (int64_t u = u0; u < units; u += gridDim.x * wpb) {
        const int64_t gu = rbase / R + u;
        const uint8_t * up = w.sm + gu * R * K; // chunk-major plane: chunk c of row r at (c * R + r) * 512
        if (u != u0) {
            HP_UNIT_RESET(w, gu)
        }
        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; rr++) { acc[rr] = 0.f; }
        // loop body = 4 rows (the code of 8 unrolled rows does not fit the instruction cache next to 31 other warps):
        // acc[0..3] always holds the current 4 rows, the array is rotated by 4 after each group
#pragma unroll 1
        for (int c = 0; c < C; c++) {
            const uint8_t * pc = up + c * R * 512;
            uint4 smv[2]; // plane rows two ahead
#pragma unroll
            for (int i = 0; i < 2; i++) { smv[i] = ld_plane((const uint4 *) (pc + i * 512) + lane); }
#pragma unroll 1
            for (int r0 = 0; r0 < R; r0 += 4) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint4 cur = smv[i % 2];
                    if (r0 + i + 2 < R) { smv[i % 2] = ld_plane((const uint4 *) (pc + (r0 + i + 2) * 512) + lane); }
                    const uint32_t smw[4] = {cur.x, cur.y, cur.z, cur.w};
                    uint32_t ex[8];
                    HP_PAIRS(ex)
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float4 xv = xs_load4(xs, c * 512 + 128 * j + 4 * lane);
                        const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                        const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                        a0 = fmaf(__uint_as_float(w0 << 16), xv.x, a0);
                        a1 = fmaf(__uint_as_float(w0 & 0xFFFF0000u), xv.y, a1);
                        a2 = fmaf(__uint_as_float(w1 << 16), xv.z, a2);
                        a3 = fmaf(__uint_as_float(w1 & 0xFFFF0000u), xv.w, a3);
                    }
                    acc[i] += (a0 + a1) + (a2 + a3);
                }
                float t4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) { t4[i] = acc[i]; }
#pragma unroll
                for (int i = 0; i + 4 < R; i++) { acc[i] = acc[i + 4]; }
#pragma unroll
                for (int i = 0; i < 4; i++) { acc[R - 4 + i] = t4[i]; }
            }
        }
#pragma unroll
        for (int rr = 0; rr < R; rr++) {
            const float sum = warp_sum(acc[rr]);
            if (lane == 0) {
                out.set(u * R + rr, sum);
            }
        }
    }
}

template <int C, int R>
__global__ void __launch_bounds__(256) k_hp_dequant_cm(const hp_ptrs w, int64_t row0, int64_t nrows, uint16_t * __restrict__ out) {
    constexpr int K = C * 512;
    __shared__ uint32_t lut[1 << HP_LP];
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { lut[i_] = w.lut[i_]; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int64_t u = blockIdx.x * 8 + warp; u < nrows / R; u += gridDim.x * 8) {
        const int64_t gu = row0 / R + u;
        HP_UNIT_BEGIN(w, gu)
        for (int c = 0; c < C; c++) {
            for (int rr = 0; rr < R; rr++) {
                const int64_t gr = gu * R + rr;
                const uint4 smv = ld_plane((const uint4 *) (w.sm + gu * R * K + (c * R + rr) * 512) + lane); // chunk-major plane
                const uint32_t smw[4] = {smv.x, smv.y, smv.z, smv.w};
                uint32_t ex[8];
                HP_PAIRS(ex)
                uint16_t * o = out + (gr - row0) * K + c * 512;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t w0 = sel_bits(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                    const uint32_t w1 = sel_bits(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                    *(uint2 *) (o + 128 * j + 4 * lane) = make_uint2(w0, w1);
                }
            }
        }
    }
}

// decoder state of one lane stream and one exponent-pair lookup (HP_UNIT_BEGIN / HP_PAIRS for an array of streams)
struct hp_dec { uint64_t buf; uint32_t wq[HP_PF]; uint32_t idx, v, pa, pb; };

__device__ __forceinline__ void hp_dec_begin(hp_dec & d, const hp_ptrs & w, int64_t gu, int lane) {
    d.idx = __ldg(w.ubase + gu) + __ldg(w.lrel + gu * 32 + lane);
    d.buf = (uint64_t) __ldg(w.stream + d.idx) << 32;
#pragma unroll
    for (int i = 0; i < HP_PF; i++) { d.wq[i] = __ldg(w.stream + d.idx + 1 + i); }
    d.idx += 1 + HP_PF;
    d.v = 32; d.pa = 0; d.pb = 0;
}

__device__ __forceinline__ uint32_t hp_pair(hp_dec & d, int q, const uint32_t * __restrict__ lst, const uint32_t * lut) {
    if (q % 2 == 0) {
        d.v = d.v - d.pa - d.pb;
        const uint32_t m = __funnelshift_r(0u, 1u, d.v);
        if (!(d.v & 32u)) {
            d.buf += (uint64_t) d.wq[0] * m;
#pragma unroll
            for (int i = 0; i + 1 < HP_PF; i++) { d.wq[i] = d.wq[i + 1]; }
            d.wq[HP_PF - 1] = __ldg(lst + d.idx);
            d.idx++;
        }
        d.v |= 32u;
    }
    const uint32_t ent = lut[(uint32_t) (d.buf >> 32) >> (32 - HP_LP)];
    d.buf = shl64w(d.buf, ent);
    if (q % 2 == 0) { d.pa = ent; } else { d.pb = ent; }
    return ent;
}

// RB > 0: the batched kernel reads its lane streams through a per-lane ring of RB 16-byte blocks in shared memory, filled by
// cp.async. Lanes refill at different pair positions, and the scoreboard is per warp register, so with a register queue
// every refill waits for the newest load into that register by any lane; an asynchronous copy has no register to wait
// on. Word i of a stream sits in block i / 4, at ring block (i / 4 + key) % RB (key spreads the lanes over the
// banks); consuming the last word of a block starts the copy of the block RB - 1 ahead into the freed slot. buf holds
// MSB-first bits, v the valid count (exact in its low 7 bits). The stream buffer has 32 words of padding for these reads.

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t) __cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <int RB> struct hp_rdec { uint64_t buf; uint32_t idx, v, pa, pb, key; uint32_t * ring; };

template <int RB>
__device__ __forceinline__ uint32_t * hp_ring_block(const hp_rdec<RB> & d, uint32_t b) { return d.ring + 4 * ((b + d.key) & (RB - 1)); }

__device__ __forceinline__ uint32_t hp_lane_start(const hp_ptrs & w, int64_t gu, int lane) {
    return __ldg(w.ubase + gu) + __ldg(w.lrel + gu * 32 + lane);
}

// start a lane stream at word idx: fill the ring; hp_ring_first takes the first word once the copies are complete
template <int RB>
__device__ __forceinline__ void hp_ring_begin(hp_rdec<RB> & d, const uint32_t * __restrict__ lst, uint32_t idx) {
    d.idx = idx; d.v = 32; d.pa = 0; d.pb = 0;
#pragma unroll
    for (int k = 0; k < RB; k++) { cp_async16(hp_ring_block(d, (idx >> 2) + k), lst + 4 * ((idx >> 2) + k)); }
}

// word idx was consumed: when it was the last word of its block, the freed slot takes the block RB - 1 ahead
template <int RB>
__device__ __forceinline__ void hp_ring_advance(hp_rdec<RB> & d, const uint32_t * __restrict__ lst) {
    d.idx++;
    if ((d.idx & 3u) == 0) {
        const uint32_t b = (d.idx >> 2) + RB - 1;
        cp_async16(hp_ring_block(d, b), lst + 4 * b);
    }
}

template <int RB>
__device__ __forceinline__ void hp_ring_first(hp_rdec<RB> & d, const uint32_t * __restrict__ lst) {
    d.buf = (uint64_t) hp_ring_block(d, d.idx >> 2)[d.idx & 3] << 32;
    hp_ring_advance(d, lst);
}

// one exponent-pair lookup; q = position within the 8 lookups of a row, a refill point when even
template <int RB>
__device__ __forceinline__ uint32_t hp_ring_pair(hp_rdec<RB> & d, int q, const uint32_t * __restrict__ lst, const uint32_t * lut) {
    if (q % 2 == 0) {
        d.v = d.v - d.pa - d.pb;
        const uint32_t m = __funnelshift_r(0u, 1u, d.v);
        if (!(d.v & 32u)) {
            d.buf += (uint64_t) hp_ring_block(d, d.idx >> 2)[d.idx & 3] * m;
            hp_ring_advance(d, lst);
        }
        d.v |= 32u;
    }
    const uint32_t ent = lut[(uint32_t) (d.buf >> 32) >> (32 - HP_LP)];
    d.buf = shl64w(d.buf, ent);
    if (q % 2 == 0) { d.pa = ent; } else { d.pb = ent; }
    return ent;
}

// ---------------------------------------------------------------- batched MUL_MAT_ID (prefill)
//
// One launch per MUL_MAT_ID; routing comes from launch_mm_ids_helper on the GPU (no host sync). A work item
// is a group of units of one expert and a tile of the tokens routed to it. Per step (a 512-column chunk of a slice of the
// units' rows; chunk-major streams make this the decode order) every warp decodes its rows into a 64-row shared slab,
// then the warps multiply the slab with the tokens' BF16 activations on the tensor cores (FP32 accumulation). Every
// weight is decoded once per token tile; the slab stays small enough to leave L1 to the lane streams.
// Within each group of 32 columns the k order is permuted identically for the slab and the activations, so a thread loads
// 16 bytes of activations for two mma steps and the slab is read with ldmatrix.

constexpr int HPM_LD = 512 + 8; // slab row stride (bf16): 16-byte row skew, conflict-free ldmatrix

struct hpm_args {
    const __nv_bfloat16 * x; int64_t x_row; // activations as BF16 rows of K
    const int32_t * ids_src1; const int32_t * ids_dst; const int32_t * bounds; // from launch_mm_ids_helper (mmid_helper.cu)
    char * dst; int64_t dst_col;           // dst column stride (elements)
    bool dst_bf16;                         // dst is BF16 (round to nearest even) instead of FP32
    int64_t rows;                          // rows per expert
    int n_experts;
};

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void * p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

// Persistent: one block per SM walks a work list of (expert, tile of TP routed tokens, group of NU units) that it builds
// from the routing bounds in shared memory. A block's NU units are decoded by its NW warps, IL per warp with interleaved
// lane streams (independent dependency chains). Per step (512-column chunk x row slice) every warp decodes RS = 64 / NU
// rows of each of its units into the 64-row slab; then all warps multiply the slab with the tokens' BF16 activations on
// the tensor cores: 2 row groups of 32 x NW / 2 token groups of WT*8 tokens; token groups without tokens split K
// instead. The accumulators of all H = R / RS slices stay in registers. The LUT is loaded once per block, and the next
// item's lane streams and plane rows are started during the last step of the current one. RB = 0: lane streams through
// the register queue of hp_dec. RB > 0: through the cp.async rings (the next item's lane starts are read at the start of
// the current one); a group is committed at each of the 4 refill points of a row, and a refill point waits until at most
// 4 (RB - 1) groups are pending, which covers every ring block it can read (started at least 4 (RB - 1) + 1 points
// earlier). The rings pay off where a warp has one stream and long items (gate, down with 1-2 units per warp); with 4
// interleaved streams per warp the register queue has enough independent chains and keeps the L1 larger.
//
// GLU: gate (w) and up (w2) of the same rows in one item, dst = silu(gate) * up (SWIGLU; the same values as MUL_MAT_ID,
// MUL_MAT_ID, GLU). In each 32-row group of the slab the first 16-row mma tile holds gate rows and the second the same up
// rows, so a thread has both values of its outputs.
template <int C, int R, int NW, int IL, int WT, int RB, bool GLU>
__global__ void __launch_bounds__(NW * 32, 1) k_hp_mm_ids(const hp_ptrs w, const hp_ptrs w2, hpm_args a) {
    constexpr int K = C * 512;
    constexpr int NU = NW * IL;            // unit slots per item (both matrices for GLU)
    constexpr int RS = 64 / NU;            // rows per unit per step
    constexpr int H = R / RS;              // row slices (steps per chunk)
    constexpr int SB = 16 / RS;            // unit slots per 16-row mma tile
    constexpr int NUM = GLU ? NU / 2 : NU; // units of each matrix per item
    constexpr int WG = RB ? 4 * (RB - 1) : 0; // cp.async groups a refill point lets pend
    static_assert(RB == 0 || RB == 2 || RB == 4, "ring of 2 or 4 blocks");
    static_assert(NU * RS == 64 && R % RS == 0, "bad block shape");
    static_assert(!GLU || (IL == 1 && 16 % RS == 0), "GLU needs one unit per warp and whole units per mma tile");
    static_assert(NW * H * 2 * WT * 4 * 32 * sizeof(float) <= 64 * HPM_LD * sizeof(uint16_t), "K-split sums exceed the slab");
    constexpr int TG = NW / 2;            // token groups (two row groups of 32 slab rows each)
    constexpr int TP = TG * WT * 8;       // tokens per item
    extern __shared__ uint32_t hpm_shm[];
    uint32_t * lut = hpm_shm;                                               // gate (or the only matrix), then up for GLU
    uint16_t * slab = (uint16_t *) (hpm_shm + (GLU ? 2 : 1) * (1 << HP_LP));
    uint32_t * rings = (uint32_t *) (slab + 64 * HPM_LD);                   // [NU][32 lanes][4 RB words]
    const int E = a.n_experts;
    int * bnd = (int *) (rings + NU * 32 * 4 * RB);                     // routing bounds, E + 1
    uint16_t * toff = (uint16_t *) (bnd + E + 1); // first token tile of each expert, E + 1 (< 2^16, checked by the launcher)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int64_t units = a.rows / R;
    const int nx = (int) ((units + NUM - 1) / NUM);
    const ydst out = {a.dst, a.dst_bf16};

    for (int i = threadIdx.x; i <= E; i += NW * 32) {
        bnd[i] = a.bounds[i];
    }
    __syncthreads();
    if (warp == 0) { // exclusive scan of the tiles per expert
        int run = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            const int e = e0 + lane;
            const int nt = e < E ? (bnd[e + 1] - bnd[e] + TP - 1) / TP : 0;
            int incl = nt;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, d);
                incl += lane >= d ? y : 0;
            }
            if (e < E) {
                toff[e] = (uint16_t) (run + incl - nt);
            }
            run += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) {
            toff[E] = (uint16_t) run;
        }
    }
    __syncthreads();
    const int n_items = toff[E] * nx; // item = tile * nx + unit group
    int item = blockIdx.x;
    if (item >= n_items) {
        return;
    }
    // item -> expert, first routed token, first unit (tile t belongs to the largest e with toff[e] <= t)
    auto locate = [&](int it, int & e, int & jb, int64_t & ub) {
        const int t = it / nx;
        int lo = 0, hi = E;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (toff[mid] <= t) { lo = mid; } else { hi = mid; }
        }
        e = lo;
        jb = bnd[lo] + (t - toff[lo]) * TP;
        ub = (int64_t) (it % nx) * NUM;
    };
#pragma unroll
    for (int i = threadIdx.x; i < (1 << HP_LP) / 4; i += NW * 32) {
        ((uint4 *) lut)[i] = __ldg((const uint4 *) w.lut + i);
        if constexpr (GLU) { ((uint4 *) (lut + (1 << HP_LP)))[i] = __ldg((const uint4 *) w2.lut + i); }
    }
    // unit slot of a warp's stream -> matrix and unit within the item (GLU: per 32-row group SB gate slots, then SB up)
    const bool second = GLU && warp % (2 * SB) >= SB;
    const int uslot = GLU ? warp / (2 * SB) * SB + warp % SB : warp * IL;
    const hp_ptrs wm = second ? w2 : w;
    const uint32_t * lst = wm.stream;
    const uint32_t * lutm = second ? lut + (1 << HP_LP) : lut;
    hp_dec qdec[IL];                // RB = 0
    hp_rdec<RB ? RB : 2> dec[IL];   // RB > 0
    const uint8_t * up[IL];
    uint4 smv[IL][RS]; // plane rows of the next step
    uint32_t nidx[IL]; // RB: lane stream starts of the next item
    if constexpr (RB > 0) {
#pragma unroll
        for (int il = 0; il < IL; il++) {
            dec[il].ring = rings + ((warp * IL + il) * 32 + lane) * 4 * RB;
            dec[il].key = (RB == 2 ? lane >> 2 : lane >> 1) & (RB - 1);
        }
    }
    auto unit_of = [&](int e, int64_t ub, int il) {
        const int64_t u = ub + uslot + il;
        return (int64_t) e * units + (u < units ? u : ub); // a missing unit decodes unit ub; never stored
    };
    auto start = [&](int e, int64_t ub, bool first) {
        if constexpr (RB > 0) {
            cp_async_wait<0>(); // the old streams' last ring blocks target the same slots
        }
#pragma unroll
        for (int il = 0; il < IL; il++) {
            const int64_t gu = unit_of(e, ub, il);
            if constexpr (RB > 0) {
                hp_ring_begin(dec[il], lst, first ? hp_lane_start(wm, gu, lane) : nidx[il]);
            } else {
                hp_dec_begin(qdec[il], wm, gu, lane);
            }
            up[il] = wm.sm + gu * R * K;
#pragma unroll
            for (int rr = 0; rr < RS; rr++) { smv[il][rr] = ld_plane((const uint4 *) (up[il] + rr * 512) + lane); }
        }
        if constexpr (RB > 0) {
            cp_async_commit();
        }
    };
    int e, jb;
    int64_t ub;
    locate(item, e, jb, ub);
    start(e, ub, true);
    const int rg = warp % 2, tg = warp / 2;
    const int scol = 32 * (lane >> 3) + 16 * (lane & 1) + 2 * ((lane >> 1) & 3);
    const uint16_t * ar = slab + (rg * 32 + (lane & 15)) * HPM_LD + (lane >> 4) * 8;
    __syncthreads();

    while (true) {
        const int j1 = bnd[e + 1];
        const int nlt = (min(TP, j1 - jb) + WT * 8 - 1) / (WT * 8);
        const int nks = TG / nlt;
        const int tge = tg % nlt, ks = tg / nlt;
        const bool live = ks < nks;
        const __nv_bfloat16 * xr[WT];
        int dc[WT][2];
#pragma unroll
        for (int i = 0; i < WT; i++) {
            const int t0 = jb + (tge * WT + i) * 8;
            xr[i] = a.x + (int64_t) a.ids_src1[min(t0 + g, j1 - 1)] * a.x_row + 8 * q;
            dc[i][0] = t0 + 2 * q < j1 ? a.ids_dst[t0 + 2 * q] : -1;
            dc[i][1] = t0 + 2 * q + 1 < j1 ? a.ids_dst[t0 + 2 * q + 1] : -1;
        }
        const int next = item + gridDim.x;
        int ne = 0, njb = 0;
        int64_t nub = 0;
        if (next < n_items) {
            locate(next, ne, njb, nub);
            if constexpr (RB > 0) {
#pragma unroll
                for (int il = 0; il < IL; il++) { nidx[il] = hp_lane_start(wm, unit_of(ne, nub, il), lane); }
            }
        }
        float acc[H][2][WT][4]; // acc[0] belongs to the slice in the slab; rotated by one slice after every step
#pragma unroll
        for (int hh = 0; hh < H; hh++) {
#pragma unroll
            for (int m = 0; m < 2; m++) {
#pragma unroll
                for (int i = 0; i < WT; i++) { acc[hh][m][i][0] = acc[hh][m][i][1] = acc[hh][m][i][2] = acc[hh][m][i][3] = 0.f; }
            }
        }
#pragma unroll 1
        for (int s = 0; s < C * H; s++) {
            const int c = s / H;
            if (RB > 0 && s == 0) { // the item's ring blocks were started during the previous item
                cp_async_wait<0>();
#pragma unroll
                for (int il = 0; il < IL; il++) { hp_ring_first(dec[il], lst); }
            }
#pragma unroll
            for (int rr = 0; rr < RS; rr++) {
                const int p = s * RS + rr; // plane row in stream order: chunk-major, (c * R + row) * 512
                uint32_t smw[IL][4], ex[IL][8];
#pragma unroll
                for (int il = 0; il < IL; il++) {
                    const uint4 cur = smv[il][rr];
                    if (p + RS < C * R) {
                        smv[il][rr] = ld_plane((const uint4 *) (up[il] + (p + RS) * 512) + lane);
                    }
                    smw[il][0] = cur.x; smw[il][1] = cur.y; smw[il][2] = cur.z; smw[il][3] = cur.w;
                }
#pragma unroll
                for (int qq = 0; qq < 8; qq++) {
                    if constexpr (RB > 0) {
                        if (qq % 2 == 0) { cp_async_wait<WG>(); }
#pragma unroll
                        for (int il = 0; il < IL; il++) { ex[il][qq] = hp_ring_pair(dec[il], qq, lst, lutm); }
                        if (qq % 2 == 1) { cp_async_commit(); }
                    } else {
#pragma unroll
                        for (int il = 0; il < IL; il++) { ex[il][qq] = hp_pair(qdec[il], qq, lst, lutm); }
                    }
                }
#pragma unroll
                for (int il = 0; il < IL; il++) {
                    uint32_t * srow = (uint32_t *) (slab + ((warp * IL + il) * RS + rr) * HPM_LD + scol);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        srow[64 * j]     = sel_bits(prmt(smw[il][j], 0, 0x9180u), ex[il][2 * j]);     // columns +0, +1
                        srow[64 * j + 4] = sel_bits(prmt(smw[il][j], 0, 0xB3A2u), ex[il][2 * j + 1]); // columns +8, +9
                    }
                }
            }
            if (s == C * H - 1 && next < n_items) { // this item's streams are consumed: start the next item's
                start(ne, nub, false);
            }
            __syncthreads();
            if (live) {
                uint4 b0[WT], b1[WT];
#pragma unroll
                for (int i = 0; i < WT; i++) {
                    b0[i] = __ldg((const uint4 *) (xr[i] + c * 512 + 32 * ks));
                    b1[i] = ks + nks < 16 ? __ldg((const uint4 *) (xr[i] + c * 512 + 32 * (ks + nks))) : make_uint4(0, 0, 0, 0);
                }
#define HPM_GROUP(B, KG)                                                                                     \
                _Pragma("unroll") for (int st = 0; st < 2; st++) {                                               \
                    uint32_t af[2][4];                                                                           \
                    ldmatrix_x4(af[0], ar + 32 * (KG) + 16 * st);                                                \
                    ldmatrix_x4(af[1], ar + 16 * HPM_LD + 32 * (KG) + 16 * st);                                  \
                    _Pragma("unroll") for (int i = 0; i < WT; i++) {                                             \
                        const uint32_t x0 = st ? (B)[i].z : (B)[i].x, x1 = st ? (B)[i].w : (B)[i].y;             \
                        mma_bf16(acc[0][0][i], af[0], x0, x1);                                                   \
                        mma_bf16(acc[0][1][i], af[1], x0, x1);                                                   \
                    }                                                                                            \
                }
                for (int kg = ks; kg < 16; kg += 2 * nks) {
                    HPM_GROUP(b0, kg)
                    if (kg + 2 * nks < 16) {
#pragma unroll
                        for (int i = 0; i < WT; i++) { b0[i] = __ldg((const uint4 *) (xr[i] + c * 512 + 32 * (kg + 2 * nks))); }
                    }
                    if (kg + nks < 16) {
                        HPM_GROUP(b1, kg + nks)
                        if (kg + 3 * nks < 16) {
#pragma unroll
                            for (int i = 0; i < WT; i++) { b1[i] = __ldg((const uint4 *) (xr[i] + c * 512 + 32 * (kg + 3 * nks))); }
                        }
                    }
                }
#undef HPM_GROUP
            }
#pragma unroll
            for (int m = 0; m < 2; m++) {
#pragma unroll
                for (int i = 0; i < WT; i++) {
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        const float t0 = acc[0][m][i][v];
#pragma unroll
                        for (int hh = 0; hh + 1 < H; hh++) { acc[hh][m][i][v] = acc[hh + 1][m][i][v]; }
                        acc[H - 1][m][i][v] = t0;
                    }
                }
            }
            __syncthreads();
        }
        if (nks > 1) { // add the K-split partial sums through the (now free) slab
            float * red = (float *) slab;
            if (live && ks > 0) {
#pragma unroll
                for (int hh = 0; hh < H; hh++) {
#pragma unroll
                    for (int m = 0; m < 2; m++) {
#pragma unroll
                        for (int i = 0; i < WT; i++) {
#pragma unroll
                            for (int v = 0; v < 4; v++) { red[(((warp * H + hh) * 2 + m) * WT * 4 + i * 4 + v) * 32 + lane] = acc[hh][m][i][v]; }
                        }
                    }
                }
            }
            __syncthreads();
            if (live && ks == 0) {
                for (int k2 = 1; k2 < nks; k2++) {
                    const int w2 = rg + 2 * (tge + nlt * k2);
#pragma unroll
                    for (int hh = 0; hh < H; hh++) {
#pragma unroll
                        for (int m = 0; m < 2; m++) {
#pragma unroll
                            for (int i = 0; i < WT; i++) {
#pragma unroll
                                for (int v = 0; v < 4; v++) { acc[hh][m][i][v] += red[(((w2 * H + hh) * 2 + m) * WT * 4 + i * 4 + v) * 32 + lane]; }
                            }
                        }
                    }
                }
            }
        }
        if (GLU && live && ks == 0) { // tile 0 of the row group = gate rows, tile 1 = the same up rows
#pragma unroll
            for (int sl = 0; sl < H; sl++) {
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const int rho = g + 8 * hh;                 // row within the tile
                    const int64_t u = ub + rg * SB + rho / RS;  // unit
                    if (u >= units) {
                        continue;
                    }
                    const int64_t row = u * R + sl * RS + rho % RS;
#pragma unroll
                    for (int i = 0; i < WT; i++) {
#pragma unroll
                        for (int t = 0; t < 2; t++) {
                            if (dc[i][t] >= 0) {
                                out.set((int64_t) dc[i][t] * a.dst_col + row,
                                    acc[sl][1][i][2 * hh + t] * silu_single(acc[sl][0][i][2 * hh + t]));
                            }
                        }
                    }
                }
            }
        }
        if (!GLU && live && ks == 0) {
#pragma unroll
            for (int sl = 0; sl < H; sl++) {
#pragma unroll
                for (int m = 0; m < 2; m++) {
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const int rho = rg * 32 + m * 16 + g + 8 * hh;                    // slab row
                        const int64_t row = (ub + rho / RS) * R + sl * RS + rho % RS;     // row within the expert
                        if (ub + rho / RS >= units) {
                            continue;
                        }
#pragma unroll
                        for (int i = 0; i < WT; i++) {
#pragma unroll
                            for (int t = 0; t < 2; t++) {
                                if (dc[i][t] >= 0) {
                                    out.set((int64_t) dc[i][t] * a.dst_col + row, acc[sl][m][i][2 * hh + t]);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (next >= n_items) {
            break;
        }
        item = next; e = ne; jb = njb; ub = nub;
        __syncthreads(); // the K-split sums were read from the slab
    }
}

// ---------------------------------------------------------------- fragment order (FLAG_FRAG_ORDER)
//
// HUFPAIR K = 512, R = 16 units (16-row bands) whose plane and lane streams are in mma.m16n8k16 A-fragment order
// (format.h): per tile pair a lane decodes 8 exponent pairs and assembles the A registers of two 16x16 tiles, rows g and
// g + 8, columns 2t, 2t + 1 and 2t + 8, 2t + 9 of each. The batched kernel multiplies them straight on the tensor cores
// (no shared-memory slab, no barrier between decoding and products); the GEMV sums them in FP32.

constexpr int FO_K = 512;

// A registers of tile pair (plane word pl, exponent entries ex[8]): a[0..3] for the even tile, a[4..7] for the odd one
__device__ __forceinline__ void fo_a(const uint4 & pl, const uint32_t (&ex)[8], uint32_t (&a)[8]) {
    a[0] = sel_bits(prmt(pl.x, 0, 0x9180u), ex[0]); a[1] = sel_bits(prmt(pl.x, 0, 0xB3A2u), ex[1]);
    a[2] = sel_bits(prmt(pl.y, 0, 0x9180u), ex[2]); a[3] = sel_bits(prmt(pl.y, 0, 0xB3A2u), ex[3]);
    a[4] = sel_bits(prmt(pl.z, 0, 0x9180u), ex[4]); a[5] = sel_bits(prmt(pl.z, 0, 0xB3A2u), ex[5]);
    a[6] = sel_bits(prmt(pl.w, 0, 0x9180u), ex[6]); a[7] = sel_bits(prmt(pl.w, 0, 0xB3A2u), ex[7]);
}

// 4 x 4 transpose of 32-bit words across the 4 lanes t of a quad: v[e] of lane t becomes word t of lane e (two
// exchange rounds, one shuffle per word pair each)
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int t) {
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (e & m) { continue; }
            const bool hi = t & m; // this lane keeps v[e ^ m] (hi) or v[e], and receives the other
            const uint32_t r = __shfl_xor_sync(0xffffffffu, hi ? v[e] : v[e ^ m], m);
            if (hi) { v[e] = r; } else { v[e ^ m] = r; }
        }
    }
}

// exact BF16 bits of bands [row0 / 16, (row0 + nrows) / 16), one warp per 16 x 512 slab unit. a[q] of tile k holds
// columns c, c + 1 of row g + 8 (q & 1), c = 16 k + 2t + 8 (q >> 1): per tile pair and row, word e of lane t is columns
// 8e + 2t, 8e + 2t + 1, so a quad transpose leaves lane t columns 8t .. 8t + 7 for one 16-byte store per row
template <int C>
__global__ void __launch_bounds__(256) k_fo_dequant(const hp_ptrs w, int64_t row0, int64_t nrows, uint16_t * __restrict__ out) {
    constexpr int K = C * FO_K, NP = FO_K / 32, PF = 2;
    __shared__ uint32_t lut[1 << HP_LP];
    for (int i_ = threadIdx.x; i_ < (1 << HP_LP); i_ += blockDim.x) { lut[i_] = w.lut[i_]; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    for (int64_t u = blockIdx.x * 8 + warp; u < nrows / 16 * C; u += gridDim.x * 8) {
        const int64_t band = u / C;
        const int slab = (int) (u % C);
        HP_UNIT_BEGIN(w, row0 / 16 * C + u)
        uint16_t * o0 = out + (band * 16 + g) * K + slab * FO_K + 8 * t;
        uint16_t * o8 = o0 + 8 * K;
        const uint4 * pp = (const uint4 *) (w.sm + (row0 / 16 + band) * 16 * K + slab * 16 * FO_K) + lane;
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
            uint32_t r0[4] = {a[0], a[2], a[4], a[6]}, r8[4] = {a[1], a[3], a[5], a[7]};
            quad_transpose(r0, t);
            quad_transpose(r8, t);
            *(uint4 *) (o0 + p * 32) = make_uint4(r0[0], r0[1], r0[2], r0[3]);
            *(uint4 *) (o8 + p * 32) = make_uint4(r8[0], r8[1], r8[2], r8[3]);
        }
    }
}

// FP32 activation v as the sum of three BF16 values, exactly: hi = v truncated to BF16, mid = (v - hi) truncated, lo =
// v - hi - mid (at most 8 significant bits left; both subtractions are exact). Part n of v as BF16 bits. Infinities
// and NaNs stay in hi (a NaN whose payload sits in the truncated bits keeps a quiet bit), mid = lo = 0.
__device__ __forceinline__ uint32_t fo_split(float v, int n) {
    const uint32_t u = __float_as_uint(v);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        return n == 0 ? (u >> 16) | ((u & 0x007FFFFFu) && !(u & 0x007F0000u) ? 0x40u : 0u) : 0u;
    }
    const float hi = __uint_as_float(u & 0xFFFF0000u);
    const float r1 = v - hi;
    const float mid = __uint_as_float(__float_as_uint(r1) & 0xFFFF0000u);
    return __float_as_uint(n == 0 ? hi : n == 1 ? mid : r1 - mid) >> 16;
}

constexpr int FO_MMV_XB = FO_K / 32 * 3 * 4; // uint4 B-register entries of one column: tile pair, split, t
inline size_t fo_mmv_shm() { return FO_MMV_XB * 16 + (4u << HP_LP); }

__device__ __forceinline__ void prefetch_l1(const void * p) { asm volatile("prefetch.global.L1 [%0];" :: "l"(p)); }

// GEMV on the tensor cores (one column per blockIdx.y, expert ids as in k_hp_mmv): a warp takes a band and multiplies
// its A fragments with a B operand whose columns 0, 1, 2 are the hi, mid and lo parts of the column's FP32 activations
// (fo_split; staged once per block in B-register order, lanes g >= 3 supply zero columns). Products are exact, sums
// FP32 on the tensor cores; y = D[:, 0] + (D[:, 1] + D[:, 2]). Against the FMA form this drops the per-weight unpack
// and FMA instructions: two mma per tile pair. With so little work between loads a band start is exposed: its stream
// position (ubase, lrel), then its first stream words, are dependent DRAM round trips (21 % of the stall samples, ncu
// T=4). The next band's position is therefore loaded at the start of the current one and its first stream line
// prefetched into L1 halfway through (T-023: 4 tokens 328 -> 290 us per layer, row-major 298; 1 token 64 -> 62).
template <typename TX>
__global__ void __launch_bounds__(HP_NT) k_fo_mmv(const hp_ptrs w, cols_t p) {
    constexpr int K = FO_K, NP = K / 32, PF = 2; // tile pairs per band, plane words in flight
    extern __shared__ __align__(16) uint32_t fov_shm[];
    uint4 * xb = (uint4 *) fov_shm;               // xb[12 q + 4 n + t]: b0, b1 of tile 2q, b0, b1 of tile 2q + 1
    uint32_t * lut = fov_shm + 4 * FO_MMV_XB;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const TX * xg; ydst out; int64_t rbase;
    col_setup(p, blockIdx.y, xg, out, rbase);
    for (int i = threadIdx.x; i < FO_MMV_XB; i += blockDim.x) {
        const int tt = i & 3, n = (i >> 2) % 3, q = i / 12;
        uint32_t r[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { // tile 2q + j / 2, k = 2t (b0) or 2t + 8 (b1) and the next column
            const int c = 32 * q + 16 * (j >> 1) + 2 * tt + 8 * (j & 1);
            r[j] = fo_split(load_x1(xg, c), n) | fo_split(load_x1(xg, c + 1), n) << 16;
        }
        xb[i] = make_uint4(r[0], r[1], r[2], r[3]);
    }
    for (int i = threadIdx.x; i < (1 << HP_LP) / 4; i += blockDim.x) { ((uint4 *) lut)[i] = __ldg((const uint4 *) w.lut + i); }
    __syncthreads();
    const uint4 * xl = xb + 4 * g + t;
    const int64_t u0 = blockIdx.x * wpb + warp, ustep = (int64_t) gridDim.x * wpb, bands = p.nrows / 16;
    uint32_t s_next = 0; // this lane's stream position in its warp's next band
    if (u0 < bands) {
        s_next = __ldg(w.ubase + rbase / 16 + u0) + __ldg(w.lrel + (rbase / 16 + u0) * 32 + lane);
    }
    for (int64_t u = u0; u < bands; u += ustep) {
        const int64_t gu = rbase / 16 + u;
        const bool has_next = u + ustep < bands;
        uint32_t ub_n = 0, lr_n = 0;
        if (has_next) {
            ub_n = __ldg(w.ubase + gu + ustep);
            lr_n = __ldg(w.lrel + (gu + ustep) * 32 + lane);
        }
        HP_UNIT_BEGIN_AT(w, s_next)
        const uint4 * pp = (const uint4 *) (w.sm + gu * 16 * K) + lane;
        uint4 ring[PF];
#pragma unroll
        for (int i = 0; i < PF; i++) { ring[i] = ld_plane(pp + i * 32); }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < NP; q++) {
            if (q == NP / 2 && has_next) {
                s_next = ub_n + lr_n;
                prefetch_l1(lst + s_next);
            }
            const uint4 pl = ring[q % PF];
            if (q + PF < NP) { ring[q % PF] = ld_plane(pp + (q + PF) * 32); }
            uint32_t ex[8], a[8];
            HP_PAIRS(ex)
            fo_a(pl, ex, a);
            const uint4 b = g < 3 ? xl[12 * q] : make_uint4(0, 0, 0, 0);
            mma_bf16(acc, *(const uint32_t (*)[4]) a, b.x, b.y);
            mma_bf16(acc, *(const uint32_t (*)[4]) (a + 4), b.z, b.w);
        }
        // lane (g, t) holds D[g][2t], D[g][2t + 1], D[g + 8][2t], D[g + 8][2t + 1]: hi, mid in t = 0, lo in t = 1
        const float lo0 = __shfl_down_sync(0xffffffffu, acc[0], 1), lo8 = __shfl_down_sync(0xffffffffu, acc[2], 1);
        if (t == 0) {
            out.set(u * 16 + g, acc[0] + (acc[1] + lo0));
            out.set(u * 16 + g + 8, acc[2] + (acc[3] + lo8));
        }
    }
}

// Dense GEMV of up to 8 BF16 columns on the tensor cores (T-077): one decode of the matrix feeds mma.m16n8k16 whose B
// columns are the tokens (exact BF16, zero columns past ncols), so 8 tokens cost about one. A group of C warps takes a
// band, warp s its slab s (K-split: bands alone are too few for 2048-row matrices); slab 0 adds the other slabs' FP32
// partial sums in slab order through shared memory. Every reduction depends on the matrix shape only (no atomics, no
// dependence on the column count or the grid), and mma keeps output columns independent: a column's result is the
// same bits whatever else is batched with it. Activations are staged row-major (rows LD apart: conflict-free ldmatrix,
// whose four 8x8 matrices are the B registers of a tile pair), only the ncols present: the lanes addressing an absent
// token's row read one 16-byte zero block instead (less shared memory, more L1 for the lane streams at 1-4 tokens).
// Warps per block (a multiple of C, at most 15 groups: one named barrier each) are the launcher's wave choice. Band
// starts are pipelined as in k_fo_mmv.
constexpr int FO8_NTOK = 8;
inline size_t fo8_shm(int C, int ncols, int nw) {
    return (4u << HP_LP) + (size_t) ncols * (C * FO_K + 8) * 2 + 16 + (C > 1 ? (size_t) nw * 32 * 16 : 0);
}

template <int C>
__global__ void __launch_bounds__(1024) k_fo_mmv8(const hp_ptrs w, cols_t p, int ncols) {
    constexpr int K = C * FO_K, LD = K + 8, NP = FO_K / 32, PF = 2;
    extern __shared__ __align__(16) uint32_t fo8_shm_raw[];
    uint32_t * lut = fo8_shm_raw;
    __nv_bfloat16 * xs = (__nv_bfloat16 *) (lut + (1 << HP_LP));
    uint4 * zero = (uint4 *) (xs + ncols * LD);
    float4 * part = (float4 *) (zero + 1); // [warps][32] slab partial sums (C > 1)
    const int nw = blockDim.x >> 5, G = nw / C;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int gid = warp / C, s = warp % C;
    for (int i = threadIdx.x; i < ncols * (K / 8); i += blockDim.x) {
        const int r = i / (K / 8), c8 = i % (K / 8);
        *(uint4 *) (xs + r * LD + c8 * 8) = __ldg((const uint4 *) (p.x + r * p.x_tok) + c8);
    }
    if (threadIdx.x == 0) { *zero = make_uint4(0, 0, 0, 0); }
    for (int i = threadIdx.x; i < (1 << HP_LP) / 4; i += blockDim.x) { ((uint4 *) lut)[i] = __ldg((const uint4 *) w.lut + i); }
    __syncthreads();
    // ldmatrix row address: token lane % 8, columns 8 (lane / 8) of tile pair 0 of this warp's slab (64 bytes per tile
    // pair), or the zero block for an absent token
    const bool present = (lane & 7) < ncols;
    const uint32_t lrow = present ? (uint32_t) __cvta_generic_to_shared(xs) + ((lane & 7) * LD + s * FO_K + 8 * (lane >> 3)) * 2
                                  : (uint32_t) __cvta_generic_to_shared(zero);
    const uint32_t lstep = present ? 64 : 0;
    const int64_t b0 = p.row0 / 16, bands = p.nrows / 16, bstep = (int64_t) gridDim.x * G;
    const int64_t u0 = (int64_t) blockIdx.x * G + gid;
    uint32_t s_next = 0; // this lane's stream position in its warp's next unit
    if (u0 < bands) {
        const int64_t un = (b0 + u0) * C + s;
        s_next = __ldg(w.ubase + un) + __ldg(w.lrel + un * 32 + lane);
    }
    for (int64_t u = u0; u < bands; u += bstep) {
        const int64_t gb = b0 + u;
        const bool has_next = u + bstep < bands;
        uint32_t ub_n = 0, lr_n = 0;
        if (has_next) {
            const int64_t un = (gb + bstep) * C + s;
            ub_n = __ldg(w.ubase + un);
            lr_n = __ldg(w.lrel + un * 32 + lane);
        }
        HP_UNIT_BEGIN_AT(w, s_next)
        const uint4 * pp = (const uint4 *) (w.sm + gb * 16 * K + s * 16 * FO_K) + lane;
        uint4 ring[PF];
#pragma unroll
        for (int i = 0; i < PF; i++) { ring[i] = ld_plane(pp + i * 32); }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < NP; q++) {
            if (q == NP / 2 && has_next) {
                s_next = ub_n + lr_n;
                prefetch_l1(lst + s_next);
            }
            const uint4 pl = ring[q % PF];
            if (q + PF < NP) { ring[q % PF] = ld_plane(pp + (q + PF) * 32); }
            uint32_t ex[8], a[8], b[4];
            HP_PAIRS(ex)
            fo_a(pl, ex, a);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(lrow + q * lstep));
            mma_bf16(acc, *(const uint32_t (*)[4]) a, b[0], b[1]);
            mma_bf16(acc, *(const uint32_t (*)[4]) (a + 4), b[2], b[3]);
        }
        if (C > 1) { // slab 0 sums the group's partials in slab order
            if (s > 0) { part[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]); }
            asm volatile("bar.sync %0, %1;" :: "r"(1 + gid), "r"(C * 32) : "memory");
            if (s == 0) {
#pragma unroll
                for (int j = 1; j < C; j++) {
                    const float4 v = part[(warp + j) * 32 + lane];
                    acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
                }
            }
            asm volatile("bar.sync %0, %1;" :: "r"(1 + gid), "r"(C * 32) : "memory");
        }
        // lane (g, t) holds D[g][2t], D[g][2t + 1], D[g + 8][2t], D[g + 8][2t + 1]: rows g, g + 8 of tokens 2t, 2t + 1
        if (s == 0) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int n = 2 * t + h;
                if (n < ncols) {
                    const ydst out = {p.dst + n * p.dst_tok, p.dst_bf16};
                    out.set(u * 16 + g, acc[h]);
                    out.set(u * 16 + g + 8, acc[2 + h]);
                }
            }
        }
    }
}

// Batched MUL_MAT_ID (prefill) over fragment-order experts. Persistent: one block per SM walks the items (expert, tile
// of up to NTOK routed tokens) built from the routing bounds in shared memory, as k_hp_mm_ids does. Per item the
// tokens' BF16 activations are staged once with cp.async (rows FOM_LD apart: conflict-free ldmatrix); the NW warps
// take the expert's bands, and per tile pair a lane decodes its A fragments and multiplies them with every 16 staged
// tokens (ldmatrix.x4 B fragments, FP32 accumulation). Every weight is decoded once per item.
constexpr int FOM_LD = FO_K + 8; // staged activation row stride (bf16)

inline size_t fo_mm_shm(int ntok, int n_experts) {
    return (4u << HP_LP) + (size_t) ntok * FOM_LD * 2 + (size_t) (n_experts + 1) * 2 * sizeof(int);
}

template <int NTOK, int NW>
__global__ void __launch_bounds__(NW * 32, 1) k_fo_mm_ids(const hp_ptrs w, hpm_args a) {
    constexpr int K = FO_K, NP = K / 32, NT = NTOK / 8; // tile pairs per band, 8-token mma tiles
    static_assert(NTOK % 16 == 0, "tokens in pairs of 8-token tiles");
    extern __shared__ __align__(16) uint32_t fom_shm[];
    uint32_t * lut = fom_shm;
    __nv_bfloat16 * xs = (__nv_bfloat16 *) (fom_shm + (1 << HP_LP));
    const int E = a.n_experts;
    int * bnd = (int *) (xs + NTOK * FOM_LD); // routing bounds, E + 1
    int * toff = bnd + E + 1;                 // first item of each expert, E + 1
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t bands = a.rows / 16;
    const ydst out = {a.dst, a.dst_bf16};
    for (int i = threadIdx.x; i <= E; i += NW * 32) { bnd[i] = a.bounds[i]; }
    for (int i = threadIdx.x; i < (1 << HP_LP) / 4; i += NW * 32) { ((uint4 *) lut)[i] = __ldg((const uint4 *) w.lut + i); }
    __syncthreads();
    if (warp == 0) { // exclusive scan of the items per expert
        int run = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            const int e = e0 + lane;
            const int nt = e < E ? (bnd[e + 1] - bnd[e] + NTOK - 1) / NTOK : 0;
            int incl = nt;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, d);
                incl += lane >= d ? y : 0;
            }
            if (e < E) { toff[e] = run + incl - nt; }
            run += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) { toff[E] = run; }
    }
    __syncthreads();
    const int n_items = toff[E];
    // ldmatrix row address of this lane: token (lane / 16) * 8 + lane % 8 of a 16-token group, column 8 ((lane / 8) % 2)
    const uint32_t xs_s = (uint32_t) __cvta_generic_to_shared(xs);
    const uint32_t lrow = xs_s + (((lane >> 4) * 8 + (lane & 7)) * FOM_LD + ((lane >> 3) & 1) * 8) * 2;
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        int lo = 0, hi = E; // the item belongs to the largest e with toff[e] <= it
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (toff[mid] <= it) { lo = mid; } else { hi = mid; }
        }
        const int e = lo;
        const int jb = bnd[e] + (it - toff[e]) * NTOK;
        const int ncol = min(NTOK, bnd[e + 1] - jb);
        const int npair = (ncol + 15) / 16;
        __syncthreads(); // the previous item's readers of xs are done
        for (int i = threadIdx.x; i < npair * 16 * (K / 8); i += NW * 32) {
            const int r = i / (K / 8), c8 = i % (K / 8);
            __nv_bfloat16 * dst = xs + r * FOM_LD + c8 * 8;
            if (r < ncol) {
                cp_async16(dst, a.x + (int64_t) a.ids_src1[jb + r] * a.x_row + c8 * 8);
            } else {
                *(uint4 *) dst = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        for (int64_t band = warp; band < bands; band += NW) {
            const int64_t gu = (int64_t) e * bands + band;
            hp_dec d;
            hp_dec_begin(d, w, gu, lane);
            float acc[NT][4];
#pragma unroll
            for (int n = 0; n < NT; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
            const uint4 * pp = (const uint4 *) (w.sm + gu * 16 * K) + lane;
            uint4 nxt = ld_plane(pp);
#pragma unroll 1
            for (int q = 0; q < NP; q++) {
                const uint4 pl = nxt;
                if (q + 1 < NP) { nxt = ld_plane(pp + (q + 1) * 32); }
                uint32_t ex[8], af[8];
#pragma unroll
                for (int j = 0; j < 8; j++) { ex[j] = hp_pair(d, j, w.stream, lut); }
                fo_a(pl, ex, af);
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    const uint32_t kofs = (uint32_t) ((2 * q + kk) * 16 * 2);
                    const uint32_t (&ak)[4] = *(const uint32_t (*)[4]) (af + 4 * kk);
#pragma unroll
                    for (int n = 0; n < NT / 2; n++) {
                        if (n < npair) {
                            uint32_t b[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                                         : "r"(lrow + n * 16 * FOM_LD * 2 + kofs));
                            mma_bf16(acc[2 * n], ak, b[0], b[1]);
                            mma_bf16(acc[2 * n + 1], ak, b[2], b[3]);
                        }
                    }
                }
            }
            const int64_t row = band * 16 + g;
#pragma unroll
            for (int n = 0; n < NT; n++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int col = n * 8 + 2 * t + h;
                    if (col < ncol) {
                        const int64_t o = (int64_t) a.ids_dst[jb + col] * a.dst_col + row;
                        out.set(o, acc[n][h]);
                        out.set(o + 8, acc[n][2 + h]);
                    }
                }
            }
        }
    }
}

} // namespace kern
} // namespace bf16n
