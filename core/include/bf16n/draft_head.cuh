// SPDX-License-Identifier: MIT
// Draft-only output head for MTP (M6): a 4-bit lm_head copy (symmetric, one FP32 scale per 128 weights) selects
// candidates, then gathered BF16N rows are re-scored in FP32. Proposals need not equal the full BF16-output head:
// candidate restriction and output rounding can change the winner. The target does not use these 4-bit weights;
// that fact alone does not establish sequence equivalence (see docs/vllm/README.md, fidelity boundaries).
// Included by launch.cu (one translation unit; keeps the build file list unchanged).
#pragma once
#include "bf16n/compat.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace bf16n {
namespace draft {

constexpr int G = 128; // weights per scale

// one thread per (row, group): q holds (value + 8) in 4 bits, even column in the low nibble
__global__ void k_int4_quantize(const __nv_bfloat16 * __restrict__ w, int64_t rows, int K, uint8_t * __restrict__ q,
                                float * __restrict__ s) {
    const int groups = K / G;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < rows * groups; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / groups;
        const int g = (int) (i % groups);
        const __nv_bfloat16 * src = w + r * K + g * G;
        float amax = 0.f;
        for (int k = 0; k < G; k++) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[k])));
        }
        const float scale = amax > 0.f ? amax / 7.f : 1.f;
        s[i] = scale;
        uint8_t * dst = q + r * (K / 2) + g * (G / 2);
        for (int k = 0; k < G; k += 2) {
            const int a = (int) fminf(fmaxf(rintf(__bfloat162float(src[k]) / scale), -8.f), 7.f) + 8;
            const int b = (int) fminf(fmaxf(rintf(__bfloat162float(src[k + 1]) / scale), -8.f), 7.f) + 8;
            dst[k / 2] = (uint8_t) (a | (b << 4));
        }
    }
}

// y[t][r] = sum_k deq(q[r][k]) x[t][k] for T <= 4 columns, K = NIT * 1024. One warp per row; per 1024 columns each
// lane loads 16 contiguous bytes (32 weights, one scale group quarter; coalesced 512 B per warp) and keeps its x
// values in registers as half2. Nibbles become half2 by the exponent trick ((n | 0x6400) = 1024 + n), products are
// summed in half2 over 32 weights and then scaled in FP32: approximate by design (candidate selection only).
template <int T, int NIT>
__global__ void __launch_bounds__(256) k_int4_gemv(const __nv_bfloat16 * __restrict__ x, const uint8_t * __restrict__ q,
                                                   const float * __restrict__ s, int64_t rows, float * __restrict__ y) {
    constexpr int K = NIT * 1024;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // xr[t][it][j]: half2 of columns it*1024 + lane*32 + {j, j+16}: matches the nibble order produced below
    __half2 xr[T][NIT][16];
#pragma unroll
    for (int t = 0; t < T; t++) {
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const __nv_bfloat16 * xp = x + t * K + it * 1024 + lane * 32;
#pragma unroll
            for (int w = 0; w < 4; w++) {        // word w of the 16 bytes holds columns 8w .. 8w+7
#pragma unroll
                for (int j = 0; j < 4; j++) {    // pair j: columns 8w + j (low halves) and 8w + j + 4 (high halves)
                    xr[t][it][4 * w + j] = __floats2half2_rn(__bfloat162float(xp[8 * w + j]), __bfloat162float(xp[8 * w + j + 4]));
                }
            }
        }
    }
    const __half2 bias = __float2half2_rn(1032.f); // 1024 + 8
    for (int64_t r = blockIdx.x * 8 + warp; r < rows; r += (int64_t) gridDim.x * 8) {
        uint4 v[NIT];
        float sc[NIT];
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            v[it] = __ldg((const uint4 *) (q + r * (K / 2) + it * 512 + lane * 16));
            sc[it] = __ldg(s + r * (K / G) + it * 8 + lane / 4);
        }
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) { acc[t] = 0.f; }
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const uint32_t words[4] = {v[it].x, v[it].y, v[it].z, v[it].w};
            __half2 hacc[T];
#pragma unroll
            for (int t = 0; t < T; t++) { hacc[t] = __float2half2_rn(0.f); }
#pragma unroll
            for (int w = 0; w < 4; w++) {
                // bytes b0..b3 of the word hold columns (2i, 2i+1) of 8w..8w+7: byte i low = col 8w+2i, high = 8w+2i+1.
                // Extract per pair j the columns 8w+j and 8w+j+4: nibble index j and j+4 of the word.
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t lo = (words[w] >> (4 * j)) & 0xFu;
                    const uint32_t hi = (words[w] >> (4 * (j + 4))) & 0xFu;
                    const uint32_t packed = (lo | (hi << 16)) | 0x64006400u;
                    const __half2 wv = __hsub2(*(const __half2 *) &packed, bias);
#pragma unroll
                    for (int t = 0; t < T; t++) { hacc[t] = __hfma2(wv, xr[t][it][4 * w + j], hacc[t]); }
                }
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float2 f = __half22float2(hacc[t]);
                acc[t] = fmaf(f.x + f.y, sc[it], acc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
            float val = acc[t];
#pragma unroll
            for (int o = 16; o; o >>= 1) { val += __shfl_xor_sync(0xffffffffu, val, o); }
            if (lane == 0) { y[t * rows + r] = val; }
        }
    }
}

} // namespace draft

using namespace draft;

void int4_quantize(const void * w_bf16, int64_t rows, int K, uint8_t * q, float * s, cudaStream_t stream) {
    BF16N_REQUIRE(K % G == 0, "int4_quantize: K must be a multiple of 128");
    const int64_t n = rows * (K / G);
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 65535);
    k_int4_quantize<<<blocks, 256, 0, stream>>>((const __nv_bfloat16 *) w_bf16, rows, K, q, s);
    BF16N_CUDA_CHECK(cudaGetLastError());
}

template <int T>
static void int4_gemv_t(const void * x, const uint8_t * q, const float * s, int64_t rows, int K, float * y,
                        unsigned blocks, cudaStream_t stream) {
    const auto * xb = (const __nv_bfloat16 *) x;
    switch (K) {
        case 1024: k_int4_gemv<T, 1><<<blocks, 256, 0, stream>>>(xb, q, s, rows, y); break;
        case 2048: k_int4_gemv<T, 2><<<blocks, 256, 0, stream>>>(xb, q, s, rows, y); break;
        default: k_int4_gemv<T, 4><<<blocks, 256, 0, stream>>>(xb, q, s, rows, y); break;
    }
}

void int4_gemv(const void * x_bf16, int T, const uint8_t * q, const float * s, int64_t rows, int K, float * y,
               int sm_count, cudaStream_t stream) {
    BF16N_REQUIRE(K == 1024 || K == 2048 || K == 4096, "int4_gemv: K must be 1024, 2048 or 4096");
    BF16N_REQUIRE(T >= 1 && T <= 4, "int4_gemv: 1 to 4 columns");
    const unsigned blocks = (unsigned) std::min<int64_t>((rows + 7) / 8, (int64_t) sm_count * 16);
    switch (T) {
        case 1: int4_gemv_t<1>(x_bf16, q, s, rows, K, y, blocks, stream); break;
        case 2: int4_gemv_t<2>(x_bf16, q, s, rows, K, y, blocks, stream); break;
        case 3: int4_gemv_t<3>(x_bf16, q, s, rows, K, y, blocks, stream); break;
        default: int4_gemv_t<4>(x_bf16, q, s, rows, K, y, blocks, stream); break;
    }
    BF16N_CUDA_CHECK(cudaGetLastError());
}

} // namespace bf16n
