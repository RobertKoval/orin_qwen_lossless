// SPDX-License-Identifier: MIT
// Routing helper for the batched MoE kernel: ported from llama.cpp ggml/src/ggml-cuda/mmid.cu (MIT License,
// Copyright (c) 2023-2026 The ggml authors), as patched by this repo; templated on the id type (int32 / int64) and
// without the inverse-map mode. For each expert (one block = one warp) it lists the (token, slot) pairs routed to it:
//   ids_src1[j] = token * sis1 + slot % nchannels_y   (activation row of the j-th routed pair)
//   ids_dst[j]  = token * n_expert_used + slot          (output column)
//   expert_bounds[e] .. expert_bounds[e + 1]            (range of expert e in j)
// with pairs sorted by expert, then by token. Runs entirely on the GPU (no host sync).
#include "bf16n/compat.cuh"

#include <climits>
#include <cstdint>

namespace bf16n {
namespace {

constexpr int WARP = 32;

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;
    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) { data = (it & 0x003FFFFF) | (iex_used << 22); }
    __device__ uint32_t it() const { return data & 0x003FFFFF; }
    __device__ uint32_t iex_used() const { return data >> 22; }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2 * mm_ids_pow2<(n + 1) / 2>::value; };
template <> struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <> struct mm_ids_pow2<0> { static constexpr int value = 1; };

// common.cuh warp_reduce_any / warp_reduce_sum (Ampere: one vote / redux instruction)
template <int width>
__device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == WARP) {
        return __any_sync(0xffffffff, x);
    }
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
    }
    return x;
}

__device__ __forceinline__ int warp_reduce_sum_int(int x) { return __reduce_add_sync(0xffffffff, x); }

template <int n_expert_used_template, typename TI>
__launch_bounds__(WARP, 1) __global__ void mm_ids_helper(
        const TI * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += WARP) {
                const int expert_used = (int) ids[it * si1 + iex]; // int indexing as upstream (checked by the launcher)
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }
            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }
            if (warp_reduce_any<WARP>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= WARP && WARP % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += WARP / neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                (int) ids[it * si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < WARP; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, WARP);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, WARP - 1, WARP);
        }
    }
    nex_prev = warp_reduce_sum_int(nex_prev);
    __syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += WARP) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it * n_expert_used + iex_used;
        ids_src1[nex_prev + itc] = it * sis1 + iex_used % nchannels_y;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int N, typename TI>
void launch_t(const TI * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int n_experts, int n_tokens,
              int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream) {
    const size_t nbytes_shared = (size_t) n_tokens * sizeof(mm_ids_helper_store);
    mm_ids_helper<N, TI><<<n_experts, WARP, nbytes_shared, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens,
                                                                      n_expert_used, nchannels_y, si1, sis1);
}

template <typename TI>
void launch(const TI * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int n_experts, int n_tokens,
            int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream) {
    switch (n_expert_used) {
#define CASE(N) case N: launch_t<N, TI>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream); break;
        CASE(2) CASE(4) CASE(6) CASE(8) CASE(10) CASE(16) CASE(32)
#undef CASE
        default: launch_t<0, TI>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream); break;
    }
    BF16N_CUDA_CHECK(cudaGetLastError());
}

template <typename TI>
void set_limits(int smem) {
#define ATTR(N) BF16N_CUDA_CHECK(cudaFuncSetAttribute(mm_ids_helper<N, TI>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    ATTR(0) ATTR(2) ATTR(4) ATTR(6) ATTR(8) ATTR(10) ATTR(16) ATTR(32)
#undef ATTR
}

} // namespace

void mm_ids_helper_init(int smem_optin) {
    set_limits<int32_t>(smem_optin);
    set_limits<int64_t>(smem_optin);
}

void launch_mm_ids_helper_i32(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream) {
    launch(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
}

void launch_mm_ids_helper_i64(const int64_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream) {
    launch(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
}

} // namespace bf16n
