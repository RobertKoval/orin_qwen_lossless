// SPDX-License-Identifier: MIT
// The few ggml-cuda helpers the BF16N kernels used (common.cuh, unary.cuh), without ggml.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#define BF16N_CUDA_CHECK(expr)                                                                             \
    do {                                                                                                   \
        const cudaError_t err_ = (expr);                                                                   \
        if (err_ != cudaSuccess) {                                                                         \
            throw std::runtime_error(std::string("bf16n: CUDA error ") + cudaGetErrorString(err_) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__) + ": " + #expr);           \
        }                                                                                                  \
    } while (0)

#define BF16N_REQUIRE(cond, msg)                                                                           \
    do {                                                                                                   \
        if (!(cond)) {                                                                                     \
            throw std::invalid_argument(std::string("bf16n: ") + (msg));                                  \
        }                                                                                                  \
    } while (0)

namespace bf16n {

// ggml_cuda_op_silu_single (unary.cuh); the extension is built with -use_fast_math like ggml-cuda, so this compiles to
// the same instructions
__device__ __forceinline__ float silu_single(float x) { return x / (1.0f + expf(-x)); }

} // namespace bf16n
