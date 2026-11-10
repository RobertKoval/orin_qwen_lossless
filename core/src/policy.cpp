// SPDX-License-Identifier: MIT
#include "bf16n/policy.h"

#include <initializer_list>

namespace bf16n {
namespace {

int dense_R(int64_t rows, policy_engine engine) noexcept {
    int best = 1;
    int64_t best_cost = 0;
    for (int r : {16, 8, 4, 2}) {
        if (rows % r != 0 || rows / r < 512) {
            continue;
        }
        const int64_t units = rows / r;
        const int64_t cost = (units / 512 + (units % 512 != 0)) * r;
        if (best == 1) {
            best = r;
            best_cost = cost;
            if (engine == policy_engine::llama) {
                break; // llama: largest R leaving at least 512 units
            }
        } else if (cost <= 0.95 * best_cost) {
            // vLLM: retain the Python wave model, including its floating-point 5% comparison.
            best = r;
            best_cost = cost;
        }
    }
    return best;
}

// rows*K*n >= threshold, without overflowing on caller-supplied dimensions.
bool at_least(int64_t rows, int64_t K, int64_t n, int64_t threshold) noexcept {
    if (threshold <= 0) {
        return true;
    }
    const int64_t per_matrix = threshold / n + (threshold % n != 0);
    const int64_t min_rows = per_matrix / K + (per_matrix % K != 0);
    return rows >= min_rows;
}

} // namespace

std::optional<policy_choice> choose(int64_t rows, int64_t K, int64_t n_matrices, policy_options opts) noexcept {
    if (rows <= 0 || n_matrices <= 0 || K <= 0 || K % CHUNK != 0 || K > K_MAX) {
        return std::nullopt;
    }
    const bool expert = opts.engine == policy_engine::llama ? opts.expert : n_matrices > 1;
    if (opts.mode == policy_mode::nibble ||
        (opts.mode == policy_mode::hybrid && !expert && !at_least(rows, K, n_matrices, opts.dense_min))) {
        return policy_choice{FMT_NIBBLE, 0, false};
    }
    // opted-in vLLM dense matrices in fragment order: the decode GEMV multiplies up to 8 tokens per decode on the tensor
    // cores (T-077, in_proj_qkvz 12288 x 2048: 4 tokens 247 -> 217 us, 8 tokens 494 -> 218 us)
    const bool fo_dense = opts.engine == policy_engine::vllm && opts.dense_frag_order && n_matrices == 1 && rows % 16 == 0;
    // vLLM's K = 2048 experts (gate / up) in fragment order, 16-row bands of 4 slabs: the decode GEMV multiplies every
    // token routed to an expert from one decode on the tensor cores, the batched prefill kernel decodes straight into
    // mma operands with SWIGLU fused (T-060 b2)
    const bool fo_gate_up = opts.engine == policy_engine::vllm && n_matrices > 1 && K == 4 * CHUNK && rows % 16 == 0;
    const int R = n_matrices > 1
        ? (fo_gate_up || (rows % 16 == 0 && rows >= 2048) ? 16 : rows % 8 == 0 ? 8 : rows % 4 == 0 ? 4 : 1)
        : fo_dense ? 16 : dense_R(rows, opts.engine);
    // vLLM's K = 512 experts (down) in fragment order: the batched prefill kernel decodes straight into mma operands
    // (T-023: down 2048 tokens 9.95 -> 4.34 ms per layer); the FO decode GEMV is 2-4 % faster than row-major at 2-4
    // tokens and within noise at 1
    const bool fo_experts = fo_gate_up || (opts.engine == policy_engine::vllm && n_matrices > 1 && K == CHUNK && R == 16);
    return policy_choice{FMT_HUFPAIR, R, n_matrices > 1 && K > CHUNK && !fo_experts, fo_experts || fo_dense};
}

} // namespace bf16n
