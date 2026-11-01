// SPDX-License-Identifier: MIT
// Engine-specific shape policy; eligibility by tensor dtype/name/device remains with the engine.
#pragma once

#include "bf16n/format.h"

#include <cstdint>
#include <optional>

namespace bf16n {

enum class policy_engine { llama, vllm };
enum class policy_mode { hybrid, huffman, nibble };

struct policy_options {
    policy_engine engine = policy_engine::vllm;
    // Historical llama "huffman" mode requests HUFPAIR, falling back to HUFFMAN in the encoder.
    policy_mode mode = policy_mode::hybrid;
    int64_t dense_min = 8ll << 20;
    // llama's loader classifies by "_exps" in the tensor name, independently of n_matrices.
    // vLLM classifies by n_matrices > 1 and ignores this flag.
    bool expert = false;
};

struct policy_choice {
    int format;
    int R; // 0 for NIBBLE
    bool chunk_major;
};

// nullopt: shape stays BF16. HUFPAIR is a request; the encoder may fall back to HUFFMAN.
// llama keeps its largest-R rule; vLLM keeps its >=5%-gain wave model. No tuning is shared implicitly.
std::optional<policy_choice> choose(int64_t rows, int64_t K, int64_t n_matrices = 1,
                                  policy_options opts = {}) noexcept;

} // namespace bf16n
