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
    // vLLM linear layers (the caller's opt-in; not embeddings, whose get_rows reads row-major units): dense HUFPAIR
    // matrices with rows % 16 == 0 in fragment order, for the tensor-core decode GEMV. Ignored by llama.
    bool dense_frag_order = false;
};

struct policy_choice {
    int format;
    int R; // 0 for NIBBLE
    bool chunk_major;
    bool frag_order = false; // vLLM experts with K = 512, opted-in vLLM dense (FLAG_FRAG_ORDER); never for llama
};

// nullopt: shape stays BF16. HUFPAIR is a request; the encoder may fall back to HUFFMAN.
// llama keeps its largest-R rule; vLLM keeps its >=5%-gain wave model. No tuning is shared implicitly.
std::optional<policy_choice> choose(int64_t rows, int64_t K, int64_t n_matrices = 1,
                                  policy_options opts = {}) noexcept;

} // namespace bf16n
