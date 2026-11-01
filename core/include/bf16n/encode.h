// SPDX-License-Identifier: MIT
// Host-side BF16N encoder (pure C++, multithreaded). See docs/FORMAT.md and format.h.
#pragma once

#include "bf16n/format.h"

#include <cstdint>
#include <vector>

namespace bf16n {

struct encoded {
    std::vector<uint8_t> plane; // sign|mantissa byte per weight (rows * K)
    std::vector<uint8_t> aux;   // header + exponent data
    int format;                 // the format actually produced (FMT_HUFPAIR may fall back to FMT_HUFFMAN)
    int R;                      // rows per unit (0 for FMT_NIBBLE)
};

// Encode `rows` x `K` BF16 bit patterns (row-major).
//   format = FMT_NIBBLE : R and chunk_major must be 0 / false
//   format = FMT_HUFFMAN: single-symbol code, R in {1,2,4,8,16}, rows % R == 0, chunk_major false
//   format = FMT_HUFPAIR: exponent-pair code (chunk_major: K > 512 and R in {4, 8, 16}); falls back to FMT_HUFFMAN when the tensor has more than HP_MAX_PAIRS
//                         distinct pairs (chunk_major is then dropped: the single-symbol format has no such layout)
// K % 512 == 0, 512 <= K <= 4096. Throws std::invalid_argument on bad arguments.
encoded encode(const uint16_t * w, int64_t rows, int64_t K, int format, int R, bool chunk_major);

} // namespace bf16n
