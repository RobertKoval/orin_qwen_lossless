// SPDX-License-Identifier: MIT
// BF16N storage format: constants and the aux header. Single source of truth for the encoder (encode.cpp), the
// kernels (kernels.cuh) and the Python side (vllm_bf16n/ops.py parses the header with the same field order).
// See docs/FORMAT.md.
#pragma once

#include <cstdint>

namespace bf16n {

// exponent formats (aux header `format`)
constexpr int FMT_NIBBLE  = 1; // 4-bit code per weight into a per-row 16-entry exponent LUT, escapes per row
constexpr int FMT_HUFFMAN = 2; // single-symbol canonical Huffman over exponents (fallback when the pair alphabet is too big)
constexpr int FMT_HUFPAIR = 3; // canonical Huffman over exponent pairs

constexpr uint32_t MAGIC    = 0x4e363142; // "B16N"
constexpr int HUF_LMAX      = 10;         // single-symbol code length limit (1024-entry LUT)
constexpr int HUF_GROUP     = 32 / HUF_LMAX;
constexpr int HP_LP         = 12;   // exponent-pair code length limit: two pair codes fit the guaranteed 32-bit window
constexpr int HP_MAX_PAIRS  = 1 << HP_LP; // more distinct exponent pairs than this -> FMT_HUFFMAN
constexpr int HP_PF         = 4;    // per-lane stream prefetch depth (words)
constexpr int HP_NT         = 1024; // one block per SM: one LUT copy, the rest of L1 for the lane streams
constexpr int CHUNK         = 512;  // weights per lane-major chunk; K must be a multiple
constexpr int K_MAX         = 4096; // escape records hold a 12-bit column
constexpr int HEADER_BYTES  = 256;

constexpr uint32_t FLAG_CHUNK_MAJOR = 1; // HUFPAIR: lane streams and plane chunk-major inside a unit
// HUFPAIR, K = 512, R = 16: a unit is a 16-row band in mma.m16n8k16 A-fragment order. Lane (g = lane / 4, t = lane % 4)
// of each 16x16 tile holds rows g, g + 8 and columns 2t, 2t + 1 (registers a0 and a1) and 2t + 8, 2t + 9 (a2, a3), each
// an exponent pair; per tile pair p the plane holds 16 bytes per lane at band + 512 p + 16 lane (tile 2p, then 2p + 1;
// a0..a3, lower column first), and the lane stream codes the lane's pairs tile by tile in register order.
constexpr uint32_t FLAG_FRAG_ORDER  = 2;

// The first 256 bytes of every aux buffer. Offsets are bytes from the aux start.
//   NIBBLE : off_a nibble codes (rows*K/2), off_b per-row LUT (rows*16), off_c esc_off u32[rows+1], off_d escapes u32;
//            n_a = escape count
//   HUFFMAN: off_a LUT u32[1024], off_b lane offsets u32[units*32] (absolute words), off_c stream; n_a units, n_b words
//   HUFPAIR: off_a LUT u32[4096], off_b unit base u32[units], off_c lane offsets u16[units*32] relative to the unit
//            base, off_d stream (16-byte aligned); n_a units, n_b words
struct aux_header {
    uint32_t magic, format, K, R;
    uint64_t rows;
    uint64_t off_a, off_b, off_c, off_d;
    uint64_t n_a, n_b;
    uint32_t flags, pad_;
};
static_assert(sizeof(aux_header) == 80, "aux_header layout is shared with vllm_bf16n/ops.py");

// lane-major position of slot s (0..15) of lane l (0..31) inside a 512-weight chunk: stored at byte 16*l + s
inline constexpr int pos_of(int l, int s) { return 128 * (s >> 2) + 4 * l + (s & 3); }

} // namespace bf16n
