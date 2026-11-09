// SPDX-License-Identifier: MIT
// Host-side BF16N encoders, extracted from the llama.cpp patch (ggml-cuda/bf16n.cu, host encoder section) without
// ggml types. Every BF16 bit pattern is reconstructed exactly by the decoders in kernels.cuh.
#include "bf16n/encode.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#define BF16N_CHECK(cond)                                                                                  \
    do {                                                                                                   \
        if (!(cond)) {                                                                                     \
            throw std::runtime_error(std::string("bf16n encode: check failed: ") + #cond);               \
        }                                                                                                  \
    } while (0)

namespace bf16n {
namespace {

template <typename F>
void parallel_for(int64_t n, F && f) {
    const int nt = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) {
        th.emplace_back([&]() {
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= n) {
                    break;
                }
                f(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

// ---------------------------------------------------------------- host encoders

void encode_sm_plane(const uint16_t * w, int64_t rows, int64_t K, uint8_t * sm) {
    const int64_t C = K / 512;
    parallel_for(rows, [&](int64_t r) {
        for (int64_t c = 0; c < C; c++) {
            const uint16_t * src = w + r * K + c * 512;
            uint8_t * dst = sm + r * K + c * 512;
            for (int l = 0; l < 32; l++) {
                for (int s = 0; s < 16; s++) {
                    const uint16_t v = src[pos_of(l, s)];
                    dst[16 * l + s] = (uint8_t) (((v >> 15) << 7) | (v & 0x7F));
                }
            }
        }
    });
}

// fragment order (FLAG_FRAG_ORDER): (row, column) of the first weight of pair q (0..3, register a_q) of lane l in a
// 16x16 tile
inline void fo_pair(int l, int q, int & r, int & c) {
    r = (l >> 2) + 8 * (q & 1);
    c = 2 * (l & 3) + 8 * (q >> 1);
}

// sign|mantissa plane of 16-row bands in fragment order: per band, tile pair p and lane l, 16 bytes (tile 2p, then
// 2p + 1; pairs a0..a3, lower column first)
void encode_sm_plane_fo(const uint16_t * w, int64_t rows, int64_t K, uint8_t * sm) {
    auto smb = [](uint16_t v) { return (uint8_t) (((v >> 15) << 7) | (v & 0x7F)); };
    parallel_for(rows / 16, [&](int64_t b) {
        uint8_t * pb = sm + b * 16 * K;
        for (int64_t p = 0; p < K / 32; p++) {
            for (int l = 0; l < 32; l++) {
                uint8_t * o = pb + p * 512 + l * 16;
                for (int kk = 0; kk < 2; kk++) {
                    for (int q = 0; q < 4; q++) {
                        int r, c;
                        fo_pair(l, q, r, c);
                        const int64_t i = (b * 16 + r) * K + (2 * p + kk) * 16 + c;
                        o[8 * kk + 2 * q] = smb(w[i]);
                        o[8 * kk + 2 * q + 1] = smb(w[i + 1]);
                    }
                }
            }
        }
    });
}

std::vector<uint8_t> encode_nibble(const uint16_t * w, int64_t rows, int64_t K) {
    const int64_t C = K / 512;
    std::vector<uint8_t> nib(rows * K / 2);
    std::vector<uint8_t> lut(rows * 16, 0);
    constexpr int64_t BLK = 256;
    const int64_t nblk = (rows + BLK - 1) / BLK;
    std::vector<std::vector<uint32_t>> esc_blk(nblk);
    std::vector<uint32_t> cnt(rows);
    parallel_for(nblk, [&](int64_t b) {
        for (int64_t r = b * BLK; r < std::min(rows, (b + 1) * BLK); r++) {
            const uint16_t * row = w + r * K;
            int hist[256] = {0};
            for (int64_t k = 0; k < K; k++) {
                hist[(row[k] >> 7) & 0xFF]++;
            }
            int idx[256];
            for (int i = 0; i < 256; i++) {
                idx[i] = i;
            }
            std::sort(idx, idx + 256, [&](int a, int c) { return hist[a] > hist[c] || (hist[a] == hist[c] && a < c); });
            int code_of[256];
            for (int i = 0; i < 256; i++) {
                code_of[i] = -1;
            }
            code_of[0] = 0;
            uint8_t * L = &lut[r * 16];
            int n = 1;
            for (int i = 0; i < 256 && n < 16; i++) {
                const int x = idx[i];
                if (hist[x] == 0) {
                    break;
                }
                if (x == 0 || x >= 128) {
                    continue; // msb-set exponents cannot go through the prmt LUT
                }
                L[n] = (uint8_t) x;
                code_of[x] = n++;
            }
            uint32_t ne = 0;
            for (int64_t c = 0; c < C; c++) {
                uint8_t * nb = &nib[(r * C + c) * 256];
                for (int l = 0; l < 32; l++) {
                    uint32_t nw[2] = {0, 0};
                    for (int s = 0; s < 16; s++) {
                        const int col = (int) (c * 512 + pos_of(l, s));
                        const uint16_t v = row[col];
                        const int ex = (v >> 7) & 0xFF;
                        int code = code_of[ex];
                        if (code < 0) {
                            const uint32_t smb = ((v >> 15) << 7) | (v & 0x7F);
                            esc_blk[b].push_back((uint32_t) col | (smb << 12) | ((uint32_t) ex << 20));
                            ne++;
                            code = 0;
                        }
                        nw[s >> 3] |= (uint32_t) code << (4 * (s & 7));
                    }
                    memcpy(nb + 8 * l, nw, 8);
                }
            }
            cnt[r] = ne;
        }
    });
    // escape slots must also read as +0 in the sign|mantissa plane: done by the caller (zero_escapes)
    std::vector<uint32_t> esc_off(rows + 1, 0);
    for (int64_t r = 0; r < rows; r++) {
        esc_off[r + 1] = esc_off[r] + cnt[r];
    }
    const uint64_t n_esc = esc_off[rows];
    aux_header h = {};
    h.magic = MAGIC; h.format = FMT_NIBBLE; h.K = (uint32_t) K; h.rows = rows;
    h.off_a = 256;                                      // nibble plane
    h.off_b = h.off_a + nib.size();                     // per-row LUT
    h.off_c = h.off_b + lut.size();                     // esc_off
    h.off_d = h.off_c + esc_off.size() * 4;             // escapes
    h.n_a = n_esc;
    std::vector<uint8_t> aux(h.off_d + std::max<uint64_t>(n_esc, 1) * 4, 0);
    memcpy(aux.data(), &h, sizeof(h));
    memcpy(aux.data() + h.off_a, nib.data(), nib.size());
    memcpy(aux.data() + h.off_b, lut.data(), lut.size());
    memcpy(aux.data() + h.off_c, esc_off.data(), esc_off.size() * 4);
    uint8_t * e = aux.data() + h.off_d;
    for (auto & v : esc_blk) {
        memcpy(e, v.data(), v.size() * 4);
        e += v.size() * 4;
    }
    return aux;
}

std::vector<int> limited_lengths(const std::vector<uint64_t> & freq, int L) {
    std::vector<int> syms;
    for (int i = 0; i < (int) freq.size(); i++) {
        if (freq[i]) {
            syms.push_back(i);
        }
    }
    std::vector<int> len(freq.size(), 0);
    if (syms.size() == 1) {
        len[syms[0]] = 1;
        return len;
    }
    struct item { uint64_t w; std::vector<int> s; };
    std::vector<item> leaves;
    for (int s : syms) {
        leaves.push_back({freq[s], {s}});
    }
    std::sort(leaves.begin(), leaves.end(), [](const item & a, const item & b) { return a.w < b.w; });
    std::vector<item> cur = leaves;
    for (int d = 1; d < L; d++) {
        std::vector<item> pk;
        for (size_t i = 0; i + 1 < cur.size(); i += 2) {
            item m{cur[i].w + cur[i + 1].w, cur[i].s};
            m.s.insert(m.s.end(), cur[i + 1].s.begin(), cur[i + 1].s.end());
            pk.push_back(std::move(m));
        }
        std::vector<item> mg;
        std::merge(pk.begin(), pk.end(), leaves.begin(), leaves.end(), std::back_inserter(mg), [](const item & a, const item & b) { return a.w < b.w; });
        cur = std::move(mg);
    }
    for (size_t i = 0; i < 2 * syms.size() - 2; i++) {
        for (int s : cur[i].s) {
            len[s]++;
        }
    }
    return len;
}

std::vector<uint8_t> encode_huffman(const uint16_t * w, int64_t rows, int64_t K, int R) {
    const int64_t C = K / 512;
    const int64_t units = rows / R;
    std::vector<uint64_t> freq(256, 0);
    {
        std::mutex m;
        constexpr int64_t BLK = 1024;
        parallel_for((rows + BLK - 1) / BLK, [&](int64_t b) {
            uint64_t h[256] = {0};
            for (int64_t i = b * BLK * K; i < std::min(rows, (b + 1) * BLK) * K; i++) {
                h[(w[i] >> 7) & 0xFF]++;
            }
            std::lock_guard<std::mutex> lk(m);
            for (int i = 0; i < 256; i++) {
                freq[i] += h[i];
            }
        });
    }
    const std::vector<int> len = limited_lengths(freq, HUF_LMAX);
    std::vector<uint32_t> code(256, 0);
    {
        uint32_t c = 0;
        for (int l = 1; l <= HUF_LMAX; l++) {
            for (int s = 0; s < 256; s++) {
                if (len[s] == l) {
                    code[s] = c++;
                }
            }
            c <<= 1;
        }
    }
    std::vector<uint32_t> lut(1 << HUF_LMAX, 0);
    for (int s = 0; s < 256; s++) {
        if (len[s]) {
            const uint32_t lo = code[s] << (HUF_LMAX - len[s]), hi = (code[s] + 1) << (HUF_LMAX - len[s]);
            for (uint32_t i = lo; i < hi; i++) {
                lut[i] = (uint32_t) len[s] | ((uint32_t) s << 7);
            }
        }
    }
    std::vector<std::vector<uint32_t>> ustream(units);
    std::vector<uint32_t> lane_rel(units * 32);
    parallel_for(units, [&](int64_t u) {
        auto & out = ustream[u];
        for (int l = 0; l < 32; l++) {
            lane_rel[u * 32 + l] = (uint32_t) out.size();
            uint64_t acc = 0;
            int nb = 0;
            for (int64_t r = u * R; r < (u + 1) * R; r++) {
                for (int64_t c = 0; c < C; c++) {
                    for (int s = 0; s < 16; s++) {
                        const int e = (w[r * K + c * 512 + pos_of(l, s)] >> 7) & 0xFF;
                        acc = (acc << len[e]) | code[e];
                        nb += len[e];
                        if (nb >= 32) {
                            out.push_back((uint32_t) (acc >> (nb - 32)));
                            nb -= 32;
                            acc &= nb ? ((1ull << nb) - 1) : 0;
                        }
                    }
                }
            }
            if (nb) {
                out.push_back((uint32_t) (acc << (32 - nb)));
            }
        }
    });
    std::vector<uint32_t> lane_off(units * 32);
    uint64_t n_words = 0;
    for (int64_t u = 0; u < units; u++) {
        for (int l = 0; l < 32; l++) {
            lane_off[u * 32 + l] = (uint32_t) (n_words + lane_rel[u * 32 + l]);
        }
        n_words += ustream[u].size();
    }
    BF16N_CHECK(n_words + 16 < (1ull << 32));
    aux_header h = {};
    h.magic = MAGIC; h.format = FMT_HUFFMAN; h.K = (uint32_t) K; h.R = (uint32_t) R; h.rows = rows;
    h.off_a = 256;                                   // LUT
    h.off_b = h.off_a + lut.size() * 4;              // lane offsets
    h.off_c = h.off_b + lane_off.size() * 4;         // stream
    h.n_a = units; h.n_b = n_words;
    std::vector<uint8_t> aux(h.off_c + (n_words + 16) * 4, 0); // +16 words: decoder lookahead
    memcpy(aux.data(), &h, sizeof(h));
    memcpy(aux.data() + h.off_a, lut.data(), lut.size() * 4);
    memcpy(aux.data() + h.off_b, lane_off.data(), lane_off.size() * 4);
    uint8_t * p = aux.data() + h.off_c;
    for (auto & v : ustream) {
        memcpy(p, v.data(), v.size() * 4);
        p += v.size() * 4;
    }
    return aux;
}

// Huffman over exponent PAIRS (positions t,t+1 of a lane slot group): adjacent exponents are strongly
// correlated (pair entropy ~2.05 vs ~2.53 bit/weight). Returns empty if the tensor has > 2^HP_LP distinct pairs.
// Lane streams of a unit (R rows) are row-major (row 0 over all K, then row 1, ...) or, with cm, chunk-major (512-column
// chunk 0 of rows 0..R-1, then chunk 1, ...). Chunk-major lets a batched kernel decode a K-slab of many rows at a time;
// such units also keep their sign|mantissa plane chunk-major (see encode() below). With fo (FLAG_FRAG_ORDER, R = 16) a
// unit is a 16-row x 512-column slab of a band (slab s of band b is unit b * K / 512 + s; K = 512: the band) and a
// lane's stream holds its pairs of the slab tile by tile, in A-register order (fo_pair).
std::vector<uint8_t> encode_huffpair(const uint16_t * w, int64_t rows, int64_t K, int R, bool cm, bool fo) {
    const int64_t C = K / 512;
    const int64_t units = fo ? rows / 16 * C : rows / R;
    auto ex = [&](int64_t i) { return (w[i] >> 7) & 0xFF; };
    std::vector<uint64_t> freq(65536, 0);
    {
        std::mutex m;
        constexpr int64_t BLK = 1024;
        parallel_for((rows + BLK - 1) / BLK, [&](int64_t b) {
            std::vector<uint64_t> h(65536, 0);
            for (int64_t i = b * BLK * K; i < std::min(rows, (b + 1) * BLK) * K; i += 2) {
                h[ex(i) | ex(i + 1) << 8]++;
            }
            std::lock_guard<std::mutex> lk(m);
            for (int i = 0; i < 65536; i++) {
                freq[i] += h[i];
            }
        });
    }
    int nsym = 0;
    for (auto f : freq) {
        nsym += f != 0;
    }
    if (nsym > (1 << HP_LP)) {
        return {};
    }
    const std::vector<int> len = limited_lengths(freq, HP_LP);
    std::vector<uint32_t> code(65536, 0);
    {
        uint32_t c = 0;
        for (int l = 1; l <= HP_LP; l++) {
            for (int s = 0; s < 65536; s++) {
                if (len[s] == l) {
                    code[s] = c++;
                }
            }
            c <<= 1;
        }
    }
    std::vector<uint32_t> lut(1 << HP_LP, 0);
    for (int s = 0; s < 65536; s++) {
        if (len[s]) {
            const uint32_t lo = code[s] << (HP_LP - len[s]), hi = (code[s] + 1) << (HP_LP - len[s]);
            for (uint32_t i = lo; i < hi; i++) {
                lut[i] = ((uint32_t) (s >> 8) << 23) | ((uint32_t) (s & 0xFF) << 7) | (uint32_t) len[s];
            }
        }
    }
    std::vector<std::vector<uint32_t>> ustream(units);
    std::vector<uint32_t> rel32(units * 32);
    parallel_for(units, [&](int64_t u) {
        auto & out = ustream[u];
        for (int l = 0; l < 32; l++) {
            rel32[u * 32 + l] = (uint32_t) out.size();
            uint64_t acc = 0;
            int nb = 0;
            auto put = [&](int64_t i) {
                const int sym = ex(i) | ex(i + 1) << 8;
                acc = (acc << len[sym]) | code[sym];
                nb += len[sym];
                if (nb >= 32) {
                    out.push_back((uint32_t) (acc >> (nb - 32)));
                    nb -= 32;
                    acc &= nb ? ((1ull << nb) - 1) : 0;
                }
            };
            if (fo) {
                const int64_t band = u / C, slab = u % C;
                for (int64_t kt = slab * 32; kt < slab * 32 + 32; kt++) {
                    for (int q = 0; q < 4; q++) {
                        int r, c;
                        fo_pair(l, q, r, c);
                        put((band * 16 + r) * K + kt * 16 + c);
                    }
                }
            } else {
                for (int64_t o = 0; o < R * C; o++) {
                    const int64_t r = u * R + (cm ? o % R : o / C), c = cm ? o / R : o % C;
                    for (int s = 0; s < 16; s += 2) {
                        put(r * K + c * 512 + pos_of(l, s));
                    }
                }
            }
            if (nb) {
                out.push_back((uint32_t) (acc << (32 - nb)));
            }
        }
    });
    std::vector<uint32_t> unit_base(units);
    std::vector<uint16_t> lane_rel(units * 32);
    uint64_t n_words = 0;
    for (int64_t u = 0; u < units; u++) {
        unit_base[u] = (uint32_t) n_words;
        for (int l = 0; l < 32; l++) {
            BF16N_CHECK(rel32[u * 32 + l] <= 0xFFFF);
            lane_rel[u * 32 + l] = (uint16_t) rel32[u * 32 + l];
        }
        n_words += ustream[u].size();
    }
    BF16N_CHECK(n_words + 32 < (1ull << 32));
    aux_header h = {};
    h.magic = MAGIC; h.format = FMT_HUFPAIR; h.K = (uint32_t) K; h.R = (uint32_t) R; h.rows = rows;
    h.off_a = 256;                                    // LUT
    h.off_b = h.off_a + lut.size() * 4;               // unit base (u32)
    h.off_c = h.off_b + unit_base.size() * 4;         // lane offsets relative to unit base (u16)
    h.off_d = (h.off_c + lane_rel.size() * 2 + 15) & ~15ull; // stream
    h.n_a = units; h.n_b = n_words; h.flags = (cm ? FLAG_CHUNK_MAJOR : 0) | (fo ? FLAG_FRAG_ORDER : 0);
    std::vector<uint8_t> aux(h.off_d + (n_words + 32) * 4, 0); // +32 words: lookahead of the decoders' prefetch and rings
    memcpy(aux.data(), &h, sizeof(h));
    memcpy(aux.data() + h.off_a, lut.data(), lut.size() * 4);
    memcpy(aux.data() + h.off_b, unit_base.data(), unit_base.size() * 4);
    memcpy(aux.data() + h.off_c, lane_rel.data(), lane_rel.size() * 2);
    uint8_t * p = aux.data() + h.off_d;
    for (auto & v : ustream) {
        memcpy(p, v.data(), v.size() * 4);
        p += v.size() * 4;
    }
    return aux;
}

} // namespace

encoded encode(const uint16_t * w, int64_t rows, int64_t K, int format, int R, bool chunk_major, bool frag_order) {
    auto bad = [](const std::string & m) { throw std::invalid_argument("bf16n encode: " + m); };
    if (K % CHUNK != 0 || K < CHUNK || K > K_MAX) {
        bad("K must be a multiple of 512 in [512, 4096], got " + std::to_string(K));
    }
    if (rows <= 0) {
        bad("rows must be positive");
    }
    if (format == FMT_NIBBLE) {
        if (R != 0 || chunk_major || frag_order) {
            bad("NIBBLE takes R = 0 and no chunk-major or fragment-order layout");
        }
    } else if (format == FMT_HUFFMAN || format == FMT_HUFPAIR) {
        if (R != 1 && R != 2 && R != 4 && R != 8 && R != 16) {
            bad("R must be one of 1, 2, 4, 8, 16, got " + std::to_string(R));
        }
        if (rows % R != 0) {
            bad("rows (" + std::to_string(rows) + ") must be a multiple of R (" + std::to_string(R) + ")");
        }
        if (format == FMT_HUFFMAN && chunk_major) {
            bad("chunk-major units exist only for HUFPAIR");
        }
        if (chunk_major && (K == CHUNK || (R != 4 && R != 8 && R != 16))) {
            bad("chunk-major units need K > 512 and R in {4, 8, 16} (the chunk-major kernels' instances)");
        }
        if (frag_order && (format != FMT_HUFPAIR || chunk_major || R != 16)) {
            bad("fragment order exists only for HUFPAIR with R = 16, without chunk-major units");
        }
    } else {
        bad("unknown format " + std::to_string(format));
    }

    encoded out;
    out.plane.resize((size_t) (rows * K));
    out.R = format == FMT_NIBBLE ? 0 : R;
    if (format == FMT_NIBBLE) {
        encode_sm_plane(w, rows, K, out.plane.data());
        out.aux = encode_nibble(w, rows, K);
        out.format = FMT_NIBBLE;
        // escaped weights must read as +0 in the plane so the dense path adds exactly nothing for them
        const aux_header * h = (const aux_header *) out.aux.data();
        const uint32_t * esc = (const uint32_t *) (out.aux.data() + h->off_d);
        const uint32_t * eo = (const uint32_t *) (out.aux.data() + h->off_c);
        for (int64_t r = 0; r < rows; r++) {
            for (uint32_t i = eo[r]; i < eo[r + 1]; i++) {
                const int col = esc[i] & 0xFFF;
                const int c = col / 512, p = col % 512;
                const int j = p / 128, l = (p % 128) / 4, t = p % 4;
                out.plane[r * K + c * 512 + 16 * l + 4 * j + t] = 0;
            }
        }
        return out;
    }
    if (format == FMT_HUFPAIR) {
        out.aux = encode_huffpair(w, rows, K, R, chunk_major, frag_order);
    }
    if (!out.aux.empty()) {
        out.format = FMT_HUFPAIR;
    } else {
        out.aux = encode_huffman(w, rows, K, R); // more distinct exponent pairs than the pair LUT holds, or forced
        out.format = FMT_HUFFMAN;
    }
    const uint32_t flags = out.format == FMT_HUFPAIR ? ((const aux_header *) out.aux.data())->flags : 0;
    if (flags & FLAG_FRAG_ORDER) {
        encode_sm_plane_fo(w, rows, K, out.plane.data());
    } else {
        encode_sm_plane(w, rows, K, out.plane.data());
    }
    if (flags & FLAG_CHUNK_MAJOR) {
        // chunk-major units also store their plane chunk-major ([chunk][row][512 B] inside each unit), so a kernel
        // walking chunk c over the unit's rows reads one contiguous block
        std::vector<uint8_t> tmp((size_t) (R * K));
        for (int64_t u = 0; u < rows / R; u++) {
            uint8_t * ub = out.plane.data() + u * R * K;
            memcpy(tmp.data(), ub, (size_t) (R * K));
            for (int r = 0; r < R; r++) {
                for (int64_t c = 0; c < K / 512; c++) {
                    memcpy(ub + (c * R + r) * 512, tmp.data() + r * K + c * 512, 512);
                }
            }
        }
    }
    return out;
}

} // namespace bf16n
