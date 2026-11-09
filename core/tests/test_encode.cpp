// SPDX-License-Identifier: MIT
// Host tests of the BF16N encoder (no CUDA): every format decodes back to the input bit patterns with a reference
// decoder written from the layout documented in format.h (independent of the CUDA kernels), argument boundaries,
// and the HUFPAIR -> HUFFMAN fallback exactly at HP_MAX_PAIRS distinct exponent pairs.
#include "bf16n/encode.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bf16n;

static int failures = 0;
#define CHECK(cond, ...)                                                                                         \
    do {                                                                                                         \
        if (!(cond)) {                                                                                           \
            failures++;                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s: ", __FILE__, __LINE__, #cond);                                 \
            std::fprintf(stderr, __VA_ARGS__);                                                                   \
            std::fprintf(stderr, "\n");                                                                          \
        }                                                                                                        \
    } while (0)

namespace {

template <typename T> T at(const std::vector<uint8_t> & b, uint64_t off) {
    T v;
    std::memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

// MSB-first bit stream over little-endian u32 words
struct bits {
    const std::vector<uint8_t> & b;
    uint64_t word; // index of the next word to load (bytes = base + 4 * word)
    uint64_t base;
    uint64_t acc = 0;
    int n = 0;
    uint32_t peek(int k) {
        while (n < k) {
            acc = (acc << 32) | at<uint32_t>(b, base + 4 * word++);
            n += 32;
        }
        return (uint32_t) ((acc >> (n - k)) & ((1ull << k) - 1));
    }
    void skip(int k) { n -= k; acc &= n ? ((1ull << n) - 1) : 0; }
};

// fragment order (format.h): register a_q of lane l in a 16x16 tile holds row l / 4 + 8 (q & 1), columns
// c, c + 1 with c = 2 (l % 4) + 8 (q / 2)
void fo_rc(int l, int q, int & r, int & c) {
    r = l / 4 + 8 * (q & 1);
    c = 2 * (l % 4) + 8 * (q / 2);
}

// sign|mantissa byte of the plane for weight (r, col), honouring a chunk-major or fragment-order unit layout
uint8_t sm_at(const encoded & e, int64_t K, int64_t r, int64_t col, bool cm, bool fo) {
    if (fo) { // band r / 16, tile pair col / 32: 16 bytes per lane (tile 2p, then 2p + 1; a0..a3, lower column first)
        const int64_t p = col / 32, kk = (col / 16) % 2, rr = r % 16, cc = col % 16;
        for (int l = 0; l < 32; l++) {
            for (int q = 0; q < 4; q++) {
                int rq, cq;
                fo_rc(l, q, rq, cq);
                if (rq == rr && (cc == cq || cc == cq + 1)) {
                    return e.plane[(r / 16) * 16 * K + p * 512 + 16 * l + 8 * kk + 2 * q + (cc - cq)];
                }
            }
        }
        return 0; // unreachable: every (row, column) of a tile belongs to one lane and register
    }
    const int64_t c = col / CHUNK, p = col % CHUNK;
    const int j = (int) (p / 128), l = (int) ((p % 128) / 4), t = (int) (p % 4);
    const int64_t in_chunk = 16 * l + 4 * j + t; // inverse of pos_of
    if (cm) {
        const int64_t R = e.R, u = r / R;
        return e.plane[u * R * K + (c * R + r % R) * CHUNK + in_chunk];
    }
    return e.plane[r * K + c * CHUNK + in_chunk];
}

uint16_t join(uint8_t sm, int ex) { return (uint16_t) (((sm >> 7) << 15) | (ex << 7) | (sm & 0x7F)); }

std::vector<uint16_t> decode(const encoded & e, int64_t rows, int64_t K) {
    const aux_header h = at<aux_header>(e.aux, 0);
    std::vector<int> ex(rows * K, -1);
    const int64_t C = K / CHUNK;
    const bool cm = h.flags & FLAG_CHUNK_MAJOR, fo = h.flags & FLAG_FRAG_ORDER;
    if (h.format == (uint32_t) FMT_NIBBLE) {
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < C; c++) {
                for (int l = 0; l < 32; l++) {
                    const uint64_t nw = at<uint64_t>(e.aux, h.off_a + (r * C + c) * 256 + 8 * l);
                    for (int s = 0; s < 16; s++) {
                        const int code = (int) ((nw >> (4 * s)) & 0xF);
                        ex[r * K + c * CHUNK + pos_of(l, s)] = e.aux[h.off_b + r * 16 + code];
                    }
                }
            }
        }
    } else if (h.format == (uint32_t) FMT_HUFFMAN) {
        for (uint64_t u = 0; u < h.n_a; u++) {
            for (int l = 0; l < 32; l++) {
                bits bs{e.aux, at<uint32_t>(e.aux, h.off_b + 4 * (u * 32 + l)), h.off_c};
                for (int64_t r = u * h.R; r < (int64_t) ((u + 1) * h.R); r++) {
                    for (int64_t c = 0; c < C; c++) {
                        for (int s = 0; s < 16; s++) {
                            const uint32_t v = at<uint32_t>(e.aux, h.off_a + 4 * bs.peek(HUF_LMAX));
                            bs.skip((int) (v & 0x7F));
                            ex[r * K + c * CHUNK + pos_of(l, s)] = (int) ((v >> 7) & 0xFF);
                        }
                    }
                }
            }
        }
    } else if (h.format == (uint32_t) FMT_HUFPAIR) {
        for (uint64_t u = 0; u < h.n_a; u++) {
            const uint32_t ub = at<uint32_t>(e.aux, h.off_b + 4 * u);
            for (int l = 0; l < 32; l++) {
                bits bs{e.aux, ub + at<uint16_t>(e.aux, h.off_c + 2 * (u * 32 + l)), h.off_d};
                // fragment order: unit u = slab u % C of band u / C; the lane's pairs of the slab tile by tile, in
                // register order
                const int64_t band = fo ? (int64_t) u / C : 0, kt0 = fo ? (int64_t) (u % C) * (CHUNK / 16) : 0;
                for (int64_t kt = kt0; fo && kt < kt0 + CHUNK / 16; kt++) {
                    for (int q = 0; q < 4; q++) {
                        int r, c;
                        fo_rc(l, q, r, c);
                        const uint32_t v = at<uint32_t>(e.aux, h.off_a + 4 * bs.peek(HP_LP));
                        bs.skip((int) (v & 0x7F));
                        ex[(band * 16 + r) * K + kt * 16 + c]     = (int) ((v >> 7) & 0xFF);
                        ex[(band * 16 + r) * K + kt * 16 + c + 1] = (int) ((v >> 23) & 0xFF);
                    }
                }
                for (int64_t o = 0; !fo && o < (int64_t) h.R * C; o++) {
                    const int64_t r = u * h.R + (cm ? o % h.R : o / C), c = cm ? o / h.R : o % C;
                    for (int s = 0; s < 16; s += 2) {
                        const uint32_t v = at<uint32_t>(e.aux, h.off_a + 4 * bs.peek(HP_LP));
                        bs.skip((int) (v & 0x7F));
                        ex[r * K + c * CHUNK + pos_of(l, s)]     = (int) ((v >> 7) & 0xFF);
                        ex[r * K + c * CHUNK + pos_of(l, s + 1)] = (int) ((v >> 23) & 0xFF);
                    }
                }
            }
        }
    }
    std::vector<uint16_t> w(rows * K);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t k = 0; k < K; k++) {
            w[r * K + k] = join(sm_at(e, K, r, k, cm, fo), ex[r * K + k]);
        }
    }
    if (h.format == (uint32_t) FMT_NIBBLE) { // escapes: column | sign|mantissa << 12 | exponent << 20
        for (int64_t r = 0; r < rows; r++) {
            for (uint32_t i = at<uint32_t>(e.aux, h.off_c + 4 * r); i < at<uint32_t>(e.aux, h.off_c + 4 * (r + 1)); i++) {
                const uint32_t x = at<uint32_t>(e.aux, h.off_d + 4 * i);
                w[r * K + (x & 0xFFF)] = join((uint8_t) ((x >> 12) & 0xFF), (int) (x >> 20));
            }
        }
    }
    return w;
}

// model-like BF16 weights (normal, scale 0.02, rare outliers) plus every special class at random positions
std::vector<uint16_t> model_like(int64_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<uint16_t> w(n);
    for (auto & v : w) {
        float f = nd(g);
        if (ud(g) < 1e-3f) {
            f *= 3000.0f; // exponents >= 128 (|w| >= 2): NIBBLE escapes
        }
        uint32_t b;
        std::memcpy(&b, &f, 4);
        v = (uint16_t) (b >> 16);
    }
    const uint16_t specials[] = {0x0000, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF, 0x3F80};
    std::uniform_int_distribution<int64_t> pos(0, n - 1);
    for (int rep = 0; rep < 4; rep++) {
        for (uint16_t s : specials) {
            w[pos(g)] = s;
        }
    }
    return w;
}

bool throws(const std::function<void()> & f) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

void round_trip(const char * name, const std::vector<uint16_t> & w, int64_t rows, int64_t K, int fmt, int R, bool cm,
                int want_fmt, bool fo = false) {
    const encoded e = encode(w.data(), rows, K, fmt, R, cm, fo);
    const aux_header h = at<aux_header>(e.aux, 0);
    CHECK(h.magic == MAGIC && h.K == K && h.rows == (uint64_t) rows, "%s: header", name);
    CHECK(e.format == want_fmt && (int) h.format == want_fmt, "%s: format %d, want %d", name, e.format, want_fmt);
    CHECK(((h.flags & FLAG_CHUNK_MAJOR) != 0) == (cm && want_fmt == FMT_HUFPAIR), "%s: chunk-major flag", name);
    CHECK(((h.flags & FLAG_FRAG_ORDER) != 0) == (fo && want_fmt == FMT_HUFPAIR), "%s: fragment-order flag", name);
    const std::vector<uint16_t> d = decode(e, rows, K);
    int64_t bad = 0, first = -1;
    for (int64_t i = 0; i < rows * K; i++) {
        if (d[i] != w[i]) {
            bad++;
            first = first < 0 ? i : first;
        }
    }
    CHECK(bad == 0, "%s: %lld of %lld weights differ, first at %lld (%04x != %04x)", name, (long long) bad,
          (long long) (rows * K), (long long) first, first < 0 ? 0 : d[first], first < 0 ? 0 : w[first]);
}

// rows x K weights whose (even, odd) exponent pairs take exactly n distinct values
std::vector<uint16_t> with_pairs(int64_t rows, int64_t K, int n) {
    std::vector<uint16_t> w(rows * K, 0);
    for (int64_t i = 0; i < rows * K / 2; i++) {
        const int p = (int) (i % n);
        w[2 * i] = (uint16_t) ((p & 0xFF) << 7);
        w[2 * i + 1] = (uint16_t) ((p >> 8) << 7);
    }
    return w;
}

} // namespace

int main() {
    // bitwise round trip, every format and layout the kernels support
    for (int64_t K : {512, 2048, 4096}) {
        const int64_t rows = 32;
        const std::vector<uint16_t> w = model_like(rows * K, (uint32_t) K);
        const std::string k = "K=" + std::to_string(K);
        round_trip(("nibble " + k).c_str(), w, rows, K, FMT_NIBBLE, 0, false, FMT_NIBBLE);
        for (int R : {1, 2, 4, 8, 16}) {
            const std::string r = k + " R=" + std::to_string(R);
            round_trip(("huffman " + r).c_str(), w, rows, K, FMT_HUFFMAN, R, false, FMT_HUFFMAN);
            round_trip(("hufpair " + r).c_str(), w, rows, K, FMT_HUFPAIR, R, false, FMT_HUFPAIR);
            if (K > CHUNK && R >= 4) {
                round_trip(("hufpair cm " + r).c_str(), w, rows, K, FMT_HUFPAIR, R, true, FMT_HUFPAIR);
            }
        }
    }

    // fragment order: bands of 16 rows, units of 512-column slabs (K = 512: one per band)
    for (int64_t K : {(int64_t) CHUNK, (int64_t) 2048, (int64_t) K_MAX}) {
        for (int64_t rows : {16, 48}) {
            const std::vector<uint16_t> w = model_like(rows * K, (uint32_t) (rows + K));
            const encoded e = encode(w.data(), rows, K, FMT_HUFPAIR, 16, false, true);
            CHECK(at<aux_header>(e.aux, 0).n_a == (uint64_t) (rows / 16 * (K / CHUNK)), "fo K=%lld: one unit per slab", (long long) K);
            round_trip(("hufpair fo K=" + std::to_string(K) + " rows=" + std::to_string(rows)).c_str(), w, rows, K,
                       FMT_HUFPAIR, 16, false, FMT_HUFPAIR, true);
        }
    }

    // HUFPAIR falls back to HUFFMAN (still lossless) above HP_MAX_PAIRS distinct exponent pairs, not at it
    for (int n : {HP_MAX_PAIRS - 1, HP_MAX_PAIRS, HP_MAX_PAIRS + 1}) {
        const int want = n <= HP_MAX_PAIRS ? FMT_HUFPAIR : FMT_HUFFMAN;
        const std::vector<uint16_t> w = with_pairs(16, 1024, n);
        round_trip(("pairs=" + std::to_string(n)).c_str(), w, 16, 1024, FMT_HUFPAIR, 4, false, want);
        round_trip(("pairs cm=" + std::to_string(n)).c_str(), w, 16, 1024, FMT_HUFPAIR, 4, true, want);
        round_trip(("pairs fo=" + std::to_string(n)).c_str(), with_pairs(32, CHUNK, n), 32, CHUNK, FMT_HUFPAIR, 16,
                   false, want, true);
    }

    // argument boundaries
    const std::vector<uint16_t> z(64 * (K_MAX + CHUNK), 0);
    auto enc = [&](int64_t rows, int64_t K, int fmt, int R, bool cm) {
        return [=, &z]() { encode(z.data(), rows, K, fmt, R, cm); };
    };
    for (int64_t K : {(int64_t) 0, (int64_t) CHUNK - 1, (int64_t) CHUNK + 1, (int64_t) K_MAX - 1, (int64_t) K_MAX + 1,
                      (int64_t) K_MAX + CHUNK}) {
        CHECK(throws(enc(16, K, FMT_HUFPAIR, 4, false)), "K=%lld accepted", (long long) K);
    }
    for (int64_t K : {(int64_t) CHUNK, (int64_t) K_MAX}) {
        CHECK(!throws(enc(16, K, FMT_HUFPAIR, 4, false)), "K=%lld rejected", (long long) K);
    }
    CHECK(throws(enc(0, 512, FMT_NIBBLE, 0, false)), "rows=0 accepted");
    CHECK(!throws(enc(1, 512, FMT_NIBBLE, 0, false)), "rows=1 rejected");
    CHECK(throws(enc(16, 512, FMT_NIBBLE, 4, false)), "NIBBLE with R accepted");
    CHECK(throws(enc(16, 1024, FMT_NIBBLE, 0, true)), "NIBBLE chunk-major accepted");
    for (int R : {0, 3, 32}) {
        CHECK(throws(enc(96, 512, FMT_HUFFMAN, R, false)), "R=%d accepted", R);
    }
    CHECK(throws(enc(15, 512, FMT_HUFPAIR, 4, false)), "rows %% R != 0 accepted (below)");
    CHECK(!throws(enc(16, 512, FMT_HUFPAIR, 4, false)), "rows %% R == 0 rejected");
    CHECK(throws(enc(17, 512, FMT_HUFPAIR, 4, false)), "rows %% R != 0 accepted (above)");
    CHECK(throws(enc(16, 1024, FMT_HUFFMAN, 4, true)), "HUFFMAN chunk-major accepted");
    CHECK(throws(enc(16, 512, FMT_HUFPAIR, 4, true)), "chunk-major at K = 512 accepted");
    CHECK(!throws(enc(16, 1024, FMT_HUFPAIR, 4, true)), "chunk-major at K = 1024 rejected");
    CHECK(throws(enc(16, 1024, FMT_HUFPAIR, 2, true)), "chunk-major with R = 2 accepted");
    CHECK(throws(enc(16, 512, 0, 4, false)) && throws(enc(16, 512, 4, 4, false)), "unknown format accepted");
    auto enc_fo = [&](int64_t rows, int64_t K, int fmt, int R, bool cm) {
        return [=, &z]() { encode(z.data(), rows, K, fmt, R, cm, true); };
    };
    CHECK(!throws(enc_fo(16, 512, FMT_HUFPAIR, 16, false)), "fragment order at K = 512, R = 16 rejected");
    CHECK(!throws(enc_fo(16, 1024, FMT_HUFPAIR, 16, false)), "fragment order at K = 1024 (two slabs) rejected");
    CHECK(throws(enc_fo(16, 512, FMT_HUFPAIR, 8, false)), "fragment order with R = 8 accepted");
    CHECK(throws(enc_fo(32, 1024, FMT_HUFPAIR, 16, true)), "fragment order with chunk-major accepted");
    CHECK(throws(enc_fo(16, 512, FMT_HUFFMAN, 16, false)), "HUFFMAN fragment order accepted");
    CHECK(throws(enc_fo(16, 512, FMT_NIBBLE, 0, false)), "NIBBLE fragment order accepted");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("bf16n encoder host tests passed\n");
    return 0;
}
