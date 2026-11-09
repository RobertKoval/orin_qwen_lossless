// SPDX-License-Identifier: MIT
// Policy contracts: actual GGUF shapes, merged vLLM projections, and semantic boundaries (host-only).
#include "bf16n/policy.h"

#include <cstdio>
#include <initializer_list>
#include <limits>

using namespace bf16n;

namespace {
int failures = 0;

void expect(int64_t rows, int64_t K, int64_t n, policy_options opts, int format, int R, bool cm) {
    const auto c = choose(rows, K, n, opts);
    if (!c || c->format != format || c->R != R || c->chunk_major != cm) {
        std::fprintf(stderr, "choose(%lld,%lld,%lld,engine=%d,mode=%d,min=%lld,expert=%d): expected %d/%d/%d\n",
                     (long long) rows, (long long) K, (long long) n, (int) opts.engine, (int) opts.mode,
                     (long long) opts.dense_min, opts.expert, format, R, cm);
        ++failures;
    }
}

void reject(int64_t rows, int64_t K, int64_t n) {
    for (auto engine : {policy_engine::llama, policy_engine::vllm}) {
        for (auto mode : {policy_mode::hybrid, policy_mode::nibble, policy_mode::huffman}) {
            if (choose(rows, K, n, {engine, mode})) {
                std::fprintf(stderr, "accepted invalid shape %lld/%lld/%lld\n",
                             (long long) rows, (long long) K, (long long) n);
                ++failures;
            }
        }
    }
}
} // namespace

int main() {
    // All ten encodable shape groups from the actual 753-tensor GGUF (484 including MTP, 472 excluding it).
    struct shape { int64_t rows, K, n; int format, R; bool cm; };
    const shape gguf[] = {
        {2048, 512, 1, FMT_NIBBLE, 0, false}, {2048, 512, 256, FMT_HUFPAIR, 16, false},
        {32, 2048, 1, FMT_NIBBLE, 0, false}, {256, 2048, 1, FMT_NIBBLE, 0, false},
        {512, 2048, 1, FMT_NIBBLE, 0, false}, {512, 2048, 256, FMT_HUFPAIR, 8, true},
        {4096, 2048, 1, FMT_HUFPAIR, 8, false}, {8192, 2048, 1, FMT_HUFPAIR, 16, false},
        {248320, 2048, 1, FMT_HUFPAIR, 16, false}, {2048, 4096, 1, FMT_HUFPAIR, 4, false},
    };
    for (auto engine : {policy_engine::llama, policy_engine::vllm}) {
        for (const auto & s : gguf) {
            expect(s.rows, s.K, s.n, {engine, policy_mode::hybrid, 8ll << 20, s.n > 1}, s.format, s.R, s.cm);
        }
    }

    const policy_options llama{policy_engine::llama};
    const policy_options vllm{policy_engine::vllm};
    // Merged projections are NOT a reason to change llama's historical policy.
    expect(12288, 2048, 1, llama, FMT_HUFPAIR, 16, false);
    expect(12288, 2048, 1, vllm, FMT_HUFPAIR, 8, false);
    expect(9216, 2048, 1, llama, FMT_HUFPAIR, 16, false);
    expect(9216, 2048, 1, vllm, FMT_HUFPAIR, 2, false);
    // 512-unit occupancy boundary, including divisibility and wave tails.
    for (auto engine : {policy_engine::llama, policy_engine::vllm}) {
        const policy_options forced{engine, policy_mode::huffman};
        for (int r : {2, 4, 8, 16}) {
            expect(512 * r - 1, 512, 1, forced, FMT_HUFPAIR, 1, false);
            expect(512 * r, 512, 1, forced, FMT_HUFPAIR, r, false);
            expect(512 * r + 1, 512, 1, forced, FMT_HUFPAIR, 1, false);
        }
    }
    // Fragment order: vLLM's K = 512 experts with 16-row bands only; never for llama (its adapter has no such kernels).
    struct fo_case { int64_t rows, K, n; policy_engine engine; bool fo; };
    for (const fo_case & f : {fo_case{2048, 512, 256, policy_engine::vllm, true}, fo_case{2048, 512, 256, policy_engine::llama, false},
                              fo_case{2040, 512, 256, policy_engine::vllm, false}, fo_case{2048, 1024, 256, policy_engine::vllm, false},
                              fo_case{16384, 512, 1, policy_engine::vllm, false}}) {
        const auto c = choose(f.rows, f.K, f.n, {f.engine, policy_mode::hybrid, 8ll << 20, f.n > 1});
        if (!c || c->frag_order != f.fo) {
            std::fprintf(stderr, "choose(%lld,%lld,%lld,engine=%d): frag_order expected %d\n", (long long) f.rows,
                         (long long) f.K, (long long) f.n, (int) f.engine, f.fo);
            ++failures;
        }
    }
    expect(64, 2048, 1, vllm, FMT_NIBBLE, 0, false);
    expect(1024, 2048, 1, vllm, FMT_NIBBLE, 0, false);
    expect(8194, 1024, 1, vllm, FMT_HUFPAIR, 2, false);
    expect(9728, 2048, 1, vllm, FMT_HUFPAIR, 4, false);
    expect(8176, 2048, 1, vllm, FMT_HUFPAIR, 8, false);
    expect(8208, 2048, 1, vllm, FMT_HUFPAIR, 2, false);
    expect(8208, 2048, 1, llama, FMT_HUFPAIR, 16, false);
    // Candidate cost just below / exactly at / just above 95% of the incumbent:
    // 34 vs .95*36; 38 vs .95*40; 42 vs .95*44. Equality must accept the smaller R.
    expect(16388, 512, 1, vllm, FMT_HUFPAIR, 2, false);
    expect(18436, 512, 1, vllm, FMT_HUFPAIR, 2, false);
    expect(20484, 512, 1, vllm, FMT_HUFPAIR, 4, false);

    for (auto engine : {policy_engine::llama, policy_engine::vllm}) {
        for (int64_t rows : {16383, 16384, 16385}) {
            const auto c = choose(rows, 512, 1, {engine});
            if (!c || c->format != (rows < 16384 ? FMT_NIBBLE : FMT_HUFPAIR)) { ++failures; }
        }
        // Expert R=16 threshold and divisibility, with R=8,4,1 fallback (never R=2).
        // K=512 keeps R=1 encodable without enshrining the deferred T-055 chunk-major mismatch.
        const policy_options expert{engine, policy_mode::hybrid, 8ll << 20, true};
        for (const auto & s : {shape{2032, 512, 2, FMT_HUFPAIR, 8, false},
                              shape{2040, 512, 2, FMT_HUFPAIR, 8, false},
                              shape{2047, 512, 2, FMT_HUFPAIR, 1, false},
                              shape{2048, 512, 2, FMT_HUFPAIR, 16, false},
                              shape{2049, 512, 2, FMT_HUFPAIR, 1, false},
                              shape{2064, 512, 2, FMT_HUFPAIR, 16, false},
                              shape{12, 512, 2, FMT_HUFPAIR, 4, false},
                              shape{6, 512, 2, FMT_HUFPAIR, 1, false}}) {
            expect(s.rows, s.K, s.n, expert, s.format, s.R, s.cm);
        }
        expect(512, 512, 2, expert, FMT_HUFPAIR, 8, false);
        expect(512, 1024, 2, expert, FMT_HUFPAIR, 8, true);
        expect(512, 4096, 2, expert, FMT_HUFPAIR, 8, true);
        // Forced nibble overrides even expert selection; forced huffman tries pairs even for tiny dense weights.
        expect(512, 2048, 2, {engine, policy_mode::nibble, 0, true}, FMT_NIBBLE, 0, false);
        expect(1, 512, 1, {engine, policy_mode::huffman}, FMT_HUFPAIR, 1, false);
    }
    // llama's name-based expert classification and configurable total-element threshold are independent of n.
    expect(512, 2048, 2, llama, FMT_NIBBLE, 0, false);
    expect(512, 2048, 2, vllm, FMT_HUFPAIR, 8, true);
    expect(512, 2048, 1, {policy_engine::llama, policy_mode::hybrid, 8ll << 20, true}, FMT_HUFPAIR, 1, false);
    for (int64_t min : {1023, 1024, 1025}) {
        expect(1, 512, 2, {policy_engine::llama, policy_mode::hybrid, min},
               min <= 1024 ? FMT_HUFPAIR : FMT_NIBBLE, min <= 1024 ? 1 : 0, false);
    }
    for (int64_t min : {-1, 0, 1}) {
        expect(1, 512, 1, {policy_engine::llama, policy_mode::hybrid, min}, FMT_HUFPAIR, 1, false);
    }
    const int64_t max = std::numeric_limits<int64_t>::max();
    expect(max - 3, 4096, max, {policy_engine::llama, policy_mode::hybrid, max}, FMT_HUFPAIR, 4, true);
    expect(max, 4096, 1, vllm, FMT_HUFPAIR, 1, false);
    expect(1, 512, 1, {policy_engine::llama, policy_mode::hybrid, max}, FMT_NIBBLE, 0, false);

    for (int64_t K : {-1, 0, 1, 256, 511, 513, 1000, 4095, 4097, 4608}) { reject(2048, K, 1); reject(2048, K, 2); }
    for (int64_t rows : {-1, 0}) { reject(rows, 512, 1); }
    for (int64_t n : {-1, 0}) { reject(1, 512, n); }
    expect(1, 512, 1, vllm, FMT_NIBBLE, 0, false);
    expect(1, 4096, 1, vllm, FMT_NIBBLE, 0, false);
    return failures ? 1 : 0;
}
