// SPDX-License-Identifier: MIT
// Test-only CUDA allocation-generation oracle. No production/private ggml state is inspected.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"
#include "cuda_alloc_probe.h"
#include "../../bench/xengine/size_limit.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

template <typename T> static T symbol(const char * name) {
    auto fn = reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
    require(fn != nullptr, (std::string("allocation observer missing: ") + name).c_str());
    return fn;
}

struct Probe {
    decltype(&s5_cuda_snapshot) snapshot = symbol<decltype(snapshot)>("s5_cuda_snapshot");
    decltype(&s5_cuda_checkpoint) checkpoint = symbol<decltype(checkpoint)>("s5_cuda_checkpoint");
    decltype(&s5_cuda_since) since = symbol<decltype(since)>("s5_cuda_since");
    decltype(&s5_cuda_report) report = symbol<decltype(report)>("s5_cuda_report");

    s5_cuda_stats stats() const {
        s5_cuda_stats result{};
        require(snapshot(&result, sizeof(result)) == 0, "observer snapshot ABI mismatch");
        require(result.tracking_errors == 0 && result.unknown_frees == 0, "observer bookkeeping/coverage incomplete");
        return result;
    }

    void reclaimed(uint64_t start, const char * phase) const {
        s5_cuda_generation_stats result{};
        require(since(start, &result, sizeof(result)) == 0, "observer generation ABI mismatch");
        require(result.checkpoint == start && result.last_id > start, "no observed allocation generation in cycle");
        require(result.tracking_errors == 0 && result.unknown_frees == 0, "observer interception incomplete");
        if (result.new_live_allocations != 0 || result.new_live_bytes != 0) {
            report(phase);
            throw std::runtime_error(std::string(phase) + ": newly allocated CUDA generations remain live after owner free");
        }
    }

    void owner_reclaimed(uint64_t start, uint64_t end, const char * phase) const {
        s5_cuda_generation_stats first{}, later{};
        require(end > start && since(start, &first, sizeof(first)) == 0 && since(end, &later, sizeof(later)) == 0,
                "owner generation interval unavailable");
        require(first.last_id == later.last_id && first.last_id >= end,
                "observer generation state changed between interval snapshots");
        require(first.unknown_frees == 0 && first.tracking_errors == 0 &&
                later.unknown_frees == 0 && later.tracking_errors == 0, "owner interception incomplete");
        // ID ranges are disjoint: equality excludes any surviving allocation in (start,end].
        // Later GET_ROWS runtime caches belong to the backend, not the tensor buffer being freed.
        if (first.new_live_allocations != later.new_live_allocations || first.new_live_bytes != later.new_live_bytes) {
            report(phase);
            throw std::runtime_error(std::string(phase) + ": owner allocation generations remain live");
        }
    }
};

static size_t memory_cap() {
    return bf16n_test::memory_budget("BF16N_TEST_MAX_DEVICE_BYTES", size_t(1) << 30);
}

using Convert = bool (*)(ggml_tensor *, const void *, int);

struct Owner {
    // Destruction order is buffer before context; the CUDA backend owns the encoded aux through this buffer.
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * weight = nullptr;
    ggml_tensor * indices = nullptr;
    ggml_tensor * rows = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<uint16_t> host;
    int K;
    int N;

    Owner(ggml_backend_t backend, const Probe & probe, Convert convert,
          int fmt, int k, int n, int experts, unsigned seed, bool gather) : K(k), N(n) {
        host.resize(size_t(K) * N * experts);
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint16_t(((118 + (i * (seed * 2 + 1) + seed) % 19) << 7) | ((i * 13 + seed) & 127));
        }
        ctx.reset(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        require(bool(ctx), "owner context allocation failed");
        weight = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16N, K, N, experts);
        ggml_set_name(weight, gather ? "reclamation-owner-B" : "reclamation-owner-A");
        if (gather) {
            require(experts == 1 && N > 1, "GET_ROWS owner must be a dense matrix with at least two rows");
            indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3);
            rows = ggml_get_rows(ctx.get(), weight, indices);
            graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, rows);
            require(ggml_backend_supports_op(backend, rows), "GET_ROWS unsupported for surviving owner");
        }
        auto buft = ggml_backend_get_default_buffer_type(backend);
        size_t tensors = 0;
        for (auto t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!t->view_src) { tensors += ggml_backend_buft_get_alloc_size(buft, t) + ggml_backend_buft_get_alignment(buft); }
        }
        // Budget estimates only: worst-case encoded aux + 64 MiB roundtrip + 256 MiB runtime reserve.
        // Global UMA availability is an admission input, NEVER the reclamation oracle.
        const size_t elements = host.size();
        const size_t aux = (fmt == 1 ? (elements * 9 + 1) / 2 : elements * 2) + size_t(N) * experts * 96 + 32768;
        const size_t estimate = tensors + aux + (size_t(320) << 20);
        const auto observed = probe.stats();
        size_t available = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &available, &total);
        const size_t cap = memory_cap();
        require(observed.live_bytes <= cap && estimate <= cap - observed.live_bytes && estimate <= available,
                "reclamation allocation exceeds admission cap/availability");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer) && weight->buffer, "owner tensor buffer allocation failed");
        require(ggml_backend_buffer_get_size(buffer.get()) <= tensors, "owner allocation exceeded estimate");
        require(convert(weight, host.data(), fmt), "owner conversion/bitwise roundtrip failed");
        if (gather) {
            const int32_t ids[] = {0, 1, N - 1};
            ggml_backend_tensor_set(indices, ids, 0, sizeof(ids));
        }
        ggml_backend_synchronize(backend);
    }

    void check_rows(ggml_backend_t backend) const {
        require(graph != nullptr, "owner has no row graph");
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "surviving owner graph failed");
        require(rows->type == GGML_TYPE_F32 && ggml_nelements(rows) == int64_t(K) * 3, "invalid gathered output shape/type");
        std::vector<float> actual(size_t(K) * 3);
        ggml_backend_tensor_get(rows, actual.data(), 0, actual.size() * sizeof(float));
        const int ids[] = {0, 1, N - 1};
        for (int r = 0; r < 3; ++r) for (int k = 0; k < K; ++k) {
            const uint32_t expected = uint32_t(host[size_t(ids[r]) * K + k]) << 16;
            const float value = actual[size_t(r) * K + k];
            require(std::isfinite(value) && memcmp(&value, &expected, 4) == 0,
                    "owner B's exact rows changed after owner A was freed");
        }
    }
};

static void conversion_cycles(ggml_backend_t backend, const Probe & probe, Convert convert, int fmt) {
    auto cycle = [&]() {
        { Owner owner(backend, probe, convert, fmt, 2048, 512, 1, 0, false); }
        ggml_backend_synchronize(backend);
    };
    for (int i = 0; i < 4; ++i) { cycle(); }
    const auto before = probe.stats();
    const auto phase = probe.checkpoint();
    for (int i = 0; i < 64; ++i) {
        const auto start = probe.checkpoint();
        cycle();
        probe.reclaimed(start, "conversion-cycle");
    }
    const auto after = probe.stats();
    probe.reclaimed(phase, "conversion-phase");
    require(after.malloc_success > before.malloc_success && after.free_success > before.free_success, "no observed CUDA API traffic");
    require(after.malloc_calls - after.malloc_success == before.malloc_calls - before.malloc_success &&
            after.free_calls - after.free_success == before.free_calls - before.free_success,
            "CUDA allocation/free failed during conversion phase");
    printf("RECLAIMED fmt=%d cycles=64 malloc=%llu free=%llu new_live=0 bytes=0\n", fmt,
           (unsigned long long) (after.malloc_success - before.malloc_success),
           (unsigned long long) (after.free_success - before.free_success));
}

static void two_owners(ggml_backend_t backend, const Probe & probe, Convert convert, int fmt) {
    // Warm the public GET_ROWS path before generation checkpoints; no cuBLAS/model/VMM workload.
    { Owner warm(backend, probe, convert, fmt, 512, 8192, 1, 11, true); warm.check_rows(backend); }
    const auto b_start = probe.checkpoint();
    uint64_t b_end = b_start;
    {
        Owner b(backend, probe, convert, fmt, 512, 8192, 1, 11, true); // HUFPAIR: R16, row-major.
        b_end = probe.checkpoint();
        b.check_rows(backend);
        for (int repeat = 0; repeat < 4; ++repeat) {
            const auto before_a = probe.stats();
            const auto a_start = probe.checkpoint();
            { Owner a(backend, probe, convert, fmt, 2048, 512, 2, unsigned(repeat + 1), false); } // HUFPAIR: R8, chunk-major.
            ggml_backend_synchronize(backend);
            probe.reclaimed(a_start, "two-owners-A-freed");
            const auto after_a = probe.stats();
            // With no new live IDs, equal cardinality preserves the entire old-ID set, not just byte totals.
            // Check BEFORE reading B: accidentally freed memory might otherwise still appear readable.
            require(after_a.live_allocations == before_a.live_allocations && after_a.live_bytes == before_a.live_bytes,
                    "freeing A changed pre-existing CUDA owners; B must not be read");
            b.check_rows(backend);
        }
    }
    ggml_backend_synchronize(backend);
    probe.owner_reclaimed(b_start, b_end, "two-owners-B-freed");
    printf("EXACT OWNERS fmt=%d: four A frees reclaim every new generation; B rows 0/1/last unchanged; B reclaimed\n", fmt);
}

int main(int argc, char ** argv) try {
    require(argc == 1 || argc == 2, "usage: reclamation [CUDA_DEVICE_ORDINAL=0]");
    (void) memory_cap(); // Reject invalid limits before loading the observer or initializing CUDA.
    int device = 0;
    if (argc == 2) {
        char * end = nullptr;
        const long parsed = strtol(argv[1], &end, 10);
        require(end != argv[1] && !*end && parsed >= 0 && parsed <= 65535, "invalid CUDA device ordinal");
        device = int(parsed);
    }
    const Probe probe; // Fail before CUDA initialization if the observer is absent.
    ggml_backend_ptr backend(ggml_backend_cuda_init(device));
    require(bool(backend), "CUDA backend unavailable");
    const auto convert = reinterpret_cast<Convert>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get())), "ggml_backend_cuda_bf16n_convert"));
    require(convert != nullptr, "BF16N conversion registry export unavailable");
    for (int fmt : {1, 2}) {
        conversion_cycles(backend.get(), probe, convert, fmt);
        two_owners(backend.get(), probe, convert, fmt);
    }
    backend.reset();
    probe.stats();
    probe.report("reclamation-complete");
    puts("PASS: exact CUDA generation reclamation and independent owner survival (no UMA leak oracle)");
    return 0;
} catch (const std::exception & e) {
    if (auto report = reinterpret_cast<decltype(&s5_cuda_report)>(dlsym(RTLD_DEFAULT, "s5_cuda_report"))) {
        report("reclamation-failed");
    }
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
