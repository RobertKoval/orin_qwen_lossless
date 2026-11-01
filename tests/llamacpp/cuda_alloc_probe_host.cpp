// SPDX-License-Identifier: MIT
// External-ABI lifecycle tests. Link cuda_alloc_probe_fake.cpp, never a CUDA runtime/SDK/device.
#include "cuda_alloc_probe.h"
#include <dlfcn.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
extern "C" int cudaMalloc(void **, size_t);
extern "C" int cudaFree(void *);
extern "C" void s5_fake_fail_next_free(void *);
extern "C" void s5_fake_reenter_next_malloc();
extern "C" void s5_fake_reuse_next_two_allocations();
extern "C" void s5_fake_zero_returns_pointer(bool);
extern "C" int s5_fake_bypass_alloc_free();

int main(int argc, char ** argv) {
    assert(argc == 2);
    auto snapshot = reinterpret_cast<int (*)(s5_cuda_stats *, size_t)>(dlsym(RTLD_DEFAULT, "s5_cuda_snapshot"));
    auto report = reinterpret_cast<void (*)(const char *)>(dlsym(RTLD_DEFAULT, "s5_cuda_report"));
    auto checkpoint = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "s5_cuda_checkpoint"));
    auto since = reinterpret_cast<int (*)(uint64_t, s5_cuda_generation_stats *, size_t)>(dlsym(RTLD_DEFAULT, "s5_cuda_since"));
    assert(snapshot && report && checkpoint && since);
    auto current = [&] { s5_cuda_stats s{}; assert(snapshot(&s, sizeof(s)) == 0); return s; };
    s5_cuda_stats scratch{};
    assert(snapshot(nullptr, sizeof(scratch)) == -1);
    assert(snapshot(&scratch, sizeof(scratch) - 1) == -1);
    assert(snapshot(&scratch, sizeof(scratch) + 1) == -1);
    assert(current().malloc_calls == 0);
    s5_cuda_generation_stats generations{};
    const uint64_t start = checkpoint();
    assert(since(start, nullptr, sizeof(generations)) == -1);
    assert(since(start, &generations, sizeof(generations) - 1) == -1);
    assert(since(start, &generations, sizeof(generations) + 1) == -1);
    assert(since(start + 1, &generations, sizeof(generations)) == -1);
    assert(since(start, &generations, sizeof(generations)) == 0 && generations.new_live_allocations == 0);
    if (!std::strcmp(argv[1], "bypass")) {
        assert(s5_fake_bypass_alloc_free() == 0);
        const auto after = current();
        assert(since(start, &generations, sizeof(generations)) == 0 && generations.new_live_allocations == 0);
        // A loaded probe plus zero new live IDs is insufficient: expected external calls must be observed.
        if (after.malloc_success == 0 || after.free_success == 0) {
            std::puts("EXPECTED REJECTION: no allocation/free interception coverage"); return 3;
        }
        return 0; // The shell requires rejection; a false pass is a failed negative control.
    } else if (!std::strcmp(argv[1], "generations")) {
        s5_fake_reuse_next_two_allocations();
        void * old_owner = nullptr; assert(cudaMalloc(&old_owner, 97) == 0);
        const auto before = current(); const uint64_t cp = checkpoint();
        assert(cudaFree(old_owner) == 0);
        void * new_owner = nullptr; assert(cudaMalloc(&new_owner, 97) == 0);
        assert(new_owner == old_owner); // Deterministic external pointer reuse, not allocator luck.
        const auto after = current();
        assert(after.live_bytes == before.live_bytes && after.live_allocations == before.live_allocations);
        assert(since(cp, &generations, sizeof(generations)) == 0);
        assert(generations.new_live_allocations == 1 && generations.new_live_bytes == 97);
        assert(generations.last_id > cp); report("equal-bytes-but-new-generation-still-live");
        assert(since(generations.last_id, &generations, sizeof(generations)) == 0 && generations.new_live_allocations == 0);
        assert(cudaFree(new_owner) == 0);
        assert(since(cp, &generations, sizeof(generations)) == 0 && generations.new_live_allocations == 0);
    } else if (!std::strcmp(argv[1], "faults")) {
        void * p = nullptr;
        assert(cudaMalloc(&p, std::numeric_limits<size_t>::max()) == 2);
        auto s = current(); assert(s.malloc_calls == 1 && s.malloc_success == 0 && s.live_bytes == 0);
        for (size_t n : {size_t(0), size_t(1), size_t(2)}) {
            assert(cudaMalloc(&p, n) == 0);
            s = current(); assert(s.live_bytes == n && s.live_allocations == size_t(n != 0));
            assert(cudaFree(p) == 0); assert(current().live_bytes == 0);
        }
        assert(cudaMalloc(&p, 12345) == 0); s5_fake_fail_next_free(p);
        assert(cudaFree(p) == 11); s = current();
        assert(s.live_bytes == 12345 && s.live_allocations == 1 && s.freed_bytes == 3);
        assert(cudaFree(p) == 0); s = current();
        assert(s.live_bytes == 0 && s.live_allocations == 0 && s.allocated_bytes == s.freed_bytes);
        assert(s.unknown_frees == 0 && s.tracking_errors == 0);
    } else if (!std::strcmp(argv[1], "zero-owner")) {
        s5_fake_zero_returns_pointer(true);
        void * p = nullptr; assert(cudaMalloc(&p, 0) == 0 && p);
        auto s = current(); assert(s.live_bytes == 0 && s.live_allocations == 1);
        assert(since(start, &generations, sizeof(generations)) == 0);
        assert(generations.new_live_bytes == 0 && generations.new_live_allocations == 1);
        assert(cudaFree(p) == 0);
        assert(since(start, &generations, sizeof(generations)) == 0 && generations.new_live_allocations == 0);
    } else if (!std::strcmp(argv[1], "unknown")) {
        void * p = std::malloc(97); assert(p); s5_fake_fail_next_free(p);
        assert(cudaFree(p) == 11);
        auto s = current(); assert(s.unknown_frees == 0 && s.free_success == 0);
        assert(cudaFree(p) == 0); s = current();
        assert(s.unknown_frees == 1 && s.live_bytes == 0 && s.live_allocations == 0);
        const uint64_t cp = checkpoint();
        assert(since(cp, &generations, sizeof(generations)) == 0 && generations.unknown_frees == 1);
    } else if (!std::strcmp(argv[1], "reentry")) {
        s5_fake_reenter_next_malloc();
        void * p = nullptr; assert(cudaMalloc(&p, 97) == 0);
        auto s = current(); assert(s.live_bytes == 97 && s.live_allocations == 1 && s.tracking_errors == 2);
        assert(cudaFree(p) == 0); s = current();
        assert(s.live_allocations == 0 && s.live_bytes == 0 && s.unknown_frees == 0 && s.tracking_errors == 2);
        const uint64_t cp = checkpoint();
        assert(since(cp, &generations, sizeof(generations)) == 0);
        assert(generations.new_live_allocations == 0 && generations.tracking_errors == 2);
    } else if (!std::strcmp(argv[1], "threads")) {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) threads.emplace_back([] {
            for (int i = 0; i < 256; ++i) {
                void * p = nullptr; assert(cudaMalloc(&p, 1 + i) == 0); assert(cudaFree(p) == 0);
            }
        });
        for (auto & t : threads) t.join();
        auto s = current();
        assert(s.malloc_success == 2048 && s.free_success == 2048);
        assert(s.live_bytes == 0 && s.live_allocations == 0 && s.allocated_bytes == s.freed_bytes);
        assert(s.unknown_frees == 0 && s.tracking_errors == 0);
    } else return 2;
    report(argv[1]); std::printf("PASS: CPU-only interposer %s behavior\n", argv[1]);
}
