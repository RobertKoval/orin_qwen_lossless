// SPDX-License-Identifier: MIT
// Positive/negative lifecycle control. Host tests link the fake ABI; GPU owners may explicitly link shared cudart.
// No CUDA headers are needed. This is not the ggml conversion/reclamation or numeric test.
#include "cuda_alloc_probe.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

extern "C" int cudaMalloc(void **, size_t);
extern "C" int cudaFree(void *);

int main() {
    auto snapshot = reinterpret_cast<int (*)(s5_cuda_stats *, size_t)>(dlsym(RTLD_DEFAULT, "s5_cuda_snapshot"));
    auto report = reinterpret_cast<void (*)(const char *)>(dlsym(RTLD_DEFAULT, "s5_cuda_report"));
    auto checkpoint = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "s5_cuda_checkpoint"));
    auto since = reinterpret_cast<int (*)(uint64_t, s5_cuda_generation_stats *, size_t)>(dlsym(RTLD_DEFAULT, "s5_cuda_since"));
    if (!snapshot || !report || !checkpoint || !since) { std::fprintf(stderr, "FAIL: probe not preloaded\n"); return 2; }
    auto check = [](bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); } };
    check(cudaFree(nullptr) == 0, "CUDA context warmup");
    s5_cuda_stats before{}, after{};
    check(snapshot(&before, sizeof(before)) == 0, "before snapshot"); report("smoke-warmed-before");
    const uint64_t generation = checkpoint();
    s5_cuda_generation_stats newly_live{};
    for (int i = 0; i < 64; ++i) {
        void * p = nullptr;
        check(cudaMalloc(&p, (1u << 20) + size_t(i)) == 0, "cycle allocation");
        check(cudaFree(p) == 0, "cycle free");
    }
    check(snapshot(&after, sizeof(after)) == 0, "after snapshot"); report("smoke-after-64-balanced-cycles");
    check(after.live_bytes == before.live_bytes && after.live_allocations == before.live_allocations, "balanced cycles liveness");
    check(since(generation, &newly_live, sizeof(newly_live)) == 0 && newly_live.new_live_allocations == 0,
          "no new allocation generations retained by balanced cycles");
    check(after.malloc_success - before.malloc_success == 64 && after.free_success - before.free_success == 64, "interception coverage");
    void * retained = nullptr;
    check(cudaMalloc(&retained, 12345) == 0, "intentional retained allocation");
    check(snapshot(&after, sizeof(after)) == 0, "retained snapshot"); report("smoke-intentionally-retained-12345");
    check(after.live_bytes == before.live_bytes + 12345 && after.live_allocations == before.live_allocations + 1,
          "retained allocation observed exactly");
    check(since(generation, &newly_live, sizeof(newly_live)) == 0 && newly_live.new_live_allocations == 1 &&
          newly_live.new_live_bytes == 12345, "retained allocation generation observed exactly");
    check(cudaFree(retained) == 0, "retained allocation cleanup");
    check(snapshot(&after, sizeof(after)) == 0, "final snapshot"); report("smoke-cleaned");
    check(after.live_bytes == before.live_bytes && after.unknown_frees == 0 && after.tracking_errors == 0, "clean final accounting");
    check(since(generation, &newly_live, sizeof(newly_live)) == 0 && newly_live.new_live_allocations == 0,
          "all new allocation generations reclaimed");
    std::puts("PASS: observed 64 balanced cycles and intentional 12345-byte retention/cleanup");
}
