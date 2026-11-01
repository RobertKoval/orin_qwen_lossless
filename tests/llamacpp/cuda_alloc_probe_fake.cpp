// SPDX-License-Identifier: MIT
// CPU-only fake external CUDA ABI. Never link/preload this library into a real GPU check.
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <mutex>

namespace {
std::mutex mutex;
void * fail_free = nullptr;
bool nested_malloc = false, zero_pointer = false;
unsigned reuse_remaining = 0;
bool reuse_live = false;
alignas(std::max_align_t) unsigned char reuse_storage[256];
}
extern "C" void s5_fake_fail_next_free(void * p) { std::lock_guard<std::mutex> lock(mutex); fail_free = p; }
extern "C" void s5_fake_reenter_next_malloc() { std::lock_guard<std::mutex> lock(mutex); nested_malloc = true; }
extern "C" void s5_fake_reuse_next_two_allocations() {
    std::lock_guard<std::mutex> lock(mutex); if (reuse_live) std::abort(); reuse_remaining = 2;
}
extern "C" void s5_fake_zero_returns_pointer(bool enabled) { std::lock_guard<std::mutex> lock(mutex); zero_pointer = enabled; }
// Simulate an allocation family outside this observer's scope. No intercepted symbol is called.
extern "C" int s5_fake_bypass_alloc_free() {
    void * p = std::malloc(97); if (!p) return 2; std::free(p); return 0;
}
extern "C" int cudaMalloc(void ** p, size_t n) {
    if (!p) return 1;
    if (n == std::numeric_limits<size_t>::max()) { *p = nullptr; return 2; }
    bool nested, nonnull_zero;
    { std::lock_guard<std::mutex> lock(mutex);
      nested = nested_malloc; nested_malloc = false; nonnull_zero = zero_pointer; }
    if (nested) {
        // Resolve through the external loader to force same-thread wrapper reentry, not a local direct call.
        auto alloc = reinterpret_cast<int (*)(void **, size_t)>(dlsym(RTLD_DEFAULT, "cudaMalloc"));
        auto release = reinterpret_cast<int (*)(void *)>(dlsym(RTLD_DEFAULT, "cudaFree"));
        void * temporary = nullptr;
        if (!alloc || !release || alloc(&temporary, 17) != 0 || release(temporary) != 0) std::abort();
    }
    if (!n && !nonnull_zero) { *p = nullptr; return 0; }
    { std::lock_guard<std::mutex> lock(mutex);
      if (reuse_remaining && n <= sizeof(reuse_storage)) {
          if (reuse_live) return 2;
          --reuse_remaining; reuse_live = true; *p = reuse_storage; return 0;
      } }
    *p = std::malloc(n ? n : 1); return *p ? 0 : 2;
}
extern "C" int cudaFree(void * p) {
    { std::lock_guard<std::mutex> lock(mutex);
      if (p && p == fail_free) { fail_free = nullptr; return 11; }
      if (p == reuse_storage) { if (!reuse_live) return 1; reuse_live = false; return 0; } }
    std::free(p); return 0;
}
