// SPDX-License-Identifier: MIT
// Test-private cudaMalloc/cudaFree requested aux/scratch ownership observer, NOT a VMM/physical-memory cap.
// Build on Linux: c++ -std=c++17 -O1 -g -fPIC -shared -pthread cuda_alloc_probe.cpp -ldl -o probe.so
// LD_PRELOAD the shared object; no CUDA SDK or CUDA library linkage is required to build it.
#if !defined(__linux__)
#error "This observer requires the Linux LD_PRELOAD/RTLD_NEXT loader contract."
#endif
#include "cuda_alloc_probe.h"
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {
using malloc_fn = int (*)(void **, size_t); // cudaError_t is an int in the CUDA runtime ABI.
using free_fn = int (*)(void *);
struct allocation { size_t bytes; uint64_t id; void * frames[16]; int depth; };
struct state {
    // Serialize the real calls and bookkeeping together so pointer reuse cannot race a successful free.
    // Recursive only to detect/report unexpected re-entry rather than deadlocking silently.
    std::recursive_mutex mutex;
    std::unordered_map<void *, allocation> live;
    s5_cuda_stats stats{};
    malloc_fn alloc = nullptr;
    free_fn release = nullptr;
    uint64_t next_id = 0;
    bool trace = true;
    bool stacks = true;
};
state & get() {
    // Deliberate host-only process-lifetime storage: CUDA library destructors may call us after our exit report.
    static state * s = new state;
    return *s;
}
thread_local unsigned depth = 0;
struct entry { entry() { ++depth; } ~entry() { --depth; } };
void resolve(state & s) {
    if (s.alloc && s.release) return;
    const char * trace = std::getenv("S5_CUDA_TRACE");
    const char * stacks = std::getenv("S5_CUDA_STACKS");
    s.trace = !trace || trace[0] != '0'; s.stacks = !stacks || stacks[0] != '0';
    s.alloc = reinterpret_cast<malloc_fn>(dlsym(RTLD_NEXT, "cudaMalloc"));
    s.release = reinterpret_cast<free_fn>(dlsym(RTLD_NEXT, "cudaFree"));
    if (!s.alloc || !s.release) {
        std::fprintf(stderr, "S5CUDA FATAL RTLD_NEXT missing cudaMalloc/cudaFree; interposition unavailable\n");
        std::fflush(stderr); _exit(126);
    }
    Dl_info alloc_info{}, free_info{};
    dladdr(reinterpret_cast<void *>(s.alloc), &alloc_info); dladdr(reinterpret_cast<void *>(s.release), &free_info);
    std::fprintf(stderr, "S5CUDA captured_apis=cudaMalloc,cudaFree scope=runtime_aux_scratch_requested_bytes_only cudaMalloc=%p malloc_object=%s cudaFree=%p free_object=%s trace=%d stacks=%d\n",
                 reinterpret_cast<void *>(s.alloc), alloc_info.dli_fname ? alloc_info.dli_fname : "?",
                 reinterpret_cast<void *>(s.release), free_info.dli_fname ? free_info.dli_fname : "?", s.trace, s.stacks);
}
void frame(void * address, int i) {
    Dl_info info{};
    if (dladdr(address, &info) && info.dli_fbase) {
        auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(info.dli_fbase);
        std::fprintf(stderr, "S5CUDA frame=%d pc=%p object=%s object_offset=0x%llx symbol=%s\n", i, address,
                     info.dli_fname ? info.dli_fname : "?", (unsigned long long) offset,
                     info.dli_sname ? info.dli_sname : "?");
    } else std::fprintf(stderr, "S5CUDA frame=%d pc=%p object=?\n", i, address);
}
void error(state & s, const char * text, void * p) {
    ++s.stats.tracking_errors;
    std::fprintf(stderr, "S5CUDA INCONCLUSIVE tracking_error=%s ptr=%p\n", text, p);
}
}

extern "C" int cudaMalloc(void ** out, size_t bytes) {
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex); resolve(s); entry guard;
    if (depth != 1) {
        error(s, "nested_cudaMalloc", nullptr);
        return s.alloc(out, bytes); // Explicitly invalidate evidence; do not alter the application's CUDA status.
    }
    ++s.stats.malloc_calls;
    const int result = s.alloc(out, bytes);
    void * p = result == 0 && out ? *out : nullptr;
    uint64_t id = 0;
    void * caller = __builtin_return_address(0);
    if (result == 0) {
        ++s.stats.malloc_success;
        if (!p && bytes) error(s, "successful_nonzero_malloc_returned_null", p);
        if (p) {
            try {
                allocation a{}; a.bytes = bytes; a.id = ++s.next_id;
                a.frames[0] = caller; a.depth = 1;
                if (s.stacks) a.depth = backtrace(a.frames, 16);
                if (bytes > std::numeric_limits<uint64_t>::max() - s.stats.live_bytes ||
                    bytes > std::numeric_limits<uint64_t>::max() - s.stats.allocated_bytes) error(s, "byte_counter_overflow", p);
                else if (!s.live.emplace(p, a).second) error(s, "duplicate_live_pointer", p);
                else {
                    id = a.id;
                    ++s.stats.live_allocations; s.stats.live_bytes += bytes; s.stats.allocated_bytes += bytes;
                    if (s.stats.live_bytes > s.stats.peak_live_bytes) s.stats.peak_live_bytes = s.stats.live_bytes;
                }
            } catch (...) { error(s, "host_metadata_allocation_failed", p); }
        }
    }
    if (s.trace) std::fprintf(stderr, "S5CUDA malloc id=%llu ptr=%p bytes=%zu status=%d live_count=%llu live_bytes=%llu\n",
                             (unsigned long long) id, p, bytes, result,
                             (unsigned long long) s.stats.live_allocations, (unsigned long long) s.stats.live_bytes);
    if (s.trace) frame(caller, -1); // Preserve the allocation call site even if the pointer is later freed.
    return result;
}

extern "C" int cudaFree(void * p) {
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex); resolve(s); entry guard;
    if (depth != 1) { error(s, "nested_cudaFree", p); return s.release(p); }
    ++s.stats.free_calls;
    auto known = s.live.find(p);
    const uint64_t id = known == s.live.end() ? 0 : known->second.id;
    size_t bytes = known == s.live.end() ? 0 : known->second.bytes;
    const int result = s.release(p);
    if (result == 0) {
        ++s.stats.free_success;
        if (p) {
            auto it = s.live.find(p);
            if (it == s.live.end()) {
                ++s.stats.unknown_frees;
                std::fprintf(stderr, "S5CUDA INCONCLUSIVE successful_free_of_untracked_ptr=%p\n", p);
            } else {
                bytes = it->second.bytes; --s.stats.live_allocations; s.stats.live_bytes -= bytes;
                s.stats.freed_bytes += bytes; s.live.erase(it);
            }
        }
    }
    if (s.trace) std::fprintf(stderr, "S5CUDA free id=%llu ptr=%p bytes=%zu status=%d live_count=%llu live_bytes=%llu\n",
                             (unsigned long long) id, p, bytes, result,
                             (unsigned long long) s.stats.live_allocations, (unsigned long long) s.stats.live_bytes);
    return result;
}

extern "C" int s5_cuda_snapshot(s5_cuda_stats * out, size_t size) {
    if (!out || size != sizeof(*out)) return -1;
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex); *out = s.stats; return 0;
}
extern "C" uint64_t s5_cuda_checkpoint() {
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex); return s.next_id;
}
extern "C" int s5_cuda_since(uint64_t checkpoint, s5_cuda_generation_stats * out, size_t size) {
    if (!out || size != sizeof(*out)) return -1;
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    if (checkpoint > s.next_id) return -1;
    s5_cuda_generation_stats result{}; result.checkpoint = checkpoint; result.last_id = s.next_id;
    result.unknown_frees = s.stats.unknown_frees; result.tracking_errors = s.stats.tracking_errors;
    for (const auto & item : s.live) if (item.second.id > checkpoint) {
        ++result.new_live_allocations; result.new_live_bytes += item.second.bytes;
    }
    *out = result; return 0;
}
extern "C" void s5_cuda_report(const char * phase) {
    state & s = get(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    const auto & n = s.stats;
    std::fprintf(stderr, "S5CUDA summary phase=%s malloc_calls=%llu malloc_success=%llu free_calls=%llu free_success=%llu live_count=%llu live_bytes=%llu peak_live_bytes=%llu allocated_bytes=%llu freed_bytes=%llu unknown_frees=%llu tracking_errors=%llu\n",
        phase ? phase : "?", (unsigned long long) n.malloc_calls, (unsigned long long) n.malloc_success,
        (unsigned long long) n.free_calls, (unsigned long long) n.free_success, (unsigned long long) n.live_allocations,
        (unsigned long long) n.live_bytes, (unsigned long long) n.peak_live_bytes, (unsigned long long) n.allocated_bytes,
        (unsigned long long) n.freed_bytes, (unsigned long long) n.unknown_frees, (unsigned long long) n.tracking_errors);
    for (const auto & item : s.live) {
        std::fprintf(stderr, "S5CUDA retained id=%llu ptr=%p bytes=%zu phase=%s\n", (unsigned long long) item.second.id,
                     item.first, item.second.bytes, phase ? phase : "?");
        for (int i = 0; i < item.second.depth; ++i) frame(item.second.frames[i], i);
    }
    if (!n.malloc_calls) std::fprintf(stderr, "S5CUDA INCONCLUSIVE no_cudaMalloc_observed (static cudart, bypassed APIs, or no allocations)\n");
    if (n.unknown_frees || n.tracking_errors) std::fprintf(stderr, "S5CUDA INCONCLUSIVE bookkeeping/coverage_errors; do not infer leak absence\n");
    std::fflush(stderr);
}
__attribute__((destructor)) static void final_report() { s5_cuda_report("interposer-destructor-not-a-test-phase-boundary"); }
