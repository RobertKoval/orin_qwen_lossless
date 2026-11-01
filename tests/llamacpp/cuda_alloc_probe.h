// SPDX-License-Identifier: MIT
// Test-private Linux LD_PRELOAD observer of cudaMalloc/cudaFree requested aux/scratch ownership only.
// No CUDA SDK dependency. NOT a VMM, physical-reclamation, process-RSS, or total GPU memory-cap oracle.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
struct s5_cuda_stats {
    uint64_t malloc_calls, malloc_success, free_calls, free_success;
    uint64_t live_allocations, live_bytes, peak_live_bytes;
    uint64_t allocated_bytes, freed_bytes;
    uint64_t unknown_frees, tracking_errors;
};
// 0: copied consistent observations; -1: null output or wrong ABI size. Does not query/synchronize CUDA.
// A query succeeding does NOT prove interception coverage: require positive/negative allocation controls,
// expected successful allocation/free deltas, and zero unknown_frees/tracking_errors before inferring reclamation.
int s5_cuda_snapshot(struct s5_cuda_stats * out, size_t size);

struct s5_cuda_generation_stats {
    uint64_t checkpoint, last_id;
    uint64_t new_live_allocations, new_live_bytes;
    uint64_t unknown_frees, tracking_errors;
};
// Take immediately before the ownership cycle. IDs distinguish reallocated addresses and equal-byte owner swaps.
uint64_t s5_cuda_checkpoint(void);
// 0: copied observations; -1: invalid output/size or a checkpoint beyond the last issued ID.
// After buffer destruction + normal synchronization, require new_live_allocations == 0, not merely equal totals.
// Counts successful non-null pointer owners even if their requested byte size is zero. Errors are cumulative and
// sticky: a later checkpoint cannot erase incomplete coverage/tracking. A zero count alone is not a coverage test.
int s5_cuda_since(uint64_t checkpoint, struct s5_cuda_generation_stats * out, size_t size);
// Logs counters and live IDs, requested sizes, allocation stacks/call sites. Never synchronizes a CUDA device.
void s5_cuda_report(const char * phase);
#ifdef __cplusplus
}
#endif
