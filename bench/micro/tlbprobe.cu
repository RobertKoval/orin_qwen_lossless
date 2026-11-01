// GPU TLB probe: pointer-chase latency vs number of distinct pages touched, for several strides.
// Each slot is one 4-byte index in its own cache line at offset k*stride (+ a varying line inside the stride block), visited
// in a random cyclic order by a single thread with L1-bypassing loads. After a warm-up pass the slots' lines sit in L2
// (slots * 128 B << L2), so the timed pass measures L2 hit + address translation; the latency step shows the TLB reach and
// the miss cost. Also prints whether the driver supports the VMM API and its allocation granularity.
//     tlbprobe [buffer_GiB=8] [vmm]
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void chase(const uint32_t * a, uint32_t start, int n, uint32_t * out_idx, long long * out_cycles) {
    uint32_t j = start;
    for (int i = 0; i < n; i++) { j = __ldcg(a + j); } // warm-up pass over every slot
    const long long t0 = clock64();
    for (int i = 0; i < n; i++) { j = __ldcg(a + j); }
    const long long t1 = clock64();
    *out_idx = j;
    *out_cycles = t1 - t0;
}

int main(int argc, char ** argv) {
    const size_t gib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 8;
    const size_t bytes = gib << 30;
    int dev = 0, vmm = 0, mhz = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaFree(nullptr)); // initialises the driver API for the cu* queries below
    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
    CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, dev));
    printf("VMM supported: %d, SM clock attr %d kHz\n", vmm, mhz);
    if (vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        size_t gmin = 0, grec = 0;
        cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        cuMemGetAllocationGranularity(&grec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        printf("VMM granularity: min %zu KiB, recommended %zu KiB\n", gmin >> 10, grec >> 10);
    }
    // vmm: physical memory from cuMemCreate in granularity-sized chunks, mapped into one reserved range
    const bool use_vmm = argc > 2 && std::string(argv[2]) == "vmm";
    uint32_t * d;
    if (use_vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        CUmemGenericAllocationHandle h;
        CUdeviceptr p;
        if (cuMemAddressReserve(&p, bytes, 0, 0, 0) != CUDA_SUCCESS || cuMemCreate(&h, bytes, &prop, 0) != CUDA_SUCCESS ||
            cuMemMap(p, bytes, 0, h, 0) != CUDA_SUCCESS) {
            fprintf(stderr, "VMM allocation failed\n");
            return 1;
        }
        CUmemAccessDesc acc = {};
        acc.location = prop.location;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(p, bytes, &acc, 1) != CUDA_SUCCESS) { fprintf(stderr, "cuMemSetAccess failed\n"); return 1; }
        d = (uint32_t *) p;
    } else {
        CK(cudaMalloc(&d, bytes));
    }
    printf("allocation: %s, %zu GiB\n", use_vmm ? "cuMemCreate + cuMemMap" : "cudaMalloc", gib);
    uint32_t * d_idx; long long * d_cyc;
    CK(cudaMalloc(&d_idx, 4)); CK(cudaMalloc(&d_cyc, 8));
    std::mt19937 rng(1);
    printf("%10s %8s %10s %12s %10s\n", "stride", "slots", "region MiB", "cycles/load", "ns/load");
    for (size_t stride : {size_t(4) << 10, size_t(64) << 10, size_t(2) << 20}) {
        for (size_t slots = 8; slots <= 16384 && slots * stride <= bytes; slots *= 2) {
            // slot k lives at k*stride + line(k)*128 (the line varies so slots do not alias one L2 set)
            std::vector<uint32_t> order(slots);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin() + 1, order.end(), rng);
            auto pos = [&](size_t k) { return (k * stride + ((k * 37) % (stride / 128)) * 128) / 4; };
            std::vector<std::pair<size_t, uint32_t>> writes;
            for (size_t i = 0; i < slots; i++) {
                writes.push_back({pos(order[i]), (uint32_t) pos(order[(i + 1) % slots])});
            }
            for (auto & w : writes) { CK(cudaMemcpy(d + w.first, &w.second, 4, cudaMemcpyHostToDevice)); }
            const int n = (int) std::max<size_t>(slots, 4096);
            chase<<<1, 1>>>(d, (uint32_t) pos(order[0]), n, d_idx, d_cyc);
            CK(cudaDeviceSynchronize());
            long long cyc; CK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
            const double c = (double) cyc / n;
            printf("%9zuK %8zu %10.1f %12.1f %10.1f\n", stride >> 10, slots, slots * stride / 1048576.0, c, c / (mhz / 1e6));
        }
    }
    return 0;
}
