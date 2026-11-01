// DRAM read-bandwidth probe for Orin: which access pattern gets closest to 204.8 GB/s?
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

template <int MODE>
__device__ __forceinline__ uint4 ld(const uint4* p) {
    uint4 v;
    if (MODE == 0) v = __ldg(p);
    else if (MODE == 1) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else if (MODE == 2) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// grid-stride, U independent 16B loads in flight per thread
template <int MODE, int U>
__global__ void k_stride(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t s = 0;
    const size_t step = (size_t)gridDim.x * blockDim.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i + (U - 1) * step < n; i += U * step) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = ld<MODE>(p + i + u * step);
#pragma unroll
        for (int u = 0; u < U; u++) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n; i += step) { uint4 v = ld<MODE>(p + i); s ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (s == 0x12345678u) out[0] = s;
}

// each warp streams its own contiguous chunk of CH bytes (DRAM page locality), U loads in flight per lane
template <int MODE, int U>
__global__ void k_chunk(const uint4* __restrict__ p, size_t n, size_t chunk16, uint32_t* out) {
    uint32_t s = 0;
    const int lane = threadIdx.x & 31;
    const size_t warps = (size_t)gridDim.x * (blockDim.x / 32);
    const size_t w0 = blockIdx.x * (size_t)(blockDim.x / 32) + (threadIdx.x >> 5);
    const size_t nch = n / chunk16;
    for (size_t c = w0; c < nch; c += warps) {
        const uint4* q = p + c * chunk16;
        for (size_t i = lane; i < chunk16; i += 32 * U) {
            uint4 v[U];
#pragma unroll
            for (int u = 0; u < U; u++) v[u] = ld<MODE>(q + i + u * 32);
#pragma unroll
            for (int u = 0; u < U; u++) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    }
    if (s == 0x12345678u) out[0] = s;
}

int main(int argc, char** argv) {
    size_t bytes = (argc > 1 ? atoll(argv[1]) : 1024) << 20;
    size_t n = bytes / 16;
    uint4* d; uint32_t* o;
    CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&o, 4));
    CK(cudaMemset(d, 1, bytes));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto run = [&](const char* name, auto fn) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaDeviceSynchronize());
        std::vector<float> v;
        for (int i = 0; i < 15; i++) { cudaEventRecord(a); fn(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); v.push_back(ms); }
        CK(cudaGetLastError());
        std::sort(v.begin(), v.end());
        printf("%-44s %7.1f GB/s (best %7.1f)\n", name, bytes / v[v.size() / 2] / 1e6, bytes / v[0] / 1e6);
    };
    char nm[128];
#define STRIDE(M, U, BPS, T) snprintf(nm, sizeof nm, "stride mode=%d U=%d blocks/SM=%d thr=%d", M, U, BPS, T); run(nm, [&]{ k_stride<M, U><<<sms * BPS, T>>>(d, n, o); });
    STRIDE(0, 1, 16, 256) STRIDE(0, 4, 4, 256) STRIDE(0, 4, 8, 256) STRIDE(0, 8, 4, 256)
    STRIDE(1, 4, 4, 256) STRIDE(1, 8, 4, 256) STRIDE(2, 4, 4, 256) STRIDE(2, 8, 4, 256) STRIDE(3, 4, 4, 256)
    STRIDE(2, 4, 8, 256) STRIDE(2, 4, 2, 512) STRIDE(2, 16, 2, 256)
#define CHUNK(M, U, KB, BPS) snprintf(nm, sizeof nm, "chunk mode=%d U=%d chunk=%dKB blocks/SM=%d", M, U, KB, BPS); run(nm, [&]{ k_chunk<M, U><<<sms * BPS, 256>>>(d, n, (size_t)KB * 1024 / 16, o); });
    CHUNK(0, 4, 16, 4) CHUNK(2, 4, 16, 4) CHUNK(2, 4, 64, 4) CHUNK(2, 8, 64, 4) CHUNK(2, 4, 256, 4) CHUNK(2, 8, 256, 2) CHUNK(2, 4, 2048, 4)
    return 0;
}
