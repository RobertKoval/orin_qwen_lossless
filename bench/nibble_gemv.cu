// Lossless BF16 "nibble" format GEMV microbench for Orin (SM87).
// Format per row: K/512 chunks of [512 B sign|mantissa bytes][256 B 4-bit exponent codes],
// per-row base exponent (code c -> exponent base+c), out-of-window weights are escapes
// (dense slot decodes to 2^(base-127), corrected exactly via a per-row escape list).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct Enc {
    int N = 0, K = 0;
    std::vector<uint8_t> data;      // N * K/512 * 768
    std::vector<uint8_t> base;      // N
    std::vector<uint32_t> esc_off;  // N + 1
    std::vector<uint32_t> esc;      // (col << 16) | bf16 bits
};

// position of (lane l, slot s) inside a 512-weight chunk
static inline int pos_of(int l, int s) { int j = s >> 2, t = s & 3; return 128 * j + 4 * l + t; }

static Enc encode(const uint16_t* w, int N, int K) {
    Enc e; e.N = N; e.K = K;
    int C = K / 512;
    e.data.assign((size_t)N * C * 768, 0);
    e.base.assign((size_t)N * 16, 0); e.esc_off.resize(N + 1);
    e.esc_off[0] = 0;
    for (int r = 0; r < N; r++) {
        const uint16_t* row = w + (size_t)r * K;
        int hist[256] = {0};
        for (int k = 0; k < K; k++) hist[(row[k] >> 7) & 0xFF]++;
        int idx[256]; for (int i = 0; i < 256; i++) idx[i] = i;
        std::sort(idx, idx + 256, [&](int a, int b) { return hist[a] > hist[b]; });
        uint8_t* lut = &e.base[(size_t)r * 16];
        int code_of[256]; for (int i = 0; i < 256; i++) code_of[i] = -1;
        code_of[0] = 0; lut[0] = 0;
        int n = 1;
        for (int i = 0; i < 256 && n < 16; i++) { int x = idx[i]; if (hist[x] == 0) break; if (x == 0 || x >= 128) continue; lut[n] = (uint8_t)x; code_of[x] = n; n++; }
        for (int ci = 0; ci < C; ci++) {
            uint8_t* sm = &e.data[((size_t)r * C + ci) * 768];
            uint8_t* nb = sm + 512;
            for (int l = 0; l < 32; l++) {
                uint32_t nw[2] = {0, 0};
                for (int s = 0; s < 16; s++) {
                    int col = ci * 512 + pos_of(l, s);
                    uint16_t v = row[col];
                    int ex = (v >> 7) & 0xFF;
                    uint8_t smb = (uint8_t)(((v >> 15) << 7) | (v & 0x7F)); int code = code_of[ex];
                    if (code < 0) { e.esc.push_back((uint32_t)col | ((uint32_t)smb << 12) | ((uint32_t)ex << 20)); smb = 0; code = 0; }
                    sm[16 * l + s] = smb;
                    nw[s >> 3] |= (uint32_t)code << (4 * (s & 7));
                }
                memcpy(nb + 8 * l, nw, 8);
            }
        }
        e.esc_off[r + 1] = (uint32_t)e.esc.size();
    }
    return e;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
    uint32_t d; asm("prmt.b32 %0, %1, 0, %2;" : "=r"(d) : "r"(a), "r"(sel)); return d;
}
__device__ __forceinline__ uint32_t prmt2(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}

// one warp per row; x (FP32) staged in shared memory; rows map to x group r / rows_per_group
template <int K>
__global__ void __launch_bounds__(256) gemv_nib(const uint8_t* __restrict__ data, const uint8_t* __restrict__ base,
        const uint32_t* __restrict__ esc_off, const uint32_t* __restrict__ esc,
        const float* __restrict__ x, float* __restrict__ y, int N, int rows_per_group, int groups) {
    constexpr int C = K / 512;
#ifdef XGLOBAL
    const float* xs = x;
#else
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K * groups; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
#endif
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * 8 + warp; r < N; r += gridDim.x * 8) {
        const float* xg = xs + (r / rows_per_group) * K;
        const uint8_t* rp = data + (size_t)r * C * 768;
        uint4 smv[C]; uint2 nbv[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            smv[c] = __ldg(reinterpret_cast<const uint4*>(rp + c * 768) + lane);
            nbv[c] = __ldg(reinterpret_cast<const uint2*>(rp + c * 768 + 512) + lane);
        }
        const uint4 L = __ldg(reinterpret_cast<const uint4*>(base) + r);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int c = 0; c < C; c++) {
            const uint32_t smw[4] = {smv[c].x, smv[c].y, smv[c].z, smv[c].w};
            const uint32_t nw[2] = {nbv[c].x, nbv[c].y};
#pragma unroll
            for (int j = 0; j < 4; j++) {
#ifdef XGLOBAL
                const float4 xv = __ldg(reinterpret_cast<const float4*>(xg + c * 512 + 128 * j + 4 * lane));
#else
                const float4 xv = *reinterpret_cast<const float4*>(xg + c * 512 + 128 * j + 4 * lane);
#endif
                const uint32_t sel = (j & 1) ? (nw[j >> 1] >> 16) : nw[j >> 1];
                const uint32_t E4 = prmt2(L.x, L.y, sel) | prmt2(L.z, L.w, sel ^ 0x8888u);
                const uint32_t w0 = (prmt(smw[j], 0x9180u) & 0x807F807Fu) | (prmt(E4, 0x4140u) << 7);
                const uint32_t w1 = (prmt(smw[j], 0xB3A2u) & 0x807F807Fu) | (prmt(E4, 0x4342u) << 7);
                a0 = fmaf(__uint_as_float(w0 << 16), xv.x, a0);
                a1 = fmaf(__uint_as_float(w0 & 0xFFFF0000u), xv.y, a1);
                a2 = fmaf(__uint_as_float(w1 << 16), xv.z, a2);
                a3 = fmaf(__uint_as_float(w1 & 0xFFFF0000u), xv.w, a3);
            }
        }
        float acc = (a0 + a1) + (a2 + a3);
        const uint32_t e_beg = esc_off[r], e_end = esc_off[r + 1];
        for (uint32_t i = e_beg + lane; i < e_end; i += 32) {
            const uint32_t rec = esc[i];
            const uint32_t smb = (rec >> 12) & 0xFF, ex = rec >> 20;
            const uint32_t bits = ((smb & 0x80) << 8) | (ex << 7) | (smb & 0x7F);
            acc = fmaf(__uint_as_float(bits << 16), xg[rec & 0xFFF], acc);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[r] = acc;
    }
}

// raw BF16 with identical position mapping / accumulation order
template <int K>
__global__ void __launch_bounds__(256) gemv_bf16(const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
        int N, int rows_per_group, int groups) {
    constexpr int C = K / 512;
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K * groups; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * 8 + warp; r < N; r += gridDim.x * 8) {
        const float* xg = xs + (r / rows_per_group) * K;
        const uint16_t* rp = w + (size_t)r * K;
        uint2 v[C][4];
#pragma unroll
        for (int c = 0; c < C; c++)
#pragma unroll
            for (int j = 0; j < 4; j++) v[c][j] = __ldg(reinterpret_cast<const uint2*>(rp + c * 512 + 128 * j + 4 * lane));
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int c = 0; c < C; c++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 xv = *reinterpret_cast<const float4*>(xg + c * 512 + 128 * j + 4 * lane);
                a0 = fmaf(__uint_as_float(v[c][j].x << 16), xv.x, a0);
                a1 = fmaf(__uint_as_float(v[c][j].x & 0xFFFF0000u), xv.y, a1);
                a2 = fmaf(__uint_as_float(v[c][j].y << 16), xv.z, a2);
                a3 = fmaf(__uint_as_float(v[c][j].y & 0xFFFF0000u), xv.w, a3);
            }
        float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[r] = acc;
    }
}

__global__ void read_bw(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i); s ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x12345678u) out[0] = s;
}

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: file offset K row0 nrows groups [reps]\n"); return 1; }
    const char* file = argv[1]; size_t off = strtoull(argv[2], 0, 10);
    int K = atoi(argv[3]); long row0 = atol(argv[4]); int N = atoi(argv[5]); int G = atoi(argv[6]);
    int reps = argc > 7 ? atoi(argv[7]) : 20;
    int rpg = N / G;
    size_t nbytes = (size_t)N * K * 2;
    std::vector<uint16_t> w((size_t)N * K);
    int fd = open(file, O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    size_t got = 0; while (got < nbytes) { ssize_t r = pread(fd, (char*)w.data() + got, nbytes - got, off + (size_t)row0 * K * 2 + got); if (r <= 0) { perror("pread"); return 1; } got += r; }
    close(fd);

    Enc e = encode(w.data(), N, K);
    size_t comp = e.data.size() + e.base.size() + e.esc_off.size() * 4 + e.esc.size() * 4;
    uint32_t maxesc = 0; for (int r = 0; r < N; r++) maxesc = std::max(maxesc, e.esc_off[r + 1] - e.esc_off[r]);
    printf("max escapes/row %u\n", maxesc);
    printf("shape N=%d K=%d groups=%d  raw=%.1f MB  comp=%.1f MB  ratio=%.4f  escapes=%zu (%.2e/weight)\n",
           N, K, G, nbytes / 1e6, comp / 1e6, (double)comp / nbytes, e.esc.size(), (double)e.esc.size() / ((double)N * K));

    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t)K * G);
    for (auto& v : x) { float f = nd(rng); uint32_t u; memcpy(&u, &f, 4); u &= 0xFFFF0000u; memcpy(&v, &u, 4); }  // bf16-representable
    std::vector<uint16_t> xb(x.size()); for (size_t i = 0; i < x.size(); i++) { uint32_t u; memcpy(&u, &x[i], 4); xb[i] = u >> 16; }

    uint8_t *d_data, *d_base; uint32_t *d_off, *d_esc, *d_dummy; uint16_t* d_w; float *d_x, *d_y1, *d_y2, *d_y3; __nv_bfloat16* d_xb;
    CK(cudaMalloc(&d_data, e.data.size())); CK(cudaMemcpy(d_data, e.data.data(), e.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_base, e.base.size())); CK(cudaMemcpy(d_base, e.base.data(), e.base.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_off, (N + 1) * 4)); CK(cudaMemcpy(d_off, e.esc_off.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_esc, std::max<size_t>(4, e.esc.size() * 4))); if (e.esc.size()) CK(cudaMemcpy(d_esc, e.esc.data(), e.esc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, nbytes)); CK(cudaMemcpy(d_w, w.data(), nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4)); CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xb, x.size() * 2)); CK(cudaMemcpy(d_xb, xb.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_y1, N * 4)); CK(cudaMalloc(&d_y2, N * 4)); CK(cudaMalloc(&d_y3, N * 4)); CK(cudaMalloc(&d_dummy, 4));

    int dev; CK(cudaGetDevice(&dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int grid = prop.multiProcessorCount * 4;
    if (getenv("GRID")) grid = atoi(getenv("GRID"));
    if (getenv("GRID_AUTO")) grid = std::max(1, std::min(prop.multiProcessorCount * 8, (N + 7) / 8));
    size_t shm = (size_t)K * G * 4;
    auto launch_nib = [&]() {
        if (K == 512) gemv_nib<512><<<grid, 256, shm>>>(d_data, d_base, d_off, d_esc, d_x, d_y1, N, rpg, G);
        else if (K == 2048) gemv_nib<2048><<<grid, 256, shm>>>(d_data, d_base, d_off, d_esc, d_x, d_y1, N, rpg, G);
        else gemv_nib<4096><<<grid, 256, shm>>>(d_data, d_base, d_off, d_esc, d_x, d_y1, N, rpg, G);
    };
    auto launch_raw = [&]() {
        if (K == 512) gemv_bf16<512><<<grid, 256, shm>>>(d_w, d_x, d_y2, N, rpg, G);
        else if (K == 2048) gemv_bf16<2048><<<grid, 256, shm>>>(d_w, d_x, d_y2, N, rpg, G);
        else gemv_bf16<4096><<<grid, 256, shm>>>(d_w, d_x, d_y2, N, rpg, G);
    };
    if (shm > 48 * 1024) {
        CK(cudaFuncSetAttribute(gemv_nib<4096>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(gemv_bf16<4096>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(gemv_nib<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(gemv_bf16<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(gemv_nib<2048>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncSetAttribute(gemv_bf16<2048>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    }
    cublasHandle_t h; cublasCreate(&h);
    const float one = 1.f, zero = 0.f;
    auto launch_cublas = [&]() {  // only meaningful for groups == 1
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &one, d_w, CUDA_R_16BF, K, d_xb, CUDA_R_16BF, K, &zero, d_y3, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    auto timeit = [&](auto fn) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaDeviceSynchronize());
        std::vector<float> v;
        for (int i = 0; i < reps; i++) { cudaEventRecord(t0); fn(); cudaEventRecord(t1); cudaEventSynchronize(t1); float ms; cudaEventElapsedTime(&ms, t0, t1); v.push_back(ms); }
        CK(cudaGetLastError());
        std::sort(v.begin(), v.end()); return v[v.size() / 2];
    };
    float tn = timeit(launch_nib), tr = timeit(launch_raw), tc = G == 1 ? timeit(launch_cublas) : -1.f;
    float tb = timeit([&]() { read_bw<<<grid * 4, 256>>>((const uint4*)d_w, nbytes / 16, d_dummy); });
    printf("read_bw  : %.3f ms  %.1f GB/s (raw bytes)\n", tb, nbytes / tb / 1e6);
    printf("bf16 gemv: %.3f ms  %.1f GB/s\n", tr, nbytes / tr / 1e6);
    if (G == 1) printf("cublas   : %.3f ms  %.1f GB/s\n", tc, nbytes / tc / 1e6);
    printf("nib gemv : %.3f ms  %.1f GB/s compressed, %.1f GB/s raw-equivalent  speedup vs bf16 kernel %.3f\n", tn, comp / tn / 1e6, nbytes / tn / 1e6, tr / tn);

    std::vector<float> y1(N), y2(N), y3(N);
    CK(cudaMemcpy(y1.data(), d_y1, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y2.data(), d_y2, N * 4, cudaMemcpyDeviceToHost));
    if (G == 1) CK(cudaMemcpy(y3.data(), d_y3, N * 4, cudaMemcpyDeviceToHost));
    double en = 0, er = 0, ec = 0; long same = 0, noesc = 0, noesc_same = 0;
    for (int r = 0; r < N; r++) {
        const float* xg = x.data() + (size_t)(r / rpg) * K;
        double ref = 0, mag = 0;
        for (int k = 0; k < K; k++) { double p = (double)bf2f(w[(size_t)r * K + k]) * xg[k]; ref += p; mag += fabs(p); }
        mag = std::max(mag, 1e-30);
        en = std::max(en, fabs(y1[r] - ref) / mag); er = std::max(er, fabs(y2[r] - ref) / mag);
        if (G == 1) ec = std::max(ec, fabs(y3[r] - ref) / mag);
        same += y1[r] == y2[r];
        if (e.esc_off[r] == e.esc_off[r + 1]) { noesc++; noesc_same += y1[r] == y2[r]; }
    }
    printf("max |y-ref|/sum|w*x| : nib %.3e  bf16 %.3e  cublas %.3e\n", en, er, ec);
    printf("nib==bf16 bitwise: %ld/%d rows; rows without escapes: %ld/%ld identical\n", same, N, noesc_same, noesc);
    return 0;
}
