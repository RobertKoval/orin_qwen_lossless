// SPDX-License-Identifier: MIT
// Torch-free driver of the extension's launchers (core/), to separate kernel/launcher effects from the torch
// process: reads one 2-D BF16 tensor from the GGUF (pread), encodes it with the production policy (HUFPAIR, dense R
// rule), uploads with cudaMalloc + cudaMemcpy and times y = W x (T = 1, FP32 x) with events, median of N.
//   nvcc -O3 -std=c++17 -use_fast_math -arch=sm_87 -I core/include -I <llama>/ggml/include standalone_mmv.cu \
//        core/src/launch.cu core/src/mmid_helper.cu core/src/encode.cpp -L <llama>/build/bin -lggml-base
//   standalone_mmv model.gguf output.weight [reps=30]
#include "bf16n/encode.h"
#include "bf16n/launch.h"
#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e_)); exit(1); } } while (0)

extern "C" float run_ptrs(const uint8_t *, const uint8_t *, int, long long, long long, int, unsigned long long,
                          unsigned long long, unsigned long long, unsigned long long, const float *, float *, int);

extern "C" int run_standalone(const char * path, const char * name, int reps) {
    if (getenv("BF16N_SCHED")) { // 1 spin, 2 yield, 4 blocking sync (cudaDeviceSchedule*)
        CK(cudaSetDeviceFlags((unsigned) atoi(getenv("BF16N_SCHED"))));
    }
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    const int64_t tid = gguf_find_tensor(g, name);
    const ggml_tensor * tm = ggml_get_tensor(meta, name);
    const int64_t K = tm->ne[0], N = tm->ne[1];
    std::vector<uint16_t> w(K * N);
    const int fd = open(path, O_RDONLY);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    size_t done = 0;
    while (done < w.size() * 2) { const ssize_t r = pread(fd, (char *) w.data() + done, w.size() * 2 - done, off + done); if (r <= 0) return 1; done += r; }
    close(fd);
    int R = 1;
    for (int r = 16; r > 1; r >>= 1) { if (N % r == 0 && N / r >= 512) { R = r; break; } }
    bf16n::encoded e = bf16n::encode(w.data(), N, K, bf16n::FMT_HUFPAIR, R, false);
    const bf16n::aux_header * h = (const bf16n::aux_header *) e.aux.data();
    {
        auto sum64 = [](const std::vector<uint8_t> & v) { unsigned long long s = 0; for (size_t i = 0; i + 8 <= v.size(); i += 8) { unsigned long long w; memcpy(&w, &v[i], 8); s = s * 1000003ull + w; } return s; };
        printf("plane %zu bytes sum %016llx, aux %zu bytes sum %016llx\n", e.plane.size(), sum64(e.plane), e.aux.size(), sum64(e.aux));
    }
    uint8_t * dp, * da; float * dx, * dy;
    CK(cudaMalloc(&dp, e.plane.size())); CK(cudaMalloc(&da, e.aux.size()));
    CK(cudaMalloc(&dx, K * 4)); CK(cudaMalloc(&dy, N * 4));
    CK(cudaMemcpy(dp, e.plane.data(), e.plane.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, e.aux.data(), e.aux.size(), cudaMemcpyHostToDevice));
    std::vector<float> x(K, 0.01f);
    CK(cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice));
    bf16n::weight W;
    W.plane = dp; W.aux = da; W.format = e.format; W.K = K; W.rows = N; W.R = e.R; W.cm = false;
    W.off_a = h->off_a; W.off_b = h->off_b; W.off_c = h->off_c; W.off_d = h->off_d;
    bf16n::mmv_args a;
    a.x = dx; a.y = dy; a.ncols = 1; a.x_tok = K * 4; a.y_tok = N * 4; a.rows_per_matrix = N; a.nrows = N;
    bf16n::init();
    if (getenv("BF16N_TIMING")) { // the run_ptrs timing (legacy stream, 200 launches of warm-up) on these buffers
        printf("run_ptrs timing on own buffers: %.3f ms\n", run_ptrs(dp, da, e.format, K, N, e.R, h->off_a, h->off_b, h->off_c, h->off_d, dx, dy, reps));
    }
    cudaStream_t s = nullptr; // BF16N_STREAM=legacy: the legacy default stream, as an eager torch op without a stream context
    if (!getenv("BF16N_STREAM")) { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    const bool dev_sync = getenv("BF16N_SYNC") != nullptr; // cudaDeviceSynchronize instead of the stream's
    const double warm = getenv("BF16N_WARM") ? atof(getenv("BF16N_WARM")) : 2.0;
    auto sync = [&]() { if (dev_sync) { CK(cudaDeviceSynchronize()); } else { CK(cudaStreamSynchronize(s)); } };
    for (const auto t0 = std::chrono::steady_clock::now();
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm;) {
        bf16n::mmv(W, a, s); sync();
    }
    std::vector<float> ms(reps);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0, s)); bf16n::mmv(W, a, s); CK(cudaEventRecord(e1, s)); sync();
        CK(cudaEventElapsedTime(&ms[r], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    printf("%s K=%lld N=%lld R=%d: %.3f ms (median of %d)\n", name, (long long) K, (long long) N, e.R, ms[reps / 2], reps);
    return 0;
}

#ifndef STANDALONE_LIB
int main(int argc, char ** argv) { return run_standalone(argv[1], argv[2], argc > 3 ? atoi(argv[3]) : 30); }
#endif

// external buffers (e.g. torch tensors): time y = W x on the legacy stream; x, y FP32
extern "C" float run_ptrs(const uint8_t * plane, const uint8_t * aux, int format, long long K, long long rows, int R,
                          unsigned long long oa, unsigned long long ob, unsigned long long oc, unsigned long long od,
                          const float * x, float * y, int reps) {
    bf16n::weight W;
    W.plane = plane; W.aux = aux; W.format = format; W.K = K; W.rows = rows; W.R = R; W.cm = false;
    W.off_a = oa; W.off_b = ob; W.off_c = oc; W.off_d = od;
    bf16n::mmv_args a;
    a.x = x; a.y = y; a.ncols = 1; a.x_tok = K * 4; a.y_tok = rows * 4; a.rows_per_matrix = rows; a.nrows = rows;
    bf16n::init();
    for (int i = 0; i < 200; i++) { bf16n::mmv(W, a, nullptr); }
    CK(cudaDeviceSynchronize());
    std::vector<float> ms(reps);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0, nullptr)); bf16n::mmv(W, a, nullptr); CK(cudaEventRecord(e1, nullptr)); CK(cudaDeviceSynchronize());
        CK(cudaEventElapsedTime(&ms[r], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    return ms[reps / 2];
}

// copy external weights into buffers allocated here (static cudart), then time them like run_ptrs
extern "C" float run_copy(const uint8_t * plane, const uint8_t * aux, long long plane_bytes, long long aux_bytes, int format,
                          long long K, long long rows, int R, unsigned long long oa, unsigned long long ob,
                          unsigned long long oc, unsigned long long od, const float * x, float * y, int reps) {
    uint8_t * p2, * a2;
    CK(cudaMalloc(&p2, plane_bytes)); CK(cudaMalloc(&a2, aux_bytes));
    CK(cudaMemcpy(p2, plane, plane_bytes, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(a2, aux, aux_bytes, cudaMemcpyDeviceToDevice));
    const float t = run_ptrs(p2, a2, format, K, rows, R, oa, ob, oc, od, x, y, reps);
    CK(cudaFree(p2)); CK(cudaFree(a2));
    return t;
}
