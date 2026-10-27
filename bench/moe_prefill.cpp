// Correctness and speed of a BF16N MUL_MAT_ID (MoE expert matrices) on real weights, through the ggml CUDA backend.
// The reference is computed on the CPU from the exact BF16 weights, accumulated in double; for more than 8 tokens the
// activations are rounded to BF16 first, as both batched (prefill) paths feed BF16 tensor cores, while the GEMV path
// (up to 8 tokens) multiplies the F32 activations.
//
//   moe_prefill <model.gguf> <tensor> <tokens> [reps=10] [skew=0] [check_cols=64]
//
// tensor: e.g. blk.20.ffn_gate_exps.weight (K x N x E). Each token picks 8 distinct experts; skew > 0 draws them with
// probability ~ (rank + 1)^-skew over a random expert order (0: uniform). GGML_BF16N_NO_MMA=1 selects the old path.
// A name with "gate_up" (blk.20.ffn_gate_up_exps.weight) times gate, up and SWIGLU as one graph (fused when the backend
// can; GGML_BF16N_NO_GLU=1 disables it), compares it bitwise with the same ops computed unfused on the GPU, and checks it
// against silu(gate) * up from the CPU reference.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static float round_bf16(float f) { // round to nearest even, as __float2bfloat16
    uint32_t u; memcpy(&u, &f, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    memcpy(&f, &u, 4); return f;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tensor tokens [reps] [skew] [check_cols]\n", argv[0]); return 1; }
    const char * path = argv[1]; const char * name = argv[2];
    const int T = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 10;
    if (reps < 1) { fprintf(stderr, "reps must be >= 1\n"); return 1; }
    const double skew = argc > 5 ? atof(argv[5]) : 0.0;
    const int check_cols = argc > 6 ? atoi(argv[6]) : 64;
    const int n_used = 8;

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    const std::string sname = name;
    const size_t gu_at = sname.find("gate_up");
    const bool glu = gu_at != std::string::npos;
    int64_t K = 0, N = 0, E = 0;
    auto read_tensor = [&](const std::string & tn, std::vector<uint16_t> & out) {
        const int64_t tid = gguf_find_tensor(g, tn.c_str());
        if (tid < 0) { fprintf(stderr, "no tensor %s\n", tn.c_str()); exit(1); }
        const ggml_tensor * tm = ggml_get_tensor(meta, tn.c_str());
        if (tm->type != GGML_TYPE_BF16 || ggml_n_dims(tm) != 3) { fprintf(stderr, "%s: need a 3-D BF16 tensor\n", tn.c_str()); exit(1); }
        if (K && (tm->ne[0] != K || tm->ne[1] != N || tm->ne[2] != E)) { fprintf(stderr, "%s: shape differs\n", tn.c_str()); exit(1); }
        K = tm->ne[0]; N = tm->ne[1]; E = tm->ne[2];
        out.resize(K * N * E);
        const int fd = open(path, O_RDONLY);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
        size_t done = 0, total = out.size() * 2;
        while (done < total) {
            const ssize_t r = pread(fd, (char *) out.data() + done, total - done, off + done);
            if (r <= 0) { perror("pread"); exit(1); }
            done += r;
        }
        close(fd);
    };
    std::vector<uint16_t> w, w2; // w2: up for gate_up
    if (glu) {
        read_tensor(std::string(sname).replace(gu_at, 7, "gate"), w);
        read_tensor(std::string(sname).replace(gu_at, 7, "up"), w2);
    } else {
        read_tensor(sname, w);
    }
    const bool down = strstr(name, "down") != nullptr; // down: one input row per (token, slot)
    const int64_t ne11 = down ? n_used : 1;
    printf("%s: K=%lld N=%lld E=%lld, tokens %d, skew %.2f\n", name, (long long) K, (long long) N, (long long) E, T, skew);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, /*no_alloc*/ true }; // 2 graphs for gate_up
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16N, K, N, E);
    ggml_tensor * W2 = glu ? ggml_new_tensor_3d(ctx, GGML_TYPE_BF16N, K, N, E) : nullptr;
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, T);
    ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * Y = glu ? ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, W, X, I), ggml_mul_mat_id(ctx, W2, X, I))
                          : ggml_mul_mat_id(ctx, W, X, I);
    // the same GLU with a second consumer of the gate result, which keeps the backend from fusing it
    ggml_tensor * G2 = glu ? ggml_mul_mat_id(ctx, W, X, I) : nullptr;
    ggml_tensor * Y2 = glu ? ggml_swiglu_split(ctx, G2, ggml_mul_mat_id(ctx, W2, X, I)) : nullptr;
    ggml_tensor * S2 = glu ? ggml_scale(ctx, G2, 1.0f) : nullptr;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    auto convert = (bool (*)(ggml_tensor *, const void *, int)) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(be)), "ggml_backend_cuda_bf16n_convert");
    if (!convert || !convert(W, w.data(), 2 /* GGML_BF16N_HUFFMAN: pair code when it fits */) ||
            (glu && !convert(W2, w2.data(), 2))) { fprintf(stderr, "convert failed\n"); return 1; }

    std::mt19937_64 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K * ne11 * T);
    for (auto & v : x) { v = nd(rng); }
    std::vector<int> order(E);
    for (int i = 0; i < E; i++) { order[i] = i; }
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<double> cdf(E);
    double acc = 0;
    for (int i = 0; i < E; i++) { acc += std::pow(i + 1.0, -skew); cdf[i] = acc; }
    std::uniform_real_distribution<double> ud(0, acc);
    std::vector<int32_t> ids(n_used * T);
    std::vector<int> tok_per_e(E, 0);
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < n_used; s++) {
            int e;
            do { e = order[std::lower_bound(cdf.begin(), cdf.end(), ud(rng)) - cdf.begin()]; }
            while (std::find(&ids[t * n_used], &ids[t * n_used] + s, e) != &ids[t * n_used] + s);
            ids[t * n_used + s] = e; tok_per_e[e]++;
        }
    }
    printf("tokens per expert: max %d, mean %.1f, experts used %lld\n", *std::max_element(tok_per_e.begin(), tok_per_e.end()),
           (double) T * n_used / E, (long long) std::count_if(tok_per_e.begin(), tok_per_e.end(), [](int c) { return c > 0; }));
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
    ggml_backend_tensor_set(I, ids.data(), 0, ggml_nbytes(I));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    // warm-up for 2 s: the GPU clock governor (nvhost_podgov) ramps up only under sustained load
    for (const auto tw = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - tw < std::chrono::seconds(2);) {
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
    }
    // median of individually synchronised repetitions: one host stall must not move the result
    std::vector<double> rep_ms(reps);
    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        rep_ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::sort(rep_ms.begin(), rep_ms.end());
    const double ms = rep_ms[reps / 2];
    const double flop = (glu ? 4.0 : 2.0) * N * K * n_used * T;
    printf("time %.3f ms  (median of %d, min %.3f, max %.3f; %.2f TFLOPS, %.1f G weights/s over the %lld used experts)\n",
           ms, reps, rep_ms.front(), rep_ms.back(), flop / ms * 1e-9,
           (double) N * K * std::count_if(tok_per_e.begin(), tok_per_e.end(), [](int c) { return c > 0; }) / ms * 1e-6,
           (long long) std::count_if(tok_per_e.begin(), tok_per_e.end(), [](int c) { return c > 0; }));

    std::vector<float> y(N * n_used * T);
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    if (glu) {
        ggml_cgraph * gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, Y2);
        ggml_build_forward_expand(gf2, S2);
        ggml_backend_graph_compute(be, gf2);
        std::vector<float> y2(y.size());
        ggml_backend_tensor_get(Y2, y2.data(), 0, ggml_nbytes(Y2));
        size_t ndiff = 0;
        for (size_t i = 0; i < y.size(); i++) { ndiff += memcmp(&y[i], &y2[i], 4) != 0; }
        printf("vs unfused GPU ops: %zu of %zu values differ bitwise\n", ndiff, y.size());
    }
    uint64_t fnv = 0xcbf29ce484222325ull; // FNV-1a of the output bytes: equal hashes = bitwise equal results
    for (size_t i = 0; i < y.size() * 4; i++) { fnv = (fnv ^ ((const uint8_t *) y.data())[i]) * 0x100000001b3ull; }
    printf("output fnv1a %016llx\n", (unsigned long long) fnv);
    double max_rel = 0, max_abs = 0; int bad = 0;
    std::uniform_int_distribution<int> cd(0, n_used * T - 1);
    for (int c = 0; c < check_cols; c++) {
        const int col = c == 0 ? 0 : cd(rng), t = col / n_used, s = col % n_used, e = ids[col];
        const float * xc = &x[((int64_t) t * ne11 + (down ? s : 0)) * K];
        for (int64_t n = 0; n < N; n++) {
            auto dot = [&](const std::vector<uint16_t> & wt, double & mg) {
                const uint16_t * wr = &wt[((int64_t) e * N + n) * K];
                double r = 0;
                mg = 0;
                for (int64_t k = 0; k < K; k++) {
                    const double p = (double) bf16_to_f32(wr[k]) * (T > 8 ? round_bf16(xc[k]) : xc[k]);
                    r += p; mg += std::fabs(p);
                }
                return r;
            };
            double mag, ref = dot(w, mag);
            if (glu) { // silu(gate) * up; first-order error bound from both dot products (|silu'| < 1.1)
                double mag_u;
                const double u = dot(w2, mag_u), sg = ref / (1.0 + std::exp(-ref));
                mag = mag_u * std::fabs(sg) + 1.1 * std::fabs(u) * mag;
                ref = sg * u;
            }
            const double d = std::fabs(y[(int64_t) col * N + n] - ref);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / (mag + 1e-30));
            bad += !(d <= 1e-5 * mag + 1e-6) || std::isnan(y[(int64_t) col * N + n]);
        }
    }
    printf("check %d columns x %lld rows: max |err| %.3g, max |err| / sum|w x| %.3g, %d outside 1e-5 -> %s\n", check_cols,
           (long long) N, max_abs, max_rel, bad, bad ? "FAIL" : "OK");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    gguf_free(g);
    return bad ? 2 : 0;
}
