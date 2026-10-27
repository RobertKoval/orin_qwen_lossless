// Accuracy and speed of the CUDA gated_delta_net op (one Qwen3.6 GDN layer: 32 value heads, 16 key heads, 128 x 128
// state) against a sequential recurrence in double precision. Inputs follow the model: q and k L2-normalised, beta =
// sigmoid(x), log-decay g = ssm_a[h] * softplus(x + dt_bias[h]) with ssm_a and dt_bias read from the model's layer.
// GGML_CUDA_GDN_NO_MMA=1 selects the sequential kernel when the chunked one would run.
//
//   gdn_check <model.gguf> <layer> <tokens> [reps=10] [state=0|1]
//
// state=1 starts from a random state (|S| ~ 0.1) instead of zeros. Prints the relative RMS error of the outputs per
// token range and of the final state, the largest relative output error, and the median time per op.
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

static std::vector<float> read_f32(const char * path, gguf_context * g, ggml_context * meta, const std::string & name, int64_t n) {
    const int64_t tid = gguf_find_tensor(g, name.c_str());
    if (tid < 0) { fprintf(stderr, "no tensor %s\n", name.c_str()); exit(1); }
    const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != n) { fprintf(stderr, "%s: need %lld F32 values\n", name.c_str(), (long long) n); exit(1); }
    std::vector<float> v(n);
    const int fd = open(path, O_RDONLY);
    if (pread(fd, v.data(), n * 4, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid)) != n * 4) { perror("pread"); exit(1); }
    close(fd);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf layer tokens [reps] [state]\n", argv[0]); return 1; }
    const char * path = argv[1];
    const int il = atoi(argv[2]);
    const int T = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 10;
    const bool rand_state = argc > 5 && atoi(argv[5]) != 0;
    if (T < 1 || reps < 1) { fprintf(stderr, "tokens and reps must be >= 1\n"); return 1; }
    const int S = 128, HV = 32, HK = 16;

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    const std::string pre = "blk." + std::to_string(il) + ".";
    const std::vector<float> ssm_a = read_f32(path, g, meta, pre + "ssm_a", HV);
    const std::vector<float> dt_b  = read_f32(path, g, meta, pre + "ssm_dt.bias", HV);

    std::mt19937 rng(1234 + il);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q((size_t) S * HK * T), k(q.size()), v((size_t) S * HV * T), gl((size_t) HV * T), beta(gl.size());
    std::vector<float> s0((size_t) S * S * HV, 0.f);
    for (auto * x : {&q, &k}) {
        for (size_t r = 0; r < x->size(); r += S) {
            double n2 = 0;
            for (int i = 0; i < S; i++) { (*x)[r + i] = nd(rng); n2 += (double) (*x)[r + i] * (*x)[r + i]; }
            const float inv = (float) (1.0 / sqrt(n2));
            for (int i = 0; i < S; i++) { (*x)[r + i] *= inv; }
        }
    }
    for (auto & x : v) { x = 0.5f * nd(rng); }
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < HV; h++) {
            const float a = nd(rng) + dt_b[h];
            gl[(size_t) t * HV + h]   = ssm_a[h] * (a > 20.f ? a : log1pf(expf(a)));
            beta[(size_t) t * HV + h] = 1.f / (1.f + expf(-nd(rng)));
        }
    }
    if (rand_state) { for (auto & x : s0) { x = 0.1f * nd(rng); } }

    // reference: S[i][c] (key i, value c), per token: S = e^g S; d = beta (v - S^T k); S += k d^T; o = scale S^T q
    const double scale = 1.0 / sqrt((double) S);
    std::vector<double> ref_o((size_t) S * HV * T), ref_s((size_t) S * S * HV);
    {
        std::vector<double> St((size_t) S * S), kv(S);
        for (int h = 0; h < HV; h++) {
            for (int c = 0; c < S; c++) { for (int i = 0; i < S; i++) { St[(size_t) i * S + c] = s0[((size_t) h * S + c) * S + i]; } }
            for (int t = 0; t < T; t++) {
                const float * qt = &q[((size_t) t * HK + h % HK) * S];
                const float * kt = &k[((size_t) t * HK + h % HK) * S];
                const float * vt = &v[((size_t) t * HV + h) * S];
                const double gd = exp((double) gl[(size_t) t * HV + h]), b = beta[(size_t) t * HV + h];
                std::fill(kv.begin(), kv.end(), 0.0);
                for (int i = 0; i < S; i++) { for (int c = 0; c < S; c++) { St[(size_t) i * S + c] *= gd; kv[c] += St[(size_t) i * S + c] * kt[i]; } }
                for (int c = 0; c < S; c++) { kv[c] = b * (vt[c] - kv[c]); }
                double * o = &ref_o[((size_t) t * HV + h) * S];
                std::fill(o, o + S, 0.0);
                for (int i = 0; i < S; i++) {
                    for (int c = 0; c < S; c++) { St[(size_t) i * S + c] += kt[i] * kv[c]; o[c] += St[(size_t) i * S + c] * qt[i]; }
                }
                for (int c = 0; c < S; c++) { o[c] *= scale; }
            }
            for (int c = 0; c < S; c++) { for (int i = 0; i < S; i++) { ref_s[((size_t) h * S + c) * S + i] = St[(size_t) i * S + c]; } }
        }
    }

    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, T, 1);
    ggml_tensor * G = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, T, 1);
    ggml_tensor * B = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, T, 1);
    ggml_tensor * S0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, 1);
    ggml_tensor * Y = ggml_gated_delta_net(ctx, Q, K, V, G, B, S0, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(Q, q.data(), 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(K, k.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, v.data(), 0, ggml_nbytes(V));
    ggml_backend_tensor_set(G, gl.data(), 0, ggml_nbytes(G));
    ggml_backend_tensor_set(B, beta.data(), 0, ggml_nbytes(B));
    ggml_backend_tensor_set(S0, s0.data(), 0, ggml_nbytes(S0));

    std::vector<double> ms;
    for (int r = 0; r < reps + 1; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        if (r > 0) { ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
    }
    std::sort(ms.begin(), ms.end());
    std::vector<float> y(ggml_nelements(Y));
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));

    // relative RMS error of the outputs per token range, and of the final state
    auto rel_rms = [](const float * a, const double * r, size_t n) {
        double e = 0, s = 0;
        for (size_t i = 0; i < n; i++) { e += (a[i] - r[i]) * (a[i] - r[i]); s += r[i] * r[i]; }
        return sqrt(e / s);
    };
    printf("layer %d T=%d state=%s mma=%s: %.3f ms median (min %.3f max %.3f, %d reps)\n", il, T, rand_state ? "rand" : "zero",
           getenv("GGML_CUDA_GDN_NO_MMA") ? "off" : "auto", ms[ms.size() / 2], ms.front(), ms.back(), reps);
    printf("  output rel RMS error by token range:");
    for (int t0 = 0; t0 < T; t0 = t0 ? t0 * 4 : 16) {
        const int t1 = std::min(T, t0 ? t0 * 4 : 16);
        printf("  [%d,%d) %.2e", t0, t1, rel_rms(&y[(size_t) t0 * HV * S], &ref_o[(size_t) t0 * HV * S], (size_t) (t1 - t0) * HV * S));
    }
    double worst = 0, omax = 0;
    for (double x : ref_o) { omax = std::max(omax, fabs(x)); }
    for (size_t i = 0; i < ref_o.size(); i++) { worst = std::max(worst, fabs(y[i] - ref_o[i])); }
    printf("\n  output max |err| / max |ref| %.2e;  final state rel RMS error %.2e\n", worst / omax,
           rel_rms(&y[(size_t) S * HV * T], ref_s.data(), ref_s.size()));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    gguf_free(g);
    return 0;
}
