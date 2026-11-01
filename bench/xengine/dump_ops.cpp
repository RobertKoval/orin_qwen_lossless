// SPDX-License-Identifier: MIT
// One real-weight BF16N graph per process: compatible with the read-only S1.5 build.
// No timing/warm-up. Output is native little-endian IEEE F32, compared byte-for-byte.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "size_limit.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static int positive(const char * s) {
    char * end = nullptr;
    long n = strtol(s, &end, 10);
    require(end != s && !*end && n > 0 && n <= 65536, "invalid positive integer");
    return int(n);
}
int main(int argc, char ** argv) try {
    require(argc == 9, "usage: dump_ops MODEL TENSOR OP TOKENS FORMAT ROUTING OUTPUT BACKEND_DEVICE");
    const size_t device_cap = bf16n_test::memory_budget("BF16N_TEST_MAX_DEVICE_BYTES", size_t(4) << 30);
    const size_t host_cap = bf16n_test::memory_budget("BF16N_TEST_MAX_HOST_BYTES", size_t(4) << 30);
    const std::string name = argv[2], op = argv[3], routing = argv[6];
    const int T = positive(argv[4]), fmt = positive(argv[5]);
    require(fmt == 1 || fmt == 2, "format must be 1 (nibble) or 2 (auto Huffman)");
    require(routing == "uniform" || routing == "concentrated", "invalid routing");
    const bool row_view = op == "get_rows_view_first" || op == "get_rows_view_last";
    const bool rows = op == "get_rows" || row_view;
    require(op == "mul_mat" || op == "mul_mat_id" || op == "glu" || rows, "invalid op");
    uint32_t endian = 1;
    require(*(uint8_t *) &endian == 1 && sizeof(float) == 4, "requires little-endian F32 host");
    ggml_context * meta = nullptr;
    gguf_context * gguf = gguf_init_from_file(argv[1], {true, &meta});
    require(gguf && meta, "cannot open GGUF");
    FILE * model = fopen(argv[1], "rb");
    require(model, "cannot open model data");
    size_t weight_bytes = 0;
    int64_t K = 0, N = 0, E = 0;
    auto read = [&](const std::string & tensor) {
        int64_t id = gguf_find_tensor(gguf, tensor.c_str());
        require(id >= 0, ("missing tensor " + tensor).c_str());
        ggml_tensor * t = ggml_get_tensor(meta, tensor.c_str());
        require(t && (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F32) && t->ne[3] == 1,
                "expected BF16 or exactly-BF16-representable F32 matrix/experts");
        if (K) { require(K == t->ne[0] && N == t->ne[1] && E == t->ne[2], "gate/up shape mismatch"); }
        K = t->ne[0]; N = t->ne[1]; E = t->ne[2];
        const size_t raw_bytes = ggml_nbytes(t);
        require(raw_bytes <= host_cap && weight_bytes <= host_cap - raw_bytes, "weight read exceeds host budget");
        weight_bytes += raw_bytes;
        std::vector<uint16_t> w(ggml_nelements(t));
        require(fseeko(model, gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id), SEEK_SET) == 0,
                "cannot seek tensor");
        if (t->type == GGML_TYPE_BF16) {
            require(fread(w.data(), 2, w.size(), model) == w.size(), "short tensor read");
        } else {
            // Match the loader's lossless F32-router conversion, without allowing rounding.
            for (auto & value : w) {
                uint32_t bits;
                require(fread(&bits, 4, 1, model) == 1, "short F32 tensor read");
                require((bits & 0xffffu) == 0, "F32 weight not exactly representable in BF16");
                value = uint16_t(bits >> 16);
            }
            weight_bytes -= raw_bytes / 2;
        }
        return w;
    };
    std::string gate = name, up = name;
    if (op == "glu") {
        auto at = name.find("gate_up");
        require(at != std::string::npos, "GLU tensor must name gate_up");
        gate.replace(at, 7, "gate"); up.replace(at, 7, "up");
    }
    auto w = read(gate);
    auto w2 = op == "glu" ? read(up) : std::vector<uint16_t>();
    fclose(model);
    const bool expert = E > 1;
    require((op != "mul_mat_id" || expert) && (op != "mul_mat" || !expert) &&
            (!rows || !expert), "operation/weight rank mismatch");
    const int64_t row0 = op == "get_rows_view_first" ? 1 : op == "get_rows_view_last" ? N - 1 : 0;
    require(!row_view || N > 1, "row view needs at least two rows");
    const int used = expert ? 8 : 1;
    require(!expert || E >= used, "need at least eight experts");
    const int slots = expert && name.find("down") != std::string::npos ? used : 1;
    ggml_backend_t be = ggml_backend_cuda_init(positive(argv[8]) - 1);
    require(be, "CUDA backend unavailable");
    ggml_context * ctx = ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    require(ctx, "context allocation failed");
    auto weight = [&](const std::string & n) {
        auto t = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16N, K, N, E);
        ggml_set_name(t, n.c_str());
        return t;
    };
    auto W = weight(gate);
    auto U = op == "glu" ? weight(up) : nullptr;
    auto X = rows ? nullptr : expert ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, slots, T)
                                                               : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    auto I = expert ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, T)
                    : rows ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T) : nullptr;
    auto product = [&](ggml_tensor * a) { return expert ? ggml_mul_mat_id(ctx, a, X, I) : ggml_mul_mat(ctx, a, X); };
    auto row_source = row_view ? ggml_view_2d(ctx, W, K, 1, W->nb[1], size_t(row0) * W->nb[1]) : W;
    auto Y = rows ? ggml_get_rows(ctx, row_source, I)
           : op == "glu" ? ggml_swiglu_split(ctx, product(W), product(U)) : product(W);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, Y);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        require(ggml_backend_supports_op(be, ggml_graph_node(graph, i)), "unsupported graph operation");
    }
    // Conservative admission estimate, not a hard CUDA RSS limiter: include padded ggml buffers,
    // worst-case encoded aux (NIBBLE escapes), full dequant fallback, input conversion,
    // roundtrip scratch and a 256 MiB cuBLAS/context reserve. No allocation if cap/free-memory fails.
    auto buft = ggml_backend_get_default_buffer_type(be);
    size_t tensor_bytes = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    for (auto t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->view_src) { tensor_bytes += ggml_backend_buft_get_alloc_size(buft, t) + alignment; }
    }
    const size_t elements = weight_bytes / 2;
    const size_t aux_bound = (fmt == 1 ? (elements * 9 + 1) / 2 : elements * 2) + size_t(N * E) * 96 * (U ? 2 : 1) + 32768;
    const size_t scratch = weight_bytes + (X ? ggml_nbytes(X) : 0) + (size_t(320) << 20);
    const size_t estimate = tensor_bytes + aux_bound + scratch;
    size_t available = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &available, &total);
    fprintf(stderr, "memory admission: device_estimate=%zu cap=%zu available=%zu host_weights=%zu\n",
            estimate, device_cap, available, weight_bytes);
    require(estimate <= device_cap && estimate <= available, "device memory estimate exceeds cap/availability");
    require(weight_bytes + (X ? ggml_nbytes(X) : 0) + ggml_nbytes(Y) <= host_cap, "host tensors exceed cap");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    require(buffer && W->buffer && (!U || U->buffer), "tensor allocation/owner buffer missing");
    require(ggml_backend_buffer_get_size(buffer) <= tensor_bytes, "allocated tensor buffer exceeded admission estimate");
    auto convert = (bool (*)(ggml_tensor *, const void *, int)) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(be)), "ggml_backend_cuda_bf16n_convert");
    require(convert && convert(W, w.data(), fmt) && (!U || convert(U, w2.data(), fmt)), "BF16N conversion failed");
    // Integer-defined PRNG and exactly representable activations, independent of the C++ standard library RNG.
    std::vector<float> x(X ? ggml_nelements(X) : 0);
    uint32_t rng = 0x1234abcd;
    for (auto & v : x) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; v = (int(rng & 8191) - 4096) / 4096.0f; }
    if (X) { ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X)); }
    std::vector<int32_t> ids(I ? ggml_nelements(I) : 0);
    for (size_t j = 0; j < ids.size(); ++j) {
        ids[j] = rows ? (row_view ? 0 : j == 0 ? 0 : j == 1 ? N - 1 : (j * 7919) % N)
               : routing == "uniform" ? j % E : j % used;
    }
    if (I) { ggml_backend_tensor_set(I, ids.data(), 0, ggml_nbytes(I)); }
    require(ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> y(ggml_nelements(Y));
    require(Y->type == GGML_TYPE_F32, "output must be F32");
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    for (float v : y) { require(std::isfinite(v), "nonfinite output"); }
    // GET_ROWS also checks independently against the raw BF16 bytes, not just A/B.
    if (rows) {
        for (int t = 0; t < T; ++t) for (int64_t k = 0; k < K; ++k) {
            uint32_t bits = uint32_t(w[(ids[t] + row0) * K + k]) << 16;
            require(memcmp(&bits, &y[t * K + k], 4) == 0, "GET_ROWS differs from raw BF16");
        }
    }
    FILE * out = fopen(argv[7], "wb");
    require(out, "cannot create dump");
    require(fwrite(y.data(), 4, y.size(), out) == y.size(), "short output write");
    require(fclose(out) == 0, "output close failed");
    printf("{\"shape\":[%lld,%lld,%lld,%lld],\"bytes\":%zu}\n", (long long) Y->ne[0],
           (long long) Y->ne[1], (long long) Y->ne[2], (long long) Y->ne[3], y.size() * 4);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx); ggml_backend_free(be); ggml_free(meta); gguf_free(gguf);
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "dump_ops: %s\n", e.what());
    return 1;
}
