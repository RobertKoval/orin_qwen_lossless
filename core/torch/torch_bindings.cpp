// SPDX-License-Identifier: MIT
// torch.ops.bf16n: BF16N encode (CPU) and the CUDA ops. Weights are passed explicitly as plane + aux uint8 tensors with
// int metadata parsed once from the aux header (vllm_bf16n/ops.py); no address registry. The CUDA ops allocate only
// their outputs / workspaces through the caching allocator, never synchronise with the host and never read the aux
// header on the host, so they can be captured into CUDA graphs. Fake (meta) implementations live in ops.py.
//
// meta = [format, K, rows, R, flags (aux header: FLAG_CHUNK_MAJOR | FLAG_FRAG_ORDER), off_a, off_b, off_c, off_d]
#include "bf16n/encode.h"
#include "bf16n/launch.h"
#include "bf16n/policy.h"

#include <pybind11/stl.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstring>
#include <tuple>

namespace bf16n {
void int4_quantize(const void * w_bf16, int64_t rows, int K, uint8_t * q, float * s, cudaStream_t stream);
void int4_gemv(const void * x_bf16, int T, const uint8_t * q, const float * s, int64_t rows, int K, float * y,
               int sm_count, cudaStream_t stream);
void int2_quantize(const void * w_bf16, int64_t rows, int K, uint8_t * q, float * s, cudaStream_t stream);
void int2_gemv(const void * x_bf16, int T, const uint8_t * q, const float * s, int64_t rows, int K, float * y,
               int sm_count, cudaStream_t stream);
} // namespace bf16n

namespace {

using at::Tensor;

constexpr int META_LEN = 9;

bf16n::weight make_weight(const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta) {
    TORCH_CHECK(meta.size() == META_LEN, "bf16n: meta must have ", META_LEN, " entries, got ", meta.size());
    TORCH_CHECK(plane.is_cuda() && aux.is_cuda(), "bf16n: plane and aux must be CUDA tensors");
    TORCH_CHECK(plane.scalar_type() == at::kByte && aux.scalar_type() == at::kByte, "bf16n: plane and aux must be uint8");
    TORCH_CHECK(plane.is_contiguous() && aux.is_contiguous(), "bf16n: plane and aux must be contiguous");
    TORCH_CHECK(plane.get_device() == aux.get_device(), "bf16n: plane and aux on different devices");
    bf16n::weight w;
    w.plane = plane.data_ptr<uint8_t>();
    w.aux = aux.data_ptr<uint8_t>();
    w.format = (int) meta[0];
    w.K = meta[1];
    w.rows = meta[2];
    w.R = (int) meta[3];
    TORCH_CHECK((meta[4] & ~(int64_t) (bf16n::FLAG_CHUNK_MAJOR | bf16n::FLAG_FRAG_ORDER)) == 0,
                "bf16n: unknown layout flags ", meta[4]);
    w.cm = (meta[4] & bf16n::FLAG_CHUNK_MAJOR) != 0;
    w.fo = (meta[4] & bf16n::FLAG_FRAG_ORDER) != 0;
    w.off_a = (uint64_t) meta[5]; w.off_b = (uint64_t) meta[6]; w.off_c = (uint64_t) meta[7]; w.off_d = (uint64_t) meta[8];
    TORCH_CHECK(plane.numel() == w.rows * w.K, "bf16n: plane has ", plane.numel(), " bytes, meta says ", w.rows, " x ", w.K);
    const uint64_t last = std::max({w.off_a, w.off_b, w.off_c, w.off_d});
    TORCH_CHECK(last < (uint64_t) aux.numel(), "bf16n: aux (", aux.numel(), " bytes) shorter than its offsets");
    TORCH_CHECK(((uintptr_t) w.plane % 16) == 0 && ((uintptr_t) w.aux % 16) == 0, "bf16n: plane/aux must be 16-byte aligned");
    return w;
}

void check_x(const Tensor & x, int64_t K, const char * what) {
    TORCH_CHECK(x.is_cuda(), "bf16n: ", what, " must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "bf16n: ", what, " must be bf16 or f32");
    TORCH_CHECK(x.size(-1) == K && x.stride(-1) == 1, "bf16n: ", what, " last dim must be K = ", K, " and contiguous");
    TORCH_CHECK(((uintptr_t) x.data_ptr() % 16) == 0, "bf16n: ", what, " must be 16-byte aligned");
    for (int d = 0; d + 1 < x.dim(); d++) {
        TORCH_CHECK((x.stride(d) * x.element_size()) % 16 == 0, "bf16n: ", what, " row strides must be multiples of 16 bytes");
    }
}

void check_ids(const Tensor & ids) {
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2, "bf16n: ids must be a 2-D CUDA tensor [tokens, n_used]");
    TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "bf16n: ids must be int32 or int64");
    TORCH_CHECK(ids.stride(1) == 1, "bf16n: ids must be contiguous along the slot dimension");
}

// ---------------------------------------------------------------- CPU

std::tuple<Tensor, Tensor> encode_cpu(const Tensor & w, int64_t format, int64_t R, bool chunk_major, bool frag_order) {
    TORCH_CHECK(w.device().is_cpu(), "bf16n.encode: weight must be on the CPU");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "bf16n.encode: weight must be bfloat16");
    TORCH_CHECK(w.dim() >= 2, "bf16n.encode: weight must be at least 2-D (rows x K)");
    const Tensor wc = w.contiguous();
    const int64_t K = wc.size(-1);
    const int64_t rows = wc.numel() / std::max<int64_t>(K, 1);
    bf16n::encoded e;
    try {
        e = bf16n::encode((const uint16_t *) wc.data_ptr(), rows, K, (int) format, (int) R, chunk_major, frag_order);
    } catch (const std::invalid_argument & ex) {
        TORCH_CHECK_VALUE(false, ex.what());
    }
    Tensor plane = at::empty({rows, K}, at::TensorOptions().dtype(at::kByte));
    Tensor aux = at::empty({(int64_t) e.aux.size()}, at::TensorOptions().dtype(at::kByte));
    memcpy(plane.data_ptr(), e.plane.data(), e.plane.size());
    memcpy(aux.data_ptr(), e.aux.data(), e.aux.size());
    return {plane, aux};
}

// ---------------------------------------------------------------- CUDA

#define BF16N_TRY(stmt)                                                                                    \
    try {                                                                                                  \
        stmt;                                                                                              \
    } catch (const std::invalid_argument & ex_) {                                                          \
        TORCH_CHECK_VALUE(false, ex_.what());                                                              \
    }

void init_op() { bf16n::init(); }

Tensor dequant_cuda(const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta, int64_t row0, int64_t nrows) {
    const c10::cuda::CUDAGuard guard(plane.device());
    const bf16n::weight w = make_weight(plane, aux, meta);
    TORCH_CHECK(row0 >= 0 && nrows >= 0 && row0 + nrows <= w.rows, "bf16n.dequant: rows out of range");
    Tensor out = at::empty({nrows, w.K}, plane.options().dtype(at::kBFloat16));
    BF16N_TRY(bf16n::dequant(w, row0, nrows, (uint16_t *) out.data_ptr(), at::cuda::getCurrentCUDAStream()));
    return out;
}

// shared by mmv and mmv_glu; glu halves the output rows
Tensor mmv_common(const Tensor & x, const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta,
                  const c10::optional<Tensor> & ids, int64_t rows_per_matrix, bool out_bf16, bool glu) {
    const c10::cuda::CUDAGuard guard(plane.device());
    const bf16n::weight w = make_weight(plane, aux, meta);
    check_x(x, w.K, "x");
    bf16n::mmv_args a;
    a.x = x.data_ptr();
    a.x_bf16 = x.scalar_type() == at::kBFloat16;
    a.y_bf16 = out_bf16;
    const auto odt = out_bf16 ? at::kBFloat16 : at::kFloat;
    const int64_t osz = out_bf16 ? 2 : 4;
    Tensor y;
    if (!ids.has_value()) {
        TORCH_CHECK(x.dim() == 2, "bf16n.mmv: dense x must be [tokens, K]");
        const int64_t rpm = rows_per_matrix > 0 ? rows_per_matrix : w.rows;
        TORCH_CHECK(rpm == w.rows, "bf16n.mmv: a dense product uses the whole matrix");
        const int64_t n_out = glu ? rpm / 2 : rpm;
        TORCH_CHECK(!glu || rpm % 2 == 0, "bf16n.mmv_glu: odd row count");
        const int64_t T = x.size(0);
        y = at::empty({T, n_out}, x.options().dtype(odt));
        if (T == 0) {
            return y;
        }
        a.y = y.data_ptr();
        a.ncols = (int) T;
        a.x_tok = x.stride(0) * x.element_size();
        a.y_tok = n_out * osz;
        a.rows_per_matrix = rpm;
        a.nrows = n_out;
    } else {
        const Tensor & id = *ids;
        check_ids(id);
        TORCH_CHECK(id.get_device() == plane.get_device(), "bf16n.mmv: ids on another device");
        const int64_t T = id.size(0), n_used = id.size(1);
        TORCH_CHECK(rows_per_matrix > 0 && w.rows % rows_per_matrix == 0, "bf16n.mmv: rows_per_matrix must divide the rows");
        TORCH_CHECK(!glu || rows_per_matrix % 2 == 0, "bf16n.mmv_glu: odd rows per matrix");
        const int64_t n_out = glu ? rows_per_matrix / 2 : rows_per_matrix;
        TORCH_CHECK((x.dim() == 2 && x.size(0) == T) || (x.dim() == 3 && x.size(0) == T && x.size(1) == n_used),
                    "bf16n.mmv: x must be [tokens, K] or [tokens, n_used, K]");
        y = at::empty({T, n_used, n_out}, x.options().dtype(odt));
        if (T == 0 || n_used == 0) {
            return y;
        }
        a.y = y.data_ptr();
        a.ncols = (int) (T * n_used);
        a.ids = id.data_ptr();
        a.ids_s0 = id.element_size();
        a.ids_s1 = id.stride(0) * id.element_size();
        a.n_used = (int) n_used;
        a.x_tok = x.stride(0) * x.element_size();
        a.x_per_slot = x.dim() == 3 ? (int) n_used : 1;
        a.x_slot = x.dim() == 3 ? x.stride(1) * x.element_size() : 0;
        a.y_tok = n_used * n_out * osz;
        a.y_slot = n_out * osz;
        a.rows_per_matrix = rows_per_matrix;
        a.nrows = n_out;
    }
    BF16N_TRY(glu ? bf16n::mmv_glu(w, a, at::cuda::getCurrentCUDAStream()) : bf16n::mmv(w, a, at::cuda::getCurrentCUDAStream()));
    return y;
}

Tensor mmv_cuda(const Tensor & x, const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta,
                const c10::optional<Tensor> & ids, int64_t rows_per_matrix, bool out_bf16) {
    return mmv_common(x, plane, aux, meta, ids, rows_per_matrix, out_bf16, false);
}

Tensor mmv_glu_cuda(const Tensor & x, const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta,
                    const c10::optional<Tensor> & ids, int64_t rows_per_matrix, bool out_bf16) {
    return mmv_common(x, plane, aux, meta, ids, rows_per_matrix, out_bf16, true);
}

// y = x W^T (+ bias) in BF16 for vLLM linear layers: up to GEMV_MAX_COLS tokens through the decode GEMV (FP32
// accumulation, BF16 out), more through exact dequant + cuBLAS BF16 (the llama.cpp dispatch, T-009). The dequant runs in
// row chunks of at most DEQUANT_CHUNK_BYTES so that lm_head-sized matrices never materialise in full.
constexpr int64_t GEMV_MAX_COLS = 8;
constexpr int64_t DEQUANT_CHUNK_BYTES = 64ll << 20;

Tensor linear_cuda(const Tensor & x, const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta,
                   const c10::optional<Tensor> & bias) {
    const c10::cuda::CUDAGuard guard(plane.device());
    const bf16n::weight w = make_weight(plane, aux, meta);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "bf16n.linear: x must be bf16");
    TORCH_CHECK(x.size(-1) == w.K, "bf16n.linear: x last dim must be K = ", w.K);
    std::vector<int64_t> oshape(x.sizes().begin(), x.sizes().end());
    oshape.back() = w.rows;
    const Tensor x2 = x.reshape({-1, w.K});
    const int64_t T = x2.size(0);
    Tensor y;
    if (T <= GEMV_MAX_COLS) {
        const Tensor xc = (x2.stride(1) == 1 && (x2.stride(0) * 2) % 16 == 0 && ((uintptr_t) x2.data_ptr() % 16) == 0) ? x2 : x2.contiguous();
        if (bias.has_value()) { // bias added in FP32 before the one rounding to BF16, as cuBLAS' epilogue does
            y = mmv_common(xc, plane, aux, meta, c10::nullopt, 0, false, false).add_(*bias).to(at::kBFloat16);
        } else {
            y = mmv_common(xc, plane, aux, meta, c10::nullopt, 0, true, false);
        }
    } else {
        y = at::empty({T, w.rows}, x.options());
        const int64_t unit = w.format == bf16n::FMT_NIBBLE ? 1 : w.R;
        int64_t chunk = std::max<int64_t>(DEQUANT_CHUNK_BYTES / (w.K * 2) / unit * unit, unit);
        chunk = std::min(chunk, w.rows);
        Tensor wdq = at::empty({chunk, w.K}, x.options());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        for (int64_t r0 = 0; r0 < w.rows; r0 += chunk) {
            const int64_t nr = std::min(chunk, w.rows - r0);
            Tensor wc = wdq.narrow(0, 0, nr);
            BF16N_TRY(bf16n::dequant(w, r0, nr, (uint16_t *) wc.data_ptr(), stream));
            if (nr == w.rows) {
                y = bias.has_value() ? at::linear(x2, wc, *bias) : at::mm(x2, wc.t());
            } else {
                Tensor ys = y.narrow(1, r0, nr);
                at::mm_out(ys, x2, wc.t());
                if (bias.has_value()) {
                    ys.add_(bias->narrow(0, r0, nr));
                }
            }
        }
    }
    return y.view(oshape);
}

// draft-only 4-bit head (draft_head.cu): symmetric 4-bit weights, one FP32 scale per 128
std::tuple<Tensor, Tensor> int4_quantize_cuda(const Tensor & w) {
    const c10::cuda::CUDAGuard guard(w.device());
    TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.is_contiguous(),
                "bf16n.int4_quantize: w must be a contiguous 2-D CUDA bf16 tensor");
    TORCH_CHECK_VALUE(w.size(1) % 128 == 0, "bf16n.int4_quantize: K must be a multiple of 128");
    Tensor q = at::empty({w.size(0), w.size(1) / 2}, w.options().dtype(at::kByte));
    Tensor s = at::empty({w.size(0), w.size(1) / 128}, w.options().dtype(at::kFloat));
    BF16N_TRY(bf16n::int4_quantize(w.data_ptr(), w.size(0), (int) w.size(1), q.data_ptr<uint8_t>(), s.data_ptr<float>(),
                                   at::cuda::getCurrentCUDAStream()));
    return {q, s};
}

Tensor int4_gemv_cuda(const Tensor & x, const Tensor & q, const Tensor & s) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                "bf16n.int4_gemv: x must be contiguous bf16 [T, K]");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kByte && q.dim() == 2 && q.size(1) * 2 == x.size(1) && q.is_contiguous(),
                "bf16n.int4_gemv: q must be uint8 [rows, K / 2]");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kFloat && s.size(0) == q.size(0) && s.size(1) * 128 == x.size(1),
                "bf16n.int4_gemv: s must be f32 [rows, K / 128]");
    Tensor y = at::empty({x.size(0), q.size(0)}, x.options().dtype(at::kFloat));
    BF16N_TRY(bf16n::int4_gemv(x.data_ptr(), (int) x.size(0), q.data_ptr<uint8_t>(), s.data_ptr<float>(), q.size(0),
                               (int) x.size(1), y.data_ptr<float>(), bf16n::sm_count(), at::cuda::getCurrentCUDAStream()));
    return y;
}

// draft-only 2-bit head (T-051): level index n in 0..3 per weight, value (n - 1.5) s, one FP32 scale per 128
std::tuple<Tensor, Tensor> int2_quantize_cuda(const Tensor & w) {
    const c10::cuda::CUDAGuard guard(w.device());
    TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.is_contiguous(),
                "bf16n.int2_quantize: w must be a contiguous 2-D CUDA bf16 tensor");
    TORCH_CHECK_VALUE(w.size(1) % 128 == 0, "bf16n.int2_quantize: K must be a multiple of 128");
    Tensor q = at::empty({w.size(0), w.size(1) / 4}, w.options().dtype(at::kByte));
    Tensor s = at::empty({w.size(0), w.size(1) / 128}, w.options().dtype(at::kFloat));
    BF16N_TRY(bf16n::int2_quantize(w.data_ptr(), w.size(0), (int) w.size(1), q.data_ptr<uint8_t>(), s.data_ptr<float>(),
                                   at::cuda::getCurrentCUDAStream()));
    return {q, s};
}

Tensor int2_gemv_cuda(const Tensor & x, const Tensor & q, const Tensor & s) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                "bf16n.int2_gemv: x must be contiguous bf16 [T, K]");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kByte && q.dim() == 2 && q.size(1) * 4 == x.size(1) && q.is_contiguous(),
                "bf16n.int2_gemv: q must be uint8 [rows, K / 4]");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kFloat && s.size(0) == q.size(0) && s.size(1) * 128 == x.size(1),
                "bf16n.int2_gemv: s must be f32 [rows, K / 128]");
    Tensor y = at::empty({x.size(0), q.size(0)}, x.options().dtype(at::kFloat));
    BF16N_TRY(bf16n::int2_gemv(x.data_ptr(), (int) x.size(0), q.data_ptr<uint8_t>(), s.data_ptr<float>(), q.size(0),
                               (int) x.size(1), y.data_ptr<float>(), bf16n::sm_count(), at::cuda::getCurrentCUDAStream()));
    return y;
}

Tensor get_rows_cuda(const Tensor & ids, const Tensor & plane, const Tensor & aux, c10::IntArrayRef meta, bool out_bf16) {
    const c10::cuda::CUDAGuard guard(plane.device());
    const bf16n::weight w = make_weight(plane, aux, meta);
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 1, "bf16n.get_rows: ids must be a 1-D CUDA tensor");
    TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "bf16n.get_rows: ids must be int32 or int64");
    const int64_t n = ids.size(0);
    Tensor out = at::empty({n, w.K}, plane.options().dtype(out_bf16 ? at::kBFloat16 : at::kFloat));
    BF16N_TRY(bf16n::get_rows(w, ids.data_ptr(), ids.stride(0) * ids.element_size(), n, out.data_ptr(), out_bf16,
                              w.K * out.element_size(), at::cuda::getCurrentCUDAStream()));
    return out;
}

Tensor moe_mm_ids_cuda(const Tensor & x, const Tensor & topk_ids, const Tensor & plane, const Tensor & aux,
                       c10::IntArrayRef meta, const c10::optional<Tensor> & up_plane, const c10::optional<Tensor> & up_aux,
                       c10::IntArrayRef up_meta, int64_t n_experts, bool out_bf16) {
    const c10::cuda::CUDAGuard guard(plane.device());
    const bf16n::weight w = make_weight(plane, aux, meta);
    const bool glu = up_plane.has_value();
    TORCH_CHECK(glu == up_aux.has_value() && glu == !up_meta.empty(), "bf16n.moe_mm_ids: give all of up_plane/up_aux/up_meta or none");
    bf16n::weight u;
    if (glu) {
        u = make_weight(*up_plane, *up_aux, up_meta);
    }
    check_ids(topk_ids);
    check_x(x, w.K, "x");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "bf16n.moe_mm_ids: x must be contiguous bf16");
    TORCH_CHECK(n_experts > 0 && w.rows % n_experts == 0, "bf16n.moe_mm_ids: rows must split into n_experts");
    const int64_t T = topk_ids.size(0), n_used = topk_ids.size(1), N = w.rows / n_experts;
    // x per token (gate/up: [T, K]) or per routed slot (down: [T, n_used, K])
    const bool per_slot = x.dim() == 3;
    TORCH_CHECK((x.dim() == 2 && x.size(0) == T) || (per_slot && x.size(0) == T && x.size(1) == n_used),
                "bf16n.moe_mm_ids: x must be [tokens, K] or [tokens, n_used, K]");
    Tensor y = at::empty({T, n_used, N}, x.options().dtype(out_bf16 ? at::kBFloat16 : at::kFloat));
    if (T == 0) {
        return y;
    }
    const auto i32 = topk_ids.options().dtype(at::kInt);
    Tensor ids_src1 = at::empty({T * n_used}, i32), ids_dst = at::empty({T * n_used}, i32), bounds = at::empty({n_experts + 1}, i32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    BF16N_TRY({
        if (!bf16n::mm_ids_supported(w, glu ? &u : nullptr)) {
            throw std::invalid_argument("bf16n.moe_mm_ids: no batched kernel for this matrix (format/K/R/layout)");
        }
        bf16n::mm_ids_route(topk_ids.data_ptr(), topk_ids.scalar_type() == at::kLong, topk_ids.stride(0),
                            ids_src1.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(), bounds.data_ptr<int32_t>(),
                            (int) n_experts, (int) T, (int) n_used, per_slot ? (int) n_used : 1, per_slot ? (int) n_used : 1, stream);
        bf16n::mm_ids(w, glu ? &u : nullptr, x.data_ptr(), w.K, ids_src1.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),
                      bounds.data_ptr<int32_t>(), y.data_ptr(), out_bf16, N, (int) n_experts, (int) T, (int) n_used, stream);
    });
    return y;
}

Tensor moe_combine_cuda(const Tensor & ys, const Tensor & topk_weights, bool out_bf16) {
    const c10::cuda::CUDAGuard guard(ys.device());
    TORCH_CHECK(ys.is_cuda() && ys.dim() == 3 && ys.scalar_type() == at::kFloat && ys.is_contiguous(),
                "bf16n.moe_combine: ys must be contiguous f32 [tokens, n_used, N]");
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.dim() == 2 && topk_weights.is_contiguous() &&
                (topk_weights.scalar_type() == at::kFloat || topk_weights.scalar_type() == at::kBFloat16),
                "bf16n.moe_combine: topk_weights must be contiguous f32/bf16 [tokens, n_used]");
    const int64_t T = ys.size(0), n_used = ys.size(1), N = ys.size(2);
    TORCH_CHECK(topk_weights.size(0) == T && topk_weights.size(1) == n_used, "bf16n.moe_combine: shape mismatch");
    Tensor y = at::empty({T, N}, ys.options().dtype(out_bf16 ? at::kBFloat16 : at::kFloat));
    BF16N_TRY(bf16n::moe_combine(ys.data_ptr<float>(), topk_weights.data_ptr(), topk_weights.scalar_type() == at::kBFloat16,
                                 y.data_ptr(), out_bf16, (int) T, (int) n_used, N, at::cuda::getCurrentCUDAStream()));
    return y;
}

Tensor swiglu_cuda(const Tensor & g, const Tensor & u) {
    const c10::cuda::CUDAGuard guard(g.device());
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.is_contiguous() && u.sizes() == g.sizes() &&
                u.scalar_type() == at::kFloat && u.is_contiguous(), "bf16n.swiglu: g and u must be contiguous f32 of one shape");
    Tensor y = at::empty_like(g);
    BF16N_TRY(bf16n::swiglu(g.data_ptr<float>(), u.data_ptr<float>(), y.data_ptr<float>(), g.numel(), at::cuda::getCurrentCUDAStream()));
    return y;
}

} // namespace

TORCH_LIBRARY(bf16n, m) {
    m.def("init() -> ()", &init_op);
    m.def("encode(Tensor w, int format, int R, bool chunk_major, bool frag_order=False) -> (Tensor, Tensor)");
    m.def("dequant(Tensor plane, Tensor aux, int[] meta, int row0, int nrows) -> Tensor");
    m.def("mmv(Tensor x, Tensor plane, Tensor aux, int[] meta, Tensor? ids, int rows_per_matrix, bool out_bf16) -> Tensor");
    m.def("mmv_glu(Tensor x, Tensor plane, Tensor aux, int[] meta, Tensor? ids, int rows_per_matrix, bool out_bf16) -> Tensor");
    m.def("linear(Tensor x, Tensor plane, Tensor aux, int[] meta, Tensor? bias) -> Tensor");
    m.def("get_rows(Tensor ids, Tensor plane, Tensor aux, int[] meta, bool out_bf16) -> Tensor");
    m.def("moe_mm_ids(Tensor x, Tensor topk_ids, Tensor plane, Tensor aux, int[] meta, Tensor? up_plane, Tensor? up_aux, "
          "int[] up_meta, int n_experts, bool out_bf16) -> Tensor");
    m.def("moe_combine(Tensor ys, Tensor topk_weights, bool out_bf16) -> Tensor");
    m.def("swiglu(Tensor g, Tensor u) -> Tensor");
    m.def("int4_quantize(Tensor w) -> (Tensor, Tensor)");
    m.def("int4_gemv(Tensor x, Tensor q, Tensor s) -> Tensor");
    m.def("int2_quantize(Tensor w) -> (Tensor, Tensor)");
    m.def("int2_gemv(Tensor x, Tensor q, Tensor s) -> Tensor");
}

TORCH_LIBRARY_IMPL(bf16n, CPU, m) {
    m.impl("encode", &encode_cpu);
}

TORCH_LIBRARY_IMPL(bf16n, CUDA, m) {
    m.impl("dequant", &dequant_cuda);
    m.impl("mmv", &mmv_cuda);
    m.impl("mmv_glu", &mmv_glu_cuda);
    m.impl("linear", &linear_cuda);
    m.impl("get_rows", &get_rows_cuda);
    m.impl("moe_mm_ids", &moe_mm_ids_cuda);
    m.impl("moe_combine", &moe_combine_cuda);
    m.impl("swiglu", &swiglu_cuda);
    m.impl("int4_quantize", &int4_quantize_cuda);
    m.impl("int4_gemv", &int4_gemv_cuda);
    m.impl("int2_quantize", &int2_quantize_cuda);
    m.impl("int2_gemv", &int2_gemv_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BF16N lossless BF16 weight ops (registered as torch.ops.bf16n)";
    m.def("choose", [](int64_t rows, int64_t K, int64_t n_matrices)
            -> std::optional<std::tuple<int, int, bool, bool>> {
        const auto c = bf16n::choose(rows, K, n_matrices); // vLLM defaults; no CUDA access
        if (!c) {
            return std::nullopt;
        }
        return std::make_tuple(c->format, c->R, c->chunk_major, c->frag_order);
    }, pybind11::arg("rows"), pybind11::arg("K"), pybind11::arg("n_matrices") = 1);
}
