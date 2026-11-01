// Lossless BF16 GEMV: raw sign|mantissa byte plane + length-limited Huffman code over exponent PAIRS
// (adjacent weights t=0,1 / t=2,3). Each warp owns a unit of R rows; each lane decodes its own bitstream.
// LUT entry (1<<LP entries): (e_odd << 23) | (e_even << 7) | len  -> bf16x2 exponent field + shift amount.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#ifndef LP
#define LP 12   // max pair code length; 2 pairs per refill check needs 2*LP <= 32
#endif
static_assert(2 * LP <= 32, "two pair codes must fit the guaranteed 32-bit window");
#ifndef IL_MINK
#define IL_MINK 1000000  // word-interleaved lane streams for K >= IL_MINK
#endif
#ifndef OFF16
#define OFF16 1  // per-lane stream offsets as u16 relative to a u32 per-unit base
#endif
#ifndef SMNEXT
#define SMNEXT 0
#endif
#ifndef XREG
#define XREG 1  // K=512: activations in registers
#endif
#ifndef PF
#define PF 4    // prefetch queue depth (words)
#endif
#ifndef NT
#define NT 1024
#endif
#ifndef BPS
#define BPS 1   // blocks per SM
#endif

static inline int pos_of(int l, int s) { int j = s >> 2, t = s & 3; return 128 * j + 4 * l + t; }

// package-merge length-limited Huffman (item trees, O(L n log n))
static std::vector<int> limited_lengths(const std::vector<uint64_t>& f, int L) {
    std::vector<int> syms; for (int i = 0; i < (int)f.size(); i++) if (f[i]) syms.push_back(i);
    std::vector<int> len(f.size(), 0); const int n = syms.size();
    if (n == 1) { len[syms[0]] = 1; return len; }
    if ((1ll << L) < n) { fprintf(stderr, "LP too small for %d symbols\n", n); exit(1); }
    std::sort(syms.begin(), syms.end(), [&](int a, int b) { return f[a] < f[b]; });
    struct It { uint64_t w; int leaf; int a; };
    std::vector<std::vector<It>> lv(L);
    std::vector<It> leaves; for (int s : syms) leaves.push_back({f[s], s, -1});
    lv[0] = leaves;
    for (int d = 1; d < L; d++) {
        std::vector<It> pk; auto& c = lv[d - 1];
        for (size_t i = 0; i + 1 < c.size(); i += 2) pk.push_back({c[i].w + c[i + 1].w, -1, (int)i});
        std::vector<It> mg; size_t i = 0, j = 0;
        while (i < pk.size() || j < leaves.size()) {
            if (j < leaves.size() && (i >= pk.size() || leaves[j].w <= pk[i].w)) mg.push_back(leaves[j++]); else mg.push_back(pk[i++]);
        }
        lv[d] = mg;
    }
    std::function<void(int, int)> walk = [&](int d, int idx) {
        const It& it = lv[d][idx];
        if (it.leaf >= 0) { len[it.leaf]++; return; }
        walk(d - 1, it.a); walk(d - 1, it.a + 1);
    };
    for (int i = 0; i < 2 * n - 2; i++) walk(L - 1, i);
    return len;
}

struct Huf {
    int N, K, R, units;
    std::vector<uint8_t> sm;        // N*K, per 512-chunk lane-major (16 B per lane)
    std::vector<uint32_t> stream;   // concatenated lane streams
    std::vector<uint32_t> lane_off; // units*32 word offsets
    std::vector<uint32_t> unit_base; // OFF16: units word offsets
    std::vector<uint16_t> lane_rel;  // OFF16: units*32 offsets relative to unit_base
    std::vector<int> unit_row;      // units+1 row boundaries (cost-balanced when BETA>0)
    std::vector<uint32_t> lut;      // 1<<LP
    double avg_len;
};

static Huf encode(const uint16_t* w, int N, int K, int R) {
    Huf h; h.N = N; h.K = K; h.R = R; h.units = (N + R - 1) / R;
    const int C = K / 512;
    auto ex = [&](size_t i) { return (w[i] >> 7) & 0xFF; };
    std::vector<uint64_t> freq(65536, 0);
    for (size_t i = 0; i < (size_t)N * K; i += 2) freq[ex(i) | ex(i + 1) << 8]++;
    std::vector<int> len = limited_lengths(freq, LP);
    double al = 0; for (int s = 0; s < 65536; s++) al += freq[s] * (double)len[s];
    h.avg_len = al / ((double)N * K);
    std::vector<uint32_t> code(65536, 0);
    { uint32_t c = 0; for (int l = 1; l <= LP; l++) { for (int s = 0; s < 65536; s++) if (len[s] == l) code[s] = c++; c <<= 1; } }
    h.lut.assign(1 << LP, 0);
    for (int s = 0; s < 65536; s++) if (len[s]) {
        uint32_t lo = code[s] << (LP - len[s]), hi = (code[s] + 1) << (LP - len[s]);
        for (uint32_t i = lo; i < hi; i++) h.lut[i] = ((uint32_t)(s >> 8) << 23) | ((uint32_t)(s & 0xFF) << 7) | (uint32_t)len[s];
    }
    h.sm.resize((size_t)N * K);
    for (int r = 0; r < N; r++) for (int c = 0; c < C; c++) for (int l = 0; l < 32; l++) for (int s = 0; s < 16; s++) {
        uint16_t v = w[(size_t)r * K + c * 512 + pos_of(l, s)];
        h.sm[(size_t)r * K + c * 512 + 16 * l + s] = (uint8_t)(((v >> 15) << 7) | (v & 0x7F));
    }
    // unit boundaries: equal rows, or balanced on cost(row) = 8*K + BETA*exponent_bits(row)
    const double beta = getenv("BETA") ? atof(getenv("BETA")) : 0.0;
    h.unit_row.resize(h.units + 1);
    if (beta > 0) {
        std::vector<double> pre(N + 1, 0);
        for (int r = 0; r < N; r++) {
            double bits = 0;
            for (size_t i = (size_t)r * K; i < (size_t)(r + 1) * K; i += 2) bits += len[ex(i) | ex(i + 1) << 8];
            pre[r + 1] = pre[r] + 8.0 * K + beta * bits;
        }
        h.unit_row[0] = 0; h.unit_row[h.units] = N;
        for (int u = 1; u < h.units; u++) {
            const double t = pre[N] * u / h.units;
            int r = std::lower_bound(pre.begin(), pre.end(), t) - pre.begin();
            if (r > 0 && t - pre[r - 1] < pre[r] - t) r--;
            r = std::max(r, h.unit_row[u - 1] + 1); r = std::min(r, N - (h.units - u));
            h.unit_row[u] = r;
        }
    } else {
        for (int u = 0; u <= h.units; u++) h.unit_row[u] = std::min(N, u * R);
    }
    h.lane_off.resize((size_t)h.units * 32);
    for (int u = 0; u < h.units; u++) for (int l = 0; l < 32; l++) {
        h.lane_off[(size_t)u * 32 + l] = (uint32_t)h.stream.size();
        uint64_t acc = 0; int nb = 0;  // MSB-first bit writer
        for (int r = h.unit_row[u]; r < h.unit_row[u + 1]; r++) for (int c = 0; c < C; c++) for (int s = 0; s < 16; s += 2) {
            size_t i = (size_t)r * K + c * 512 + pos_of(l, s);
            int sym = ex(i) | ex(i + 1) << 8;
            acc = (acc << len[sym]) | code[sym]; nb += len[sym];
            if (nb >= 32) { h.stream.push_back((uint32_t)(acc >> (nb - 32))); nb -= 32; acc &= (nb ? ((1ull << nb) - 1) : 0); }
        }
        if (nb) h.stream.push_back((uint32_t)(acc << (32 - nb)));
    }
    {   // padding cost if lanes of a unit were padded to the longest lane (word-interleaved layout)
        double pad = 0, tot = 0;
        for (int u = 0; u < h.units; u++) {
            uint32_t mx = 0; std::vector<uint32_t> L(32);
            for (int l = 0; l < 32; l++) { uint32_t e = (u * 32 + l + 1 < (int)h.lane_off.size()) ? h.lane_off[u * 32 + l + 1] : h.stream.size(); L[l] = e - h.lane_off[u * 32 + l]; mx = std::max(mx, L[l]); }
            for (int l = 0; l < 32; l++) { pad += mx - L[l]; tot += L[l]; }
        }
        printf("interleave padding: %.0f words = %.2f%% of stream, %.4f b/w\n", pad, 100 * pad / tot, pad * 32 / ((double)N * K));
    }
    if (K >= IL_MINK) {  // word-interleaved layout: word i of lane l at unit_base + 32*i + l (lanes padded to the unit's longest)
        std::vector<uint32_t> il, ub(h.units);
        for (int u = 0; u < h.units; u++) {
            uint32_t L[32], mx = 0;
            for (int l = 0; l < 32; l++) { uint32_t e = (u * 32 + l + 1 < (int)h.lane_off.size()) ? h.lane_off[u * 32 + l + 1] : h.stream.size(); L[l] = e - h.lane_off[u * 32 + l]; mx = std::max(mx, L[l]); }
            ub[u] = il.size();
            for (uint32_t i = 0; i < mx; i++) for (int l = 0; l < 32; l++) il.push_back(i < L[l] ? h.stream[h.lane_off[u * 32 + l] + i] : 0u);
        }
        h.stream = il; h.lane_off = ub;
        for (int i = 0; i < 32 * 8; i++) h.stream.push_back(0);
    }
#if OFF16
    if (K < IL_MINK) {
        h.unit_base.resize(h.units); h.lane_rel.resize((size_t)h.units * 32);
        for (int u = 0; u < h.units; u++) {
            h.unit_base[u] = h.lane_off[(size_t)u * 32];
            for (int l = 0; l < 32; l++) {
                const uint32_t d = h.lane_off[(size_t)u * 32 + l] - h.unit_base[u];
                if (d > 0xFFFF) { fprintf(stderr, "OFF16 overflow\n"); exit(1); }
                h.lane_rel[(size_t)u * 32 + l] = (uint16_t)d;
            }
        }
    }
#endif
    for (int i = 0; i < 8; i++) h.stream.push_back(0);  // slack: prefetch reads ahead
    return h;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ uint32_t compose(uint32_t p, uint32_t e) {  // (p & 0x807F807F) | (e & ~0x807F807F)
    uint32_t d; asm("lop3.b32 %0, %1, %2, 0x807F807F, 0xE4;" : "=r"(d) : "r"(p), "r"(e)); return d;
}

#ifndef SMLD
#define SMLD 1  // 0: ld.global.nc   1: + L1::no_allocate   2: ld.global.cg (L2 only)   3: ld.global.cs
#endif
#ifndef STLD
#define STLD 0  // 0: ld.global.nc   1: + L2::256B prefetch   2: + L2::128B
#endif
__device__ __forceinline__ uint4 ld_sm(const uint4* p) {
    uint4 v;
#if SMLD == 1
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#elif SMLD == 4
    uint64_t pol; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
#elif SMLD == 2
    v = __ldcg(p);
#elif SMLD == 3
    v = __ldcs(p);
#else
    v = __ldg(p);
#endif
    return v;
}
__device__ __forceinline__ uint32_t ld_st(const uint32_t* p) {
    uint32_t v;
#if STLD == 1
    asm("ld.global.nc.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));
#elif STLD == 2
    asm("ld.global.nc.L2::128B.u32 %0, [%1];" : "=r"(v) : "l"(p));
#elif STLD == 4
    uint64_t pol; asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    asm("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
#elif STLD == 3
    asm("ld.global.nc.L1::evict_last.u32 %0, [%1];" : "=r"(v) : "l"(p));
#else
    v = __ldg(p);
#endif
    return v;
}

// {hi:lo} <<= (n & 31), kept as one aligned 64-bit register pair
__device__ __forceinline__ uint64_t shl64w(uint64_t b, uint32_t n) {
    uint64_t d;
    asm("{.reg .b32 l, h; mov.b64 {l, h}, %1; shf.l.wrap.b32 h, l, h, %2; shf.l.wrap.b32 l, 0, l, %2; mov.b64 %0, {l, h};}" : "=l"(d) : "l"(b), "r"(n));
    return d;
}
__device__ __forceinline__ uint32_t hi32(uint64_t b) { return (uint32_t)(b >> 32); }

template <int K>
__global__ void __launch_bounds__(NT) gemv_huf(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream,
        const uint32_t* __restrict__ lane_off, const uint16_t* __restrict__ lane_rel, const uint32_t* __restrict__ lut_g, const float* __restrict__ x,
        float* __restrict__ y, const int* __restrict__ unit_row, int units, int rows_per_group, int groups, unsigned long long* __restrict__ diag) {
    constexpr int C = K / 512;
    constexpr int WPB = NT / 32;
    unsigned long long t_start; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_start));
    extern __shared__ float shm[];
    float* xs = shm;
    uint32_t* lut = reinterpret_cast<uint32_t*>(shm + K * groups);
    for (int i = threadIdx.x; i < K * groups; i += NT) xs[i] = x[i];
    for (int i = threadIdx.x; i < (1 << LP); i += NT) lut[i] = lut_g[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int u = blockIdx.x * WPB + warp; u < units; u += gridDim.x * WPB) {
        constexpr bool ILK = (K >= IL_MINK);
        constexpr int SW = ILK ? 32 : 1;  // word stride of one lane's stream
        const uint32_t* lst = ILK ? stream + lane_off[u] + lane : stream;
#if OFF16
        uint32_t idx = ILK ? 0u : __ldg(lane_off + u) + __ldg(lane_rel + (size_t)u * 32 + lane);
#else
        uint32_t idx = ILK ? 0u : lane_off[(size_t)u * 32 + lane];
#endif
        // bit buffer {hi:lo}, MSB-first; v = valid bits (exact in low 7 bits, garbage above); v>=32 after each check
        uint64_t buf = (uint64_t)ld_st(lst + SW * idx) << 32;
        uint32_t wq[PF];
#pragma unroll
        for (int i = 0; i < PF; i++) wq[i] = ld_st(lst + SW * (idx + 1 + i));
        idx += 1 + PF;
        uint32_t v = 32, pa = 0, pb = 0;
        const int r0 = __ldg(unit_row + u), r_end = __ldg(unit_row + u + 1);
        int g = r0 / rows_per_group, g_next = (g + 1) * rows_per_group;
        const float* xg = xs + g * K;
#if XREG
        constexpr bool XR = (C == 1);  // K=512: this lane's 16 activations stay in registers
        float4 xr[4];
        if (XR) {
#pragma unroll
            for (int j = 0; j < 4; j++) xr[j] = *reinterpret_cast<const float4*>(xg + 128 * j + 4 * lane);
        }
#endif
#if SMNEXT
        constexpr bool SN = (C == 1);  // K=512: sign|mantissa chunk of the next row loaded one row ahead
        uint4 smn = SN ? ld_sm(reinterpret_cast<const uint4*>(sm + (size_t)r0 * K) + lane) : make_uint4(0, 0, 0, 0);
#endif
        for (int r = r0; r < r_end; r++) {
            if (r == g_next) {
                xg += K; g_next += rows_per_group;
#if XREG
                if (XR) {
#pragma unroll
                    for (int j = 0; j < 4; j++) xr[j] = *reinterpret_cast<const float4*>(xg + 128 * j + 4 * lane);
                }
#endif
            }
            const uint8_t* rp = sm + (size_t)r * K;
            uint4 smv[C];
#if SMNEXT
            if (SN) {
                smv[0] = smn;
                smn = ld_sm(reinterpret_cast<const uint4*>(sm + (size_t)min(r + 1, r_end - 1) * K) + lane);
            } else
#endif
            {
#pragma unroll
            for (int c = 0; c < C; c++) {
#ifdef ABL_NOSM
                smv[c] = make_uint4(r * 77 + c, r ^ lane, r + lane, c);
#else
                smv[c] = ld_sm(reinterpret_cast<const uint4*>(rp + c * 512) + lane);
#endif
            }
            }
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
            for (int c = 0; c < C; c++) {
                const uint32_t smw[4] = {smv[c].x, smv[c].y, smv[c].z, smv[c].w};
                uint32_t ex[8];
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    if (q % 2 == 0) {
                        v = v - pa - pb;  // low 7 bits exact (entries have len in bits 0..4, zeros in 5..6)
                        const uint32_t m = __funnelshift_r(0u, 1u, v);  // 2^(32-v) for v in [1,31]
                        if (!(v & 32u)) {
                            buf += (uint64_t)wq[0] * m;  // lo == 0 here: insert word right after the v valid bits
#pragma unroll
                            for (int i = 0; i + 1 < PF; i++) wq[i] = wq[i + 1];
#ifdef ABL_NOSTREAM
                            wq[PF - 1] = wq[0] ^ idx;
#else
                            wq[PF - 1] = ld_st(lst + SW * idx);
#endif
                            idx++;
                        }
                        v |= 32u;
                    }
                    const uint32_t ent = lut[hi32(buf) >> (32 - LP)];
                    buf = shl64w(buf, ent);
                    ex[q] = ent;
                    if (q % 2 == 0) pa = ent; else pb = ent;
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
#ifdef ABL_NOX
                    const float4 xv = make_float4(1.f, 2.f, 3.f, 4.f);
#elif XREG
                    const float4 xv = XR ? xr[j] : *reinterpret_cast<const float4*>(xg + c * 512 + 128 * j + 4 * lane);
#else
                    const float4 xv = *reinterpret_cast<const float4*>(xg + c * 512 + 128 * j + 4 * lane);
#endif
                    const uint32_t w0 = compose(prmt(smw[j], 0, 0x9180u), ex[2 * j]);
                    const uint32_t w1 = compose(prmt(smw[j], 0, 0xB3A2u), ex[2 * j + 1]);
                    a0 = fmaf(__uint_as_float(w0 << 16), xv.x, a0);
                    a1 = fmaf(__uint_as_float(w0 & 0xFFFF0000u), xv.y, a1);
                    a2 = fmaf(__uint_as_float(w1 << 16), xv.z, a2);
                    a3 = fmaf(__uint_as_float(w1 & 0xFFFF0000u), xv.w, a3);
                }
            }
            float acc_f = (a0 + a1) + (a2 + a3);
#pragma unroll
            for (int o = 16; o; o >>= 1) acc_f += __shfl_xor_sync(0xffffffffu, acc_f, o);
            if (lane == 0) y[r] = acc_f;
        }
    }
    if (diag && lane == 0) {
        unsigned long long t_end; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_end));
        unsigned smid; asm("mov.u32 %0, %smid;" : "=r"(smid));
        const int gw = blockIdx.x * WPB + warp;
        diag[3 * gw] = t_start; diag[3 * gw + 1] = t_end; diag[3 * gw + 2] = smid;
    }
}

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

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: file offset K row0 nrows groups R [reps]\n"); return 1; }
    const char* file = argv[1]; size_t off = strtoull(argv[2], 0, 10);
    int K = atoi(argv[3]); long row0 = atol(argv[4]); int N = atoi(argv[5]); int G = atoi(argv[6]); int R = atoi(argv[7]);
    int reps = argc > 8 ? atoi(argv[8]) : 20;
    int rpg = N / G;
    size_t nbytes = (size_t)N * K * 2;
    std::vector<uint16_t> w((size_t)N * K);
    int fd = open(file, O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    size_t got = 0; while (got < nbytes) { ssize_t r = pread(fd, (char*)w.data() + got, nbytes - got, off + (size_t)row0 * K * 2 + got); if (r <= 0) { perror("pread"); return 1; } got += r; }
    close(fd);

    Huf h = encode(w.data(), N, K, R);
    const bool off16 = !h.lane_rel.empty();
    const size_t offbytes = off16 ? h.unit_base.size() * 4 + h.lane_rel.size() * 2 : h.lane_off.size() * 4;
    size_t comp = h.sm.size() + h.stream.size() * 4 + offbytes + h.lut.size() * 4 + h.unit_row.size() * 4;
    printf("PAIR LP=%d N=%d K=%d G=%d R=%d  raw=%.1f MB comp=%.1f MB ratio=%.4f  exp avg_len=%.4f  stream=%.3f b/w offs=%.3f b/w\n",
           LP, N, K, G, R, nbytes / 1e6, comp / 1e6, (double)comp / nbytes, h.avg_len,
           h.stream.size() * 32.0 / ((double)N * K), offbytes * 8.0 / ((double)N * K));

    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t)K * G);
    for (auto& v : x) { float f = nd(rng); uint32_t u; memcpy(&u, &f, 4); u &= 0xFFFF0000u; memcpy(&v, &u, 4); }

    uint8_t* d_sm; uint32_t *d_stream, *d_off, *d_lut; uint16_t* d_w; float *d_x, *d_y1, *d_y2;
    CK(cudaMalloc(&d_sm, h.sm.size())); CK(cudaMemcpy(d_sm, h.sm.data(), h.sm.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_stream, h.stream.size() * 4)); CK(cudaMemcpy(d_stream, h.stream.data(), h.stream.size() * 4, cudaMemcpyHostToDevice));
    uint16_t* d_rel = nullptr;
    if (off16) {
        CK(cudaMalloc(&d_off, h.unit_base.size() * 4)); CK(cudaMemcpy(d_off, h.unit_base.data(), h.unit_base.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_rel, h.lane_rel.size() * 2)); CK(cudaMemcpy(d_rel, h.lane_rel.data(), h.lane_rel.size() * 2, cudaMemcpyHostToDevice));
    } else {
        CK(cudaMalloc(&d_off, h.lane_off.size() * 4)); CK(cudaMemcpy(d_off, h.lane_off.data(), h.lane_off.size() * 4, cudaMemcpyHostToDevice));
    }
    int* d_ur; CK(cudaMalloc(&d_ur, h.unit_row.size() * 4)); CK(cudaMemcpy(d_ur, h.unit_row.data(), h.unit_row.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_lut, h.lut.size() * 4)); CK(cudaMemcpy(d_lut, h.lut.data(), h.lut.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, nbytes)); CK(cudaMemcpy(d_w, w.data(), nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4)); CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_y1, N * 4)); CK(cudaMalloc(&d_y2, N * 4));

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    unsigned long long* d_diag = nullptr;
    int grid = prop.multiProcessorCount * BPS, grid_raw = prop.multiProcessorCount * 4;
    size_t shm_raw = (size_t)K * G * 4, shm_huf = shm_raw + (4u << LP);
#define SETSHM(KK) CK(cudaFuncSetAttribute(gemv_huf<KK>, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024)); CK(cudaFuncSetAttribute(gemv_bf16<KK>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    SETSHM(512) SETSHM(2048) SETSHM(4096)
    {
        // carveout: only what BPS resident blocks need, rest stays L1 (per-lane streams live there)
        const int pct = (int)std::min<size_t>(100, (BPS * (shm_huf + 1024) * 100 + prop.sharedMemPerMultiprocessor - 1) / prop.sharedMemPerMultiprocessor);
        CK(cudaFuncSetAttribute(gemv_huf<512>, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
        CK(cudaFuncSetAttribute(gemv_huf<2048>, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
        CK(cudaFuncSetAttribute(gemv_huf<4096>, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
        int nb = 0;
        if (K == 512) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gemv_huf<512>, NT, shm_huf));
        else if (K == 2048) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gemv_huf<2048>, NT, shm_huf));
        printf("occupancy: %d blocks/SM of %d threads, shm/block=%zu\n", nb, NT, shm_huf);
    }
    auto launch_huf = [&]() {
        if (K == 512) gemv_huf<512><<<grid, NT, shm_huf>>>(d_sm, d_stream, d_off, d_rel, d_lut, d_x, d_y1, d_ur, h.units, rpg, G, d_diag);
        else if (K == 2048) gemv_huf<2048><<<grid, NT, shm_huf>>>(d_sm, d_stream, d_off, d_rel, d_lut, d_x, d_y1, d_ur, h.units, rpg, G, d_diag);
        else gemv_huf<4096><<<grid, NT, shm_huf>>>(d_sm, d_stream, d_off, d_rel, d_lut, d_x, d_y1, d_ur, h.units, rpg, G, d_diag);
    };
    auto launch_raw = [&]() {
        if (K == 512) gemv_bf16<512><<<grid_raw, 256, shm_raw>>>(d_w, d_x, d_y2, N, rpg, G);
        else if (K == 2048) gemv_bf16<2048><<<grid_raw, 256, shm_raw>>>(d_w, d_x, d_y2, N, rpg, G);
        else gemv_bf16<4096><<<grid_raw, 256, shm_raw>>>(d_w, d_x, d_y2, N, rpg, G);
    };
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    auto timeit = [&](auto fn) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaDeviceSynchronize());
        std::vector<float> v;
        for (int i = 0; i < reps; i++) { cudaEventRecord(t0); fn(); cudaEventRecord(t1); cudaEventSynchronize(t1); float ms; cudaEventElapsedTime(&ms, t0, t1); v.push_back(ms); }
        CK(cudaGetLastError());
        std::sort(v.begin(), v.end());
        printf("  times ms: min %.4f p25 %.4f med %.4f p75 %.4f max %.4f\n", v[0], v[v.size() / 4], v[v.size() / 2], v[3 * v.size() / 4], v.back());
        return v[v.size() / 2];
    };
    float th = timeit(launch_huf), tr = timeit(launch_raw);
    printf("bf16 gemv: %.3f ms  %.1f GB/s\n", tr, nbytes / tr / 1e6);
    printf("huf  gemv: %.3f ms  %.1f GB/s compressed, %.1f Gweights/s  speedup vs bf16 %.3f\n", th, comp / th / 1e6, (double)N * K / th / 1e6, tr / th);
    if (getenv("DIAG")) {
        const int nw = grid * (NT / 32);
        CK(cudaMalloc(&d_diag, nw * 3 * 8)); CK(cudaMemset(d_diag, 0, nw * 3 * 8));
        launch_huf(); CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> dg(nw * 3); CK(cudaMemcpy(dg.data(), d_diag, nw * 3 * 8, cudaMemcpyDeviceToHost));
        unsigned long long t0m = ~0ull, t1m = 0; for (int i = 0; i < nw; i++) { t0m = std::min(t0m, dg[3 * i]); t1m = std::max(t1m, dg[3 * i + 1]); }
        std::vector<double> st, en; for (int i = 0; i < nw; i++) { st.push_back((dg[3 * i] - t0m) / 1e3); en.push_back((dg[3 * i + 1] - t0m) / 1e3); }
        std::sort(st.begin(), st.end()); std::sort(en.begin(), en.end());
        printf("DIAG warps=%d span=%.1f us  start p50=%.1f max=%.1f  end min=%.1f p10=%.1f p50=%.1f p90=%.1f max=%.1f\n", nw, (t1m - t0m) / 1e3,
               st[nw / 2], st.back(), en[0], en[nw / 10], en[nw / 2], en[9 * nw / 10], en.back());
        std::vector<double> sm_end(64, 0); for (int i = 0; i < nw; i++) { int s = dg[3 * i + 2]; sm_end[s] = std::max(sm_end[s], (dg[3 * i + 1] - t0m) / 1e3); }
        printf("DIAG per-SM last end:"); for (int s = 0; s < prop.multiProcessorCount; s++) printf(" %.0f", sm_end[s]); printf("\n");
        if (h.units == nw && K < IL_MINK) {  // one unit per warp: correlate end time with unit stream size
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            for (int i = 0; i < nw; i++) {
                double xx = (i + 1 < nw ? h.lane_off[(i + 1) * 32] : h.stream.size()) - h.lane_off[i * 32];
                double yy = (dg[3 * i + 1] - t0m) / 1e3;
                sx += xx; sy += yy; sxx += xx * xx; syy += yy * yy; sxy += xx * yy;
            }
            const double n = nw, cov = sxy / n - sx / n * sy / n, vx = sxx / n - sx * sx / n / n, vy = syy / n - sy * sy / n / n;
            printf("DIAG corr(end, unit words)=%.3f  unit words mean=%.0f sd=%.0f\n", cov / sqrt(vx * vy), sx / n, sqrt(vx));
        }
        cudaFree(d_diag); d_diag = nullptr;
    }
    std::vector<float> y1(N), y2(N);
    CK(cudaMemcpy(y1.data(), d_y1, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y2.data(), d_y2, N * 4, cudaMemcpyDeviceToHost));
    long same = 0; for (int r = 0; r < N; r++) same += (y1[r] == y2[r]) || (std::isnan(y1[r]) && std::isnan(y2[r]));
    printf("huf==bf16 bitwise: %ld/%d rows %s\n", same, N, same == N ? "OK" : "MISMATCH");
    return 0;
}
