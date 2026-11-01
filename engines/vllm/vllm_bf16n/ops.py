"""torch.ops.bf16n wrappers: encoding, metadata, fake (meta) implementations and the dispatch between the decode GEMV
and the batched / dequant paths.

Importing this module loads the extension (vllm_bf16n._C) but does not initialise CUDA. Call `init()` once per process
before the first launch and before CUDA-graph capture (kernel attributes are set there, never inside an op).

BF16N_REFERENCE=1 (read at import) routes every product through exact dequant + stock BF16 torch kernels.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

from . import _C  # noqa: F401  (registers torch.ops.bf16n)

FMT_NIBBLE, FMT_HUFFMAN, FMT_HUFPAIR = 1, 2, 3
FORMAT_NAMES = {FMT_NIBBLE: "nibble", FMT_HUFFMAN: "huffman", FMT_HUFPAIR: "hufpair"}
MAGIC = 0x4E363142
HEADER = struct.Struct("<IIIIQQQQQQQII")  # core/include/bf16n/format.h aux_header
FLAG_CHUNK_MAJOR = 1

# llama.cpp dispatch thresholds: up to 8 tokens through the GEMV kernels, more through the batched MoE kernel /
# dequant + cuBLAS (the C++ linear op applies the same bound for dense matrices)
GEMV_MAX_TOKENS = 8

REFERENCE = os.environ.get("BF16N_REFERENCE", "0") not in ("", "0")

# Long prefill chunks (>= MOE_DEQUANT_MIN_TOKENS tokens): exact dequant of the layer's experts into a BF16 workspace +
# vLLM's Triton fused_experts beats the decode-in-shared-memory kernel (layer 0, 8192 tokens: 17 + 30.6 ms vs 70 ms
# with vLLM's default Triton config, 17 + 24 ms with the Orin-tuned one in vllm_bf16n/configs/fused_moe; 4096 tokens:
# 17 + 15 ms vs 38 ms; 3072 tokens: the batched kernel wins). The workspace holds 1/MOE_DEQUANT_SPLIT of the experts
# (split 2: 0.8 GB; each part masks the others' tokens through expert_map, partial outputs summed in FP32).
MOE_DEQUANT_MIN_TOKENS = int(os.environ.get("BF16N_MOE_DEQUANT_MIN_TOKENS", "4096"))
MOE_DEQUANT_SPLIT = int(os.environ.get("BF16N_MOE_DEQUANT_SPLIT", "2"))

_ops = torch.ops.bf16n


@dataclass(frozen=True)
class Meta:
    """Host-side metadata of one encoded matrix, parsed once from its aux header."""

    format: int
    K: int
    rows: int
    R: int
    chunk_major: bool
    off_a: int
    off_b: int
    off_c: int
    off_d: int

    def as_list(self) -> list[int]:
        return [self.format, self.K, self.rows, self.R, int(self.chunk_major), self.off_a, self.off_b, self.off_c,
                self.off_d]


def parse_header(aux: torch.Tensor) -> Meta:
    """Metadata from the first 256 bytes of a CPU aux tensor; raises on a malformed header."""
    if aux.device.type != "cpu" or aux.dtype != torch.uint8 or aux.numel() < 256:
        raise ValueError("bf16n: aux must be a CPU uint8 tensor of at least 256 bytes")
    (magic, fmt, K, R, rows, a, b, c, d, _na, _nb, flags, _pad) = HEADER.unpack(bytes(aux[: HEADER.size].tolist()))
    if magic != MAGIC or fmt not in FORMAT_NAMES:
        raise ValueError(f"bf16n: bad aux header (magic {magic:#x}, format {fmt})")
    return Meta(fmt, K, rows, R, bool(flags & FLAG_CHUNK_MAJOR), a, b, c, d)


@dataclass
class Weight:
    """One encoded matrix on the device: plane (rows x K sign|mantissa bytes) + aux, and its metadata."""

    plane: torch.Tensor
    aux: torch.Tensor
    meta: Meta

    @property
    def m(self) -> list[int]:
        return self.meta.as_list()

    @property
    def nbytes(self) -> int:
        return self.plane.numel() + self.aux.numel()


def init() -> None:
    """Device attributes and kernel shared-memory limits; once per process, before any launch / graph capture."""
    _ops.init()


def encode(w: torch.Tensor, fmt: int, R: int, chunk_major: bool) -> tuple[torch.Tensor, torch.Tensor, Meta]:
    """Encode a BF16 CPU tensor (..., K) on all host cores. FMT_HUFPAIR falls back to FMT_HUFFMAN when the tensor has
    more than 4096 distinct exponent pairs; the returned Meta holds the format actually produced."""
    plane, aux = _ops.encode(w, fmt, R, chunk_major)
    return plane, aux, parse_header(aux)


def upload(plane: torch.Tensor, aux: torch.Tensor, meta: Meta, device: torch.device | str) -> Weight:
    return Weight(plane.to(device), aux.to(device), meta)


def dequant(w: Weight, row0: int = 0, nrows: int | None = None) -> torch.Tensor:
    return _ops.dequant(w.plane, w.aux, w.m, row0, w.meta.rows - row0 if nrows is None else nrows)


def dequant_into(out: torch.Tensor, w: Weight, row0: int, nrows: int, group_rows: int = 0, group_stride: int = 0) -> None:
    """Rows [row0, row0 + nrows) into `out` (BF16, contiguous, rows of K); group_rows > 0: matrix row i goes to output
    row (i // group_rows) * group_stride + i % group_rows."""
    _ops.dequant_into(out, w.plane, w.aux, w.m, row0, nrows, group_rows, group_stride)


def roundtrip_equal(w: Weight, src: torch.Tensor, chunk_bytes: int = 64 << 20) -> bool:
    """Bitwise check of the GPU decode against the BF16 source (any device), in row chunks of whole units."""
    K, rows = w.meta.K, w.meta.rows
    unit = max(w.meta.R, 1)
    src2 = src.reshape(rows, K)
    step = max(chunk_bytes // (2 * K) // unit * unit, unit)
    for r0 in range(0, rows, step):
        nr = min(step, rows - r0)
        got = dequant(w, r0, nr).view(torch.int16)
        ref = src2[r0 : r0 + nr].to(got.device).view(torch.int16)
        if not torch.equal(got, ref):
            return False
    return True


def mmv(x: torch.Tensor, w: Weight, ids: torch.Tensor | None = None, rows_per_matrix: int = 0,
        out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _ops.mmv(x, w.plane, w.aux, w.m, ids, rows_per_matrix, out_dtype == torch.bfloat16)


def mmv_glu(x: torch.Tensor, w: Weight, ids: torch.Tensor | None = None, rows_per_matrix: int = 0,
            out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """silu(gate x) * (up x) for a NIBBLE matrix whose rows per matrix are [gate; up]."""
    return _ops.mmv_glu(x, w.plane, w.aux, w.m, ids, rows_per_matrix or w.meta.rows, out_dtype == torch.bfloat16)


def linear(x: torch.Tensor, w: Weight, bias: torch.Tensor | None = None) -> torch.Tensor:
    if REFERENCE:
        return torch.nn.functional.linear(x, dequant(w), bias)
    return _ops.linear(x, w.plane, w.aux, w.m, bias)


def get_rows(ids: torch.Tensor, w: Weight, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return _ops.get_rows(ids, w.plane, w.aux, w.m, out_dtype == torch.bfloat16)


def embedding(ids: torch.Tensor, w: Weight) -> torch.Tensor:
    flat = ids.reshape(-1)
    return get_rows(flat, w).view(*ids.shape, w.meta.K)


def mm_ids_supported(w: Weight, up: Weight | None = None) -> bool:
    """Mirror of launch.cu mm_ids_supported: shapes with a batched (prefill) MoE kernel instance."""
    m = w.meta
    ok = m.format == FMT_HUFPAIR and ((m.K == 2048 and m.R == 8 and m.chunk_major) or (m.K == 512 and m.R == 16))
    if up is None:
        return ok
    u = up.meta
    return ok and m.K == 2048 and (u.format, u.K, u.R, u.chunk_major, u.rows) == (m.format, m.K, m.R, True, m.rows)


def moe_mm_ids(x: torch.Tensor, topk_ids: torch.Tensor, w: Weight, n_experts: int, up: Weight | None = None,
               out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _ops.moe_mm_ids(x, topk_ids, w.plane, w.aux, w.m, None if up is None else up.plane,
                           None if up is None else up.aux, [] if up is None else up.m, n_experts,
                           out_dtype == torch.bfloat16)


def int4_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft-only 4-bit copy of a BF16 matrix (symmetric, FP32 scale per 128 weights): (q uint8 [R, K/2], s f32 [R, K/128])."""
    return _ops.int4_quantize(w)


def int4_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Approximate y = x W^T (FP32 [T, R]) from the 4-bit copy; T <= 4."""
    return _ops.int4_gemv(x, q, s)


DRAFT_CANDIDATES = int(os.environ.get("BF16N_DRAFT_CANDIDATES", "32"))


def draft_logits(h: torch.Tensor, w: Weight, q: torch.Tensor, s: torch.Tensor, vocab: int,
                 candidates: int = DRAFT_CANDIDATES) -> torch.Tensor:
    """Draft-token logits: the 4-bit head selects candidates; gathered BF16N rows are re-scored in FP32, others get -inf.
    Candidate coverage alone does not guarantee the full BF16-output head's argmax: output rounding can change ties.
    This is a proposal path, not a claim of full-head or sequence equivalence."""
    n, K = h.shape
    h = h.contiguous()
    cand = int4_gemv(h, q, s).topk(candidates, dim=-1).indices                     # [n, C]
    rows = get_rows(cand.reshape(-1), w).view(n, candidates, K)                    # exact BF16 rows
    exact = torch.einsum("nck,nk->nc", rows.float(), h.float())
    out = torch.full((n, vocab), float("-inf"), dtype=torch.float32, device=h.device)
    return out.scatter_(1, cand, exact)


def moe_combine(ys: torch.Tensor, topk_weights: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return _ops.moe_combine(ys, topk_weights, out_dtype == torch.bfloat16)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _ops.swiglu(g, u)


def moe_experts(x: torch.Tensor, gate: Weight, up: Weight, down: Weight, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Routed experts of one MoE layer: sum_k w_k * down(silu(gate x) * up x) over the top-k experts, BF16 out.
    Up to GEMV_MAX_TOKENS tokens (or shapes without a batched kernel): expert GEMVs with FP32 activations between
    them; more: the batched kernel (fused gate/up + SWIGLU written as BF16, then down), as llama.cpp does; from
    MOE_DEQUANT_MIN_TOKENS on: exact dequant + vLLM's Triton fused_experts (moe_experts_dequant)."""
    T = x.shape[0]
    N = gate.meta.rows // n_experts
    H = down.meta.rows // n_experts
    if REFERENCE:
        return moe_experts_reference(x, gate, up, down, topk_weights, topk_ids, n_experts)
    x = x.contiguous()
    if T >= MOE_DEQUANT_MIN_TOKENS and n_experts % MOE_DEQUANT_SPLIT == 0:
        return moe_experts_dequant(x, gate, up, down, topk_weights, topk_ids, n_experts, MOE_DEQUANT_SPLIT)
    if T > GEMV_MAX_TOKENS and mm_ids_supported(gate, up) and mm_ids_supported(down):
        a = moe_mm_ids(x, topk_ids, gate, n_experts, up=up, out_dtype=torch.bfloat16)
        ys = moe_mm_ids(a, topk_ids, down, n_experts, out_dtype=torch.float32)
    else:
        g = mmv(x, gate, topk_ids, N)
        u = mmv(x, up, topk_ids, N)
        ys = mmv(swiglu(g, u), down, topk_ids, H)
    return moe_combine(ys, topk_weights, torch.bfloat16)


_workspaces: dict = {}


def _workspace(name: str, shape: tuple, device) -> torch.Tensor:
    """Persistent BF16 buffers of the dequant prefill path, one per (name, shape, device), allocated on first use
    (prefill chunks run outside CUDA graphs)."""
    key = (name, shape, str(device))
    if key not in _workspaces:
        _workspaces[key] = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return _workspaces[key]


def _expert_map(n_experts: int, e0: int, n: int, device) -> torch.Tensor:
    """expert_map selecting experts [e0, e0 + n) as local 0..n-1 (others -1), cached per part."""
    key = ("emap", n_experts, e0, n, str(device))
    if key not in _workspaces:
        m = torch.full((n_experts,), -1, dtype=torch.int32, device=device)
        m[e0:e0 + n] = torch.arange(n, dtype=torch.int32, device=device)
        _workspaces[key] = m
    return _workspaces[key]


def moe_experts_dequant(x, gate, up, down, topk_weights, topk_ids, n_experts, split=2):
    """Exact dequant of 1/split of the experts at a time into a BF16 workspace in vLLM's layout (w13 [E][gate; up][K],
    w2 [E][H][N]) + vLLM's unquantized Triton fused_experts restricted to those experts by expert_map."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    N, H, K = gate.meta.rows // n_experts, down.meta.rows // n_experts, gate.meta.K
    eg = n_experts // split
    w13 = _workspace("w13", (eg, 2 * N, K), x.device)
    w2 = _workspace("w2", (eg, H, N), x.device)
    out = None
    for g in range(split):
        e0 = g * eg
        dequant_into(w13, gate, e0 * N, eg * N, N, 2 * N)
        dequant_into(w13.view(-1)[N * K:], up, e0 * N, eg * N, N, 2 * N)
        dequant_into(w2, down, e0 * H, eg * H)
        if split == 1:
            return fused_experts(x, w13, w2, topk_weights, topk_ids)
        emap = _expert_map(n_experts, e0, eg, x.device)
        y = fused_experts(x, w13, w2, topk_weights, topk_ids, global_num_experts=n_experts, expert_map=emap).float()
        out = y if out is None else out.add_(y)
    return out.to(torch.bfloat16)


def moe_experts_reference(x, gate, up, down, topk_weights, topk_ids, n_experts):
    """Exact-weight reference in FP32 (dequant per used expert): slow, for tests and BF16N_REFERENCE."""
    N = gate.meta.rows // n_experts
    H = down.meta.rows // n_experts
    xf = x.float()
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=x.device)
    for e in torch.unique(topk_ids).tolist():
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        g = dequant(gate, e * N, N).float()
        u = dequant(up, e * N, N).float()
        d = dequant(down, e * H, H).float()
        a = torch.nn.functional.silu(xf[tok] @ g.t()) * (xf[tok] @ u.t())
        out.index_add_(0, tok, (a @ d.t()) * topk_weights[tok, slot].float()[:, None])
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------- fake (meta) implementations

@torch.library.register_fake("bf16n::encode")
def _encode_fake(w, fmt, R, chunk_major):
    ctx = torch.library.get_ctx()
    K = w.shape[-1]
    rows = w.numel() // K
    return w.new_empty((rows, K), dtype=torch.uint8), w.new_empty((ctx.new_dynamic_size(),), dtype=torch.uint8)


@torch.library.register_fake("bf16n::dequant_into")
def _dequant_into_fake(out, plane, aux, meta, row0, nrows, group_rows, group_stride):
    return None


@torch.library.register_fake("bf16n::dequant")
def _dequant_fake(plane, aux, meta, row0, nrows):
    return plane.new_empty((nrows, meta[1]), dtype=torch.bfloat16)


def _mmv_shape(x, meta, ids, rows_per_matrix, glu):
    if ids is None:
        rows = meta[2]
        return (x.shape[0], rows // 2 if glu else rows)
    return (ids.shape[0], ids.shape[1], rows_per_matrix // 2 if glu else rows_per_matrix)


@torch.library.register_fake("bf16n::mmv")
def _mmv_fake(x, plane, aux, meta, ids, rows_per_matrix, out_bf16):
    return x.new_empty(_mmv_shape(x, meta, ids, rows_per_matrix, False),
                       dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.register_fake("bf16n::mmv_glu")
def _mmv_glu_fake(x, plane, aux, meta, ids, rows_per_matrix, out_bf16):
    return x.new_empty(_mmv_shape(x, meta, ids, rows_per_matrix, True),
                       dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.register_fake("bf16n::linear")
def _linear_fake(x, plane, aux, meta, bias):
    return x.new_empty((*x.shape[:-1], meta[2]))


@torch.library.register_fake("bf16n::get_rows")
def _get_rows_fake(ids, plane, aux, meta, out_bf16):
    return plane.new_empty((ids.shape[0], meta[1]), dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.register_fake("bf16n::moe_mm_ids")
def _moe_mm_ids_fake(x, topk_ids, plane, aux, meta, up_plane, up_aux, up_meta, n_experts, out_bf16):
    return x.new_empty((topk_ids.shape[0], topk_ids.shape[1], meta[2] // n_experts),
                       dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.register_fake("bf16n::moe_combine")
def _moe_combine_fake(ys, topk_weights, out_bf16):
    return ys.new_empty((ys.shape[0], ys.shape[2]), dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.register_fake("bf16n::int4_quantize")
def _int4_quantize_fake(w):
    return (w.new_empty((w.shape[0], w.shape[1] // 2), dtype=torch.uint8),
            w.new_empty((w.shape[0], w.shape[1] // 128), dtype=torch.float32))


@torch.library.register_fake("bf16n::int4_gemv")
def _int4_gemv_fake(x, q, s):
    return x.new_empty((x.shape[0], q.shape[0]), dtype=torch.float32)


@torch.library.register_fake("bf16n::swiglu")
def _swiglu_fake(g, u):
    return torch.empty_like(g)
