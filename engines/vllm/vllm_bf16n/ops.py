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
FLAG_CHUNK_MAJOR, FLAG_FRAG_ORDER = 1, 2

# llama.cpp dispatch thresholds: up to 8 tokens through the GEMV kernels, more through the batched MoE kernel /
# dequant + cuBLAS (the C++ linear op applies the same bound for dense matrices)
GEMV_MAX_TOKENS = 8

REFERENCE = os.environ.get("BF16N_REFERENCE", "0") not in ("", "0")

_ops = torch.ops.bf16n


@dataclass(frozen=True)
class Meta:
    """Host-side metadata of one encoded matrix, parsed once from its aux header."""

    format: int
    K: int
    rows: int
    R: int
    chunk_major: bool
    frag_order: bool
    off_a: int
    off_b: int
    off_c: int
    off_d: int

    def as_list(self) -> list[int]:
        flags = (FLAG_CHUNK_MAJOR if self.chunk_major else 0) | (FLAG_FRAG_ORDER if self.frag_order else 0)
        return [self.format, self.K, self.rows, self.R, flags, self.off_a, self.off_b, self.off_c, self.off_d]


def parse_header(aux: torch.Tensor) -> Meta:
    """Metadata from the first 256 bytes of a CPU aux tensor; raises on a malformed header."""
    if aux.device.type != "cpu" or aux.dtype != torch.uint8 or aux.numel() < 256:
        raise ValueError("bf16n: aux must be a CPU uint8 tensor of at least 256 bytes")
    (magic, fmt, K, R, rows, a, b, c, d, _na, _nb, flags, _pad) = HEADER.unpack(bytes(aux[: HEADER.size].tolist()))
    if magic != MAGIC or fmt not in FORMAT_NAMES:
        raise ValueError(f"bf16n: bad aux header (magic {magic:#x}, format {fmt})")
    if flags & ~(FLAG_CHUNK_MAJOR | FLAG_FRAG_ORDER):
        raise ValueError(f"bf16n: unknown layout flags {flags:#x}")
    return Meta(fmt, K, rows, R, bool(flags & FLAG_CHUNK_MAJOR), bool(flags & FLAG_FRAG_ORDER), a, b, c, d)


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


def encode(w: torch.Tensor, fmt: int, R: int, chunk_major: bool, frag_order: bool = False
           ) -> tuple[torch.Tensor, torch.Tensor, Meta]:
    """Encode a BF16 CPU tensor (..., K) on all host cores. FMT_HUFPAIR falls back to FMT_HUFFMAN when the tensor has
    more than 4096 distinct exponent pairs; the returned Meta holds the format and layout actually produced."""
    plane, aux = _ops.encode(w, fmt, R, chunk_major, frag_order)
    return plane, aux, parse_header(aux)


def upload(plane: torch.Tensor, aux: torch.Tensor, meta: Meta, device: torch.device | str) -> Weight:
    return Weight(plane.to(device), aux.to(device), meta)


def dequant(w: Weight, row0: int = 0, nrows: int | None = None) -> torch.Tensor:
    return _ops.dequant(w.plane, w.aux, w.m, row0, w.meta.rows - row0 if nrows is None else nrows)


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
    """Mirror of launch.cu mm_ids_supported: shapes with a batched (prefill) MoE kernel instance (K = 512, R = 16: the
    row-major or the fragment-order kernel)."""
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


def int2_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft-only 2-bit copy of a BF16 matrix: per 128 weights s = amax / 1.5 and each weight the nearest of
    -1.5 s, -0.5 s, 0.5 s, 1.5 s: (q uint8 [R, K/4], s f32 [R, K/128])."""
    return _ops.int2_quantize(w)


def int2_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Approximate y = x W^T (FP32 [T, R]) from the 2-bit copy; T <= 4, K = 2048."""
    return _ops.int2_gemv(x, q, s)


DRAFT_HEADS = {"int2": 2, "int4": 4}  # BF16N_DRAFT_HEAD values with a k-bit copy; "full" scores with the exact head
# Candidates re-scored exactly. T-051, 1152 MTP draft states of the production model: with these counts the exact
# top-1 was always among the candidates, and the exact softmax mass left outside was 4.1e-2 (2-bit, 128) and 5.8e-2
# (4-bit, 32) at temperature 1, 7.6e-3 / 1.2e-2 at 0.7.
DEFAULT_CANDIDATES = {2: 128, 4: 32}


@dataclass(frozen=True)
class DraftHead:
    """Draft-only k-bit copy of an lm_head (symmetric, one FP32 scale per 128 weights) whose top `candidates` scores
    are re-scored exactly by draft_logits."""
    bits: int
    q: torch.Tensor
    s: torch.Tensor
    candidates: int

    def __post_init__(self):
        if self.bits not in DEFAULT_CANDIDATES:
            raise ValueError(f"draft head: {self.bits} bits (2 or 4)")
        if self.q.shape[1] * (8 // self.bits) != self.s.shape[1] * 128 or self.q.shape[0] != self.s.shape[0]:
            raise ValueError(f"draft head: q {tuple(self.q.shape)} and s {tuple(self.s.shape)} disagree for {self.bits} bits")
        if not 1 <= self.candidates <= self.q.shape[0]:
            raise ValueError(f"draft head: {self.candidates} candidates for {self.q.shape[0]} rows")

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        return (int2_gemv if self.bits == 2 else int4_gemv)(h, self.q, self.s)


def draft_head_bits() -> int | None:
    """Bits of the draft head chosen by BF16N_DRAFT_HEAD (int2 by default, int4; full = the exact head, None)."""
    v = os.environ.get("BF16N_DRAFT_HEAD", "int2")
    if v == "full":
        return None
    if v not in DRAFT_HEADS:
        raise ValueError(f"BF16N_DRAFT_HEAD must be one of int2, int4, full; got {v!r}")
    return DRAFT_HEADS[v]


def make_draft_head(w: torch.Tensor, bits: int) -> DraftHead:
    """The k-bit draft copy of BF16 w with BF16N_DRAFT_CANDIDATES candidates (default per bits)."""
    q, s = (int2_quantize if bits == 2 else int4_quantize)(w)
    env = os.environ.get("BF16N_DRAFT_CANDIDATES")
    return DraftHead(bits, q, s, int(env) if env is not None else DEFAULT_CANDIDATES[bits])


def draft_logits(h: torch.Tensor, w: Weight, head: DraftHead, vocab: int) -> torch.Tensor:
    """Draft-token logits: the k-bit head selects candidates; gathered BF16N rows are re-scored in FP32, others get -inf.
    Candidate coverage alone does not guarantee the full BF16-output head's argmax: output rounding can change ties.
    This is a proposal path, not a claim of full-head or sequence equivalence."""
    n, K = h.shape
    h = h.contiguous()
    cand = head.scores(h).topk(head.candidates, dim=-1).indices                    # [n, C]
    rows = get_rows(cand.reshape(-1), w).view(n, head.candidates, K)              # exact BF16 rows
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
    them; more: the batched kernels (fused gate/up + SWIGLU written as BF16, then down: in fragment order for vLLM's
    policy), as llama.cpp does. Every prefill chunk takes the batched kernels: the exact dequant + Triton fused_experts
    path that long chunks used before was 7-10 % slower end to end in the production configuration (T-023)."""
    T = x.shape[0]
    N = gate.meta.rows // n_experts
    H = down.meta.rows // n_experts
    if REFERENCE:
        return moe_experts_reference(x, gate, up, down, topk_weights, topk_ids, n_experts)
    x = x.contiguous()
    if T > GEMV_MAX_TOKENS and mm_ids_supported(gate, up) and mm_ids_supported(down):
        a = moe_mm_ids(x, topk_ids, gate, n_experts, up=up, out_dtype=torch.bfloat16)
        ys = moe_mm_ids(a, topk_ids, down, n_experts, out_dtype=torch.float32)
    else:
        g = mmv(x, gate, topk_ids, N)
        u = mmv(x, up, topk_ids, N)
        ys = mmv(swiglu(g, u), down, topk_ids, H)
    return moe_combine(ys, topk_weights, torch.bfloat16)


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
def _encode_fake(w, fmt, R, chunk_major, frag_order=False):
    ctx = torch.library.get_ctx()
    K = w.shape[-1]
    rows = w.numel() // K
    return w.new_empty((rows, K), dtype=torch.uint8), w.new_empty((ctx.new_dynamic_size(),), dtype=torch.uint8)


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


@torch.library.register_fake("bf16n::int2_quantize")
def _int2_quantize_fake(w):
    return (w.new_empty((w.shape[0], w.shape[1] // 4), dtype=torch.uint8),
            w.new_empty((w.shape[0], w.shape[1] // 128), dtype=torch.float32))


@torch.library.register_fake("bf16n::int2_gemv")
def _int2_gemv_fake(x, q, s):
    return x.new_empty((x.shape[0], q.shape[0]), dtype=torch.float32)


@torch.library.register_fake("bf16n::swiglu")
def _swiglu_fake(g, u):
    return torch.empty_like(g)
