"""Kernels against FP64 references on the exact (dequantised) weights.

Bounds: FP32 GEMV / MMA accumulation |err| <= 1e-5 * sum|w x| per output (plan M1); BF16 outputs additionally carry one
rounding (<= 2^-9 relative, checked as 2^-8). Where two paths must agree exactly (BF16 vs F32 staging of the same values,
multi-column vs single column, BF16 output vs rounding of the F32 output, fused GLU vs unfused) the test is bitwise.
"""
import pytest
import torch
import torch.nn.functional as F

from conftest import BF16_REL, GEMV_REL, bits, encode_upload, gemv_ref, weights_like_model

NIB, HUF, HP = 1, 2, 3
DENSE = {"nibble": (NIB, 0, 256), "huffman": (HUF, 8, 256), "hufpair": (HP, 16, 512)}  # fmt, R, rows
TOKENS = [1, 2, 7, 8, 9]


def act(T, K, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(T, K, generator=g).to(dtype).cuda()


@pytest.fixture(scope="module")
def dense(ops):
    out = {}
    for name, (fmt, R, rows) in DENSE.items():
        for K in (512, 2048, 4096):
            w = weights_like_model((rows, K), seed=K + fmt)
            out[name, K] = (w, encode_upload(ops, w, fmt, R, False))
    return out


def assert_within(y, ref, S, rel=GEMV_REL, out_rel=0.0):
    err = (y.double() - ref).abs()
    bound = rel * S + out_rel * ref.abs() + 1e-30
    worst = (err / bound).max().item()
    assert worst <= 1.0, f"error {worst:.3g} x the bound"


@pytest.mark.parametrize("name", list(DENSE))
@pytest.mark.parametrize("K", [512, 2048, 4096])
@pytest.mark.parametrize("T", TOKENS)
def test_mmv_dense(ops, dense, name, K, T):
    w, enc = dense[name, K]
    x = act(T, K, seed=T)
    ref, S = gemv_ref(w, x)
    y32 = ops.mmv(x, enc, out_dtype=torch.float32)
    assert_within(y32, ref, S)
    # BF16 staging is exact: the same values as F32 give bitwise the same products and sums
    assert torch.equal(ops.mmv(x.float(), enc, out_dtype=torch.float32), y32)
    # BF16 output = round-to-nearest-even of the FP32 result
    assert torch.equal(bits(ops.mmv(x, enc, out_dtype=torch.bfloat16)), bits(y32.to(torch.bfloat16)))
    # every column of a multi-column launch is bitwise its single-column result (batch invariance)
    if T > 1:
        single = torch.cat([ops.mmv(x[t : t + 1], enc) for t in range(T)])
        assert torch.equal(y32, single)


@pytest.mark.parametrize("K", [512, 2048, 4096])
def test_mmv_dense_full_blocks(ops, K):
    """>= 512 units: 1024-thread blocks, whose prologue stages x with one vector load per thread (the other launch
    shapes stage element by element)."""
    w = weights_like_model((8192, K), seed=K)
    enc = encode_upload(ops, w, HP, 16)
    x = act(1, K)
    ref, S = gemv_ref(w, x)
    y = ops.mmv(x, enc)
    assert_within(y, ref, S)
    assert torch.equal(ops.mmv(x.float(), enc), y)


def test_mmv_dense_nibble_escapes(ops):
    """NIBBLE rows with escaped weights (exponents >= 128 and > 15 distinct): the escape list adds the exact products."""
    w = weights_like_model((64, 2048), seed=5).view(torch.int16).clone()
    w[:, ::5] = (0x4000 | torch.arange(w[:, ::5].numel()).view(64, -1) % 128).to(torch.int16)  # |w| in [2, 4)
    w[:, 1::97] = ((torch.arange(w[:, 1::97].numel()).view(64, -1) % 40 + 70) << 7).to(torch.int16)  # many exponents
    w = w.view(torch.bfloat16)
    enc = encode_upload(ops, w, NIB)
    x = act(3, 2048)
    ref, S = gemv_ref(w, x)
    assert_within(ops.mmv(x, enc), ref, S)


@pytest.mark.parametrize("name", ["nibble", "hufpair"])
@pytest.mark.parametrize("T", TOKENS)
def test_linear_dispatch(ops, dense, name, T):
    """bf16n.linear: GEMV up to 8 tokens, dequant + cuBLAS above; BF16 out, optional bias, leading dims kept."""
    w, enc = dense[name, 2048]
    x = act(T, 2048, seed=T)
    bias = torch.randn(w.shape[0]).to(torch.bfloat16).cuda()
    y = ops.linear(x, enc)
    yb = ops.linear(x.view(1, T, 2048), enc, bias)
    assert y.dtype == torch.bfloat16 and y.shape == (T, w.shape[0]) and yb.shape == (1, T, w.shape[0])
    if T > ops.GEMV_MAX_TOKENS:
        # exact weights + the stock BF16 GEMM: bitwise what vLLM's unquantized layer computes
        wc = w.cuda()
        assert torch.equal(bits(y), bits(F.linear(x, wc)))
        assert torch.equal(bits(yb[0]), bits(F.linear(x, wc, bias)))
    else:
        ref, S = gemv_ref(w, x)
        assert_within(y, ref, S, out_rel=BF16_REL)
        # bias is added in FP32 before the single BF16 rounding (cuBLAS epilogue semantics)
        assert_within(yb[0], ref + bias.double(), S + bias.double().abs(), out_rel=BF16_REL)


@pytest.mark.parametrize("T", [9, 17])
def test_linear_chunked_dequant(ops, T):
    """A matrix above the 64 MiB dequant chunk (lm_head-like) goes through several chunks. Stock torch BF16 GEMMs allow
    reduced-precision (BF16) split-K reductions (allow_bf16_reduced_precision_reduction, default on), so the bound is
    the stock GEMM's: 2^-9 of sum |w x| (a wrong or missing chunk is off by O(1))."""
    w = weights_like_model((17408, 2048), seed=11)  # 71 MB of BF16
    enc = encode_upload(ops, w, HP, 16)
    x = act(T, 2048)
    ref, S = gemv_ref(w, x)
    y = ops.linear(x, enc)
    assert_within(y, ref, S, rel=2.0 ** -9, out_rel=BF16_REL)
    assert_within(F.linear(x, w.cuda()), ref, S, rel=2.0 ** -9, out_rel=BF16_REL)  # the bound holds for stock too


# ---------------------------------------------------------------- GLU (shared expert) and get_rows


@pytest.mark.parametrize("T", [1, 2, 8])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mmv_glu_equals_unfused(ops, T, input_dtype, output_dtype):
    """Fused NIBBLE GLU on a merged [gate; up] matrix is bitwise silu(gate x) * (up x) from the unfused ops."""
    N, K = 512, 2048
    w = weights_like_model((2 * N, K), seed=21)
    enc = encode_upload(ops, w, NIB)
    x = act(T, K, seed=T, dtype=input_dtype)
    gu = ops.mmv(x, enc, out_dtype=torch.float32)
    unfused = ops.swiglu(gu[:, :N].contiguous(), gu[:, N:].contiguous())
    actual = ops.mmv_glu(x, enc, out_dtype=output_dtype)
    assert actual.dtype == output_dtype
    assert torch.equal(bits(actual), bits(unfused.to(output_dtype)))


@pytest.mark.parametrize("K", [1024, 2048])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mmv_glu_with_ids(ops, K, input_dtype, output_dtype):
    """Routed token/slot byte strides, padded inputs, repeated experts and a partial final row warp."""
    E, N, T, used = 4, 257, 3, 2
    w = weights_like_model((E, 2 * N, K), seed=22)
    enc = encode_upload(ops, w, NIB)
    x = act(T, used * (K + 8), dtype=input_dtype).view(T, used, K + 8)[..., :K]
    ids = torch.tensor([[0, 3], [1, 1], [2, 1], [0, 0], [3, 3]], dtype=torch.int64).cuda()[::2]
    actual = ops.mmv_glu(x, enc, ids, 2 * N, out_dtype=output_dtype)
    # Each reference column has zero token/slot offset. A shared routed-stride error cannot mask itself.
    expected = []
    for token in range(T):
        slots = []
        for slot in range(used):
            column = x[token, slot].contiguous().view(1, K)
            expert = ids[token, slot].reshape(1, 1)
            gu = ops.mmv(column, enc, expert, 2 * N, out_dtype=torch.float32).reshape(1, 2 * N)
            slots.append(ops.swiglu(gu[:, :N].contiguous(), gu[:, N:].contiguous()).reshape(N))
        expected.append(torch.stack(slots))
    expected = torch.stack(expected).to(output_dtype)
    assert actual.dtype == output_dtype and actual.shape == (T, used, N)
    assert torch.equal(bits(actual), bits(expected))


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("boundary", ["base", "row_stride"])
def test_mmv_glu_input_alignment_boundary(ops, input_dtype, delta, boundary):
    """Torch's 16-byte boundary: adjacent representable addresses/strides reject, equality computes correctly."""
    T, N, K = 2, 32, 512
    enc = encode_upload(ops, weights_like_model((2 * N, K), seed=29), NIB)
    elements_per_alignment = 16 // torch.empty((), dtype=input_dtype).element_size()
    offset = elements_per_alignment + delta
    if boundary == "base":
        storage = act(1, T * K + elements_per_alignment + 1, dtype=input_dtype).reshape(-1)
        x = storage[offset:offset + T * K].view(T, K)
        message = "must be 16-byte aligned"
    else:
        x = act(T, K + offset, dtype=input_dtype)[:, :K]
        message = "row strides must be multiples of 16 bytes"
    if delta:
        with pytest.raises(RuntimeError, match=message):
            ops.mmv_glu(x, enc)
    else:
        expected = []
        for token in range(T):
            gu = ops.mmv(x[token:token + 1].contiguous(), enc)
            expected.append(ops.swiglu(gu[:, :N].contiguous(), gu[:, N:].contiguous()))
        assert torch.equal(ops.mmv_glu(x, enc), torch.cat(expected))


@pytest.mark.parametrize("K", [512, 2048])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mmv_glu_scalar_escape_columns(ops, K, input_dtype, output_dtype):
    """Escapes at 1,2,3,4,K-1 use scalar loads, not aligned-vector addressing or rounded column indices."""
    N = 32
    columns = torch.tensor([1, 2, 3, 4, K - 1])
    values = torch.tensor([0.125, 0.25, 0.5, 0.75, 1.25], dtype=torch.float32)
    row = torch.arange(2 * N).view(-1, 1)
    col = torch.arange(K).view(1, -1)
    # Sixteen common exponents occur >=27 times each. Each target's distinct exponent occurs once,
    # so these five columns necessarily escape the sixteen-entry per-row LUT.
    raw = (((112 + col % 16) << 7) | ((row * 7 + col * 3) % 128)).to(torch.int16)
    for index, column in enumerate(columns.tolist()):
        raw[:, column] = (((128 + index) << 7) | ((row[:, 0] * 5 + index) % 128)).to(torch.int16)
    raw |= ((row % 3 == 0).to(torch.int16) * -32768)
    w = raw.view(torch.bfloat16)
    enc = encode_upload(ops, w, NIB)
    x = torch.zeros((len(columns), K), dtype=input_dtype)
    x[torch.arange(len(columns)), columns] = values.to(input_dtype)
    # One nonzero product per column: independently exact FP32 products, no GEMV accumulation oracle.
    gu = (w[:, columns].float().T * values[:, None]).cuda()
    expected = ops.swiglu(gu[:, :N].contiguous(), gu[:, N:].contiguous()).to(output_dtype)
    actual = ops.mmv_glu(x.cuda(), enc, out_dtype=output_dtype)
    assert torch.equal(bits(actual), bits(expected))


def test_swiglu_matches_torch(ops):
    g = torch.randn(4096, device="cuda") * 4
    u = torch.randn(4096, device="cuda")
    torch.testing.assert_close(ops.swiglu(g, u), F.silu(g) * u, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fmt, R", [(NIB, 0), (HP, 16)])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_get_rows_exact(ops, fmt, R, ids_dtype):
    from test_roundtrip import special_values

    w = special_values(64, 2048, seed=4)
    enc = encode_upload(ops, w, fmt, R)
    ids = torch.tensor([0, 15, 16, 63, 31, 0, 47], dtype=ids_dtype).cuda()  # unit starts / ends, repeats
    rows = ops.get_rows(ids, enc)
    assert torch.equal(bits(rows.cpu()), bits(w[ids.cpu()]))
    as_f32 = ops.get_rows(ids, enc, out_dtype=torch.float32)
    assert torch.equal(as_f32.view(torch.int32), rows.float().view(torch.int32))
    emb = ops.embedding(ids.view(1, -1), enc)
    assert emb.shape == (1, 7, 2048) and torch.equal(bits(emb[0]), bits(rows))


def test_get_rows_rejects_chunk_major(ops):
    enc = encode_upload(ops, weights_like_model((16, 1024)), HP, 8, True)
    with pytest.raises(ValueError, match="chunk-major"):
        ops.get_rows(torch.zeros(1, dtype=torch.int32, device="cuda"), enc)


def test_get_rows_rejects_frag_order(ops):
    enc = encode_upload(ops, weights_like_model((16, 512)), HP, 16, False, True)
    with pytest.raises(ValueError, match="fragment-order"):
        ops.get_rows(torch.zeros(1, dtype=torch.int32, device="cuda"), enc)


@pytest.mark.parametrize("T", [1, 2, 4, 9])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mmv_dense_frag_order(ops, T, input_dtype, output_dtype):
    """Fragment-order GEMV without ids (columns = tokens) against the FP64 reference; BF16 output = rounded FP32."""
    w = weights_like_model((64, 512), seed=T)
    enc = encode_upload(ops, w, HP, 16, False, True)
    x = act(T, 512, seed=T, dtype=input_dtype)
    y = ops.mmv(x, enc)
    ref, S = gemv_ref(w, x)
    assert_within(y, ref, S)
    if output_dtype == torch.bfloat16:
        assert torch.equal(bits(ops.mmv(x, enc, out_dtype=output_dtype)), bits(y.to(torch.bfloat16)))


# ---------------------------------------------------------------- MoE

E, N, H, TOPK = 16, 512, 2048, 8  # production per-expert shapes (gate/up 512 x 2048, down 2048 x 512), 16 experts


@pytest.fixture(scope="module", params=["row_major", "frag_order"])
def experts(ops, request):
    """Production expert layouts; down row-major (llama.cpp's) or in fragment order (vLLM's policy)."""
    gate = weights_like_model((E, N, H), seed=31)
    up = weights_like_model((E, N, H), seed=32)
    down = weights_like_model((E, H, N), seed=33)
    fo = request.param == "frag_order"
    enc = (
        encode_upload(ops, gate, HP, 8, True),
        encode_upload(ops, up, HP, 8, True),
        encode_upload(ops, down, HP, 16, False, fo),
    )
    assert enc[2].meta.frag_order is fo
    return (gate.cuda(), up.cuda(), down.cuda()), enc


def routing(T, mode, seed=0, dtype=torch.int32):
    g = torch.Generator().manual_seed(seed)
    if mode == "uniform":
        ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)])
    elif mode == "one_expert_all":  # expert 0 gets every token (tile boundaries at 127 / 128 / 129)
        ids = torch.stack([torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(E - 1, generator=g)[: TOPK - 1]])
                           for _ in range(T)])
    elif mode == "half_empty":  # experts 8..15 get no token
        ids = torch.stack([torch.randperm(TOPK, generator=g) for _ in range(T)])
    else:
        raise ValueError(mode)
    w = torch.rand(T, TOPK, generator=g) + 0.1
    return ids.to(dtype).cuda(), (w / w.sum(1, keepdim=True)).cuda()


def per_slot_ref(w, ids, a):
    """FP64 y[t, k] = W[ids[t, k]] a[t, k] and S = |W| |a|, one expert at a time (no T x k x rows x K gather)."""
    T, k = ids.shape
    ref = torch.zeros(T, k, w.shape[1], dtype=torch.float64, device="cuda")
    S = torch.zeros_like(ref)
    for e in torch.unique(ids).tolist():
        t, s = (ids == e).nonzero(as_tuple=True)
        We, ae = w[e].double(), a[t, s].double()
        ref[t, s] = ae @ We.t()
        S[t, s] = ae.abs() @ We.abs().t()
    return ref, S


def moe_ref(x, ids, wts, gate, up, down, mma):
    """FP64 reference and per-output error bound of the routed experts."""
    xd = x.double()
    T = x.shape[0]
    y = torch.zeros(T, H, dtype=torch.float64, device="cuda")
    bound = torch.zeros_like(y)
    for e in torch.unique(ids).tolist():
        t, k = (ids == e).nonzero(as_tuple=True)
        G, U, D = gate[e].double(), up[e].double(), down[e].double()
        g, u = xd[t] @ G.t(), xd[t] @ U.t()
        Sg, Su = xd[t].abs() @ G.abs().t(), xd[t].abs() @ U.abs().t()
        s = g * torch.sigmoid(g)
        a = s * u
        # error of the activation: FP32 accumulation of g and u through silu, plus its BF16 rounding on the MMA path
        da = 2 * GEMV_REL * (1.1 * Sg * u.abs() + s.abs() * Su) + (BF16_REL if mma else 1e-6) * a.abs()
        wk = wts[t, k].double()[:, None]
        y.index_add_(0, t, wk * (a @ D.t()))
        bound.index_add_(0, t, wk.abs() * ((da + 1.1 * GEMV_REL * a.abs()) @ D.abs().t()))
    return y, bound


@pytest.mark.parametrize("T, mode", [(1, "uniform"), (2, "uniform"), (7, "half_empty"), (8, "one_expert_all"),
                                     (9, "uniform"), (9, "one_expert_all"), (48, "uniform"), (49, "uniform"),
                                     (96, "uniform"), (97, "uniform"), (127, "one_expert_all"),
                                     (128, "one_expert_all"), (129, "one_expert_all"), (129, "half_empty"),
                                     (2048, "uniform")])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_moe_experts(ops, experts, T, mode, ids_dtype):
    """Routed experts end to end at the GEMV / batched switch (8 / 9 tokens), the down kernel's interleave switches
    (routed tokens 24 / 48 per expert: T = 48 / 49, 96 / 97 at 16 experts), the 128-token tile (one expert with 127,
    128, 129 tokens), experts without tokens, and int32 / int64 ids."""
    (gate, up, down), (eg, eu, ed) = experts
    x = act(T, H, seed=T)
    ids, wts = routing(T, mode, seed=T, dtype=ids_dtype)
    y = ops.moe_experts(x, eg, eu, ed, wts, ids, E)
    assert y.dtype == torch.bfloat16 and y.shape == (T, H)
    ref, bound = moe_ref(x, ids, wts, gate, up, down, mma=T > ops.GEMV_MAX_TOKENS)
    err = (y.double() - ref).abs()
    worst = (err / (bound + BF16_REL * ref.abs() + 1e-30)).max().item()
    assert worst <= 1.0, f"error {worst:.3g} x the bound"


@pytest.mark.parametrize("T", [9, 64, 129, 1024])  # down interleave 4 / 2 / 1 / 1 at 16 experts
def test_moe_mm_ids_down_stage(ops, experts, T):
    """Batched down kernel alone (per-slot BF16 inputs): FP32 accumulation bound, every routed slot written; the BF16
    output is the rounding of the FP32 one."""
    (_, _, down), (_, _, ed) = experts
    ids, _ = routing(T, "uniform", seed=T)
    a = act(T * TOPK, N, seed=T).view(T, TOPK, N)
    y = ops.moe_mm_ids(a, ids, ed, E)
    ref, S = per_slot_ref(down, ids, a)
    assert_within(y, ref, S)
    assert torch.equal(bits(ops.moe_mm_ids(a, ids, ed, E, out_dtype=torch.bfloat16)), bits(y.to(torch.bfloat16)))


@pytest.mark.parametrize("T", [15, 16, 17, 95, 96, 97, 193])
def test_moe_mm_ids_down_token_tiles(ops, experts, T):
    """One expert gets every token: below / at / above a 16-token mma pair and the fragment-order kernel's 96-token
    items (one, two and three items for that expert); every routed slot written."""
    (_, _, down), (_, _, ed) = experts
    ids, _ = routing(T, "one_expert_all", seed=T)
    a = act(T * TOPK, N, seed=T).view(T, TOPK, N)
    y = ops.moe_mm_ids(a, ids, ed, E)
    ref, S = per_slot_ref(down, ids, a)
    assert_within(y, ref, S)


@pytest.mark.parametrize("T", [9, 128, 300])
def test_moe_mm_ids_gate_alone_stage(ops, experts, T):
    """Batched gate kernel without GLU fusion (the non-fused 2048 x 8 instance)."""
    (gate, _, _), (eg, _, _) = experts
    ids, _ = routing(T, "one_expert_all", seed=T)
    x = act(T, H, seed=T)
    y = ops.moe_mm_ids(x, ids, eg, E)
    ref, S = per_slot_ref(gate, ids, x[:, None, :].expand(T, TOPK, H))
    assert_within(y, ref, S)


def test_moe_mm_ids_rejects_unsupported_layout(ops, experts):
    (_, _, _), (eg, eu, ed) = experts
    ids, _ = routing(16, "uniform")
    with pytest.raises(ValueError, match="no batched kernel"):
        ops.moe_mm_ids(act(16, N), ids, ed, E, up=ed)  # GLU needs chunk-major 2048 x 8 matrices


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_moe_combine(ops, w_dtype):
    T, k, n = 5, 8, 2048
    ys = torch.randn(T, k, n, device="cuda")
    w = torch.rand(T, k, device="cuda").to(w_dtype)
    y = ops.moe_combine(ys, w, torch.float32)
    ref = torch.einsum("tk,tkn->tn", w.double(), ys.double())
    S = torch.einsum("tk,tkn->tn", w.double().abs(), ys.double().abs())
    assert_within(y, ref, S, rel=1e-6)
    assert torch.equal(bits(ops.moe_combine(ys, w)), bits(y.to(torch.bfloat16)))


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_mmv_ids_down_per_slot(ops, experts, ids_dtype):
    """Expert GEMV with per-slot activations (down after the GLU) against the per-slot reference."""
    (_, _, down), (_, _, ed) = experts
    T = 3
    ids, _ = routing(T, "uniform", seed=9, dtype=ids_dtype)
    a = torch.randn(T, TOPK, N, device="cuda")
    y = ops.mmv(a, ed, ids, H)
    ref, S = per_slot_ref(down, ids, a)
    assert_within(y, ref, S)

