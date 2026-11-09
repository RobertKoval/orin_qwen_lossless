"""Bitwise round trip: encode (CPU) -> GPU dequant == source, for every format and the format boundaries."""
import os

import pytest
import torch

from conftest import CKPT, bits, encode_upload, weights_like_model

NIB, HUF, HP = 1, 2, 3
# (requested format, R, chunk_major) for the four layouts
LAYOUTS = {
    "nibble": (NIB, 0, False),
    "huffman": (HUF, 8, False),
    "hufpair": (HP, 8, False),
    "hufpair_cm": (HP, 8, True),
}


def roundtrip(ops, w, fmt, R, cm):
    enc = encode_upload(ops, w, fmt, R, cm)
    back = ops.dequant(enc)
    return enc, back


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("K", [512, 2048, 4096])
@pytest.mark.parametrize("units", [1, 2])
def test_formats_bitwise(ops, layout, K, units):
    fmt, R, cm = LAYOUTS[layout]
    if cm and K == 512:
        pytest.skip("chunk-major needs K > 512 (test_chunk_major_shape_limits)")
    rows = units * (R or 8)
    w = weights_like_model((rows, K), seed=K + units)
    enc, back = roundtrip(ops, w, fmt, R, cm)
    assert enc.meta.format == fmt  # the requested format was produced (no silent fallback)
    assert enc.meta.chunk_major is cm
    assert torch.equal(bits(back.cpu()), bits(w))


@pytest.mark.parametrize("layout", ["huffman", "hufpair", "hufpair_cm"])
def test_rows_not_a_multiple_of_R_rejected(ops, layout):
    fmt, R, cm = LAYOUTS[layout]
    w = weights_like_model((R - 1, 2048))
    with pytest.raises(ValueError, match="multiple of R"):
        ops.encode(w, fmt, R, cm)


@pytest.mark.parametrize("R", [1, 2, 4, 16])
def test_hufpair_unit_sizes(ops, R):
    w = weights_like_model((2 * R, 1024), seed=R)
    enc, back = roundtrip(ops, w, HP, R, False)
    assert torch.equal(bits(back.cpu()), bits(w))


@pytest.mark.parametrize("R", [4, 16])
def test_hufpair_chunk_major_unit_sizes(ops, R):
    w = weights_like_model((2 * R, 2048), seed=R)
    enc, back = roundtrip(ops, w, HP, R, True)
    assert torch.equal(bits(back.cpu()), bits(w))


@pytest.mark.parametrize("K, R, ok", [(512, 8, False), (1024, 8, True), (2048, 2, False), (2048, 4, True)])
def test_chunk_major_shape_limits(ops, K, R, ok):
    """Chunk-major units exist for K > 512 and R in {4, 8, 16}: anything else is rejected at encode time instead of
    failing later at the first launch."""
    w = weights_like_model((16, K))
    if ok:
        enc, back = roundtrip(ops, w, HP, R, True)
        assert enc.meta.chunk_major and torch.equal(bits(back.cpu()), bits(w))
    else:
        with pytest.raises(ValueError, match="chunk-major"):
            ops.encode(w, HP, R, True)


@pytest.mark.parametrize("K, R", [(256, 8), (4608, 8), (2048, 3), (2048, 32)])
def test_bad_shapes_rejected(ops, K, R):
    with pytest.raises(ValueError):
        ops.encode(weights_like_model((32, K)), HP, R, False)


def special_values(rows, K, seed=0):
    """Model-like weights salted with every special BF16 class: +-0, subnormals, exponent 1 and 254, +-inf, NaN with
    payloads (quiet / signalling / all ones)."""
    w = weights_like_model((rows, K), seed=seed).view(torch.int16).clone()
    specials = torch.tensor(
        [0x0000, 0x8000, 0x0001, 0x007F, 0x8001, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1,
         0x7F81, 0xFFFF, 0x7FFF, 0xFFC1, 0x3F80, 0xBF80],
        dtype=torch.int32,
    ).to(torch.int16)
    g = torch.Generator().manual_seed(seed + 1)
    idx = torch.randperm(rows * K, generator=g)[: 64 * len(specials)]
    w.view(-1)[idx] = specials.repeat(64)
    w[0, : len(specials)] = specials  # also in the first lane slots
    return w.view(torch.bfloat16)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_special_values_bitwise(ops, layout):
    fmt, R, cm = LAYOUTS[layout]
    w = special_values(16, 2048)
    enc, back = roundtrip(ops, w, fmt, R, cm)
    assert torch.equal(bits(back.cpu()), bits(w))


def nibble_rows_with_exponents(n_exp, K=512, rows=4):
    """Rows using exactly n_exp distinct non-zero exponents (< 128) plus exponent 0; frequencies distinct so the LUT
    choice is deterministic (the least frequent exponent is the one that escapes when n_exp > 15)."""
    w = torch.empty(rows, K, dtype=torch.int16)
    exps = list(range(100, 100 + n_exp))
    counts = [K // 2 // n_exp + (n_exp - i) for i in range(n_exp)]
    for r in range(rows):
        vals = []
        for e, c in zip(exps, counts):
            vals += [(e << 7) | ((r + j) & 0x7F) | ((j & 1) << 15) for j in range(c)]
        vals += [(j & 0x7F) for j in range(K - len(vals))]  # exponent 0: zeros and subnormals
        w[r] = torch.tensor(vals, dtype=torch.int32).to(torch.int16)[torch.randperm(K)]
    return w.view(torch.bfloat16), counts


@pytest.mark.parametrize("n_exp", [14, 15, 16, 17])
def test_nibble_escape_boundary(ops, n_exp):
    w, counts = nibble_rows_with_exponents(n_exp)
    enc, back = roundtrip(ops, w, NIB, 0, False)
    assert torch.equal(bits(back.cpu()), bits(w))
    # precondition of this boundary test: up to 15 non-zero exponents fit the LUT, more escape (least frequent first)
    import struct

    n_esc = struct.unpack_from("<Q", bytes(enc.aux[56:64].cpu().tolist()))[0]  # header n_a
    expect = 0 if n_exp <= 15 else sum(counts[15:]) * w.shape[0]
    assert n_esc == expect


def test_nibble_exponents_with_msb_escape(ops):
    """Exponents >= 128 (|w| >= 2) cannot go through the prmt LUT and always escape."""
    w = weights_like_model((8, 1024)).view(torch.int16).clone()
    w[:, ::7] = 0x4000 | (torch.arange(w[:, ::7].numel()).view(8, -1) & 0x7F).to(torch.int16)  # exponent 128
    w = w.view(torch.bfloat16)
    enc, back = roundtrip(ops, w, NIB, 0, False)
    assert torch.equal(bits(back.cpu()), bits(w))


def pairs_tensor(n_pairs, K=512):
    """A tensor with exactly n_pairs distinct exponent pairs (positions 2i, 2i+1 within a row)."""
    pairs = [(a, b) for a in range(60, 160) for b in range(60, 160)][:n_pairs]
    n_w = 2 * len(pairs)
    rows = -(-n_w // K)
    rows += -rows % 8
    seq = torch.tensor([e for p in pairs for e in p], dtype=torch.int32)
    flat = seq.repeat(-(-rows * K // n_w))[: rows * K]  # cyclic padding repeats existing pairs only
    mant = torch.arange(rows * K, dtype=torch.int32) % 128
    w = ((flat << 7) | mant).to(torch.int16).view(rows, K)
    return w.view(torch.bfloat16)


@pytest.mark.parametrize("n_pairs, expected", [(4095, HP), (4096, HP), (4097, HUF)])
def test_pair_alphabet_fallback_boundary(ops, n_pairs, expected):
    w = pairs_tensor(n_pairs)
    ex = (w.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    assert torch.unique(ex.view(-1, 2)[:, 0] | ex.view(-1, 2)[:, 1] << 8).numel() == n_pairs  # precondition
    enc, back = roundtrip(ops, w, HP, 8, False)
    assert enc.meta.format == expected
    assert torch.equal(bits(back.cpu()), bits(w))


@pytest.mark.parametrize("bands", [1, 2, 3])
def test_frag_order_bitwise(ops, bands):
    w = weights_like_model((16 * bands, 512), seed=40 + bands)
    enc = encode_upload(ops, w, HP, 16, False, True)
    assert enc.meta.format == HP and enc.meta.frag_order and not enc.meta.chunk_major
    assert torch.equal(bits(ops.dequant(enc).cpu()), bits(w))


def test_frag_order_special_values_bitwise(ops):
    w = special_values(32, 512, seed=5)
    enc = encode_upload(ops, w, HP, 16, False, True)
    assert torch.equal(bits(ops.dequant(enc).cpu()), bits(w))


def test_frag_order_same_code_as_row_major(ops):
    """Fragment order reorders the same pair code: identical LUT and stream length up to per-lane word padding."""
    w = weights_like_model((64, 512), seed=9)
    rm, fo = encode_upload(ops, w, HP, 16), encode_upload(ops, w, HP, 16, False, True)
    lut = slice(rm.meta.off_a, rm.meta.off_a + 4 * 4096)
    assert torch.equal(rm.aux[lut].cpu(), fo.aux[lut].cpu())
    assert abs(rm.aux.numel() - fo.aux.numel()) <= 4 * 32 * 4  # one padding word per lane and band at most


@pytest.mark.parametrize("fmt, K, R, cm, ok", [(HP, 512, 16, False, True), (HP, 1024, 16, False, False),
                                               (HP, 512, 8, False, False), (HP, 2048, 16, True, False),
                                               (HUF, 512, 16, False, False), (NIB, 512, 0, False, False)])
def test_frag_order_shape_limits(ops, fmt, K, R, cm, ok):
    """Fragment order exists for HUFPAIR K = 512, R = 16 bands only: anything else is rejected at encode time."""
    w = weights_like_model((32, K))
    if ok:
        assert encode_upload(ops, w, fmt, R, cm, True).meta.frag_order
    else:
        with pytest.raises(ValueError, match="fragment order|fragment-order"):
            ops.encode(w, fmt, R, cm, True)


def test_frag_order_pair_alphabet_fallback(ops):
    """More than 4096 exponent pairs: the single-symbol fallback has no fragment order, and says so in its meta."""
    w = pairs_tensor(4097)  # rows a multiple of 8
    if w.shape[0] % 16:
        w = torch.cat([w, w[:8]])
    enc = encode_upload(ops, w, HP, 16, False, True)
    assert enc.meta.format == HUF and not enc.meta.frag_order
    assert torch.equal(bits(ops.dequant(enc).cpu()), bits(w))


def test_frag_order_row_ranges_match_full(ops):
    w = weights_like_model((64, 512), seed=8)
    enc = encode_upload(ops, w, HP, 16, False, True)
    full = ops.dequant(enc)
    for r0, n in [(0, 16), (16, 16), (16, 48)]:
        assert torch.equal(bits(ops.dequant(enc, r0, n)), bits(full[r0 : r0 + n]))
    with pytest.raises(ValueError, match="whole units"):
        ops.dequant(enc, 8, 16)


def test_dequant_row_ranges_match_full(ops):
    """Dequant of a unit-aligned row range (an expert view) equals that slice of the full decode."""
    w = weights_like_model((64, 2048), seed=7)
    enc = encode_upload(ops, w, HP, 8, True)
    full = ops.dequant(enc)
    for r0, n in [(0, 8), (8, 8), (24, 16), (56, 8)]:
        assert torch.equal(bits(ops.dequant(enc, r0, n)), bits(full[r0 : r0 + n]))
    with pytest.raises(ValueError, match="whole units"):
        ops.dequant(enc, 4, 8)


def test_roundtrip_equal_detects_a_flipped_bit(ops):
    w = weights_like_model((16, 512), seed=3)
    enc = encode_upload(ops, w, HP, 8, False)
    assert ops.roundtrip_equal(enc, w)
    bad = w.clone()
    bad.view(torch.int16)[5, 77] ^= 1
    assert not ops.roundtrip_equal(enc, bad)


LAYER0 = "model.language_model.layers.0."


@pytest.mark.skipif(not os.path.isdir(CKPT), reason="checkpoint not available")
@pytest.mark.parametrize(
    "name",
    [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight",
        "linear_attn.in_proj_a.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ],
)
def test_real_layer0_tensors(ops, name):
    """Layer 0 of the BF16 checkpoint through the production policy, as vLLM will load it."""
    import json

    from safetensors import safe_open

    from vllm_bf16n.policy import choose

    full = LAYER0 + name
    shard = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"][full]
    with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
        w = f.get_tensor(full)
    assert w.dtype == torch.bfloat16
    parts = [w]
    if name == "mlp.experts.gate_up_proj":  # vLLM splits w13 into gate and up before encoding
        n = w.shape[1] // 2
        parts = [w[:, :n].contiguous(), w[:, n:].contiguous()]
    for p in parts:
        n_mat = p.shape[0] if p.dim() == 3 else 1
        c = choose(p.shape[-2], p.shape[-1], n_mat)
        enc = encode_upload(ops, p, c.format, c.R, c.chunk_major)
        assert enc.meta.format == c.format
        assert ops.roundtrip_equal(enc, p)
        assert enc.nbytes < 0.8 * 2 * p.numel()  # compression actually happens
