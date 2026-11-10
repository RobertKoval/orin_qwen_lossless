# SPDX-License-Identifier: MIT
"""Shape -> format policy (docs/vllm/README.md "Module -> format", docs/FORMAT.md "Conversion policy"). Pure, no GPU."""
import pytest

from vllm_bf16n.policy import FMT_HUFPAIR, FMT_NIBBLE, Choice, choose

HP, NIB = FMT_HUFPAIR, FMT_NIBBLE


@pytest.mark.parametrize(
    "name, rows, K, n, expected",
    [
        # Qwen3.6-35B-A3B modules (docs/vllm/README.md)
        ("experts gate / up", 512, 2048, 256, Choice(HP, 16, False, True)),  # fragment order (grouped decode GEMV)
        ("experts down", 2048, 512, 256, Choice(HP, 16, False, True)),  # fragment order (batched prefill kernel)
        ("self_attn.qkv_proj (merged)", 9216, 2048, 1, Choice(HP, 2, False)),  # 576 units of 16 = 2 waves
        ("o_proj / linear_attn.out_proj", 2048, 4096, 1, Choice(HP, 4, False)),
        ("linear_attn.in_proj_qkvz", 12288, 2048, 1, Choice(HP, 8, False)),  # 768 units of 16 = 1.5 waves
        ("llama.cpp attn_qkv / attn_q (same R as llama.cpp)", 8192, 2048, 1, Choice(HP, 16, False)),
        ("llama.cpp attn_gate", 4096, 2048, 1, Choice(HP, 8, False)),
        ("linear_attn.in_proj_ba", 64, 2048, 1, Choice(NIB, 0, False)),
        ("shared_expert.gate_up_proj", 1024, 2048, 1, Choice(NIB, 0, False)),
        ("shared_expert.down_proj", 2048, 512, 1, Choice(NIB, 0, False)),
        ("lm_head / embed_tokens", 248320, 2048, 1, Choice(HP, 16, False)),
        ("mtp.fc", 2048, 4096, 1, Choice(HP, 4, False)),
    ],
)
def test_model_modules(name, rows, K, n, expected):
    assert choose(rows, K, n) == expected


@pytest.mark.parametrize(
    "rows, K, expected_format",
    [(16383, 512, NIB), (16384, 512, HP), (16385, 512, HP)],  # 8M - 512, 8M, 8M + 512 weights
)
def test_dense_size_threshold(rows, K, expected_format):
    assert choose(rows, K).format == expected_format


@pytest.mark.parametrize(
    "rows, K, R",
    [
        (8176, 2048, 8),    # R = 16 would leave 511 units (< 512); R = 8: 2 waves x 8 = R = 4 and 2 (tie: largest)
        (8192, 2048, 16),   # exactly 512 units of 16: one wave
        (8208, 2048, 2),    # 513 units of 16 = 2 waves (cost 32) -> 8: 24 -> 4: 20 -> 2: 18 (each >= 5 % better)
        (16385, 512, 1),    # odd rows: only R = 1 divides
        (8194, 1024, 2),    # rows = 2 * odd: R = 2 (4097 units)
        (248320, 2048, 16),  # lm_head: 30.3 waves of 16; smaller R saves < 5 % (496 -> 488)
        (16388, 512, 2),   # candidate cost 34 < .95 * 36
        (18436, 512, 2),   # candidate cost 38 == .95 * 40 (equality accepts smaller R)
        (20484, 512, 4),   # candidate cost 42 > .95 * 44
        (9728, 2048, 4),    # 608 units of 16 = 2 waves (32) -> 8: 3 waves (24) -> 4: 5 waves (20); 2: 10 waves
                            # (20) is not >= 5 % better than 4
    ],
)
def test_dense_unit_size(rows, K, R):
    c = choose(rows, K)
    assert (c.format, c.R, c.chunk_major) == (HP, R, False)
    assert rows % c.R == 0


@pytest.mark.parametrize(
    "per, R",
    [(2032, 8), (2047, 1), (2048, 16), (2049, 1), (2064, 16), (2040, 8), (12, 4), (6, 1)],  # 16 needs >= 2048 rows and 16 | rows
)
def test_expert_unit_size(per, R):
    c = choose(per, 512, 8)
    assert c == Choice(HP, R, False, R == 16)  # 16-row bands of K = 512 experts: fragment order
    assert per % c.R == 0


@pytest.mark.parametrize("K, cm", [(512, False), (1024, True), (2048, False), (4096, True)])  # 2048: fragment order
def test_expert_chunk_major_only_above_one_chunk(K, cm):
    assert choose(512, K, 4).chunk_major is cm


@pytest.mark.parametrize("K, ok", [(-1, False), (0, False), (1, False), (256, False), (511, False), (512, True), (513, False),
                 (1000, False), (4095, False), (4096, True), (4097, False), (4608, False)])
def test_unencodable_k_stays_bf16(K, ok):
    assert (choose(4096, K) is not None) is ok
    assert (choose(4096, K, 8) is not None) is ok


@pytest.mark.parametrize("rows, n", [(0, 1), (-1, 1), (512, -1), (512, 0)])
def test_degenerate_shapes_rejected(rows, n):
    assert choose(rows, 2048, n) is None


def test_smallest_and_largest_signed_shapes():
    assert choose(1, 512) == Choice(NIB, 0, False)
    assert choose(1, 4096) == Choice(NIB, 0, False)
    assert choose((1 << 63) - 1, 4096) == Choice(HP, 1, False)


@pytest.mark.parametrize(
    "rows, K, n, frag_order",
    [
        (2048, 512, 256, True),    # vLLM down experts
        (2040, 512, 256, False),   # rows % 16 != 0: R = 8 bands do not exist in fragment order
        (512, 2048, 256, True),    # vLLM gate / up experts: 16-row bands although rows < 2048
        (520, 2048, 256, False),   # rows % 16 != 0: chunk-major R = 8
        (16, 2048, 2, True),       # the smallest band count, two matrices
        (2048, 1024, 256, False),  # K = 1024 / 4096: no fragment-order expert kernels, chunk-major
        (512, 4096, 256, False),
        (16384, 512, 1, False),    # dense K = 512 HUFPAIR (R = 16): GEMV only, row-major
        (512, 2048, 1, False),     # one matrix of the gate / up shape is dense (NIBBLE)
    ],
)
def test_frag_order_experts(rows, K, n, frag_order):
    c = choose(rows, K, n)
    assert c.frag_order is frag_order
    assert not (c.frag_order and c.chunk_major)
    assert not c.frag_order or c.R == 16


@pytest.mark.parametrize(
    "rows, K, expected",
    [
        (12288, 2048, Choice(HP, 16, False, True)),  # linear_attn.in_proj_qkvz
        (9216, 2048, Choice(HP, 16, False, True)),   # self_attn.qkv_proj
        (2048, 4096, Choice(HP, 16, False, True)),   # o_proj / out_proj / mtp.fc
        (248320, 2048, Choice(HP, 16, False, True)),
        (2056, 4096, Choice(HP, 2, False)),          # rows % 16 != 0: no bands, the row-major wave model's R
        (1024, 2048, Choice(NIB, 0, False)),         # small dense: NIBBLE whatever the opt-in
        (64, 2048, Choice(NIB, 0, False)),
    ],
)
def test_dense_frag_order_opt_in(rows, K, expected):
    """Linear layers opt in to fragment order (tensor-core decode GEMV); without the opt-in (embeddings, get_rows) the
    same shape keeps the row-major choice."""
    assert choose(rows, K, 1, dense_frag_order=True) == expected
    assert choose(rows, K, 1).frag_order is False
