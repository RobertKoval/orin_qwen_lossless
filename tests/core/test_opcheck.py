"""torch.library.opcheck on every op: schema / mutation annotations, fake (meta) kernels against the real ones, and
AOT-dispatch with dynamic shapes (what torch.compile in vLLM relies on)."""
import pytest
import torch

from conftest import encode_upload, weights_like_model

NIB, HP = 1, 3


@pytest.fixture(scope="module")
def m(ops):
    return {
        "hp": encode_upload(ops, weights_like_model((512, 2048), seed=1), HP, 16),
        "nib": encode_upload(ops, weights_like_model((1024, 2048), seed=2), NIB),
        "gate": encode_upload(ops, weights_like_model((8, 512, 2048), seed=3), HP, 8, True),
        "up": encode_upload(ops, weights_like_model((8, 512, 2048), seed=4), HP, 8, True),
        "down": encode_upload(ops, weights_like_model((8, 2048, 512), seed=5), HP, 16),
        "down_fo": encode_upload(ops, weights_like_model((8, 2048, 512), seed=6), HP, 16, False, True),
        "gate_fo": encode_upload(ops, weights_like_model((8, 512, 2048), seed=7), HP, 16, False, True),
        "up_fo": encode_upload(ops, weights_like_model((8, 512, 2048), seed=8), HP, 16, False, True),
    }


def cases(m):
    b = torch.ops.bf16n
    x4 = torch.randn(4, 2048, device="cuda").to(torch.bfloat16)
    x12 = torch.randn(12, 2048, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(8)[:4] for _ in range(12)]).to(torch.int32).cuda()
    hp, nib, gate, up, down, dfo = m["hp"], m["nib"], m["gate"], m["up"], m["down"], m["down_fo"]
    gfo, ufo = m["gate_fo"], m["up_fo"]
    a12 = torch.randn(12, 4, 512, device="cuda").to(torch.bfloat16)
    w2 = (torch.randn(64, 2048, device="cuda") * 0.02).to(torch.bfloat16)  # rows of a draft-head copy
    return {
        "encode": (b.encode, (torch.randn(16, 512).to(torch.bfloat16), HP, 8, False)),
        "dequant": (b.dequant, (hp.plane, hp.aux, hp.m, 16, 32)),
        "mmv_dense": (b.mmv, (x4, hp.plane, hp.aux, hp.m, None, 0, True)),
        "mmv_ids": (b.mmv, (x12, gate.plane, gate.aux, gate.m, ids, 512, False)),
        "mmv_glu": (b.mmv_glu, (x4, nib.plane, nib.aux, nib.m, None, 0, True)),
        "linear_gemv": (b.linear, (x4, hp.plane, hp.aux, hp.m, None)),
        "linear_gemm": (b.linear, (x12.view(3, 4, 2048), hp.plane, hp.aux, hp.m, None)),
        "get_rows": (b.get_rows, (ids[:, 0].long(), hp.plane, hp.aux, hp.m, True)),
        "moe_mm_ids_glu": (b.moe_mm_ids, (x12, ids, gate.plane, gate.aux, gate.m, up.plane, up.aux, up.m, 8, True)),
        "moe_mm_ids_down": (b.moe_mm_ids, (torch.randn(12, 4, 512, device="cuda").to(torch.bfloat16), ids, down.plane,
                                           down.aux, down.m, None, None, [], 8, False)),
        "encode_fo": (b.encode, (torch.randn(16, 512).to(torch.bfloat16), HP, 16, False, True)),
        "dequant_fo": (b.dequant, (dfo.plane, dfo.aux, dfo.m, 2048, 2048)),
        "mmv_ids_fo": (b.mmv, (a12, dfo.plane, dfo.aux, dfo.m, ids, 2048, False)),
        "moe_mm_ids_down_fo": (b.moe_mm_ids, (a12, ids, dfo.plane, dfo.aux, dfo.m, None, None, [], 8, False)),
        "mmv_ids_fo_gate": (b.mmv, (x12[:5], gfo.plane, gfo.aux, gfo.m, ids[:5], 512, False)),
        "moe_mm_ids_glu_fo": (b.moe_mm_ids, (x12, ids, gfo.plane, gfo.aux, gfo.m, ufo.plane, ufo.aux, ufo.m, 8, True)),
        "moe_combine": (b.moe_combine, (torch.randn(12, 4, 2048, device="cuda"), torch.rand(12, 4, device="cuda"), True)),
        "swiglu": (b.swiglu, (torch.randn(12, 4, 512, device="cuda"), torch.randn(12, 4, 512, device="cuda"))),
        "int2_quantize": (b.int2_quantize, (w2,)),
        "int2_gemv": (b.int2_gemv, (x4[:1], *b.int2_quantize(w2))),
        "int4_quantize": (b.int4_quantize, (w2,)),
        "int4_gemv": (b.int4_gemv, (x4[:1], *b.int4_quantize(w2))),
    }


NAMES = ["encode", "dequant", "mmv_dense", "mmv_ids", "mmv_glu", "linear_gemv", "linear_gemm", "get_rows",
         "moe_mm_ids_glu", "moe_mm_ids_down", "moe_combine", "swiglu", "encode_fo", "dequant_fo", "mmv_ids_fo",
         "moe_mm_ids_down_fo", "mmv_ids_fo_gate", "moe_mm_ids_glu_fo", "int2_quantize", "int2_gemv", "int4_quantize",
         "int4_gemv"]


@pytest.mark.parametrize("name", NAMES)
def test_opcheck(ops, m, name):
    op, args = cases(m)[name]
    tests = ("test_schema", "test_autograd_registration", "test_faketensor")
    if not name.startswith("encode"):  # encode's aux size depends on the data: no AOT tracing (it never runs under compile)
        tests += ("test_aot_dispatch_dynamic",)
    torch.library.opcheck(op, args, test_utils=tests)


def test_compiled_module_fullgraph(ops, m):
    """Two linear ops in a torch.compile(fullgraph=True) module trace without graph breaks and match eager (R5)."""
    hp, nib = m["hp"], m["nib"]

    def f(x):
        h = ops.linear(x, hp)
        return ops.linear(torch.cat([h, h, h, h], dim=-1), nib)

    x = torch.randn(3, 2048, device="cuda").to(torch.bfloat16)
    cf = torch.compile(f, fullgraph=True, dynamic=True)
    assert torch.equal(cf(x).view(torch.int16), f(x).view(torch.int16))
    x9 = torch.randn(9, 2048, device="cuda").to(torch.bfloat16)  # other side of the GEMV / GEMM switch, same graph
    assert torch.equal(cf(x9).view(torch.int16), f(x9).view(torch.int16))
