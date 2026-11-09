"""Shared fixtures and helpers for the BF16N extension tests (run on the Orin: pytest tests -q)."""
from __future__ import annotations

import os

import pytest
import torch

# Checkpoint-specific tests skip unless the caller supplies a local BF16 safetensors directory.
CKPT = os.environ.get("BF16N_CKPT", "")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the Orin GPU")


@pytest.fixture(scope="session")
def ops():
    if not torch.cuda.is_available():
        pytest.skip("needs the Orin GPU")
    from vllm_bf16n import ops as o

    torch.backends.cuda.matmul.allow_tf32 = False
    # unified memory: a runaway allocation must fail in the allocator instead of freezing the board (it rebooted once)
    torch.cuda.set_per_process_memory_fraction(0.4)
    o.init()
    return o


def weights_like_model(shape, seed=0, scale=0.02) -> torch.Tensor:
    """BF16 weights with a model-like exponent distribution (normal, a few outliers)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * scale
    out = torch.rand(shape, generator=g) < 1e-4
    w[out] *= 50.0
    return w.to(torch.bfloat16)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def encode_upload(ops, w: torch.Tensor, fmt: int, R: int = 0, cm: bool = False, fo: bool = False):
    plane, aux, meta = ops.encode(w, fmt, R, cm, fo)
    return ops.upload(plane, aux, meta, "cuda")


def gemv_ref(w_bf16: torch.Tensor, x: torch.Tensor):
    """FP64 reference y = W x and the bound scale S = sum |w x| per output (W: rows x K, x: T x K)."""
    wd = w_bf16.to("cuda").double()
    xd = x.to("cuda").double()
    return xd @ wd.t(), xd.abs() @ wd.abs().t()


# FP32 reassociation bound of the GEMV kernels relative to sum |w x| (plan M1); BF16 rounding of an output: half ulp
GEMV_REL = 1e-5
BF16_REL = 2.0 ** -8
