"""CUDA-graph capture: the ops allocate only through the caching allocator, never sync and set no attributes, so a
captured graph replays bitwise what eager execution computes, also after the inputs change in place."""
import pytest
import torch

from conftest import encode_upload, weights_like_model

NIB, HP = 1, 3
E, N, H, TOPK = 16, 512, 2048, 8


@pytest.fixture(scope="module")
def mats(ops):
    return {
        "qkv": encode_upload(ops, weights_like_model((1024, 2048), seed=1), HP, 16),
        "nib": encode_upload(ops, weights_like_model((256, 2048), seed=2), NIB),
        "shexp": encode_upload(ops, weights_like_model((2 * 512, 2048), seed=3), NIB),
        "gate": encode_upload(ops, weights_like_model((E, N, H), seed=4), HP, 16, False, True),  # vLLM: fragment order
        "up": encode_upload(ops, weights_like_model((E, N, H), seed=5), HP, 16, False, True),
        "down": encode_upload(ops, weights_like_model((E, H, N), seed=6), HP, 16, False, True),
    }


def step(ops, m, x, ids, wts):
    """One decode-like step through every op that runs inside vLLM's graphs."""
    return [
        ops.linear(x, m["qkv"]),
        ops.linear(x, m["nib"]),
        ops.mmv_glu(x, m["shexp"], out_dtype=torch.bfloat16),
        ops.moe_experts(x, m["gate"], m["up"], m["down"], wts, ids, E),
        ops.embedding(ids[:, 0].long(), m["qkv"]),
    ]


def inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 2048, generator=g).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, TOPK, generator=g)
    return x.cuda(), ids.cuda(), (w / w.sum(1, keepdim=True)).cuda()


@pytest.mark.parametrize("T", [1, 2, 4])
def test_capture_replay_bitwise(ops, mats, T):
    x, ids, wts = inputs(T, seed=T)
    step(ops, mats, x, ids, wts)  # warm-up outside capture (cuBLAS handles, allocator pools)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = step(ops, mats, x, ids, wts)
    for seed in (100 + T, 200 + T):  # new inputs written in place, as vLLM does before each replay
        nx, nids, nw = inputs(T, seed)
        x.copy_(nx); ids.copy_(nids); wts.copy_(nw)
        g.replay()
        torch.cuda.synchronize()
        eager = step(ops, mats, x, ids, wts)
        for a, b in zip(outs, eager):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_capture_batched_moe_path(ops, mats):
    """The prefill MoE path (routing helper + batched kernels, workspaces from the allocator) is capturable too."""
    T = 64
    x, ids, wts = inputs(T, seed=7)
    f = lambda: ops.moe_experts(x, mats["gate"], mats["up"], mats["down"], wts, ids, E)  # noqa: E731
    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = f()
    nx, nids, nw = inputs(T, seed=8)
    x.copy_(nx); ids.copy_(nids); wts.copy_(nw)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), f().view(torch.int16))
