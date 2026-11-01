"""The vLLM plugin skeleton: registration through vLLM's own plugin loader, method selection per layer type, and real
vLLM layers loading BF16 weights through the layerwise (meta-device) path into BF16N and computing with them."""
import os
import socket
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import BF16_REL, GEMV_REL, gemv_ref, weights_like_model


def test_vllm_plugin_loader_registers_bf16n():
    """A fresh process, as every vLLM process does: load_general_plugins() -> `--quantization bf16n` resolves and the
    platform accepts it."""
    code = (
        "from vllm.plugins import load_general_plugins\n"
        "load_general_plugins()\n"
        "from vllm.model_executor.layers.quantization import get_quantization_config, QUANTIZATION_METHODS\n"
        "from vllm.platforms import current_platform\n"
        "c = get_quantization_config('bf16n')\n"
        "current_platform.verify_quantization('bf16n')\n"
        "print(c.__module__, c.__name__, c().get_name(), 'bf16n' in QUANTIZATION_METHODS, 'verified')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "VLLM_PLUGINS"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-5:] == ["vllm_bf16n.config", "BF16NConfig", "bf16n", "True", "verified"]


def test_plugin_is_inert_when_disallowed():
    """VLLM_PLUGINS without bf16n (how other deployments in the shared venv can opt out): nothing is registered."""
    code = (
        "from vllm.plugins import load_general_plugins\n"
        "load_general_plugins()\n"
        "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS\n"
        "print('bf16n' in QUANTIZATION_METHODS)\n"
    )
    env = dict(os.environ, VLLM_PLUGINS="none")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-1] == "False"


@pytest.fixture(scope="module")
def vllm_env(ops):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (cleanup_dist_env_and_memory, init_distributed_environment,
                                  initialize_model_parallel)

    import vllm_bf16n

    vllm_bf16n.register()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
        initialize_model_parallel(1, 1)
        yield
        cleanup_dist_env_and_memory()


def cfg():
    from vllm.model_executor.layers.quantization import get_quantization_config

    return get_quantization_config("bf16n")()


def test_quant_method_per_layer_type(vllm_env):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    from vllm_bf16n.embedding import BF16NEmbeddingMethod
    from vllm_bf16n.linear import BF16NLinearMethod

    c = cfg()
    with torch.device("cuda"):
        lin = ReplicatedLinear(2048, 512, bias=False, quant_config=c, prefix="t.lin")
        emb = VocabParallelEmbedding(1024, 2048, quant_config=c, prefix="t.emb")
        head = ParallelLMHead(1024, 2048, quant_config=c, prefix="t.head")
    assert isinstance(lin.quant_method, BF16NLinearMethod)
    assert isinstance(emb.quant_method, BF16NEmbeddingMethod)
    assert isinstance(head.quant_method, BF16NEmbeddingMethod)
    assert c.get_quant_method(torch.nn.LayerNorm(8), "t.norm") is None
    # the vision tower stays stock BF16 (M5): its linears get vLLM's unquantized method
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    for prefix in ("visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1"):
        assert isinstance(c.get_quant_method(lin, prefix), UnquantizedLinearMethod)


def layer_bytes(layer):
    return sum(t.numel() * t.element_size() for t in list(layer.parameters()) + list(layer.buffers()))


@pytest.mark.parametrize("rows, K", [(2048, 4096), (512, 2048)])  # HUFPAIR R = 4 (o_proj) / NIBBLE
def test_row_parallel_linear_layerwise_load(vllm_env, rows, K):
    from vllm.model_executor.layers.linear import RowParallelLinear

    w = weights_like_model((rows, K), seed=rows)
    with torch.device("cuda"):
        layer = RowParallelLinear(K, rows, bias=False, params_dtype=torch.bfloat16, quant_config=cfg(),
                                  prefix=f"t.o_proj{rows}")
    assert layer.weight.device.type == "meta"  # nothing materialised before loading
    layer.weight.weight_loader(layer.weight, w)  # complete -> vLLM materialises, loads, encodes this layer
    assert layer_bytes(layer) < 0.8 * w.numel() * 2  # the BF16 copy is gone, the compressed one is smaller
    x1 = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    y1, _ = layer(x1)
    ref, S = gemv_ref(w, x1)
    err = (y1.double() - ref).abs()
    assert (err <= GEMV_REL * S + BF16_REL * ref.abs()).all()
    x9 = torch.randn(9, K, device="cuda").to(torch.bfloat16)
    y9, _ = layer(x9)
    assert torch.equal(y9.view(torch.int16), F.linear(x9, w.cuda()).view(torch.int16))


def test_merged_column_linear_shards(vllm_env):
    """gate_up of the shared expert: two checkpoint shards; processing waits for both (numel accounting)."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    g, u = weights_like_model((512, 2048), seed=1), weights_like_model((512, 2048), seed=2)
    with torch.device("cuda"):
        layer = MergedColumnParallelLinear(2048, [512, 512], bias=False, params_dtype=torch.bfloat16,
                                           quant_config=cfg(), prefix="t.gate_up")
    layer.weight.weight_loader(layer.weight, g, 0)
    assert layer.weight.device.type == "meta"  # half loaded: still waiting
    layer.weight.weight_loader(layer.weight, u, 1)
    x = torch.randn(2, 2048, device="cuda").to(torch.bfloat16)
    y, _ = layer(x)
    ref, S = gemv_ref(torch.cat([g, u]), x)
    assert ((y.double() - ref).abs() <= GEMV_REL * S + BF16_REL * ref.abs()).all()


def test_fused_moe_accepts_bf16n_method(vllm_env, ops):
    """A synthetic 1-layer FusedMoE with BF16NMoEMethod: vLLM's expert loaders fill w13/w2 per expert, the layer is
    encoded once complete, apply() matches the exact-weight reference, and modular-kernel init leaves it alone."""
    from vllm.model_executor.layers.fused_moe import FusedMoE

    from vllm_bf16n.moe import BF16NMoEMethod

    E, top_k, H, N = 16, 8, 2048, 512
    with torch.device("cuda"):
        moe = FusedMoE(num_experts=E, top_k=top_k, hidden_size=H, intermediate_size=N, params_dtype=torch.bfloat16,
                       quant_config=cfg(), prefix="model.layers.0.mlp.experts")
    assert isinstance(moe.quant_method, BF16NMoEMethod)
    gate = weights_like_model((E, N, H), seed=1)
    up = weights_like_model((E, N, H), seed=2)
    down = weights_like_model((E, H, N), seed=3)
    for e in range(E):
        moe.w13_weight.weight_loader(moe.w13_weight, gate[e], "experts.w13_weight", shard_id="w1", expert_id=e)
        moe.w13_weight.weight_loader(moe.w13_weight, up[e], "experts.w13_weight", shard_id="w3", expert_id=e)
    assert moe.w13_weight.device.type == "meta"  # w2 still missing: not processed yet
    for e in range(E):
        moe.w2_weight.weight_loader(moe.w2_weight, down[e], "experts.w2_weight", shard_id="w2", expert_id=e)
    assert layer_bytes(moe) < 0.8 * 3 * gate.numel() * 2
    moe.maybe_init_modular_kernel()
    assert isinstance(moe.quant_method, BF16NMoEMethod)
    T = 3
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)]).to(torch.int32).cuda()
    wts = torch.softmax(torch.randn(T, top_k, device="cuda"), -1)
    y = moe.quant_method.apply(moe, x, wts, ids, None)
    from vllm_bf16n.encode_layer import weight_of

    ref = ops.moe_experts_reference(x, weight_of(moe, "bf16n_gate"), weight_of(moe, "bf16n_up"),
                                    weight_of(moe, "bf16n_down"), wts, ids, E)
    # the encoded experts are the loaded ones: decode them and compare with the source bitwise
    assert torch.equal(ops.dequant(weight_of(moe, "bf16n_gate")).view(torch.int16).cpu(),
                       gate.reshape(E * N, H).view(torch.int16))
    assert torch.equal(ops.dequant(weight_of(moe, "bf16n_down")).view(torch.int16).cpu(),
                       down.reshape(E * H, N).view(torch.int16))
    torch.testing.assert_close(y.float(), ref.float(), rtol=2 ** -7, atol=1e-3)


@pytest.mark.parametrize("layout", ["wheel", "tree"])
@pytest.mark.parametrize("changed", ["format.h", "encode.cpp", "policy.h", "policy.cpp", "policy.py"])
def test_cache_key_tracks_encoding_and_policy_bytes_and_requires_them(tmp_path, monkeypatch, layout, changed):
    import vllm_bf16n as p

    ref = p.cache_key()
    files = {n: p._key_source(n) for n in p._KEY_FILES}
    pkg = tmp_path / "engines" / "vllm" / "vllm_bf16n"
    pkg.mkdir(parents=True)
    (pkg / "policy.py").write_bytes((p._PKG / "policy.py").read_bytes())
    where = {n: pkg / "_keysrc" / n if layout == "wheel" else tmp_path / rel for n, rel in p._KEY_FILES.items()}
    for n, b in files.items():
        where[n].parent.mkdir(parents=True, exist_ok=True)
        where[n].write_bytes(b)
    monkeypatch.setattr(p, "_PKG", pkg)
    assert p.cache_key() == ref                       # same bytes in either layout -> same key
    changed_path = pkg / "policy.py" if changed == "policy.py" else where[changed]
    original = changed_path.read_bytes()
    changed_path.write_bytes(original + b"\n")
    assert p.cache_key() != ref                       # changes cannot reuse stale compiled metadata
    changed_path.write_bytes(original)
    assert p.cache_key() == ref
    if changed != "policy.py":
        changed_path.unlink()
        with pytest.raises(RuntimeError):             # never silently omit a missing core input
            p.cache_key()


def test_cache_key_reference_mode_is_distinct(monkeypatch):
    import vllm_bf16n as p

    monkeypatch.delenv("BF16N_REFERENCE", raising=False)
    default = p.cache_key()
    monkeypatch.setenv("BF16N_REFERENCE", "0")
    assert p.cache_key() == default
    monkeypatch.setenv("BF16N_REFERENCE", "1")
    assert p.cache_key() != default


@pytest.mark.parametrize("n, rows", [(1, 256), (255, 256), (256, 256), (257, 256), (271, 256), (272, 256),
                                     (273, 256), (700, 1), (700, 333)])
def test_sliced_prompt_logprobs_equal_unsliced(n, rows):
    """Prompt logprobs computed `rows` tokens at a time (T-045) are bitwise those of the one-shot computation: inputs
    just below / at / above one slice, tails just below / at / above the 16-row minimum (271 / 272 / 273 = 256 + 15 /
    16 / 17), and a slice size below the minimum (1, raised to 16)."""
    from vllm.v1.sample.sampler import Sampler

    from vllm_bf16n.prompt_logprobs import sliced_gather

    g = torch.Generator(device="cuda").manual_seed(n)
    w = torch.randn(4096, 64, device="cuda", generator=g).to(torch.bfloat16)
    hs = torch.randn(n, 64, device="cuda", generator=g).to(torch.bfloat16)
    tgt = torch.randint(0, 4096, (n,), device="cuda", generator=g)
    logits = lambda h: h @ w.t()
    full = Sampler.gather_logprobs(Sampler.compute_logprobs(logits(hs)), 5, tgt)
    part = sliced_gather(logits, Sampler.compute_logprobs, Sampler.gather_logprobs, hs, 5, tgt, rows)
    for a, b in zip(full[:3], part[:3]):
        assert torch.equal(a, b)
