"""The Qwen3.5-MoE override and the host-wait scheduling hook (M2, T-033 / T-035 / T-036).

The model is constructed from the real checkpoint's config with 4 decoder layers on the meta device (no weights), with
and without `--quantization bf16n`: with it, embed_tokens / lm_head get BF16N methods and every shared expert runs the
FP32-intermediate forward; without it, the override builds exactly the stock module tree."""
import os
import socket
import subprocess
import sys

import pytest
import torch

from conftest import CKPT, gemv_ref, weights_like_model

pytestmark = pytest.mark.skipif(not os.path.isdir(CKPT), reason="checkpoint not available")


def run_py(code, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("BF16N_")}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.split("\n")


SCHED_CODE = (
    "import vllm_bf16n.sched as s\n"
    "print(s.apply_from_env())\n"
    "import torch; torch.zeros(1, device='cuda')\n"
    "print(s.primary_ctx_state())\n"
)


@pytest.mark.parametrize("mode, flag", [("yield", 2), ("spin", 1)])
def test_sched_applies_before_the_context_exists(mode, flag):
    out = run_py(SCHED_CODE, BF16N_CUDA_SCHED=mode)
    assert "set to" in out[0]
    assert out[1] == f"({flag}, True)"  # torch's context was created with the requested scheduling


def test_sched_is_inert_without_the_variable():
    out = run_py(SCHED_CODE)
    assert out[0] == "None" and out[1] == "(0, True)"


def test_sched_reports_an_already_active_context():
    code = "import torch; torch.zeros(1, device='cuda')\nimport vllm_bf16n.sched as s\nprint(s.apply_from_env())\n"
    assert "already active" in run_py(code, BF16N_CUDA_SCHED="yield")[0]


def test_sched_rejects_unknown_modes():
    e = dict(os.environ, BF16N_CUDA_SCHED="fast")
    out = subprocess.run([sys.executable, "-c", "import vllm_bf16n.sched as s; s.apply_from_env()"],
                         capture_output=True, text=True, env=e, timeout=120)
    assert out.returncode != 0 and "BF16N_CUDA_SCHED must be one of" in out.stderr


@pytest.fixture(scope="module")
def dist(ops):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel

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


def build(quantization, model_cls=None):
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models import ModelRegistry

    args = EngineArgs(model=CKPT, quantization=quantization, language_model_only=True, max_model_len=4096,
                      enforce_eager=True, dtype="bfloat16", hf_overrides={"text_config": {"num_hidden_layers": 4}})
    cfg = args.create_engine_config()
    cls = model_cls or ModelRegistry._try_load_model_cls(cfg.model_config.architectures[0])
    with set_current_vllm_config(cfg), torch.device("meta"):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            return cls(vllm_config=cfg, prefix="")
        finally:
            torch.set_default_dtype(prev)


def test_override_with_bf16n(dist):
    from vllm.model_executor.models import qwen3_5

    from vllm_bf16n.embedding import BF16NEmbeddingMethod
    from vllm_bf16n.model import Qwen3_5MoeForConditionalGeneration as Ours

    stock = (qwen3_5.VocabParallelEmbedding, qwen3_5.ParallelLMHead)
    m = build("bf16n")
    assert type(m) is Ours  # what the registry hands vLLM (same class name as the stock one)
    lm = m.language_model
    assert isinstance(lm.model.embed_tokens.quant_method, BF16NEmbeddingMethod)
    assert isinstance(lm.lm_head.quant_method, BF16NEmbeddingMethod)
    assert m.bf16n_fused_shared_experts == 4  # one per MoE block
    assert (qwen3_5.VocabParallelEmbedding, qwen3_5.ParallelLMHead) == stock  # construction-time patch undone


def test_override_is_inert_without_bf16n(dist):
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    ours = build(None)
    stock = build(None, Qwen3_5MoeForConditionalGeneration)
    shapes = lambda mod: [(n, tuple(p.shape), p.dtype) for n, p in mod.named_parameters()]  # noqa: E731
    assert shapes(ours) == shapes(stock)
    assert [n for n, _ in ours.named_modules()] == [n for n, _ in stock.named_modules()]
    assert isinstance(ours.language_model.model.embed_tokens.quant_method, UnquantizedEmbeddingMethod)
    assert not hasattr(ours, "bf16n_fused_shared_experts")


@pytest.mark.parametrize("T", [1, 8, 9])
def test_fused_shared_expert_matches_reference(dist, ops, T):
    """FP32-intermediate shared expert (T <= 8) and the stock path (T > 8) against FP64 silu(g) * u -> down."""
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from vllm_bf16n.config import BF16NConfig
    from vllm_bf16n.model import fuse_shared_experts

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_expert = Qwen2MoeMLP(2048, 512, "silu", quant_config=BF16NConfig(), reduce_results=False,
                                             prefix="t.shared_expert")

    from vllm.config import VllmConfig, set_current_vllm_config

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # vLLM constructs models under the model dtype
    try:
        with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
            blk = Block()
    finally:
        torch.set_default_dtype(prev)
    mlp = blk.shared_expert
    g, u, d = (weights_like_model(s, seed=i) for i, s in enumerate([(512, 2048), (512, 2048), (2048, 512)]))
    mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, g, 0)
    mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, u, 1)
    mlp.down_proj.weight.weight_loader(mlp.down_proj.weight, d)
    assert fuse_shared_experts(blk) == 1
    x = torch.randn(T, 2048, device="cuda").to(torch.bfloat16)
    y = mlp(x)
    gd, ud = gemv_ref(g, x)[0], gemv_ref(u, x)[0]
    a = gd * torch.sigmoid(gd) * ud
    if T > 8:  # stock path: BF16 activation between the linears
        a = a.to(torch.bfloat16).double()
    ref, S = gemv_ref(d, a)
    # T > 8: g, u, the activation and the output are each rounded to BF16 on the stock path
    tol = 1e-4 * S + 2 ** -7 * ref.abs() + (2 ** -6 * (a.abs() @ d.cuda().double().abs().t()) if T > 8 else 0)
    assert ((y.double() - ref).abs() <= tol).all()
    if T <= 8:  # exactly the composition of the FP32-intermediate ops
        from vllm_bf16n.encode_layer import weight_of

        a32 = ops.mmv_glu(x, weight_of(mlp.gate_up_proj, "bf16n"), out_dtype=torch.float32)
        assert torch.equal(y.view(torch.int16), ops.mmv(a32, weight_of(mlp.down_proj, "bf16n"),
                                                         out_dtype=torch.bfloat16).view(torch.int16))


def test_mtp_override_with_bf16n(dist):
    """The MTP draft never materialises its own vocab matrices (vLLM shares the target's) and its shared expert runs
    the FP32-intermediate forward."""
    from vllm_bf16n.model import Qwen3_5MoeMTP, _FromTarget

    m = build("bf16n", Qwen3_5MoeMTP)
    assert isinstance(m.model.embed_tokens, _FromTarget) and isinstance(m.lm_head, _FromTarget)
    assert m.bf16n_fused_shared_experts == 1
    assert not any(248320 in p.shape for p in m.parameters())  # no vocab-sized matrix anywhere in the draft


def test_mtp_override_is_inert_without_bf16n(dist):
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP as Stock

    from vllm_bf16n.model import Qwen3_5MoeMTP

    ours, stock = build(None, Qwen3_5MoeMTP), build(None, Stock)
    shapes = lambda mod: [(n, tuple(p.shape), p.dtype) for n, p in mod.named_parameters()]  # noqa: E731
    assert shapes(ours) == shapes(stock)
    assert type(ours).__name__ == type(stock).__name__  # vLLM dispatches on class names
