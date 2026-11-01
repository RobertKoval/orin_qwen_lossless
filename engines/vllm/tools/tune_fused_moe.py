#!/usr/bin/env python3
"""Tune vLLM's unquantized (BF16) Triton fused_experts for Qwen3.6-35B-A3B's experts on this GPU (no tuned config
ships for Orin; the default config is used otherwise). Real layer-0 experts, uniform top-8 routing.
Writes <out_dir>/E=256,N=512,device_name=<dev>.json (and E=128), the files vLLM reads from VLLM_TUNED_CONFIG_FOLDER (production: `vllm_bf16n/configs/fused_moe`).
  tune_fused_moe.py <out_dir> [M list=1024,2048,3072,4096,6144,8192]
SPLIT=<s> (T-048): time the call the BF16N dequant path makes with BF16N_MOE_DEQUANT_SPLIT=s: E/s local experts with
an expert_map over all E (tokens routed elsewhere are masked), and write only the E/s file."""
import itertools, json, os, sys, time
import torch

out_dir = sys.argv[1]
Ms = [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "1024,2048,3072,4096,6144,8192").split(",")]
sys.argv = ["x"]; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bench"))
from bench_ops import load  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402

if not os.environ.get("BF16N_GGUF"):
    raise SystemExit("Set BF16N_GGUF to a local BF16 GGUF checkpoint")
r = GGUFReader(os.environ["BF16N_GGUF"])
E, N, H, TOPK = 256, 512, 2048, 8
w13 = torch.cat([load(r, "blk.0.ffn_gate_exps.weight"), load(r, "blk.0.ffn_up_exps.weight")], 1).cuda()
w2 = load(r, "blk.0.ffn_down_exps.weight").cuda()
s = torch.cuda.current_stream()


def timeit(f, n=5):
    f(); s.synchronize()
    v = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); s.synchronize(); v.append(e0.elapsed_time(e1))
    return sorted(v)[n // 2]


grid = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm, SPLIT_K=1, num_warps=nw, num_stages=ns)
        for bm, bn, bk, gm, nw, ns in itertools.product((64, 128), (64, 128, 256), (32, 64), (1, 16), (4, 8), (3, 4))]
# (a first pass over BLOCK_K 128 and 2 stages at M = 1024 found nothing faster; every config costs a ~20 s Triton
# compile on the Orin CPU, so the grid is kept to 96)
with set_current_vllm_config(VllmConfig()):
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, get_config_file_name
    from vllm.platforms import current_platform

    split = int(os.environ.get("SPLIT", "0"))
    if split:
        el = E // split
        emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        emap[:el] = torch.arange(el, dtype=torch.int32, device="cuda")
        w13, w2 = w13[:el].contiguous(), w2[:el].contiguous()
        _fe = fused_experts
        fused_experts = lambda x, a, b, w, ids: _fe(x, a, b, w, ids, global_num_experts=E, expert_map=emap)
    best = {}
    for M in Ms:
        x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(torch.int32).cuda()
        w = torch.softmax(torch.randn(M, TOPK, device="cuda"), -1)
        base = timeit(lambda: fused_experts(x, w13, w2, w, ids))
        res = []
        for cfg in grid:
            try:
                with override_config(cfg):
                    res.append((timeit(lambda: fused_experts(x, w13, w2, w, ids), 3), cfg))
            except Exception:  # shared memory / register limits
                pass
        t, cfg = min(res, key=lambda r: r[0])
        best[str(M)] = cfg
        print(f"M={M}: default {base:.2f} ms, best {t:.2f} ms {cfg}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    # the BF16N dequant path runs fused_experts on 1/split of the experts at a time (vllm_bf16n.ops.MOE_DEQUANT_SPLIT,
    # default 2): same per-expert GEMMs, so the same configs serve E/2
    for e in ((E // split,) if split else (E, E // 2)):
        name = get_config_file_name(e, N, None)
        json.dump(best, open(os.path.join(out_dir, name), "w"), indent=1)
        print("wrote", os.path.join(out_dir, name), current_platform.get_device_name())
