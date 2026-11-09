# vLLM + BF16N: results (M1–M6)

Measured 2026-10-30 on the Jetson AGX Orin 64GB (pinned clocks unless stated, production stopped). Comparisons are
BF16 only: vLLM-BF16N against llama.cpp-BF16N (same client where stated), vLLM-BF16N plain against MTP, and per op
against stock BF16 kernels. Design and knobs: [README.md](README.md); headline table for both engines:
[../RESULTS.md](../RESULTS.md).

This is a record: file paths are those of the time (pre-monorepo layout, e.g. `vllm/bench_openai.py` is now
`bench/serving/bench_openai.py`, `csrc/bf16n/` is `core/`, `tests/vllm_bf16n/` is `tests/{core,vllm}/`; full map in
the monorepo refactor commits). The milestone plan these results answer is in git
at tag `pre-monorepo`, `docs/VLLM_PORT_PLAN.md`.

## M1 result (2026-10-30, measured)

Built (working tree, not committed): `csrc/bf16n/` (format.h, encode.{h,cpp}, kernels.cuh, launch.{h,cu}, mmid_helper.cu,
torch_bindings.cpp, compat.cuh), `vllm_bf16n/` (ops, policy, config, linear, moe, embedding, encode_layer), `setup.py` /
`pyproject.toml` (`vllm_bf16n._C`, sm_87, -O3, -use_fast_math like ggml-cuda, entry point `vllm.general_plugins: bf16n`),
`tests/vllm_bf16n/` (6 files), `bench/vllm_bf16n/` (bench_ops.py, ggml_ref.cpp, kern_stats.py, run_bench.sh,
standalone_mmv.cu). Installed into `trt_llm/.venv` with `pip install -e . --no-build-isolation --no-deps` (pip freeze diff:
only `vllm-bf16n`); pytest lives in a side dir (`.m1deps`, PYTHONPATH), not in the venv.

* Ops (`torch.ops.bf16n`): encode (CPU), dequant, mmv (dense / expert ids, BF16 or F32 in and out), mmv_glu (NIBBLE
  [gate; up]), linear (GEMV up to 8 tokens, dequant + cuBLAS above, in-op dispatch so Dynamo sees one op), get_rows,
  moe_mm_ids (routing helper + batched kernel, fused GLU), moe_combine, swiglu, init. Fake impls in ops.py.
  Kernel attributes in `init()` only; ops allocate through the caching allocator and never sync.
* Tests: **235 passed, 2 skipped** (`pytest tests/vllm_bf16n`, 50 s): bitwise round trips (4 layouts × K 512/2048/4096,
  R−1 rejected, specials, NIBBLE 14/15/16/17 exponents, 4095/4096/4097 pairs, layer-0 tensors of the checkpoint),
  kernels vs FP64 with the 1e-5·Σ|wx| bound (T 1/2/7/8/9, MoE T 1…2048 incl. 127/128/129 on one expert, empty experts,
  int32/int64 ids, the down kernel's interleave switches 48/49, 96/97), bitwise invariants (BF16 vs F32 staging, column
  batching, BF16 output = rounded F32, fused = unfused GLU), opcheck on all ops + `torch.compile(fullgraph)`, CUDA graph
  capture/replay bitwise (T 1/2/4 decode step, batched MoE), policy table + boundaries, plugin (vLLM plugin loader,
  `VLLM_PLUGINS` opt-out, RowParallel/MergedColumn linears and a 16-expert FusedMoE loaded through vLLM's layerwise
  meta-device path, `maybe_init_modular_kernel`). Two mutations (BF16 widening order, tile-edge store) fail 70+ tests.
* Microbench, real blk.0 / output tensors from the GGUF, identical routing, nsys kernel durations, pinned clocks,
  production stopped, mean of 2 runs (µs):

| op (kernel) | llama.cpp | vLLM-BF16N, BF16 x | F32 x | vs llama | BF16 staging |
|---|---|---|---|---|---|
| attn_qkv 8192×2048, T=1 (k_hp_mmv<4>) | 154.7 | 143.2 | 142.9 | −7.4 % | +0.2 % |
| attn_gate 4096×2048 (k_hp_mmv<4>) | 61.2 | 60.3 | 60.8 | −1.4 % | −0.7 % |
| ssm_out 2048×4096 (k_hp_mmv<8>) | 69.8 | 71.6 | 71.6 | +2.5 % | 0 % |
| ffn_gate_exps, 8 slots (k_hp_mmv_cm<4,8>) | 59.6 | 59.0 | 59.1 | −0.9 % | 0 % |
| ffn_down_exps, 8 slots (k_hp_mmv<1>) | 66.1 | 63.5 | 63.5 | −3.9 % | 0 % |
| lm_head 248320×2048 (k_hp_mmv<4>) | 4258 | 4229 | 4230 | −0.7 % | 0 % |
| ssm_alpha 32×2048 (k_nibble_mmv<4>) | 4.23 | 4.03 | 4.10 | −4.7 % | −1.7 % |
| shexp down 2048×512 (k_nibble_mmv<1>) | 7.96 | 8.19 | 7.89 | +2.9 % | **+3.7 %** |
| shexp gate_up GLU (k_nibble_mmv_glu<4>; llama: 2 × gate-only) | 13.5 | 12.0 | 12.8 | −11 % | — |
| attn_qkv T=2 (k_hp_mmv_mc<4,2>) | 157.7 | 159.7 | 160.5 | +1.3 % | −0.5 % |
| ffn_gate_exps T=2, 16 slots | 137.1 | 140.3 | 140.4 | +2.3 % | 0 % |
| gate_up GLU batched, 512 tok (k_hp_mm_ids<4,8,…,true>) | 7082 | 7158 | — | +1.1 % | — |
| gate_up GLU batched, 2048 tok | 10203 | 10052 | — | −1.5 % | — |
| down batched, 512 tok | 4325 | 4366 | — | +0.9 % | — |
| down batched, 2048 tok | 8352 | 8430 | — | +0.9 % | — |
| routing helper 512 / 2048 tok | 34.6 / 131.7 | 36.1 / 133.7 | — | +4 % / +1.5 % | — |
| F32→BF16 activation convert 512 gate / 2048 down | 30.3 / 263.7 | none (BF16 in) | — | — | — |

* **Verdict: GO.** Every op within 3 % of llama.cpp (worst +2.5 %, run-to-run spread ±2 %), BF16 staging ≤ 1 % on all
  HUFPAIR kernels (the bulk of decode). One exception: the K = 512 NIBBLE GEMV (shared-expert down) pays +3.7 %
  (0.3 µs × 40 layers ≈ 12 µs/token, 0.04 %); with FP32 input it is at parity, so M2 keeps the shared-expert GLU output
  in FP32 (llama.cpp's numerics too). Same SASS as llama.cpp for the hot kernels; a torch-free driver of our launchers
  (standalone_mmv.cu) gives lm_head 4.21 ms = llama.cpp 4.23 ms.
* **Finding for M2 (high impact):** how the host waits changes GPU kernel speed on Orin. With `cudaDeviceSynchronize`
  under the default context flags, or with `cudaDeviceScheduleBlockingSync`, lm_head takes 5.1–5.8 ms instead of 4.22
  (same buffers, same launch); spin/yield scheduling or `cudaStreamSynchronize` keep 4.22. torch's
  `torch.cuda.synchronize()` hits the slow mode. M2: measure how the vLLM worker waits (nsys) and, if it blocks, set
  spin/yield scheduling on the primary context at plugin init (`cuDevicePrimaryCtxSetFlags`).
* Other findings: Python dispatch of one op ≈ 70 µs host time on Orin (irrelevant under FULL CUDA graphs, matters for
  eager prefill); stock torch BF16 GEMMs allow reduced-precision split-K reduction (T > 8 dense path error up to
  ≈ 2^-12·Σ|wx|, identical to stock vLLM; check in the M2 KLD run); `k_hp_mmv_mc<4,2>` spills 8 bytes (llama.cpp: 0)
  and NIBBLE kernels use more registers (runtime BF16/F32 store flag), measured +1.3 % / ±0.
* Deferred: llama.cpp patch on the shared headers (needs a separate tree build, bitwise decode check and tg64 in a
  downtime window; kernels.cuh differs from the patch in templates and SASS), model override for embed/lm_head (M2:
  must stay inert for other users of the shared venv), routed-expert fusion of SWIGLU into the down GEMV staging.
* Incident: the first kernel-test run allocated a T×k×rows×K FP64 gather (68 GB) and froze the board; it rebooted at
  01:26–01:34, run-models.service restarted llama-server at boot despite `.watchdog-paused`, and it was stopped again
  at 02:06. Tests now reference per expert and cap the process at 40 % of memory.

## M2 result (2026-10-30, measured)

vLLM 0.20 serves Qwen3.6-35B-A3B from the original BF16 safetensors with BF16N weights; text only, BF16 KV cache,
ctx 32768, 1 seq, MTP off. Launcher `vllm/run_bf16n.sh` (port 8010); session helper `vllm/m2_session.sh` (drop
caches, start, memory watchdog `vllm/mem_watch.sh`, client, stop via `vllm/stop_bf16n.sh`).

* **Load:** layer by layer through vLLM's meta-device path, every matrix encoded on 12 host threads and checked bitwise
  against the loaded BF16 on the GPU: **312/312 tensors `roundtrip=OK`**, 0 FAIL (embed_tokens and lm_head included).
  "Model loading took 43.5 GiB", weights 160–165 s, engine ready after ≈ 4 min with a cached compile (52–75 s init),
  ≈ 7 min with a fresh one.
* **Model override (T-035):** `vllm_bf16n/model.py`, registered over `Qwen3_5MoeForConditionalGeneration` by the
  plugin. Inert without `--quantization bf16n`: same module tree and parameters as the stock class (tested), stock
  embedding method. With bf16n: embed_tokens / lm_head get BF16NEmbeddingMethod; the 40 shared experts run a fused
  NIBBLE GLU-GEMV with an **FP32 intermediate** into the down GEMV for ≤ 8 tokens (T-036), stock path above.
* **T-033:** vLLM's decode loop waits on CUDA events, which are fast under the default flags. Measured in-process:
  only `cudaDeviceSynchronize` slows lm_head (5.3–5.6 ms vs 4.25 ms; events, stream sync and `.cpu()` stay at 4.25).
  Decode tok/s is the same under the default flags and under yield (30.57 both). The flags can only be set before the
  primary context exists (afterwards: `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`). So the plugin sets them from
  `BF16N_CUDA_SCHED` (launcher: `yield`) in every vLLM process before CUDA init. The worker logs `flags 0x2`, and
  under nsys lm_head in the decode graph takes **4.27 ms** (llama.cpp: 4.25). This also covers vLLM's
  `_sync_device()` (prompt logprobs).
* **Decode** (`vllm/bench_openai.py`, 4 greedy prompts × 256 tokens, pinned clocks):

| engine | decode tok/s | prefill tok/s (23k tokens) |
|---|---|---|
| **vLLM-BF16N (M2)** | **32.7** (32.4–32.8; GPU step 30.6 ms under nsys) | ≈ 1200 median (855–1615 over 8 runs) |
| llama.cpp BF16N, same client (llama-server, MTP off, ctx 32768) | 28.6 | 1192 |
| llama.cpp BF16N, llama-bench tg64 (README) | 31.2 | — |

  First measured at 30.6 tok/s. nsys showed the merged vLLM projections running a half-empty second wave of units
  (qkvz 12288 rows = 768 units of 16; attention qkv 9216 rows = 576). The dense R now comes from a wave model
  (`policy.dense_R`; identical to llama.cpp for llama.cpp's shapes; qkvz R = 8, attention qkv R = 2): qkvz 255 → 219 µs,
  qkv 255 → 167 µs, **30.6 → 32.7 tok/s**.
* **Correctness vs llama.cpp BF16N** (`vllm/ref_llama.sh` + `vllm/compare_llama.py`, run_prompts.sh prompts, 96
  tokens, top-5 by token id). Prompt token ids are identical in both engines. Teacher-forced on llama.cpp's own
  tokens: **top-1 agreement 382/383 (99.74 %)**, mean KL (top-5 estimate) 1.6–2.3e-3, |Δlogprob| median 1e-4,
  p99 0.15–0.22, max 0.27–0.34. Greedy text: 2–3 of 4 prompts identical to 96 tokens. Divergences: prompt 1 at
  token 47 (llama.cpp top-2 gap 0.027) and prompt 3 at token 65 (gaps 0.02 / 0.0), both near-ties. In one restart,
  prompt 3 instead diverged at token 17 with gaps 0.38 / 0.13 (not a near-tie). That is the one teacher-forced top-1
  miss, on the prefill path.
* **Fused vs reference within vLLM** (`REFERENCE=1`: exact dequant + F.linear / an FP32 per-expert reference,
  enforce-eager; outputs from `vllm/ref_vllm.py`). Greedy 3/4 identical; prompt 3 diverges at token 65 at an exact tie
  in BF16N (gap 0.0). Teacher-forced top-1 99.74 %, KL 1.8e-3. Plan go criteria (KLD ≤ 0.003, top-1 ≥ 98 %): met.
* **Memory** (`mem_watch.sh`, used = total − free − cached − nvmap pool, desktop running; baseline 2.3 GiB after
  dropping caches):

| phase | used (GiB, incl. desktop) |
|---|---|
| end of load | 50.4–51.2 |
| serving idle / decode / prefill | 54.2–54.8 |
| peak, during prompt-logprob requests (compare script, full-vocab FP32 logits) | 58.8–59.3 |

  The board total is 61.4 GiB. vLLM's start check compares against MemAvailable, which excludes the nvmap page pool
  (7.7 GB after earlier runs), so the launcher needs `drop_caches` first; it uses `--kv-cache-memory-bytes 2 GiB` with
  `--gpu-memory-utilization 0.55`.
* **Verdict: GO** (plan: c=1 ≥ 26 tok/s; within-vLLM KLD ≤ 0.003 and top-1 ≥ 98 %; load peak below the total).
  Decode beats llama.cpp through the same client (32.7 vs 28.6) and llama-bench (31.2). Prefill matches llama.cpp
  (≈ 1200); M3 works on it. Memory headroom when serving is ≈ 6.5 GiB, ≈ 2 GiB at the
  prompt-logprob peak.
* Tests: 249 passed, 2 skipped (new: `tests/vllm_bf16n/test_model.py` — scheduling hook, override with/without bf16n,
  fused shared expert; policy wave-model cases).

## M3 result (2026-10-30, measured)

* **T-038 (prefill variance 855–1615 tok/s) resolved.** Clocks were pinned the whole time (GPU 1300.5 MHz, EMC 3199
  MHz, checked with `jetson_clocks --show`). `bench_openai.py` timed a single cold 23k-token prompt, and the variance
  came from two effects:
  1. **First request at a new length per process:** 13–17 % slower (2k: 1034 vs 1186; 8k: 1367 vs 1646). The
     caching allocator grows, about 0.9 GB is swapped out to zram during that request, and Triton JITs / autotunes
     the new shapes.
  2. **Prefix caching in mamba "align" mode** (vLLM sets the attention block to 1056 tokens) cuts every prompt at a
     multiple of 1056 tokens and runs the remainder as an extra chunk. Every MoE layer pays a whole extra pass for
     that tail.

  New `vllm/bench_prefill.py`: unique-prefix prompts of fixed lengths, a warm-up run per length, then the median.
  After warm-up the spread within a session is ≤ 0.4 %.
* **Changes:**
  * MoE chunks of ≥ 4096 tokens use exact dequant into a BF16 workspace plus vLLM's Triton `fused_experts`, one half of
    the experts at a time (`expert_map`, 0.8 GB workspace). New op `bf16n.dequant_into` writes gate / up straight
    into the interleaved w13 layout.
  * Orin-tuned Triton configs (`vllm/tune_fused_moe.py` → `vllm/fused_moe_configs`, 20–24 % faster than vLLM's
    default config).
  * Below 4096 tokens the BF16N batched kernel stays, because it is faster there (layer 0: 22 vs 29.5 ms at 2048
    tokens).
* **Prefill**, tok/s (median of 5 warm runs, min–max), pinned clocks, same client:

| | 2k | 4k | 8k | 23k |
|---|---|---|---|---|
| M2 (start of M3) | 1184 | — | 1645 | 1559 |
| **M3** | **1183** (1032–1186) | **1483** | **1855** (1805–1857) | **1756** (1756–1758) |
| M3, prefix caching off | 1507 | — | 1899 (1402–2059) | 1762 |
| llama.cpp BF16N | pp8192 830–1480 (llama-bench, ub 512–4096); llama-server 23k: 1192 | | | |

  **≥ llama.cpp: yes** (8k +25 % over llama-bench at ub 4096; 23k +47 % over llama-server). Where the time goes at 8k (nsys): about 17 ms of dequant per layer per
  chunk, the align-mode tail chunk on the batched kernel (≈ 0.5 s per prompt), and FP32 partial sums of the
  two-half workspace.
* **T-039 decided: not a BF16N bug.** Numerics differ between an in-process full inductor compile and graphs loaded
  from the compile cache (AOT, or a "fresh" start that hits inductor's own cache). Each mode is deterministic. The
  AOT-loaded run and a cache-hit "fresh" run gave identical greedy text and identical metrics against the eager
  reference (KL 1.98e-3). Two full compiles (M2 wave2 / final) were identical to each other, and both modes are about
  equally close to the eager `BF16N_REFERENCE` run (KL 1.77e-3 vs 1.98e-3). Our ops are bitwise reproducible
  (graph-replay tests; identical artifacts give identical outputs). Reference numbers should be taken in cache-loaded
  mode, as production will run. Not traced to a specific inductor kernel (deferred).
* Memory: the prefill bench peaks at 55.2 GiB including the desktop (+0.8 GB workspace). Tests: 262 passed,
  2 skipped (new: `dequant_into`, dequant-path MoE with 1/2/4 expert parts, threshold boundary 63/64/65).
* **Verdict: GO** (≥ llama.cpp prefill; stable measurement).

## M4 result (2026-10-30, measured)

* vLLM's native MTP (`MTP=<n>` in `vllm/run_bf16n.sh`) runs with BF16N. The draft's MTP layer is encoded like the
  target (320/320 tensors roundtrip OK). The override registered over `Qwen3_5MoeMTP` gives the draft placeholder
  embed_tokens / lm_head, so it never loads its own 2 GB BF16 copies (vLLM shares the target's). The draft's shared
  expert uses the FP32 intermediate. The overrides now carry the stock class names: vLLM's proposer dispatches on
  `Qwen3_5MoeForConditionalGeneration` by name.
* **Measurement fix:** `bench_openai.py` counted stream chunks. With speculative decoding one chunk carries every
  token accepted in a step, so MTP runs were under-counted. It now counts tokens.
* **Decode** (`bench_openai.py`, same 4 prompts × 256 tokens: code / Ukrainian / arithmetic / 1.7k-token summary):

| config | mean tok/s | per prompt | mean acceptance length |
|---|---|---|---|
| vLLM-BF16N, no MTP | 32.7 | | |
| vLLM-BF16N MTP n=1 | 34.8 | 36.5 / 34.9 / 35.6 / 32.2 | 1.90 |
| vLLM-BF16N MTP n=2 | 35.0 | 38.2 / 33.5 / 38.9 / 29.2 | 2.60 |
| **vLLM-BF16N MTP n=3** | **36.2** | **42.0 / 32.6 / 41.1 / 29.3** | 3.14 |
| llama.cpp BF16N MTP n=1 (same client) | 35.5 | 36.8 / 35.4 / 36.5 / 33.3 | ≈ 1.78 |

  * The draft vocab slice (`BF16N_MTP_VOCAB=65536`, llama.cpp's trick) cuts acceptance from 0.90 to 0.73: 58 % of
    the Ukrainian prompt's tokens have ids ≥ 65536 (up to 242k). Left off by default.
  * `disable_padded_drafter_batch`: no gain.
* **Step analysis** (n=1, nsys, per step): about 50 ms of kernels; the plain decode step is 30.6 ms.
  * The 2-token verify is ≈ 42 ms, on par with llama.cpp's 41.2: dense multi-column GEMV 18.3 ms, expert GEMVs over
    16 (token, slot) columns 11.4 ms, down 5.8 ms.
  * The draft is ≈ 6 ms, of which 4.2 ms is the full-vocab lm_head.
  * vLLM's serial spec-decode host loop leaves ≈ 4 ms of GPU idle per step without the profiler. nsys inflates this
    to 10 ms through 1.3 ms per `cudaGraphLaunch` under node tracing.
  * So an MTP step costs ≈ 1.77 plain steps (target 1.2). The excess is the draft lm_head and vLLM's host loop, not
    the verify kernels.
* **Correctness:** greedy output with MTP n=3 is identical to non-MTP vLLM on 3/4 prompts; the fourth diverges at a
  near-tie (top-2 gap < 0.05). Teacher-forced top-1 agreement 100 %, KL 1.85e-3. Against llama.cpp: 99.74 %,
  KL 1.68e-3.
* Memory: MTP bench peak 56.6 GiB. With prompt-logprob requests and `--kv-cache-memory-bytes 1.25 GiB`: 58.8 GiB.
* Draft 32K SWA: not available for vLLM's MTP proposer. The local patch's `draft_attention_policy` is DFlash-only,
  and SWA is not needed at ctx 32768. Deferred to M5.
* **Verdict: NO-GO on the ≥ 40 tok/s mean target** (36.2; ≥ 40 only on the code and arithmetic prompts with n=3),
  and the 1.2× step target is not met. **GO on parity:** lossless MTP that matches or beats llama.cpp MTP on the same
  bench (36.2 vs 35.5), +11 % over our plain decode. Levers:
  * a cheaper draft head that is not an id prefix (frequency-ranked token subset with a full-vocab fallback, or a
    draft-only head);
  * expert-column merging in the verify GEMV when the two tokens share experts;
  * vLLM's spec-decode host loop (needs vLLM changes).

## M5 result (2026-10-30, measured)

Production candidate: `vllm/run_qwen3_a3b_vllm_bf16n.sh`, installed as `trt_llm/models/run_qwen3_a3b_vllm_bf16n.sh`.
It follows the llama.cpp launcher's contract (`[port] --detach --hf-offline`; pid / log `logs/vllm_bf16n_<port>.*`;
`BF16N_API_KEY`).
* **Serving config:** original BF16 safetensors with BF16N, ONE sequence, `--max-model-len 262144`, BF16 KV
  (`--kv-cache-memory-bytes 6 GiB`), vision on (stock BF16 tower), MTP 3, FlashInfer attention.
* **API features:**
  * `qwen3_coder` tool parser, `qwen3` reasoning parser, the generation-config override of the host's other vLLM launchers;
  * the MTP loop breaker (env patch of the rejection sampler; vLLM 0.20 rejects `--logits-processors` with MTP);
  * served as `qwen3-a3b-30b`.
* **Environment:** settings use `VBF16N_*` names, because run_models.sh exports Ornith settings into every 30B
  launcher.
* **Start-up guards:**
  * drops the page cache;
  * kills orphaned BF16N engine cores (parent pid 1 and our cache key in their environment).
* **Memory** (GiB of 61.36, used = total − MemFree − Cached; the desktop running):

| item | GiB |
|---|---|
| desktop + system after `drop_caches` | 2.3 |
| weights ("Model loading took 45.46 GiB": text 43.5, MTP layer ≈ 1.0, BF16 vision ≈ 0.9); torch allocated 44.48 / reserved 44.59 | 44.6 |
| KV + GDN state blocks for 262144 tokens (vLLM: block 1072 tokens, "maximum concurrency 1.02x") | 6.0 |
| engine-core host memory and API server (RSS beyond the GPU pool) | ≈ 2.1 |
| CUDA context, activation reserve (4288-token chunks, vision encoder profiled at 4096 image tokens), FlashInfer, CUDA-graph pools | ≈ 3.0 |
| **idle margin after start** | **3.36** |
| margin after a 3840×2160 image / after a 248K-token request / after a second image / idle after all of them | 2.61 / 2.61 / 2.53 / **2.52** |
| peak during start-up (KV allocation) | 60.3k MiB used ≈ 2.0 GiB margin |

  Fixes needed to fit:
  1. A separate `torch.cuda.MemPool` for the encoded weights. Before it, torch reserved stayed 2.8 GiB above what was
     allocated after loading: encoded tensors landed in split blocks of the previous layer's BF16 staging.
  2. One `empty_cache` after loading.
  3. 4288-token chunks: 4 × the 1072-token block, still ≥ 4096 so long prefills use the dequant path.
  4. MoE dequant workspace split into 4 parts (0.4 GB).
  5. Images capped at 4.19 MP. At native 4K (8160 image tokens) the vision encoder's activations OOM-killed the engine
     next to the full window. The cap is passed as the processor's `size` dict: vLLM counts tokens from a
     `max_pixels` override that the fast HF processor ignores, so requests failed against the encoder cache.
* **Long context** (needle at 50 % depth found at every length; MTP 3; acceptance length 2.5–3.6):

| context | prefill tok/s (TTFT) | decode tok/s | llama.cpp BF16N MTP |
|---|---|---|---|
| short (bench_openai) | 1298–1462 at 23k | 33.9 (38.5 / 30.7 / 37.6 / 28.8) | 35.5 (same bench) |
| 68.5K | 1014 (68 s) | 25.4 | 27.1 at 64K |
| 222K | 662 (336 s) | 22.1 | 22.6 at 224K (prefill 585–616) |
| 248K (full window) | 619 (400 s) | 19.8–21.1 | — |

  FlashAttention 2, vLLM's default backend, decodes at 18.4 / 10.7 / 10.2 tok/s at the same lengths. It does not pack
  the 8 query heads of a KV head when there is more than one query token (MTP verification). Microbench at 200K: 12.7 ms
  per call for 2–4 tokens vs 2.4 ms for one token. FlashInfer fixes long context but has no FULL CUDA graph with spec
  decode (PIECEWISE only), so short-context decode is 33.9 vs 36.2 tok/s.
* **Vision:** a 3840×2160 test image is downscaled to 4111 image tokens; shapes, colours and text are read correctly
  (17 s cold, 6 s warm).
* **API checks** (`vllm/m5_api_checks.py`):
  * Tool call: `get_weather({"city": "Kyiv"})`, `finish_reason=tool_calls`.
  * Reasoning: split into `reasoning` (vLLM 0.20 field name).
  * API key: `/v1/*` gives 401 without the key and 200 with it; `/health` gives 200 without it (the watchdog
    needs that).
  * Loop breaker: fires with MTP 3 (target rows 0–2 plus bonus logged); the repetition prompt stops after 3 lines.
* **Service flow** (run-models.service, `models.conf` key `vllm_bf16n`, `STARTUP_WAIT 480`; `DEFAULT_30B` set
  temporarily for the test and restored to `bf16n`):
  * health check passed at the first check;
  * a SIGKILLed API server left its engine core as an orphan (PPID 1): the watchdog restarted the launcher, which
    killed the orphan, and the server was healthy again ≈ 4.5 min later;
  * a service restart's cleanup removes vLLM completely.
* Start-up: ≈ 4.5 min warm (encode 3 min, compile 7 s with the compile cache), ≈ 8 min with a cold compile cache.
  Tests: 262 passed, 2 skipped.
* **Verdict: GO as cutover candidate.**
  * Memory margin ≥ 2 GiB at idle after a 4K image and after a full-window request.
  * Long-context decode at parity with llama.cpp (−2 % at 222K, −6 % at 64K), prefill at long context slightly
    faster.
  * Lossless weights, vision, tools, reasoning, API key and watchdog integration all work.
* **Caveats:**
  * images above 4.19 MP are downscaled;
  * short-context decode (33.9) is a little below llama.cpp MTP (35.5);
  * the MTP draft attends the full window: vLLM's MTP has no SWA option, and long-context decode is still at parity.

## M6 result: decode performance (BF16N only)

Goal: short-context MTP decode ≥ 40 tok/s (cutover measured 33.3), long context as high as possible, lossless
outputs, ≥ 2 GiB free at idle with the 262144 window + vision. Bench: `vllm/bench_openai.py`, 4 greedy prompts × 256
tokens (code / Ukrainian / arithmetic / 1.7k-token summary), warm, second of two passes (`REPEAT=2`); "nolp" =
without logprobs, as real clients (`LOGPROBS=0`). All numbers are vLLM-BF16N or llama.cpp-BF16N.

**Result: GO. Production runs the M6 configuration: 40.9 tok/s short-context decode (nolp), 40.4 with logprobs.**

| | before (cutover, FlashInfer) | M6 | llama.cpp BF16N MTP |
|---|---|---|---|
| short, MTP 3, nolp | 33.9 (38.5 / 30.7 / 37.6 / 28.8) | **41.1** (45.8 / 38.0 / 45.9 / 34.9) | 35.5 |
| 68.5K decode / prefill | 25.4 / 1014 | **28.0 / 1076** | 27.1 at 64K |
| 200K decode / prefill | 22.1 / 662 (222K) | **23.6 / 705** | 22.6 at 224K |
| 248K decode / prefill | 19.8–21.1 / 619 | **21.7 / 620** | — |
| prefill 23k | 1298–1462 | 1304 | — |

MTP depth in the M6 configuration (nolp mean; per-prompt draft acceptance in brackets): n=1 37.6 (0.98 / 0.85 /
0.93 / 0.84), n=2 38.4, **n=3 41.1** (0.87 / 0.62 / 0.85 / 0.57), n=4 31.3. Plain decode without MTP: 31.1
(was 32.7 with FlashInfer FULL graphs; production does not use it).

### Where the time went (nsys per-step timeline, `vllm/step_timeline.py`; host side py-spy, `vllm/pyspy_tree.py`)

One MTP n=3 step before M6 (FlashInfer, prefix caching): 97.7 ms = target forward (4 tokens) 68.4 GPU-busy + draft
(3 × MTP layer + full-vocabulary head) 15.6 + small kernels 1.6 + **GPU idle 16.2**. Replacing the draft head
(below) cut the draft to 8.1 ms but the step stayed at 98.5 ms: idle grew to 24 ms, i.e. the step was host-bound.
Host profile (ms per step): sync wait in `_update_states_after_model_execute` 42.3, target-forward launch
(piecewise graphs, eager GDN core) 29.6, drafter 12.5 (of which FlashInfer `build_for_drafting` 4.5 incl. a
`seq_lens_cpu` GPU→CPU sync per draft step 1.8, draft sampling 2.6), input preparation + attention metadata ≈ 10.

Hypotheses and the experiment that decided each:

| hypothesis | experiment | result |
|---|---|---|
| H1 align-mode `.cpu()` sync after rejection sampling serialises host and GPU | no prefix caching (`mamba_cache_mode none`, where vLLM defers the read to the next step) | FlashInfer: −4.5 ms/step only; Triton: 36.1 → 41.0. Confirmed, but only together with H2 |
| H2 FlashInfer's per-draft-step planning + `seq_lens_cpu` sync and piecewise graphs keep the drafter host-bound | TRITON_ATTN (FULL graphs for uniform batches, no planning, no sync) | drafts become back-to-back on the GPU; short decode 33.9 → 36.1 (with H1's sync still present) |
| H3 draft head is a large GPU cost | 4-bit candidate head | −7.5 ms GPU per step; +12 % only once the step is GPU-bound (36.7 → 41.1) |
| H4 expert GEMVs re-decode experts shared by the 4 verify tokens | measured routing: 63.8 % distinct experts per verify batch; expert-grouped GEMV kernel (bitwise equal) | rejected: the HUFPAIR decoder is issue-bound, the extra activation columns cost about what the saved decodes save (T=4, 64 % distinct: down 272 vs 264 µs per-pair; gate/up slower, register spills) |
| H5 Triton attention is slow at long context | 64K / 200K decode | 16.6 / 9.1 tok/s: the 2D kernel runs (3 × 2) programs for a 4-token verify. Fixed by the split-KV path (below) |

### Changes (all lossless; target logits unchanged)

1. **Attention: TRITON_ATTN with routing** (`vllm_bf16n/triton_spec.py`, launcher default `VBF16N_ATTN=TRITON_ATTN`).
   * MTP verification and the first draft step (2–8 query tokens) use Triton's split-KV ("3D") kernel, which vLLM
     uses only for single-token queries. The kernel indexes segment partials by query token and masks per row, so it
     is query-length agnostic. 200K decode: 9.1 → 24.0 tok/s.
   * Prefill chunks (> 8 query tokens) run FlashInfer's paged prefill kernel on the same KV cache (both use the
     `[blocks, 2, block, kv_heads, dim]` layout). Triton's prefill was 1.6–2.2× slower at 68K–200K.
   * Tests: `tests/vllm_bf16n/test_attention_routing.py` (FP32 reference; 1, 2, 4, 8 and 9 query tokens; segment
     boundaries where rows of a segment are fully masked; FlashInfer plan with a cached prefix and two requests).
2. **Asynchronous accepted-token bookkeeping in align mode** (`vllm_bf16n/async_align.py`): the accepted counts go to
   pinned memory with an event; the CPU part (count readback, GDN state copies into completed prefix-cache blocks)
   runs at the start of the next `execute_model`, after the drafts are queued. Inputs that the next step changes
   are captured at the original call; finished or preempted requests are skipped. Short decode 36.1 → 39.7–40.0
   with prefix caching kept.
   * Check (`vllm/prefix_cache_check.py`): a 2-turn conversation whose second turn hits 2144 cached tokens, one
     block of them completed during the first turn's decode. Patched and stock code give identical turn-2 tokens.
3. **4-bit draft head** (`csrc/bf16n/draft_head.cuh`, `ops.draft_logits`, `BF16N_DRAFT_HEAD=full` to disable):
   * A draft-only symmetric 4-bit copy of lm_head (one FP32 scale per 128 weights, +270 MB) scores the full
     vocabulary in 1.71 ms instead of 4.2 ms.
   * The top 32 candidates are re-scored in FP32 from exactly reconstructed BF16N weight rows; all other draft
     logits are `-inf`. Candidate omission can change the draft token. Even within the set, FP32 rescoring is not
     numerically identical to the ordinary BF16-output full head, so inclusion alone does not guarantee equality.
   * **Measured on these four prompts only:** per-prompt acceptance with the full and 4-bit heads was identical
     (0.867 / 0.623 / 0.847 / 0.574), with identical outputs. This is not proof of universal draft-token equality,
     sampling-distribution preservation, or identical same-seed sequences.

### Correctness and memory

**Claim boundary:** BF16N reconstructs the target BF16 weights exactly. The draft-only 4-bit candidate selector is
approximate, and arithmetic kernels can differ despite identical weights. Correct speculative rejection can preserve
the target sampling distribution with an approximate proposal, provided it uses the actual proposal probabilities
(or a point mass for deterministic drafting): the production MTP path drafts deterministically and the rejection sampler
receives no draft probabilities, i.e. the point-mass branch. Results measured before 11-09 had the host's external MTP
loop breaker enabled (it penalises target logits on detected repetition cycles); since 11-09 the launcher keeps it off
and target sampling is unmodified. An explicit evaluation can also select the full draft head (`BF16N_DRAFT_HEAD=full`). See
[README.md](README.md#fidelity-boundaries).

* **vs llama.cpp BF16N** (`compare_llama.py`, 32K window because the prompt-logprob requests do not fit next to the
  full window, see the finding below):
  * Greedy: 3/4 prompts identical (M4: 2/4). Prompt 3 diverges at token 86, where the top-2 gaps are 0.109
    (llama.cpp) and 0.125 nats (vLLM), one BF16 step of the logit.
  * Teacher-forced: top-1 agreement 99.48 % (M4: 99.74 %; 2 vs 1 of 383 tokens), KL 2.0e-3 (M4: 1.7e-3).
* **Memory, full window + vision** (MemFree + Cached; M5 in brackets):

| state | GiB |
|---|---|
| idle after start | 3.28 (3.36) |
| after a 4K image | 2.95 (2.61) |
| after a 248K request | 2.78 (2.61) |
| idle after all | 2.73 (2.52) |

  Start-up peak 60.2 GiB used.
* Tests: 273 passed + 24 routing tests, 2 skipped.

### Findings ledger (M6)

* **Fixed:**
  * the align-mode host sync;
  * FlashInfer's drafter syncs and piecewise graphs (backend switch);
  * Triton's 2D kernel for multi-token decode at long context;
  * Triton's slow long-context prefill (FlashInfer route);
  * the draft head cost;
  * `bench_openai.py` now reports per-prompt draft acceptance (`/metrics`) and a no-logprobs mode.
* **Rejected:**
  * the expert-grouped GEMV (H4, measured no gain; kernel removed);
  * MTP n=4 (31.3).
* **Deferred:**
  * T-045: prompt-logprob requests OOM-kill the engine at the full window. compare_llama's teacher-forced requests
    did; the logits of the whole prompt are FP32 × 248320 vocabulary. The watchdog restarts the server.
  * T-046: rare single-request decode stalls, 3 of ~40 bench passes, one prompt at 16–24 tok/s; cause not
    identified.
  * T-040 (rest): the remaining host idle per step, ≈ 5 ms in align mode vs no prefix caching.

## After M6: production follow-ups (2026-10-30/31, measured)

Production configuration throughout (MTP 3, full window, vision, prefix caching); A/B = two server sessions on the
same board, identical except the change.

* **T-045 — prompt logprobs at the full window.** The M6 OOM (teacher-forced `compare_llama.py`: FP32 logits of the
  whole prompt × 248320) is gone: prompt logprobs are computed in 256-row slices (`prompt_logprobs.py`, wrapping the
  three calls of vLLM's `_get_prompt_logprobs_dict`; no slice below 16 rows, so the lm_head kernel stays the one of
  the unsliced chunk). A unit test checks the sliced result bitwise against the one-shot computation for 1, 255, 256,
  257 and 700 prompt tokens. At the full window: `compare_llama.py` teacher-forced runs (top-1 99.74 %, KL 1.58e-3 vs llama.cpp),
  `prompt_logprobs=5` over 71K and 241K prompts (89 s / 456 s), three 4K images in one request (12277 tokens).
  Minimum MemFree + Cached during the run 2.17 GiB, idle 3.32 GiB. Decode outputs (text and top-5 logprobs)
  unchanged.
* **T-048 — tuned Triton MoE config for the production split.** With `BF16N_MOE_DEQUANT_SPLIT=4` vLLM asks for an
  `E=64,N=512` config, which did not exist (default config used). Tuned with `tune_fused_moe.py` in SPLIT mode
  (expert-map timing, M = 4096 / 4288): 7.56 → 6.22 ms per call at M = 4288. Prefill A/B:

| prefill tok/s | default config | tuned `E=64` |
|---|---|---|
| 8k | 1398 | 1439 |
| 23k | 1305 | 1367 |
| 71K | 1051 | 1093 |
| 193K | 707 (700–719) | 727 (723–735) |

  Prompt logprobs at 8k / 23k identical between the two (top-1 1.0, KL 1e-17), decode outputs identical, memory
  margin unchanged. Shipped as package data (`configs/fused_moe/E=64,N=512,…json`).
* **T-049 — nothing BF16N-specific is baked into compiled graphs besides the cache key's inputs.** The cached
  graphs contain only `bf16n.linear` (its metadata come from the policy and the encoder, both key inputs; the
  `BF16N_REFERENCE` branch is a key input) and opaque ops (`moe_forward_shared`, attention, GDN), inside which the
  other `BF16N_*` knobs act eagerly; the draft head, the async bookkeeping, prompt-logprob slicing and the CUDA
  scheduling flags act outside the compiled region. The key is unchanged.
* **T-046 — rare decode stalls.** Not reproduced in 300 probe requests (`stall_probe.py`: longest step gap per
  request 77–89 ms) over three server instances, then seen again in the first `bench_openai.py` run after some server
  starts. Open; the bench now records each request's longest chunk gap and its time.
* **T-040 — host idle per MTP step is ≈ 3 ms, not 5–7.** Step time from the bench (prompt 0: 3.6 tokens/step at
  45.6 tok/s = 79 ms; prompt 3: 2.72 at 34.9 = 78 ms) against a GPU span of 75.4 ms (nsys inflates host gaps to
  7.3 ms). The host waits 57 ms on the accepted-count event, then prepares the next step in ≈ 13 ms while the drafts
  (11 ms GPU) run. Decode levers measured and dropped:
  * dense `R = 2` (and `R = 4` for lm_head-sized matrices) at 4 tokens (`bench_dense_R.py`): up to 19 µs per out
    projection in the kernel, end to end 41.13 → 41.08 tok/s (no logprobs), prefill 23k 1372 → 1366; reverted;
  * expert-ordered (token, slot) pairs for L2 reuse (`bench_expert_order.py`): gate 243 → 231 µs, down 262 → 261 µs,
    ≈ 0.5 ms per step after the extra gathers; not implemented.

  Production after these: short-context decode 40.39 tok/s with logprobs, 40.84 without, outputs equal to the S1
  release. What is left for decode is GPU work (T-051: target forward 68.5 ms, of which BF16N GEMVs 63.3).

## S5: ordered micro results and strict-N1 attribution budget

The original-cubin small A/B passed strict weight/output/RNE/ordering checks.
Shared pair improved ≈ 8.1% with rotating buffers, but 40 source-projected calls
weight that to only ≈ 126 µs/token (≈ 0.4% of the earlier 31–32 ms interval), before
whole-graph overlap/interactions. Thirty BA calls add ≈ 25 µs in the same conditional
screen; this does not explain the positive 0.57–1.11 ms engine interval gap or justify
BA-only optimisation for a1–2% bar. Dense MC N2 is not a strict-N1 explanation
without observed padding. Paging/EMC qualifications and no-end-to-end boundary
remain explicit. See [the numerical budget and one next discriminator](S5_ATTRIBUTION_BUDGET.md).

## Sliding window for the MTP draft layer (2026-11-04, measured, dropped)

At long context the draft layer's three passes per step read its whole KV cache (≈ 1.2 GB per step at 200K). A
32K window would cut that, and draft attention never changes target outputs.

* **Kernel (microbench, one attention layer, vLLM's Triton split-KV launch, ms per call for 1 / 4 query tokens).**
  Full context: 0.92 / 1.00 at 64K, 1.67 / 1.83 at 128K, 2.49 / 2.77 at 200K, 3.07 / 3.43 at 250K. A plain 32K
  window saves only 0.15–1.1 ms. The kernel prunes tiles outside the window, but it still splits the full length into
  16 segments, so at 200K only ~3 segments have work. With the block table shifted by whole blocks to the window
  start, a call costs 0.55 / 0.66 ms at any length. Windowed outputs match FP32 attention over the last 32K keys.
* **End to end** (production configuration, the plugin swapped in through the launcher, identical prompts on both
  servers):

| context | decode tok/s, full → 32K window | draft acceptance, full → 32K window |
|---|---|---|
| short (4 prompts) | 39.0 → 38.7 | identical |
| 72K | 28.6 → 26.2 | 0.565 → 0.498 |
| 130K | 24.2 → 23.1 | 0.519 → 0.468 |
| 197K | 22.6 → 21.5 | 0.570 → 0.473 |

  Target texts were identical at 72K and 130K. At 197K they part near the end at a near-tie, because the verify
  batches differ. The same window without the block-table shift gives acceptance 0.493 at 72K and an identical text,
  so the drop comes from the window, not from the shift.
* **Verdict:** dropped. The MTP layer uses long-range context: acceptance falls 12–17 % at every long length, more
  than the ≈ 6 ms per step saved at 197K. The plugin code was removed.

## 2026-11-09 release: batched MoE prefill, GDN state fixes, unmodified sampling (measured)

Production wheel `0.1.0+ge2e458f10ab4` against the 11-01 release, same launcher configuration:

* **Batched MoE prefill.** Every MoE prefill chunk runs the batched BF16N kernels; the exact dequant into a BF16
  workspace plus Triton `fused_experts` for chunks of ≥ 4096 tokens is gone (with its 0.4 GB workspace and tuned
  configs). `bench_prefill.py`, median of 3: 8K 1439 → **1516** tok/s, 23K 1367 → **1488**. Long context
  (`long_ctx_test.py`, haystack from the llama.cpp tree): prefill 1156 tok/s at 72K and 840 at 197K. The
  fragment-order down layout halves the batched down kernel (2048 tokens 9.95 → 4.34 ms per layer) but its 4-token
  decode GEMV is 7 % slower than row-major, so the policy does not choose it.
* **GDN state correctness** (`hybrid_state.py`). On the 11-01 release `tests/serving/state_checks.py` failed 7 checks:
  the first 4-token prompt after a start returned NaN (`!!!!!!!!`), `2+2=` depended on the previous request, and five
  prefix-cache hits were off by 91–98 nats against a no-cache run. All pass now; real hits are bitwise equal to the
  no-cache run.
* **Sampling.** The host's MTP loop breaker (≈ 245 interventions in 5775 requests over 11-05..11-09) is off: target
  sampling is the model's distribution.
* **Decode** (`bench_openai.py`, 4 prompts × 256 tokens): **41.4** tok/s without logprobs (41.2 before), 40.6 with
  (40.4). Long context: 29.9 tok/s at 72K (draft acceptance 0.56), 26.5 at 197K (0.63); the 11-04 runs on the same
  haystack with another prompt salt gave 28.6 / 22.6 at acceptance 0.565 / 0.570, so part of the 197K gain is the
  different continuation.
* **Vision.** The FA2 library that runs the ViT attention on this host is a rebuilt one; at the ViT's shapes (16 heads
  × 72, BF16, 1K–16K tokens) its outputs are bitwise equal to the stock wheel's library.
