# llama.cpp engine: BF16N as a patch series

llama.cpp at the commit pinned in [`versions.env`](../../versions.env) (`LLAMACPP_COMMIT`) plus the series in
`engines/llamacpp/patches/` gets a new tensor type `GGML_TYPE_BF16N`: an ordinary BF16 GGUF is converted at load
(`LLAMA_BF16N=hybrid`), checked bit for bit, and multiplied by fused BF16N kernels for `MUL_MAT`, `MUL_MAT_ID` and
`GET_ROWS`. Format: [../FORMAT.md](../FORMAT.md); results and research history: [RESULTS.md](RESULTS.md).

## Build

```bash
engines/llamacpp/build.sh [workdir=engines/llamacpp/src]   # clone + checkout the pin, git am the series, cmake (SM87)
```

`build.sh` refuses a work tree whose history is not exactly the pin plus the series (a stale or hand-edited tree
must be rebuilt from scratch or exported first, below). The binaries land in `<workdir>/build/bin`. The production
copy on the board is a release directory, not a dev tree (`deploy/orin/run_qwen3_a3b_bf16n.sh`, `BF16N_LLAMA_DIR`).

## Patch series

The series is `git format-patch` output against `LLAMACPP_COMMIT`, one patch per topic, so each can be reviewed and
rebased on its own and upstreamed backports drop out on a rebase:

| patch | topic |
|---|---|
| `0001` | ggml: the `GGML_TYPE_BF16N` type |
| `0002` | ggml-cuda: thin shared-core adapter (tensor views, buffer ownership, workspaces, environment switches, conversion) |
| `0003` | llama: convert to BF16N at load (`LLAMA_BF16N`, shared policy bridge, F32-router exactness check, embedding on the GPU) |
| `0004` | ggml-cuda: gated delta net reads recurrent-state rows in place; tensor-core GDN on Orin |
| `0005` | ggml-cuda: tiled transposed concat (conv state, 2-token MTP batches) + test-backend-ops cases |
| `0006` | ggml-cuda, llama: integrated-GPU graph inputs in place (`sync_host_inputs`), `GGML_CUDA_VMM_POOL_GB` |
| `0007` | llama: MTP draft sliding window and vocabulary subset (`LLAMA_MTP_SWA`, `LLAMA_MTP_VOCAB`) |
| `0008` | exact BF16N backend tests (products, routed products, GLU, base/view row gather, repeated buffer reuse) |
| `0009` | separate FP32 shared-expert GLU at K2048/N512, batches 1/2/8 |

The upstream backports of the research history (chunked gated delta net, norms and conversions) are part of the pinned
commit, so the series carries none. Before S5, `pin + series` was byte-identical to the original production tree
`503384f4311e`. S5 compiles `core/src/{encode.cpp,policy.cpp,launch.cu,mmid_helper.cu}` directly into ggml-cuda;
there is no copied encoder or kernel implementation in the patch. `build.sh` passes `BF16N_CORE_DIR` automatically;
a direct CMake build must provide `-DBF16N_CORE_DIR=/absolute/path/to/core` (a missing core is a configure error).
The loader retains llama's original tuning and flags; vLLM's merged-projection wave policy remains separate.

Work in the tree `build.sh` made (branch `bf16n`): edit, commit to the right topic (`git commit --fixup` +
`git rebase -i --autosquash $LLAMACPP_COMMIT`), then regenerate the series and check it:

```bash
engines/llamacpp/export-patches.sh [workdir]   # git format-patch --zero-commit --no-signature, deterministic
make check-patches                             # fresh pin checkout + git am of the series must succeed
```

Rebase onto a newer llama.cpp: `git rebase --onto <new> $LLAMACPP_COMMIT bf16n` in the work tree, drop the backports
upstream now contains, build, verify (test-backend-ops BF16N cases, bitwise per-op hashes against the previous build,
KLD/top-1 end to end, speed with alternating A/B), export, bump `versions.env` and the README badge.

## Server flags

For the `llama-server` command of the main README's quick start: with the full window, the projector and the MTP draft
the server process holds 56.4 GB after a 4K image and about 3 GB of the 61.4 GiB stay free (desktop
running, the kernel keeps 1 GB in reserve), so the server gets one slot (`-np 1`).
What the flags are for:

* `--fit off`: llama.cpp's automatic fitting does not see the encoded exponent streams (≈ 11 GB) and would shrink
  the context; set `-c` yourself (KV costs 20 KiB per token, 5 GiB for the full window).
* `-ub 2048`: larger micro-batches lift prefill from ≈ 830 to ≈ 1320 tok/s at the price of a larger compute buffer.
* `-cram 0`: no prompt cache in RAM, which the unified memory cannot spare.
* `GGML_CUDA_VMM_POOL_GB=4`: the projector's CUDA context needs its own pool; the default 32 GB address reservation
  per pool fails next to the model.
* Draft: the MTP layer is part of the GGUF and runs on the model's own weights (embedding and output matrix shared,
  the layer itself BF16N like the rest). One draft token: the dense matrices verify 2 tokens for ≈ 1.1× the cost
  of one, but each extra token brings its own experts, so 2–3 draft tokens are slower (36.7 / 37.8 vs 38.8 tok/s,
  also with a `--spec-draft-p-min` threshold); `LLAMA_MTP_VOCAB=65536` lets the draft score only the first 65536
  token ids, so its head reads a quarter of the output matrix (+5 %, 0.831 instead of 0.853 drafts accepted);
  F16 draft KV, because a quantized one makes flash attention keep F16 copies of the whole window.
  `LLAMA_MTP_SWA=32768` lets the draft attend to its last 32768 positions only. Its KV and attention mask then stay
  small, so it takes the prompt in full 2048-token micro-batches: the draft costs 3–6 % of prefill instead of 14–24 %
  with 64-token ones, the process stays at the same size, and beyond 32K the cheaper draft step outweighs the lower
  acceptance (224K: 22.6 vs 21.9 tok/s with a full-window draft).
* The attention mask of the model itself is 1 GiB at `-ub 2048`. On Orin the GPU reads it, like every graph input,
  straight from host memory instead of a device copy (−1.06 GB); upstream switches this off because llama.cpp
  overwrote inputs the GPU was still reading (#15034), which the patch prevents by synchronizing before it writes
  them. `-ub 1024` would save the same again but costs 13–17 % of prefill speed.

Close desktop applications for best speed: on Orin the compositor and remote desktop share the GPU and DRAM.

| variable | effect |
|---|---|
| `LLAMA_BF16N=hybrid` | enable (recommended): pair-Huffman for experts and large tensors, 4-bit codes for small ones |
| `LLAMA_BF16N=huffman` / `nibble` | one format for everything (experiments) |
| `LLAMA_BF16N_DENSE_MIN=<n>` | size threshold for entropy-coding dense tensors (default 8388608 weights) |
| `LLAMA_BF16N_INPUT_CPU=1` | keep the embedding on the CPU as upstream does |
| `GGML_BF16N_REFERENCE=1` | route every product through exact dequant + stock cuBLAS (slow; for verification) |
| `GGML_BF16N_NO_MMA=1` | prefill MoE through per-expert dequant + cuBLAS instead of the batched kernel |
| `GGML_CUDA_NO_GDN_ROWS=1` | disable the in-place recurrent-state read |
| `GGML_CUDA_VMM_POOL_GB=<n>` | virtual address space each CUDA memory pool reserves (default 32); `4` lets the image projector's context start next to the model |
| `GGML_BF16N_NO_MULTI_COL=1` | decode a dense matrix once per column again (the GEMV before the multi-column kernel) |
| `GGML_CUDA_NO_INTEGRATED=1` | copy graph inputs to the device as upstream does, instead of reading them from host memory in place |
| `LLAMA_MTP_VOCAB=<n>` | the MTP draft scores only token ids below `n` (a multiple of 256); others are never drafted |
| `LLAMA_MTP_SWA=<n>` | the MTP draft attends to its last `n` positions (sliding window); its KV holds `n` + micro-batch cells |
| `GGML_BF16N_HP_GLU=1` | fused gate/up/SWIGLU for pair-Huffman experts in decode (measured slower on Orin) |
| `GGML_BF16N_NO_GLU=1` | no gate/up/SWIGLU fusion at all (decode GEMV and prefill MoE kernel) |


## Benchmarks and checks

`L=engines/llamacpp/src` below (the `build.sh` work tree). S5's strict old–old/old–new op and five-prompt
verification, including MTP configuration receipts, is documented in [the cross-engine harness](../../bench/xengine/README.md).
`LLAMA_BUILD_TESTS=ON make build-llamacpp`, then `cmake --build "$L/build" --target test-backend-ops` and
`make test-llamacpp`; it also runs the independent allocation-lifetime observer in `tests/llamacpp/`.
`make verify-xengine` requires explicit old/new paths.

```bash
MODEL=qwen3.6-35b-a3b-bf16.gguf engines/llamacpp/tools/bench.sh          # llama-bench decode + prefill
MODEL=... bench/xengine/run_prompts.sh fast LLAMA_BF16N=hybrid           # greedy outputs + top-5 logprobs
MODEL=... bench/xengine/run_prompts.sh ref  LLAMA_BF16N=hybrid GGML_BF16N_REFERENCE=1
python3 bench/xengine/compare_outputs.py out/prompts-fast.json out/prompts-ref.json
# where a prefill forward goes: nsys of llama-bench, then the per-kernel split of the last forward
LLAMA_BF16N=hybrid nsys profile -t cuda -s none -o prof llama-bench -m $MODEL -ngl 99 -fa 1 -p 2048 -n 0 -b 2048 -ub 2048 -r 3
nsys export --type sqlite -o prof.sqlite prof.nsys-rep && python3 tools/analysis/fwd_breakdown.py prof.sqlite
```

Kernel microbenchmarks on real weights live in [`bench/micro/`](../../bench/micro): `bwprobe.cu` (DRAM read bandwidth), `nibble_gemv.cu` and
`hufpair_gemv.cu` (format + kernel; each checks its output bit-for-bit against a raw-BF16 kernel with the same summation
order), and in `tools/analysis/`: `locate_tensor.py`, `exponent_coverage.py`, `kv_entropy.py` (lossless compressibility of a saved
llama-server KV cache). Build with `nvcc -O3 -arch=sm_87 -std=c++17 -o X bench/micro/X.cu -lcublas`.

`engines/llamacpp/bench/moe_prefill.cpp` runs one BF16N `MUL_MAT_ID` of a real expert tensor (or `MUL_MAT` of a 2-D one) through the patched
ggml CUDA backend (any token count, uniform or skewed routing), reports the median of individually synchronised
repetitions after a 2 s GPU warm-up, prints a hash of the output (equal hashes = bitwise equal results) and checks
sampled (or all) output columns against a CPU reference:

```bash
g++ -O2 -std=c++17 -I $L/ggml/include engines/llamacpp/bench/moe_prefill.cpp -o moe_prefill \
    -L $L/build/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,$PWD/$L/build/bin
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_gate_exps.weight 2048    # GGML_BF16N_NO_MMA=1: previous path
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_down_exps.weight 4096 30 1 32768   # 30 reps, skewed, every column
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.ffn_gate_up_exps.weight 2048 # gate + up + SWIGLU, fused vs unfused bitwise
./moe_prefill qwen3.6-35b-a3b-bf16.gguf blk.20.attn_qkv.weight 1 300        # a 2-D tensor: MUL_MAT, the decode GEMV at 1 token
```

`engines/llamacpp/bench/gdn_check.cpp` runs one gated delta net layer (model-like inputs, decay from the layer's own `ssm_a` and `dt_bias`)
through the same backend and compares outputs and final state with a double-precision recurrence:

```bash
g++ -O2 -std=c++17 -I $L/ggml/include engines/llamacpp/bench/gdn_check.cpp -o gdn_check \
    -L $L/build/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,$PWD/$L/build/bin
./gdn_check qwen3.6-35b-a3b-bf16.gguf 0 2048        # layer 0, 2048 tokens; [reps] [1 = random initial state]
```

