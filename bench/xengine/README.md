# S5 exact llama backend verification (T-052/T-053)

No command below rebuilds or modifies the old production source/build. GPU commands are operator-only; do not run
next to a model unless its memory budget has been checked. No perf loops, clock changes, SSH, service control or
implicit production endpoints are involved.

## Per-op old–old, then old–new

Supply **source roots** containing `ggml/include` and **build roots** containing `bin/libggml*`:

```bash
make verify-xengine XENGINE_ARGS='--old-source /readonly/old-llama --old-build /readonly/old-llama/build \
  --new-source /readonly/old-llama --new-build /readonly/old-llama/build \
  --model /models/model-bf16.gguf --out out/s5-old-old'

make verify-xengine XENGINE_ARGS='--old-source /readonly/old-llama --old-build /readonly/old-llama/build \
  --new-source engines/llamacpp/src --new-build /scratch/candidate/build \
  --model /models/model-bf16.gguf --out out/s5-old-new'
```

The same `dump_ops.cpp` is compiled against each side's headers and libraries, **only into the fresh output directory**.
It uses production's existing `ggml_backend_cuda_bf16n_convert(tensor, host_bf16, format)` registry export. Format 1
requests NIBBLE; 2 requests Huffman with automatic pair-Huffman selection (3 is an internal tag, not the request).
Each graph has a fresh subprocess, containing the old release's auxiliary-buffer leak. Activations use integer-defined
PRNG arithmetic, routing is deterministic, and both sides read the same real GGUF weights. F32 router weights must
be exactly representable in BF16; rounding is forbidden. GET_ROWS additionally checks against raw BF16 on the host,
including single-row views at row0=1 and the final row (non-unit-aligned in pair-Huffman).

Coverage comes from the audited Qwen GGUF shapes: blocks 0 and 20, regular-attention block 3, embedding/output,
M1 dense/expert kernels, NIBBLE shared experts and GLU, dense multi-column NC=1/2/3/4 and remainder groups, decode
cutoff T=7/8/9, uniform/concentrated expert routing, batched T=512/2048/4096, and real down-kernel interleave boundaries
T=767/768/769 and 1535/1536/1537. Explicit additional variants set NO_MULTI_COL, NO_GLU, HP_GLU, NO_MMA and REFERENCE.
Inherited flags are never cleared. These are requested graph/dispatch cases, not a kernel-trace certification; inspect
logs and deployment flags before claiming every instantiation executed. Opt-in HP_GLU is not a production requirement
in the plan, but its explicit comparison is strict too: a difference fails and requires disposition.

```bash
make verify-xengine XENGINE_ARGS=--list      # CPU-only case inventory
# A regex subset is always reported PARTIAL, never a full gate:
make verify-xengine XENGINE_ARGS='... --case output.*mul_mat-t1-'
```

Outputs must have identical shape and **every F32 byte**. Size differences, nonfinite outputs, missing symbols,
unsupported operations, child errors and even a one-bit or signed-zero mismatch fail. Logs and `manifest.json`
(recording environment overlays, shapes and SHA256) are retained. Matching raw dumps are deleted after exact comparison;
use `--keep-dumps` to retain them. Failure dumps are always retained. The manifest's `passed` remains false on failure.
Existing result directories are refused.

`BF16N_TEST_MAX_DEVICE_BYTES` defaults to 4294967296 (4 GiB). Before tensor allocation the runner checks padded ggml
buffer sizes plus conservative auxiliary storage, full-dequant/input-conversion scratch, 64 MiB roundtrip scratch and
a 256 MiB context/cuBLAS reserve against both this cap and reported available memory. `W->buffer` must exist for
ownership cleanup. `BF16N_TEST_MAX_HOST_BYTES` defaults to 4 GiB and checks weight/input/output sizes. These are
**admission estimates, not a hard CUDA RSS or host allocator limit**; library-internal allocations/fragmentation and
other processes can change actual memory use. The parent should also enforce an external process/device budget if a
hard bound is required. Old and new processes run serially; only a current case's pair of dumps occupies disk by default.

## Candidate backend behavioral tests

Build the candidate with tests enabled first, then explicitly run:

```bash
make test-llamacpp LLAMA_WORK=/scratch/candidate LLAMA_BUILD=/scratch/candidate/build LLAMA_TEST_BACKEND=CUDA0
# Compile helper + test-only observer without executing a GPU binary:
make test-llamacpp LLAMA_WORK=/scratch/candidate LLAMA_BUILD=/scratch/candidate/build LLAMA_TEST_ARGS=--build-only
# Numeric portion only (not the complete resource gate):
/scratch/candidate/build/bin/test-backend-ops bf16n -b CUDA0
```

`LLAMA_WORK` selects the source containing `ggml/include`; `LLAMA_BUILD` selects the build containing `bin`.
Specify both when overriding the layout; the local source default remains `engines/llamacpp/src`. The remote S5
layout has the monorepo at `$S5/src` and the ignored llama source/build beside it:

```bash
S5=/path/to/s5
make -C "$S5/src" test-llamacpp LLAMA_WORK="$S5/llama" LLAMA_BUILD="$S5/llama/build"
```

The dedicated upstream test mode is necessary because the generic CPU comparison cannot copy CUDA BF16N auxiliary
state. It checks one-hot MUL_MAT/MUL_MAT_ID against independent exact host answers; GET_ROWS against raw host BF16;
and GLU bitwise against native-BF16 CUDA graphs whose intermediate products are independently verified and marked
outputs to prevent fusion. Both formats, K=512/1024/2048/4096, escapes, index endpoints, decode boundaries and synthetic
interleave boundaries are covered. GET_ROWS views use N=8192/R=16 with row0=1 and N-1, indices 0.
A separate K512/N8192/R16 GLU dataset enumerates 65*65=4225 exponent pairs, forcing the single-symbol HUFFMAN fallback;
its native-BF16 reference products and nonlinear outputs are exact too. Repeated allocate/convert/compute/free cases
remain numerical checks. No missing backend, unsupported operation or failed case is skipped.

The numeric suite defaults to a 1 GiB **cumulative admission estimate** (override with
`BF16N_TEST_MAX_DEVICE_BYTES`). The original 512 MiB default refused a valid run when its conservative reserve plus
system-wide availability delta exceeded that budget; the 1 GiB default passes the complete Orin fixture, not a
measurement or proof of maximum physical memory use. Its accounting includes the next case's padded ggml buffers, worst-case aux/scratch and free-memory consumption since
backend initialization. On Orin that free-memory query is global UMA `MemAvailable` in `/proc/meminfo`, not process-owned CUDA bytes:
it may conservatively refuse admission due to unrelated OS/device activity, but **must not classify a leak**. This is
not an allocator-enforced RSS cap. The former UMA-delta reclamation check was invalid and has been removed, not relaxed.

`tests/llamacpp/run_tests.py` builds only into `LLAMA_TEST_OUT` (default `build/tests/llamacpp`), never into the selected
llama source/build. It compiles the test-only `cuda_alloc_probe.cpp` observer without CUDA headers/libraries, compiles
`reclamation.cpp` against the selected llama headers/libraries, runs the numeric binary, then preloads the observer
**only into the external reclamation helper**. Numerical failure prevents the reclamation launch. `--observer-library`
can select an already-built observer; no observer ABI dependency is added to the engine patch.

Both `BF16N_TEST_MAX_DEVICE_BYTES` and `BF16N_TEST_MAX_HOST_BYTES` require positive ASCII decimal integers fitting
native `size_t`: zero, signs, whitespace, suffixes and overflow are rejected. Leading zeros are accepted; the wrapper
canonicalizes them in child environments without changing the caller's environment. Validation precedes directory
creation, compilation and GPU execution. Native harness parsers independently enforce the same boundary using
`std::from_chars`; the standalone engine patch keeps its own small parser, without a monorepo header dependency.
Values 1 and SIZE_MAX are syntactically valid budgets, not guarantees that a workload passes admission.

Reclamation uses successful CUDA allocation **generation IDs**, not equal byte totals or global UMA deltas. Per format,
4 warm-ups precede 64 allocation/conversion/free cycles; every measured cycle must have observed new allocations and
then exactly **zero** new live IDs/bytes after owner destruction and synchronization. Missing preload/API traffic,
unknown frees or tracking errors fail closed. A separate two-owner check keeps dense B (K512/N8192/R16, row-major)
alive while expert A (K2048/N512/E2/R8, chunk-major for pair-Huffman) is allocated/converted/freed four times. Each A
cycle must reclaim every new generation **and** preserve the baseline live allocation count/bytes before any B read.
Because allocation IDs are monotonic and no new IDs survive, equal live cardinality preserves the entire old-ID set;
this catches an accidentally freed B before potentially reading freed-but-still-mapped memory. B's rows 0/1/last must
then still match raw host BF16 bitwise. Freeing B must reclaim the generations minted during its allocation/conversion;
subsequent GET_ROWS backend caches are excluded using generation-interval snapshots, not mistaken for tensor-owner leaks. The helper uses a 1 GiB admission estimate based on observed live requested allocations,
next buffers/aux/roundtrip scratch, runtime reserve, and global availability (availability is not the leak oracle).

The observer covers intercepted `cudaMalloc`/`cudaFree` requested ownership only: it does not establish VMM, physical
page release, CUDA-internal allocation coverage, process RSS or a hard GPU-memory cap. Run its positive/negative
allocation controls and an old immutable-library negative reclamation control before claiming coverage. Linux CPU-only
observer controls are `bash tests/llamacpp/run_cuda_alloc_probe_host.sh`; the actual CUDA-runtime smoke is
`cuda_alloc_probe_smoke.cpp` (build/run owned by the GPU lease operator). Example old/new controls:

```bash
# No numeric mode is required in the old production binary. Neither tree/build is modified.
python3 tests/llamacpp/run_tests.py --source /readonly/old-llama --build /readonly/old-llama/build \
  --out build/tests/llamacpp-old --reclamation-only   # known leaking old release MUST fail
python3 tests/llamacpp/run_tests.py --source /scratch/candidate --build /scratch/candidate/build \
  --out build/tests/llamacpp-new --reclamation-only   # candidate MUST pass
```

A reclamation-only control is explicitly **not** the complete numeric/resource gate. No numerical comparison or leak
threshold is loosened: output bytes remain exact, and observed new live allocation generations must be zero.

The new `tests/test-backend-ops-bf16n.h` and edits to `tests/test-backend-ops.cpp` live in the local llama source;
the main owner exports them into the series. No supplemental patch was generated here.

## Five frozen prompt captures, MTP off and on

Use an explicitly launched isolated server with matching weights, flags, context (32768), microbatch (2048), compile
mode and loop-breaker settings. The collector never launches/stops anything:

```bash
python3 bench/xengine/collect_outputs.py --engine llama --url http://127.0.0.1:8099 \
  --model qwen3-a3b-30b --mtp off --out out/s5-old-off-1.json
# repeat old-off-2, then new-off; repeat all three with --mtp on and the corresponding server launch
python3 bench/xengine/compare_outputs.py out/s5-old-off-1.json out/s5-old-off-2.json
python3 bench/xengine/compare_outputs.py out/s5-old-off-1.json out/s5-new-off.json
```

`--engine vllm` uses `/v1/completions`, model `qwen3-a3b-30b` by default, and token-ID responses. `--request-json FILE`
records extra request options; frozen greedy/96-token/top-5/seed fields cannot be overridden. `--server-config FILE`
records behavioral launch/compile configuration; use identical receipts across A/B (not different build paths).
MTP mode and launch configuration are operator declarations, not inferred from the server. Server URLs may differ.

The first four published prompts are unchanged; the fifth is
`chat('Summarize this document in three sentences.\n\n' + serving_long_doc())` from the frozen corpus. Full raw responses,
prompt IDs, selected IDs/logprobs and ranked full top-5 objects are saved. The comparator rejects wrong prompt identities,
missing/extra prompts/tokens/top-5 entries, lossy historical pairs, duplicate JSON fields, missing IDs, nonfinite values,
reordered ranks, different flags/configuration and different parsed IEEE binary64 logprob bits (including signed zero).
It rejects context truncation and early length stops; fewer than 96 tokens require an explicit EOS termination receipt.
This verifies the logprobs exposed by JSON, not hidden full-vocabulary logits. No near-tie, KLD or tolerance exception exists.
Load logs (472 base tensors, plus MTP tensors when loaded) and deployment/performance gates remain separate operator checks.
`--no-include-long`/comparator `--prompts 4` are explicitly reduced captures, not the five-prompt gate.

`run_prompts.sh` is an optional MTP-off isolated-child launcher using the same collector. Never point its port at an
existing server. For MTP-on, use the operator's launch and the collector directly.

## CPU-only verification

```bash
make test-python-host PYTHON='uv run --no-project --with pytest python'
c++ -std=c++17 -fsyntax-only -I engines/llamacpp/src/ggml/include bench/xengine/dump_ops.cpp
c++ -std=c++17 -fsyntax-only -I engines/llamacpp/src/ggml/include engines/llamacpp/src/tests/test-backend-ops.cpp
```

Host pytest disables the CUDA fixtures in `tests/conftest.py`; these tests need neither torch nor a GPU. Policy binding
tests stay in GPU `test-core`; pure C++ policy semantics stay in `test-core-host`.
