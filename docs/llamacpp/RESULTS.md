# llama.cpp measured results

BF16N uses original BF16 weights, encoded and reconstructed bit-exactly. Historical measurements used Jetson AGX Orin64GB, SM87, MAXN with controlled GPU/CPU clocks, CUDA12.6/L4T R36.5, and llama.cpp commit `338fce6511c744893cc0d3d958b275f40552c7a4` plus the BF16N patches. All results are workload/configuration specific.

## Historical engine-specific baseline

These figures precede the shared-core candidate; they are **not** evidence that the current source was deployed with identical performance.

| Measurement | Observation | Scope |
|---|---:|---|
| Resident encoded weights | 46.5GB from69.3GB, ratio0.671 | 472/472 tensors passed the loading roundtrip check; decimal GB |
| Plain decode | about31.2 tokens/s | Short-context `llama-bench` generation workload |
| MTP decode | about40.2 tokens/s | One draft token; four400-token server responses |
| 8192-token prefill | 832 /1323 /1481 tokens/s | Micro-batches512 /2048 /4096, historical benchmark configuration |
| Plain long decode | about25.1 at64K;16.6 at224K tokens/s | Different context-specific benchmark/server samples |
| MTP long decode | about27.1 at64K;22.6 at224K tokens/s | Free-text and copy workloads, respectively |
| Long prefill | about648 tokens/s at224K;585 with MTP | Historical prompt and server recipe |

The long-window model plus vision/projector and runtime state occupied substantially more memory than weights alone. No free-memory observation from these runs is a universal admission or safety threshold.

**Historical prefill and long-context prompts are not the new public corpus.** Public clients and haystack inputs require a fresh baseline. These values must not be advertised as exact reproduction targets for different public prompts, even at the same token count.

## Shared-core candidate

Moving format/encoding/CUDA code into the shared core was checked against retained engine-specific outputs and encoded weights:

- 212 compared per-operation old/candidate outputs were byte-exact in the tested matrix/activation cases.
- 472 compared base tensors retained identical encoded planes and auxiliary bytes.
- Serving checks matched ordered top5 output records for the tested deterministic prompts. This is finite numerical evidence, not a general proof that every backend/configuration produces identical logits.

Measured throughput changes from two paired starts were approximately:

| Workload | Candidate change versus earlier build |
|---|---:|
| TG64, pooled nested tasks | −0.12% |
| Prefill8192, micro-batch512 | −0.57% |
| Prefill8192, micro-batch2048 | −0.54% |
| Prefill8192, micro-batch4096 | −0.22% |
| Separate native MTP comparison | about0.37% slower in one paired result |

These are small, limited-sample observations—not statistical equivalence or an unconditional no-regression claim. An earlier larger GLU regression was corrected, but the remaining measured differences were not erased as noise. Common-pointer kernel replay did not reproduce the whole-model residual, leaving working-set/placement effects unresolved.

The current shared-core source has **not replaced the earlier production build**. Its correctness evidence and performance tradeoffs are separate from the [vLLM candidate](../vllm/RESULTS.md), where the MTP-off gap was larger.

## Reproduction boundary

The pinned upstream commit and patch series are public build inputs. Use `engines/llamacpp/build.sh` from the repository root and an original BF16 GGUF converted from the pinned model checkpoint. See the [top-level build/run examples](../../README.md#build-and-run) and [format description](../FORMAT.md).

Use `llama-bench --help` / `llama-server --help` from that exact build for supported options. Record model revision, build/compiler, tensor layout, context, batch/micro-batch, attention/cache dtype, draft settings and clocks with every new measurement. Exact weight storage does not imply identical reduction order between CUDA kernels or engines. Draft acceptance and any sampling modifications need their own disclosure.
