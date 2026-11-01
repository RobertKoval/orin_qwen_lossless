# vLLM measured results

All figures below are BF16-model measurements on Jetson AGX Orin64GB, SM87, CUDA12.6/L4T R36.5, single sequence. Historical runs used controlled GPU/CPU clocks; independent EMC pinning was not established for every diagnostic. GB and GiB are not interchangeable.

## Historical serving baseline

These are results from the **earlier engine-specific plugin**, not a deployed release of the current shared-core candidate. Its environment included local dependency/model-definition patches; version strings alone do not reproduce it. See [compatibility limits](README.md#compatibility-boundary).

| Measurement | Historical observation | Scope |
|---|---:|---|
| Text weights | about43.5GiB | BF16N storage, excluding additional vision/MTP components and runtime buffers |
| Model loading footprint | about45.46GiB reported by loader | Recipe included additional MTP/vision components; not total process/UMA usage |
| Plain decode | about31–33 tokens/s | Configuration-dependent; approximately31.1 in the later full-feature recipe |
| MTP3 short decode | about40–41 tokens/s | 40.4 with the measured logging recipe;40.9–41.1 without logprobs |
| MTP3 long decode | 28.0 at68.5K;23.6 at200K;21.7 at248K tokens/s | Different context-specific workloads; not a universal curve |
| Prefill | about1439 at8K;1367 at23K tokens/s | Historical full-feature configuration and corpus |
| Long prefill | about1093 at71K;727 at193K tokens/s | Historical prompts and cache history |

MTP results depend on draft acceptance, verification and sampling configuration. Historical configurations included an approximate candidate head and, in some runs, repetition interventions. Do not equate these with unchanged sampling solely because the target's weight encoding is lossless.

**Prefill reproduction caveat:** the historical prompt corpus is not the public corpus. Public prefill/haystack clients need a new baseline; matching requested length does not reproduce tokenization, cache reuse, expert routing or historical throughput. No public-client run is claimed to have re-established these figures.

## Shared-core candidate: correctness versus performance

The candidate retained all472 compared base tensors' encoded plane/auxiliary bytes. Installed old/candidate serving comparisons matched complete ordered top5 results over480 generated tokens with MTP and479 without, for the tested prompts/configurations. These are finite behavioral checks, not a universal proof of identical model arithmetic or sampling.

A bounded, warmed comparison with MTP off found:

| Interval-rate measurement | Earlier plugin | Shared candidate | Change |
|---|---:|---:|---:|
| Nonstream,816 post-first-token intervals | 32.220/s | 31.641/s | −1.795% |
| SSE,756 post-first-token intervals | 32.199/s | 31.087/s | −3.453% |

The server span includes GPU work **and engine-host pacing**, not pure kernel time. SSE chunks were not counted as tokens. This was one old/new start pair with fixed ordering; it does not support a population confidence interval or a complete causal diagnosis. Other installed comparisons also observed an MTP-off latency tradeoff.

Static inspection found changed BF16-input NIBBLE address generation and output-specialized shared-expert/MC kernels, alongside unchanged HP scalar/CM, routed-MoE and routing/dequantization instruction streams in the compared binaries. Smaller instruction/register counts are not a measured speedup; unchanged instructions shift investigation toward runtime placement/cache/launch behavior rather than a compiler-body explanation for those kernels. No single cause of the whole-model gap was established.

**Release status:** correctness checks passed for the tested cases, but the MTP-off loss remains a measured tradeoff. The shared-core candidate has not replaced the earlier production baseline. Native-engine MTP measurements cannot waive a vLLM MTP-off regression.

## Interpretation

Lossless BF16 weight reconstruction, approximate draft behavior, attention/cache dtype and sampling interventions must be reported separately. These measurements establish neither cross-engine bit-identical logits nor safe memory limits for other machines or contexts. Experimental offline KV compression results are [reported separately](../RESULTS.md#experimental-bf16-kv-reference), with no resident-vLLM or throughput claim.
