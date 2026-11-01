# Results and limits

Measured on Jetson AGX Orin64GB, SM87, CUDA12.6/L4T R36.5, Qwen3.6-35B-A3B, single sequence. Results are BF16-model comparisons. Weight-storage bytes, process RSS, GPU allocation and physical UMA usage are different quantities.

## Historical baselines versus current source

| Historical engine-specific build | Encoded weights | Short decode | With native MTP |
|---|---:|---:|---:|
| llama.cpp | 46.5GB from69.3GB | about31.2 tokens/s | about40.2 tokens/s, one draft token |
| vLLM | about43.5GiB text weights; vision/MTP extra | about31–33 tokens/s | about40–41 tokens/s, three draft tokens |

These are earlier engine-specific baselines, **not a deployed shared-core release**. Draft settings, approximate candidate heads and any sampler interventions are separate from lossless target weights. Full methodology and scope: [llama.cpp](llamacpp/RESULTS.md), [vLLM](vllm/RESULTS.md).

The shared-core candidate passed the tested encoded-weight/per-operation/serving comparisons but retains performance tradeoffs. Native generation was near the earlier baseline in limited paired measurements; native prefill changed by roughly−0.2 to−0.6%. A bounded vLLM MTP-off comparison observed32.220→31.641/s nonstream and32.199→31.087/s SSE post-first-token interval rates (−1.8% /−3.45%). Those spans include engine-host pacing. No universal performance parity or complete cause is established.

**Shared-core vLLM deployment:** one final matched warm comparison of the default MTP-on configuration measured **40.88 → 41.21 tokens/s (+0.80%)** with matching texts/token counts on four prompts. The shared-core release is now serving; health, tools, reasoning and vision checks passed. The MTP-off tradeoff above remains, and one pair is not a general speedup guarantee.

## Prefill and long-context measurements

Historical8K llama.cpp prefill ranged832–1481 tokens/s across micro-batches512–4096; historical full-feature vLLM prefill was about1439 at8K and1367 at23K tokens/s. Longer-context figures are in the engine result pages.

**The public corpus differs from the historical inputs. New public prefill/haystack runs require a new baseline.** Neither matching token count nor using the same client name establishes exact historical reproduction. No fresh public-client validation is implied by these tables.

The vLLM environment also contained local dependency/model-definition patches. [Version labels alone do not reproduce it](vllm/README.md#compatibility-boundary); compatible external definitions/patch sources remain a reproduction dependency.

## Experimental BF16 KV reference

A CPU reference inspired by [GAAC](https://arxiv.org/abs/2610.38210) was evaluated on **real post-RoPE BF16 K and BF16 V**, not converted lower-precision cache data. It is an independent reference format, not official GAAC code or reproduction.

English calibration used4096 input tokens; held-out evaluation used8192. Only attention layers3 and39 were sampled, both KV heads/all256 channels, with2048 held-out rows per128-channel partition. Calibration and evaluation texts were distinct. Raw snapshots are private and are not distributed.

| Held-out sample | Raw bytes | Accounted reference capacity |
|---|---:|---:|
| Layer3 K | 2,097,152 | 1,555,520 |
| Layer3 V | 2,097,152 | 1,556,608 |
| Layer39 K | 2,097,152 | 1,556,608 |
| Layer39 V | 2,097,152 | 1,555,520 |
| **Total** | **8,388,608** | **6,224,256** |

**25.80% smaller,1.3477× raw/capacity. All4,194,304 uint16 values reconstructed exactly across2048 blocks.**

Accounting includes6,036,512 used bytes plus187,744 bytes of compressed-slot slack. Metadata totals192,640 bytes:49,152 headers,8,192 block-table entries,4,352 calibration tables and130,944 group tags. Two raw-fallback blocks consume8,240 bytes including their headers, already included in the total. The chosen compressed slot is3032 bytes including its24-byte header; it was not retuned on the evaluation data.

**No per-group offset table exists in this sequential CPU format** (zero such serialized bytes); a future random-access GPU layout would have additional, unmeasured constraints/costs. The block-table/tail capacities are explicit reference-model accounting, not observed vLLM allocations. Audit-file metadata is separate disk provenance, not codec resident storage.

This establishes short-English/two-layer **reference storage** savings only: no all-layer, code/Ukrainian,64K/200K, throughput or resident-vLLM memory claim. There is no production cache integration. Public tools can process an appropriately declared capture, but the historical ratio cannot be reproduced without the original private raw snapshots.
