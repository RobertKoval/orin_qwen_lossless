# Research directions

The implemented engines share the BF16N format, encoder, policy and CUDA kernels. These are possible research directions,
not completed features or promised performance gains. See [results](RESULTS.md) for measured outcomes.

- **Fragment-order expert layout:** the standalone prototype in `bench/micro/fo_moe.cu` reduces down-expert prefill
  latency. The first target is vLLM's prefill path, where chunks of at least 4096 tokens dequantise every expert to BF16
  (about 17 ms per layer per chunk) before Triton's fused MoE. Integration still requires whole-model correctness, decode
  and prefill comparisons.
- **Lossless BF16 KV:** the experimental offline codec reduces stored bytes by 25.8% on an English 8K, two-layer
  sample. This does not establish resident cache savings or attention throughput; paged allocation, incremental
  updates, prefill and speculative rollback require separate validation. Earlier F16-cache findings do not transfer
  directly to BF16 because the exponent and mantissa widths differ. The capture tool now covers all ten attention
  layers; the next measurement samples positions near 64K and 200K tokens in English, code and Ukrainian text.
- **Shared-core performance:** preserve numerical behaviour while investigating engine-specific dispatch and
  working-set effects. A microkernel gain must be weighted by actual invocation counts and confirmed end to end.
- **Deeper drafts (measured or modelled, not adopted):** each extra verified token costs about 8.3 ms of expert GEMVs,
  and a fifth token decodes every dense matrix a second time (MTP with four draft tokens: 31.3 tokens/s, with three:
  41.1). An eight-token block drafter such as DFlash2 would need about 5.4–5.8 accepted tokens per step to match MTP
  with three drafts; its model card reports 4.41–5.34 on other hardware. A 32K attention window for the MTP draft
  layer was measured and dropped: draft acceptance fell 12–17 % at 72K–197K tokens and decode slowed by 4–8 %
  ([results](vllm/RESULTS.md#draft-attention-window-measured-not-adopted)).
- **Batched expert reuse and dense prefill:** reduce repeated weight decoding only when the benefit exceeds extra
  scheduling, memory and arithmetic costs.

Validation must retain original BF16 weight bits, distinguish sampling policy from weight fidelity, and compare
matched workloads with controlled clocks, memory pressure and compilation modes. No overclocking is required.
