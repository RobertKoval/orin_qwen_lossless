# Research directions

The implemented engines share the BF16N format, encoder, policy and CUDA kernels. These are possible research directions,
not completed features or promised performance gains. See [results](RESULTS.md) for measured outcomes.

- **Fragment-order expert layout:** the standalone prototype in `bench/micro/fo_moe.cu` reduces down-expert prefill
  latency. Integration still requires whole-model correctness, decode and prefill comparisons in both engines.
- **Lossless BF16 KV:** the experimental offline codec reduces stored bytes by 25.8% on an English 8K, two-layer
  sample. This does not establish resident cache savings or attention throughput; paged allocation, incremental
  updates, prefill and speculative rollback require separate validation. Earlier F16-cache findings do not transfer
  directly to BF16 because the exponent and mantissa widths differ.
- **Shared-core performance:** preserve numerical behaviour while investigating engine-specific dispatch and
  working-set effects. A microkernel gain must be weighted by actual invocation counts and confirmed end to end.
- **Batched expert reuse and dense prefill:** reduce repeated weight decoding only when the benefit exceeds extra
  scheduling, memory and arithmetic costs.

Validation must retain original BF16 weight bits, distinguish sampling policy from weight fidelity, and compare
matched workloads with controlled clocks, memory pressure and compilation modes. No overclocking is required.
