# Shared-core microkernel attribution

These measurements compare earlier engine-specific kernels with the shared-core candidate on SM87. They are not an
end-to-end serving speedup claim. Original cubins ran in a common context on identical real weight packets and seeded
activations, with rotating working sets larger than L2, CUDA-event timing and both module-load orders. Reconstruction
and operator outputs passed strict comparisons.

## Shared-expert pair

| Operation | Earlier → shared-core, µs | Paired time change |
|---|---:|---:|
| GLU, BF16 input → FP32 intermediate | 24.67 → 21.56 | −12.60% |
| Down, FP32 intermediate → BF16 output | 15.98 → 14.60 | −8.68% |
| Measured two-kernel pair | 38.99 → 35.84 | −8.07% |

The shared expert is called once in each of 40 decoder layers. Applying the measured pair delta to those calls gives
approximately 126–136 µs per token, roughly 0.4% of a 31–32 ms interval, **before overlap and working-set effects**.
Do not add the separately measured GLU and down improvements: their timings are not additive across different cache
and launch contexts.

## Narrow repeat

The following repeat retained whole ABBA rounds with no observed global swap-I/O change, owned swap/major-fault
change, or GPU clock change at the recorded boundaries. Reverse module-load order preserved the directions.
EMC frequency/pinning was not verified; this selected-window control does not remove all environmental uncertainty.

| Operation | Earlier → shared-core, µs | Paired time change |
|---|---:|---:|
| GDN BA, 64×2048, BF16 | 7.8205 → 8.6584 | +10.71% |
| QKV scalar N1 control, 9216×2048 | 158.8874 → 158.8632 | −0.015% |
| QKV MC N2, 9216×2048 | 176.3225 → 181.5518 | +2.97% |

Positive changes mean slower. Compute buffers were explicit non-managed device allocations; Jetson's shared physical
DRAM is not CPU matrix computation or CPU offload. Host enqueue/counter reads were outside the CUDA event interval.

There are 30 BA projections per strict one-row target forward. The observed BA delta therefore suggests about
25 µs per token (approximately 0.08%), not a 10% whole-model slowdown. MC N2 has no role in strict physical N1 unless
runtime padding or batching actually selects it; configured graph buckets alone do not establish that selection.

## Interpretation

The conditional pair-plus-BA budget suggests about 100–129 µs improvement per token. It does not explain the earlier
bounded MTP-off serving comparison, which observed about 0.57–1.11 ms longer engine decode intervals. Most other
compared kernel instruction streams were unchanged. Full-model allocator/cache/clock effects and per-token engine
pacing remain possible; there is no demonstrated causal attribution from these microbenchmarks.

See [serving results](RESULTS.md) for the separate full-model observations and their limitations. Numerical parity,
selected-window kernel timings, and whole-model performance are distinct claims.
