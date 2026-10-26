# Jetson AGX Orin notes (measured / researched while building this)

Everything below was measured on one AGX Orin 64GB dev kit (L4T R36.5) or read from NVIDIA's public BSP and sources.
Speculation is marked as such.

## Memory bandwidth is the budget

* LPDDR5 at 3199 MHz × 256 bit = 204.8 GB/s theoretical. `bench/bwprobe.cu` reaches **180 GB/s** (88 %) with ≥ 4 independent
  16-byte loads in flight per thread; `L1::no_allocate` / `L2::256B` hints change nothing for a pure stream.
* Check the clocks instead of assuming them: for weeks this board ran MAXN with EMC at 3199 MHz but the GPU on the
  `nvhost_podgov` governor (306–1300.5 MHz) and the CPU on `schedutil` from 729 MHz. Pinned (`devfreq/.../min_freq` =
  1300.5 MHz, CPU `scaling_min_freq` 2.2 GHz) decode was 1–3 % faster and the host-bound old prefill path 16–22 %.
  The measurement scripts print `/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/{min,max}_freq` and
  `/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq` before every run.
* Single-stream decode of a 3B-active MoE moves ≈ 4.0 GB of compressed weights per token here, so ≈ 22 ms is the floor at
  180 GB/s; everything else (≈ 12 ms today) is decode ALU work, small ops and launch gaps.
* The desktop is not free: the compositor/remote-desktop share GPU time and DRAM. One open GUI window cost ~13 % decode speed
  in our measurements (see RESULTS.md).

## GPU (GA10B, SM87) facts that shaped the kernels

* `ncu` needs root on Jetson; without it, SASS inspection (`cuobjdump -sass`) and ablations were enough.
* INT32 and `IMAD` share one 16-lane datapath per SM sub-partition; only FFMA can use the second one. Entropy decoding is
  therefore INT-bound long before it is FP-bound: count non-FFMA instructions per weight.
* L1 and shared memory are one array. Per-lane bitstreams rely on L1 hits, so: stream the sign/mantissa plane with
  `L1::no_allocate`, keep one LUT copy per SM (1024-thread blocks), and avoid big replicated tables.

## Accelerators

DLA, PVA, NVJPG/NVDEC, VIC and the CPU all share the same DRAM. Decoding weights anywhere but inside the GEMV adds traffic
(write decoded BF16, read it again), and DLA has no BF16. Not useful for this workload.

## `gpu-unlock` (Zagitta/gpu-unlock)

It lifts the GeForce half-rate FP16/BF16/TF32 tensor-core MMA limit on PCIe GA102–GA106 via a feature-override register
protected by SEC2 firmware. GA10B is an integrated platform device with a different, signed GSP-RISC-V boot chain; the module
does not apply, and decode is bandwidth-bound anyway (its own README reports no decode gain).

## EMC (DRAM) clock

* The EMC DVFS table lives in plaintext in the BPMP device tree (`bootloader/generic/tegra234-bpmp-3701-0005-*.dtb` in the
  BSP), per DRAM variant (here SK hynix, `MR5=6`), with 204 / 665.6 / 2133 / 3199 MHz entries plus derated copies, and a
  `max-rate-maxn` cap of 3.2 GHz. The DTB is signed but not encrypted; on an unfused dev kit it flashes with the NULL key.
* The table follows the Tegra210 EMC timing-struct layout in its header; timing fields are `ceil(t_ns · f_CK)` and PLL fields
  follow `38.4/M · (N + ½ + SDM/8192) / P` — this reproduces NVIDIA's values for all four rates.
* What is not public: NVIDIA's table generator (`t234_emc_reg_tool`) and the binned I/O, ODT, DFE and training-window fields;
  what the encrypted BPMP firmware validates. JEDEC LPDDR5 stops at 6400 MT/s; the SK hynix part has no public datasheet.
* Assessment: a 3400 MHz table is conceivable (≤ +6 % bandwidth) but carries a real risk of silent data corruption. Any
  attempt needs a multi-day memory-integrity test (stressapptest + GPU pattern tests + weight/logit hashing) before trusting
  bit-exact inference. Not attempted here.
* NVIDIA's `maxn` BPMP DTB only raises the memory-hub clock (933 → 1067 MHz), not the DRAM rate (speculative 0–3 %).
