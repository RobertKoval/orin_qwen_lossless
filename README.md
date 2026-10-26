# orin_qwen_lossless

Bit-exact BF16 inference of **Qwen3.6-35B-A3B** on an **NVIDIA Jetson AGX Orin 64GB** with llama.cpp, using a lossless
compressed weight format that is decoded inside the matrix–vector kernels.

The BF16 model (64.6 GiB) does not fit into Orin's 61 GiB of unified memory. This patch keeps every weight bit-identical
but stores it 33 % smaller, and decodes it on the fly at memory bandwidth, so decoding is faster than it would be with raw
BF16 even if that fitted.

| | |
|---|---|
| resident weights | **46.5 GB** instead of 69.3 GB (ratio 0.671), all 472 tensors verified bit-exact at load |
| decode, 1 sequence | **≈ 30 tok/s** (mean 30.3 at context 0 and 2048; single runs 28.9–31.0) |
| prefill | **305–311 tok/s** (2048-token prompt ≈ 6.6 s) |
| outputs | greedy text identical to llama.cpp's own dequant + cuBLAS BF16 path |
| CPU | no weights and no model ops: the whole graph, embedding included, runs on the GPU (the host only samples) |

Details and methodology: [docs/RESULTS.md](docs/RESULTS.md). Format: [docs/FORMAT.md](docs/FORMAT.md).
Platform findings (bandwidth, EMC, gpu-unlock, accelerators): [docs/ORIN_NOTES.md](docs/ORIN_NOTES.md).

## How it works

* A new ggml type `GGML_TYPE_BF16N` keeps the sign and mantissa of each BF16 weight as one raw byte (they are
  incompressible) and entropy-codes only the exponent.
* Large tensors use a per-tensor Huffman code over **pairs** of adjacent exponents (≈ 10.7 bit/weight), decoded with one
  4096-entry table lookup per two weights; small dense tensors use a fixed-width 4-bit code with a per-row exponent table
  (≈ 12.1 bit/weight). Both are exact; there is no quantisation anywhere.
* Weights are converted while llama.cpp loads a normal BF16 GGUF, then decoded back on the GPU and compared bit-for-bit.
* CUDA kernels: fused decode + GEMV for `MUL_MAT` / `MUL_MAT_ID` (MoE experts selected on the GPU, CUDA-graph friendly),
  `GET_ROWS` for the embedding, exact dequantisation to BF16 + the stock cuBLAS path for prefill. The gated-delta-net kernel
  reads its recurrent state in place instead of a gathered copy, and the GEMV prologue issues all of its independent loads
  (code table, activations, first bitstream words) before the first use.

## Quick start (on the Orin)

```bash
git clone https://github.com/RobertKoval/orin_qwen_lossless && cd orin_qwen_lossless
scripts/build_llama.sh                 # clones llama.cpp b29c606, applies the patch, builds for SM87

# BF16 GGUF of Qwen/Qwen3.6-35B-A3B (llama.cpp convert_hf_to_gguf.py --outtype bf16)
LLAMA_BF16N=hybrid ./llama.cpp/build/bin/llama-server -m qwen3.6-35b-a3b-bf16.gguf -ngl 99 -fa on -c 8192
```

Loading reads the 69 GB GGUF and encodes it (≈ 3–4 min on the Orin's NVMe and CPU). Run fully offloaded (`-ngl 99`).

| variable | effect |
|---|---|
| `LLAMA_BF16N=hybrid` | enable; HUFPAIR for experts and big dense tensors, NIBBLE for small ones (recommended) |
| `LLAMA_BF16N=huffman` / `nibble` | one family for everything (for experiments) |
| `LLAMA_BF16N_DENSE_MIN=<n>` | dense-tensor size threshold for HUFPAIR (default 8388608 weights) |
| `LLAMA_BF16N_INPUT_CPU=1` | keep the embedding on the CPU as upstream does |
| `GGML_BF16N_REFERENCE=1` | route every product through exact dequant + stock cuBLAS (slow, for verification) |
| `GGML_CUDA_NO_GDN_ROWS=1` | disable the in-place recurrent-state read |
| `GGML_BF16N_HP_GLU=1` | fused gate/up/SWIGLU for HUFPAIR experts (measured slower on Orin) |

## Reproducing the numbers

```bash
MODEL=qwen3.6-35b-a3b-bf16.gguf scripts/bench.sh                         # llama-bench decode + prefill
MODEL=... scripts/run_prompts.sh fast LLAMA_BF16N=hybrid                   # greedy outputs + logprobs
MODEL=... scripts/run_prompts.sh ref  LLAMA_BF16N=hybrid GGML_BF16N_REFERENCE=1
python3 scripts/compare_outputs.py out/prompts-fast.json out/prompts-ref.json
```

Microbenchmarks on real weights (`nvcc -O3 -arch=sm_87 -std=c++17 -o X bench/X.cu -lcublas`):
`bench/bwprobe.cu` (DRAM read bandwidth), `bench/nibble_gemv.cu`, `bench/hufpair_gemv.cu` (format + kernel, each checks its
output bit-for-bit against a raw-BF16 kernel with the same summation order), `bench/locate_tensor.py` (offsets of a tensor
in a safetensors snapshot), `bench/exponent_coverage.py`.

Close desktop applications before measuring: on Orin the compositor and remote desktop share GPU and DRAM (one open window
cost ~13 % decode speed here).

## Limitations

* Tested on one configuration: Qwen3.6-35B-A3B, AGX Orin 64GB (SM87), CUDA 12.6, llama.cpp `b29c606`. Rows must be a
  multiple of 512 and at most 4096 wide (true for this model).
* The aux buffers are allocated outside ggml's buffers, so llama.cpp's automatic memory fitting does not see them; size the
  context explicitly.
* Prefill dequantises each used expert and synchronises once per MoE layer (not CUDA-graph capturable); fine at 305 tok/s
  but not optimised.
* The in-place recurrent-state read is enabled only for single-sequence batches.

## License

MIT, same as llama.cpp. See [LICENSE](LICENSE).

Ideas this builds on: exponent entropy coding for lossless BF16 (DFloat11, ZipNN), on-the-fly decode inside GEMM/GEMV
(ZipServ), and llama.cpp/ggml.
