#!/usr/bin/env bash
# Decode/prefill throughput with lossless BF16N weights, as reported in docs/RESULTS.md.
#   MODEL=/path/qwen3.6-35b-a3b-bf16.gguf LLAMA=./llama.cpp/build/bin scripts/bench.sh
set -euo pipefail
: "${MODEL:?set MODEL to the BF16 GGUF}"
LLAMA="${LLAMA:-./llama.cpp/build/bin}"

# decode at an empty context and after 2048 tokens; every process re-encodes and round-trip checks all weights
LLAMA_BF16N="${LLAMA_BF16N:-hybrid}" "$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -t 2 -p 0 -n 64 -d 0,2048 -r 3

# prefill
LLAMA_BF16N="${LLAMA_BF16N:-hybrid}" "$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -t 2 -p 512,2048 -n 0 -r 1
