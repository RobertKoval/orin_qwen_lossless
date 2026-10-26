#!/usr/bin/env bash
# Decode/prefill throughput with lossless BF16N weights, as reported in docs/RESULTS.md.
#   MODEL=/path/qwen3.6-35b-a3b-bf16.gguf LLAMA=./llama.cpp/build/bin scripts/bench.sh
# The numbers in RESULTS.md also drop the page cache and compact memory before each process
# (echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory) and alternate variants.
set -euo pipefail
: "${MODEL:?set MODEL to the BF16 GGUF}"
LLAMA="${LLAMA:-./llama.cpp/build/bin}"

# dynamic vs pinned clocks change the results by up to 22 % (docs/RESULTS.md), so record them
gpu=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
cpu=/sys/devices/system/cpu/cpu0/cpufreq
echo "GPU clock min/max $(cat $gpu/min_freq)/$(cat $gpu/max_freq) Hz, CPU min $(cat $cpu/scaling_min_freq) kHz"

# decode at an empty context and after 2048 tokens; every process re-encodes and round-trip checks all weights
LLAMA_BF16N="${LLAMA_BF16N:-hybrid}" "$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -t 2 -p 0 -n 64 -d 0,2048 -r 3

# prefill of an 8192-token prompt by micro-batch size
LLAMA_BF16N="${LLAMA_BF16N:-hybrid}" "$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -t 2 -p 8192 -n 0 -b 8192 \
    -ub 512,2048,4096 -r 2
