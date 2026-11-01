#!/usr/bin/env bash
# llama.cpp at LLAMACPP_COMMIT (versions.env) + the BF16N patch series (patches/), built for Jetson AGX Orin (SM87).
#   engines/llamacpp/build.sh [workdir=engines/llamacpp/src]      (LLAMACPP_MIRROR=<local clone>: fetch without network)
# A new workdir gets the pin and `git am` of the series on branch bf16n. An existing one is built only if its HEAD tree
# is exactly pin + series and nothing is uncommitted (a hand-edited tree: export-patches.sh first, or start afresh).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
dir="${1:-$here/src}"

if [ ! -d "$dir/.git" ]; then
    mkdir -p "$dir"
    g init -q "$dir"
    fetch_pin "$dir"
    g -C "$dir" checkout -q -B bf16n "$LLAMACPP_COMMIT"
    g -C "$dir" am -q "${series[@]}"
fi
want="$(series_tree "$dir")"
have="$(g -C "$dir" rev-parse 'HEAD^{tree}')"
if [ "$have" != "$want" ] || [ -n "$(g -C "$dir" status --porcelain --untracked-files=no)" ]; then
    echo "build.sh: $dir is not $LLAMACPP_COMMIT + the series (HEAD tree $have, expected $want, or uncommitted changes)" >&2
    exit 1
fi

cd "$dir"
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBF16N_CORE_DIR="$root/core" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DLLAMA_BUILD_TESTS="${LLAMA_BUILD_TESTS:-OFF}" \
    -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j "${MAX_JOBS:-$(nproc)}" --target llama-server llama-bench llama-cli
echo "built: $PWD/build/bin"
