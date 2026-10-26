#!/usr/bin/env bash
# Fetch llama.cpp at the tested commit, apply the BF16N patch and build the CUDA binaries for Jetson AGX Orin (SM87).
#   scripts/build_llama.sh [workdir]        (default: ./llama.cpp)
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
dir="${1:-llama.cpp}"
commit=b29c606e28a01b1bc8c1351026a0fa6e616bf6c4

if [ ! -d "$dir/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$dir"
fi
cd "$dir"
git checkout --quiet "$commit"
if git apply --check "$here/patches/llama.cpp-b29c606-bf16n.patch" 2>/dev/null; then
    git apply "$here/patches/llama.cpp-b29c606-bf16n.patch"
else
    git apply --reverse --check "$here/patches/llama.cpp-b29c606-bf16n.patch" # already applied, else fail loudly
fi

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j "$(nproc)" --target llama-server llama-bench llama-cli
echo "built: $PWD/build/bin"
