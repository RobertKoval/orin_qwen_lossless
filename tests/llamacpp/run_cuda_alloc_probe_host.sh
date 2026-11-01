#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Linux CPU-only external-ABI observer tests. No CUDA SDK/runtime/driver/GPU is used.
# Optional argument: build directory to retain artifacts; default is an automatically cleaned temporary directory.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    printf '%s\n' 'SKIP: allocation observer requires Linux LD_PRELOAD/RTLD_NEXT; macOS DYLD is a different loader contract.' >&2
    exit 77
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$#" -gt 1 ]]; then printf 'Usage: %s [build-dir]\n' "$0" >&2; exit 2; fi
if [[ "$#" == 1 ]]; then
    mkdir -p "$1"; BUILD=$(cd -- "$1" && pwd)
else
    BUILD=$(mktemp -d); trap 'rm -rf -- "$BUILD"' EXIT
fi
CXX=${CXX:-c++}
"$CXX" --version
common=(-std=c++17 -O1 -g -fno-omit-frame-pointer -Wall -Wextra -Wpedantic -pthread)
"$CXX" "${common[@]}" -shared -fPIC "$HERE/cuda_alloc_probe.cpp" -ldl -o "$BUILD/libs5_cuda_probe.so"
"$CXX" "${common[@]}" -shared -fPIC "$HERE/cuda_alloc_probe_fake.cpp" -ldl -o "$BUILD/libs5_fake_cuda.so"
for name in smoke host; do
    "$CXX" "${common[@]}" -UNDEBUG "$HERE/cuda_alloc_probe_$name.cpp" -L "$BUILD" -Wl,-rpath,"$BUILD" -ls5_fake_cuda -ldl -o "$BUILD/$name"
done
"$CXX" "${common[@]}" "$HERE/cuda_alloc_probe_missing.cpp" -ldl -o "$BUILD/missing-next"
run() { env LD_PRELOAD="$BUILD/libs5_cuda_probe.so" S5_CUDA_TRACE=0 S5_CUDA_STACKS=1 "$@"; }
expect_status() {
    local expected=$1 status; shift
    if "$@"; then status=0; else status=$?; fi
    if [[ "$status" != "$expected" ]]; then
        printf 'FAIL: negative control expected exit %s, got %s\n' "$expected" "$status" >&2; exit 1
    fi
}
# Negative controls must reject missing/bypassed interception, not interpret zero observations as reclamation.
expect_status 2 env -u LD_PRELOAD "$BUILD/smoke"
expect_status 126 run "$BUILD/missing-next"
expect_status 3 run "$BUILD/host" bypass
printf '%s\n' 'PASS: absent/missing-next/bypassed interception controls fail closed'
run "$BUILD/smoke"
for scenario in faults unknown threads generations zero-owner reentry; do run "$BUILD/host" "$scenario"; done
# Exercise trace-on allocation/free IDs and caller reporting, too.
env LD_PRELOAD="$BUILD/libs5_cuda_probe.so" S5_CUDA_TRACE=1 S5_CUDA_STACKS=0 "$BUILD/host" generations
printf '%s\n' 'PASS: Linux external-ABI lifecycle observer controls (fake CUDA only; no GPU)'
