// SPDX-License-Identifier: MIT
// Negative control: no fake/real CUDA runtime is linked. A loaded probe must fail resolving RTLD_NEXT.
#include <cstddef>
#include <dlfcn.h>
int main() {
    auto alloc = reinterpret_cast<int (*)(void **, size_t)>(dlsym(RTLD_DEFAULT, "cudaMalloc"));
    if (!alloc) return 2;
    void * p = nullptr;
    return alloc(&p, 1); // Interposer exits 126 when no next implementation exists; no CUDA/device is involved.
}
