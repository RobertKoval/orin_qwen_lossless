// SPDX-License-Identifier: MIT
// CPU-only executable of the same parser used by dump_ops and reclamation.
#include "../../bench/xengine/size_limit.h"
#include <cstdio>
#include <limits>

int main(int argc, char ** argv) try {
    if (argc != 2) { return 2; }
    if (std::strcmp(argv[1], "--max") == 0) {
        std::printf("%zu\n", std::numeric_limits<std::size_t>::max());
    } else {
        std::printf("%zu\n", bf16n_test::memory_budget(argv[1], std::size_t(1) << 30));
    }
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
