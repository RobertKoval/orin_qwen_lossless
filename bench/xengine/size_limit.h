// SPDX-License-Identifier: MIT
// Root test harnesses only. The standalone engine patch keeps its own small boundary parser.
#pragma once
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace bf16n_test {
inline std::size_t memory_budget(const char * name, std::size_t fallback) {
    const char * value = std::getenv(name);
    if (!value) { return fallback; }
    const char * end = value + std::strlen(value);
    std::size_t parsed = 0;
    const auto result = std::from_chars(value, end, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != end || parsed == 0) {
        throw std::invalid_argument(std::string(name) + ": expected positive decimal bytes fitting size_t");
    }
    return parsed;
}
} // namespace bf16n_test
