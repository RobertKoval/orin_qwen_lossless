# SPDX-License-Identifier: MIT
# Engine-independent BF16N core: bf16n_host (encoder, no CUDA) and bf16n_cuda (launchers + kernels, SM87), host tests.
#   cmake -S core -B build/core -DBF16N_CUDA=OFF && cmake --build build/core && ctest --test-dir build/core
# The vLLM extension (setup.py) and llama.cpp (patch series) compile these sources with their own build systems.
cmake_minimum_required(VERSION 3.18)
project(bf16n_core LANGUAGES CXX)

option(BF16N_CUDA "build bf16n_cuda (needs nvcc; Jetson AGX Orin = SM87)" ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

add_library(bf16n_host STATIC src/encode.cpp src/policy.cpp)
target_include_directories(bf16n_host PUBLIC include)
target_link_libraries(bf16n_host PUBLIC Threads::Threads)

if(BF16N_CUDA)
    # before enable_language: it would otherwise default the architectures to nvcc's (sm_52, no FP16 arithmetic)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 87)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_library(bf16n_cuda STATIC src/launch.cu src/mmid_helper.cu)
    # same device flags as setup.py and ggml-cuda
    target_compile_options(bf16n_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -lineinfo>)
    target_link_libraries(bf16n_cuda PUBLIC bf16n_host)
endif()

enable_testing()
add_executable(test_encode tests/test_encode.cpp)
target_link_libraries(test_encode PRIVATE bf16n_host)
add_test(NAME encode COMMAND test_encode)
add_executable(test_policy tests/test_policy.cpp)
target_link_libraries(test_policy PRIVATE bf16n_host)
add_test(NAME policy COMMAND test_policy)
