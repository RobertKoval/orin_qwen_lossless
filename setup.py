# SPDX-License-Identifier: MIT
# vllm-bf16n: BF16N lossless BF16 weights for vLLM (torch extension vllm_bf16n._C + quantization plugin).
# Build on Orin (CUDA 12.6, gcc-11, torch 2.11); see docs/vllm/README.md:
#   BF16N_LOCAL_VERSION=<sha12> TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4 pip wheel . --no-build-isolation --no-deps --no-index
# The kernels are compiled with -use_fast_math, like ggml-cuda, so they are the same instructions as in llama.cpp.
import os
import pathlib
import shutil

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = pathlib.Path(__file__).resolve().parent
# The bytes of these sources define the BF16N encoding and are part of vllm_bf16n.cache_key(). They are copied into
# the package so that an installed wheel computes the key of the sources its _C was built from.
KEY_SOURCES = (  # = vllm_bf16n._KEY_FILES, in the same hash order
    "core/include/bf16n/format.h", "core/src/encode.cpp",
    "core/include/bf16n/policy.h", "core/src/policy.cpp",
)
keysrc = ROOT / "engines" / "vllm" / "vllm_bf16n" / "_keysrc"
keysrc.mkdir(exist_ok=True)
for s in KEY_SOURCES:
    shutil.copyfile(ROOT / s, keysrc / pathlib.Path(s).name)
local = os.environ.get("BF16N_LOCAL_VERSION", "")

setup(
    name="vllm-bf16n",
    version="0.1.0" + (f"+g{local}" if local else ""),
    description="BF16N lossless compressed BF16 weights for vLLM",
    license="MIT",
    package_dir={"": "engines/vllm"},
    packages=find_packages(where="engines/vllm", include=["vllm_bf16n", "vllm_bf16n.*"]),
    package_data={"vllm_bf16n": ["_keysrc/*"]},
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            name="vllm_bf16n._C",
            sources=[
                "core/torch/torch_bindings.cpp",
                "core/src/encode.cpp",
                "core/src/policy.cpp",
                "core/src/launch.cu",
                "core/src/mmid_helper.cu",
            ],
            include_dirs=[str(ROOT / "core" / "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-use_fast_math", "-lineinfo",
                         "--generate-code=arch=compute_87,code=sm_87"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    entry_points={"vllm.general_plugins": ["bf16n = vllm_bf16n:register"]},
)
