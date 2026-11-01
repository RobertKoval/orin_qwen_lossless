#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit Linux/CUDA test entry point: exact numeric cases, then externally observed reclamation.

Compilation writes only to --out, never into the selected llama source/build. LD_PRELOAD is added only to
reclamation's environment, never the numerical test process. --build-only executes no GPU binary.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIZE_MAX = (1 << (8 * struct.calcsize("P"))) - 1


def budget_environment():
    env = dict(os.environ)
    for name in ("BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES"):
        if name not in env:
            continue
        value = env[name]
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"{name}: expected positive decimal bytes fitting size_t")
        parsed = int(value, 10)
        if not 1 <= parsed <= SIZE_MAX:
            raise ValueError(f"{name}: expected positive decimal bytes fitting size_t")
        env[name] = str(parsed)  # Canonical ASCII decimal in every child, without changing the caller's environment.
    return env


def run(args):
    env = budget_environment()  # Reject before build-directory creation, compiler invocation or GPU initialization.
    source = args.source.resolve(strict=True)
    build = args.build.resolve(strict=True)
    out = args.out.resolve()
    if out.is_relative_to(source) or out.is_relative_to(build):
        raise ValueError("--out must be outside the selected llama source/build")
    match = re.fullmatch(r"CUDA(\d+)", args.backend)
    if not match:
        raise ValueError("BF16N reclamation requires a CUDA<n> backend")
    device = int(match.group(1))
    if device > 65535:
        raise ValueError("invalid CUDA device ordinal")
    include = source / "ggml/include"
    lib = build / "bin"
    if not (include / "ggml.h").is_file() or not lib.is_dir():
        raise ValueError("expected source/ggml/include and build/bin")
    compiler = shlex.split(args.cxx)
    if not compiler:
        raise ValueError("empty C++ compiler command")
    out.mkdir(parents=True, exist_ok=True)
    if args.observer_library:
        observer = args.observer_library.resolve(strict=True)
    else:
        observer = out / "libcuda_alloc_probe.so"
        subprocess.run(compiler + ["-std=c++17", "-O1", "-g", "-fno-omit-frame-pointer", "-Wall", "-Wextra",
                                   "-Wpedantic", "-fPIC", "-shared", "-pthread", str(HERE / "cuda_alloc_probe.cpp"),
                                   "-ldl", "-o", str(observer)], check=True, env=env)
    if any(c.isspace() or c == ":" for c in str(observer)):
        raise ValueError("observer path cannot contain LD_PRELOAD separators (whitespace/colon)")
    helper = out / "reclamation"
    subprocess.run(compiler + ["-O2", "-g", "-std=c++17", "-Wall", "-Wextra", "-I", str(include),
                               str(HERE / "reclamation.cpp"), "-o", str(helper), "-L", str(lib),
                               "-lggml", "-lggml-base", "-lggml-cuda", "-ldl", f"-Wl,-rpath,{lib}"], check=True, env=env)
    if args.build_only:
        print(f"BUILT ONLY (no GPU execution): {helper}; observer={observer}")
        return
    env["LD_LIBRARY_PATH"] = str(lib) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if not args.reclamation_only:
        subprocess.run([str(lib / "test-backend-ops"), "bf16n", "-b", args.backend], env=env, check=True)
    # Never inject the test observer into production or the numeric binary.
    observed_env = {**env, "LD_PRELOAD": str(observer) + (" " + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")}
    subprocess.run([str(helper), str(device)], env=observed_env, check=True)
    print("PASS: " + ("reclamation-only control" if args.reclamation_only else "numeric BF16N + exact generation reclamation"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "build/tests/llamacpp")
    p.add_argument("--backend", default="CUDA0")
    p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    p.add_argument("--observer-library", type=Path, help="use an already-built test observer instead of compiling it")
    p.add_argument("--build-only", action="store_true", help="compile only; execute neither GPU test binary")
    p.add_argument("--reclamation-only", action="store_true", help="reclamation control only, NOT the complete gate")
    args = p.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"test-llamacpp: FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
