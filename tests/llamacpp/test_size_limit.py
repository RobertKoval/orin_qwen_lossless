# SPDX-License-Identifier: MIT
"""Execute the native root cap parser at representable boundaries; no CUDA/GGML dependency."""
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import pytest

SIZE_MAX = (1 << (8 * struct.calcsize("P"))) - 1
KEYS = ("BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES")


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("native cap-parser test requires a host C++17 compiler")
    binary = tmp_path_factory.mktemp("native-cap") / "size_limit"
    subprocess.run(compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror", str(Path(__file__).with_name("size_limit_host.cpp")),
                               "-o", str(binary)], check=True, capture_output=True, text=True)
    maximum = subprocess.run([str(binary), "--max"], check=True, capture_output=True, text=True)
    assert int(maximum.stdout) == SIZE_MAX
    return binary


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value,expected", [(None, 1 << 30), ("1", 1), ("2", 2), ("0001", 1),
                                            (str(SIZE_MAX - 1), SIZE_MAX - 1), (str(SIZE_MAX), SIZE_MAX)])
def test_positive_size_boundaries_execute_native_parser(parser, key, value, expected):
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    if value is not None:
        env[key] = value
    result = subprocess.run([str(parser), key], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == expected


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", ["", "0", "00", "-1", " -1", "\t-1", "+1", " 1", "1 ", "1\n", "1x", "1_000",
                                    "0x10", "1e3", "١", str(SIZE_MAX + 1), str(SIZE_MAX + 2), "9" * 100])
def test_invalid_sizes_fail_native_parser(parser, key, value):
    result = subprocess.run([str(parser), key], env={**os.environ, key: value}, capture_output=True, text=True)
    assert result.returncode == 1
    assert key in result.stderr
