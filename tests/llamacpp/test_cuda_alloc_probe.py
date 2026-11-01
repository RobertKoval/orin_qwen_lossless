# SPDX-License-Identifier: MIT
"""Linux CPU-only preload/RTLD_NEXT lifecycle tests; no CUDA SDK, GPU, torch, or engine build.

Run: python -m pytest --noconftest tests/llamacpp/test_cuda_alloc_probe.py -q -rs
Or on Linux without pytest: bash tests/llamacpp/run_cuda_alloc_probe_host.sh
Portable CPU container (official GCC image; no GPU passthrough):
  docker run --rm --network none -v "$PWD:/repo:ro" -w /repo gcc:12 \
    bash tests/llamacpp/run_cuda_alloc_probe_host.sh
The C++ controls assert external allocation/ownership effects, not implementation/source strings.
"""
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="allocation observer requires Linux LD_PRELOAD/RTLD_NEXT; macOS DYLD is a different loader contract",
)


def test_external_cuda_allocation_lifecycles(tmp_path):
    runner = Path(__file__).with_name("run_cuda_alloc_probe_host.sh")
    result = subprocess.run(
        ["bash", str(runner), str(tmp_path / "probe-build")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
