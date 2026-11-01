"""CPU-only tests of the exact numeric gate and isolated subprocess orchestration."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.xengine.verify_ops import compare_dumps  # noqa: E402


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (-0.0, -0.0), (0.0, -0.0), (1.0, 1.0000001192092896)])
def test_float_bits_not_tolerance(a, b, tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    left.write_bytes(struct.pack("<f", a))
    right.write_bytes(struct.pack("<f", b))
    if left.read_bytes() == right.read_bytes():
        assert len(compare_dumps(left, right)) == 64
    else:
        with pytest.raises(ValueError, match=r"F32\[0\]"):
            compare_dumps(left, right)


@pytest.mark.parametrize("size", [0, 3, 4, 5])
def test_raw_f32_size_boundary(size, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"\0" * size)
    b.write_bytes(a.read_bytes())
    if size == 4:
        assert compare_dumps(a, b)
    else:
        with pytest.raises(ValueError, match="sizes"):
            compare_dumps(a, b)


def test_extra_output_and_late_mismatch_fail(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"\0" * (1024 * 1024 + 4))
    b.write_bytes(a.read_bytes() + b"\0" * 4)
    with pytest.raises(ValueError, match="sizes"):
        compare_dumps(a, b)
    b.write_bytes(a.read_bytes()[:-4] + struct.pack("<f", 1.0))
    with pytest.raises(ValueError, match=r"F32\[262144\]"):
        compare_dumps(a, b)


@pytest.mark.parametrize("mode", ["equal", "mismatch", "failed_process", "wrong_metadata"])
def test_orchestrator_old_old_and_failure_receipts(mode, tmp_path):
    # A stand-in compiler/runner exercises the actual CLI, subprocesses, output files,
    # fail-closed metadata and receipts, without loading GGUF/CUDA or building llama.
    source, build = tmp_path / "source", tmp_path / "build"
    (source / "ggml/include").mkdir(parents=True)
    (source / "ggml/include/ggml.h").touch()
    (build / "bin").mkdir(parents=True)
    model = tmp_path / "model"
    model.write_bytes(b"test-double model")
    runner = tmp_path / "runner.py"
    runner.write_text('''#!/usr/bin/env python3
import json, os, pathlib, struct, sys
mode = os.environ['DOUBLE_MODE']
new = pathlib.Path(sys.argv[0]).name.startswith('new')
if new and mode == 'failed_process': sys.exit(7)
value = 1.0000001192092896 if new and mode == 'mismatch' else 1.0
pathlib.Path(sys.argv[7]).write_bytes(struct.pack('<f', value))
print(json.dumps({'shape':[1,1,1,1], 'bytes':8 if mode == 'wrong_metadata' else 4}))
''')
    compiler = tmp_path / "compiler.py"
    compiler.write_text('''#!/usr/bin/env python3
import os, pathlib, shutil, sys
out = pathlib.Path(sys.argv[sys.argv.index('-o')+1])
shutil.copyfile(os.environ['DOUBLE_RUNNER'], out)
out.chmod(0o755)
''')
    out = tmp_path / "results"
    command = [sys.executable, str(ROOT / "bench/xengine/verify_ops.py"),
               "--old-source", str(source), "--new-source", str(source), "--old-build", str(build),
               "--new-build", str(build), "--model", str(model), "--out", str(out),
               "--cxx", f"{sys.executable} {compiler}", "--case", r"^output\.weight-mul_mat-t1-"]
    env = dict(os.environ, DOUBLE_MODE=mode, DOUBLE_RUNNER=str(runner))
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    receipt = json.loads((out / "manifest.json").read_text())
    assert receipt["partial"] is True
    assert receipt["passed"] is (mode == "equal")
    assert result.returncode == (0 if mode == "equal" else 1)
    if mode == "equal":
        assert "PARTIAL PASS" in result.stdout and len(receipt["cases"]) == 1
        # A stale directory must fail rather than reusing previous passing output.
        repeat = subprocess.run(command, env=env, capture_output=True, text=True)
        assert repeat.returncode == 1
    else:
        assert "FAIL:" in result.stderr
