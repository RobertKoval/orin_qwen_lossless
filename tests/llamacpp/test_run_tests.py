"""Public subprocess effects of the llama test entry point; no CUDA/model/compiler execution."""
import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("llama_test_entry", Path(__file__).with_name("run_tests.py"))
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
SIZE_MAX = ctypes.c_size_t(-1).value


@pytest.fixture
def command(tmp_path):
    source, build = tmp_path / "source", tmp_path / "selected-build"
    (source / "ggml/include").mkdir(parents=True)
    (source / "ggml/include/ggml.h").touch()
    (build / "bin").mkdir(parents=True)
    observer = tmp_path / "existing-probe.so"
    observer.touch()
    return ["--source", str(source), "--build", str(build), "--out", str(tmp_path / "test-build"),
            "--observer-library", str(observer), "--backend", "CUDA2"]


@pytest.mark.parametrize("failure", [None, "compile", "numeric", "reclamation"])
def test_numeric_then_helper_with_test_only_preload_and_failures(monkeypatch, command, failure):
    effects = []
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setenv("GGML_BF16N_NO_MULTI_COL", "1")

    def child(argv, *, check, env=None):
        name = Path(argv[0]).name
        kind = "compile" if "-o" in argv else "numeric" if name == "test-backend-ops" else "reclamation"
        effects.append((kind, env, argv))
        if kind == failure:
            raise subprocess.CalledProcessError(7, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ENTRY.subprocess, "run", child)
    assert ENTRY.main(command) == (1 if failure else 0)
    expected = ["compile", "numeric", "reclamation"]
    if failure:
        expected = expected[:expected.index(failure) + 1]
    assert [e[0] for e in effects] == expected
    for kind, env, argv in effects:
        if kind == "numeric":
            assert "LD_PRELOAD" not in env
            assert env["GGML_BF16N_NO_MULTI_COL"] == "1"
        if kind == "reclamation":
            assert env["LD_PRELOAD"].endswith("existing-probe.so")
            assert env["GGML_BF16N_NO_MULTI_COL"] == "1"
            assert argv[1:] == ["2"]


def test_build_only_executes_no_gpu_binary(monkeypatch, command):
    effects = []

    def child(argv, **kwargs):
        effects.append(argv)
        assert "-o" in argv  # The only permitted effect is compilation, never either GPU executable.
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ENTRY.subprocess, "run", child)
    assert ENTRY.main(command + ["--build-only"]) == 0
    assert len(effects) == 1


def test_old_library_control_does_not_run_missing_numeric_mode(monkeypatch, command):
    effects = []

    def child(argv, **kwargs):
        effects.append(argv)
        assert Path(argv[0]).name != "test-backend-ops"
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ENTRY.subprocess, "run", child)
    assert ENTRY.main(command + ["--reclamation-only"]) == 0
    assert len(effects) == 2


@pytest.mark.parametrize("target", ["source", "selected-build"])
def test_no_writes_inside_selected_source_or_build(monkeypatch, command, target):
    def forbidden(*args, **kwargs):
        pytest.fail("compiler or GPU binary must not run for unsafe output placement")
    monkeypatch.setattr(ENTRY.subprocess, "run", forbidden)
    at = command.index("--out") + 1
    base = Path(command[at]).parent
    command[at] = str(base / target / "would-mutate")
    assert ENTRY.main(command) == 1
    assert not Path(command[at]).exists()


def test_build_only_compiles_both_observer_and_helper_without_launching_them(monkeypatch, command):
    at = command.index("--observer-library")
    del command[at:at + 2]
    effects = []

    def child(argv, **kwargs):
        effects.append(argv)
        assert "-o" in argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ENTRY.subprocess, "run", child)
    assert ENTRY.main(command + ["--build-only"]) == 0
    assert len(effects) == 2


@pytest.mark.parametrize("key", ["BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES"])
@pytest.mark.parametrize("value", ["", "0", "00", "-1", " -1", "\t-1", "+1", " 1", "1 ", "1\n", "1x", "1_000",
                                    "0x10", "1e3", "١", str(SIZE_MAX + 1), str(SIZE_MAX + 2), "9" * 100])
def test_invalid_limit_fails_before_any_build_or_gpu_effect(monkeypatch, command, capsys, key, value):
    for name in ("BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid resource limit must not launch a compiler or GPU binary")

    monkeypatch.setattr(ENTRY.subprocess, "run", forbidden)
    assert ENTRY.main(command) == 1
    assert key in capsys.readouterr().err
    assert not Path(command[command.index("--out") + 1]).exists()


@pytest.mark.parametrize("key", ["BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES"])
@pytest.mark.parametrize("value", ["1", "2", "0001", str(SIZE_MAX - 1), str(SIZE_MAX), "000" + str(SIZE_MAX)])
def test_valid_limit_is_canonical_in_all_child_environments(monkeypatch, command, key, value):
    for name in ("BF16N_TEST_MAX_DEVICE_BYTES", "BF16N_TEST_MAX_HOST_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)
    effects = []

    def child(argv, *, env, **kwargs):
        effects.append(argv)
        assert env[key] == str(int(value, 10))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ENTRY.subprocess, "run", child)
    assert ENTRY.main(command) == 0
    assert len(effects) == 3  # compilation, numeric binary, observed reclamation binary
    assert os.environ[key] == value  # The wrapper does not mutate the invoking session's environment.
