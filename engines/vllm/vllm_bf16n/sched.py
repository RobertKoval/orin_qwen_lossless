"""CUDA host-wait scheduling for the vLLM worker (T-033).

On Orin, how the host thread waits changes GPU kernel speed: under the default context flags cudaDeviceSynchronize
blocks and long bandwidth-bound kernels run up to ~35 % slower meanwhile (lm_head 4.22 -> 5.6 ms); spin / yield
scheduling keeps them at full speed. The flags of the primary context can only be set before it is created
(cuDevicePrimaryCtxSetFlags returns CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE afterwards), so this runs from the plugin entry
point, which vLLM calls in every process before the worker initialises CUDA.

Opt-in by environment (inert for everyone else): BF16N_CUDA_SCHED = spin | yield | blocking | auto. The process must
not fork after this (cuInit is not fork-safe): launchers set VLLM_WORKER_MULTIPROC_METHOD=spawn.
"""
from __future__ import annotations

import ctypes
import os

FLAGS = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708


def _cuda():
    return ctypes.CDLL("libcuda.so.1")


def primary_ctx_state(device: int = 0) -> tuple[int, bool]:
    """(flags, active) of the device's primary context."""
    cu = _cuda()
    if cu.cuInit(0) != 0:
        raise RuntimeError("bf16n: cuInit failed")
    flags, active = ctypes.c_uint(), ctypes.c_int()
    if cu.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)) != 0:
        raise RuntimeError("bf16n: cuDevicePrimaryCtxGetState failed")
    return flags.value, bool(active.value)


def apply_from_env(device: int = 0) -> str | None:
    """Set the primary context's scheduling flags from BF16N_CUDA_SCHED; returns a status line, None if not requested.
    Raises on an unknown value. If the context is already active, leaves it alone and says so (the caller logs it)."""
    mode = os.environ.get("BF16N_CUDA_SCHED")
    if not mode:
        return None
    if mode not in FLAGS:
        raise ValueError(f"BF16N_CUDA_SCHED must be one of {sorted(FLAGS)}, got {mode!r}")
    want = FLAGS[mode]
    flags, active = primary_ctx_state(device)
    if (flags & 7) == want:
        return f"bf16n: CUDA scheduling already {mode} (pid {os.getpid()})"
    if active:
        return f"bf16n: CUDA primary context already active with flags {flags:#x}; cannot set {mode} (pid {os.getpid()})"
    rc = _cuda().cuDevicePrimaryCtxSetFlags(device, want)
    if rc != 0:
        raise RuntimeError(f"bf16n: cuDevicePrimaryCtxSetFlags({mode}) failed with {rc}")
    return f"bf16n: CUDA scheduling set to {mode} (pid {os.getpid()})"
