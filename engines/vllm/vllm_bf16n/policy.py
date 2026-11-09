# SPDX-License-Identifier: MIT
"""vLLM shape policy, implemented by the shared host-only C++ core.

The core retains vLLM's wave-model tuning separately from llama.cpp's largest-R rule.
Import stays cheap; choosing a format loads the extension but never touches CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass

FMT_NIBBLE, FMT_HUFFMAN, FMT_HUFPAIR = 1, 2, 3


@dataclass(frozen=True)
class Choice:
    format: int
    R: int
    chunk_major: bool
    frag_order: bool = False


def choose(rows_per_matrix: int, K: int, n_matrices: int = 1) -> Choice | None:
    """Choose encoding for n_matrices of rows_per_matrix x K; None leaves the weights BF16."""
    import torch  # noqa: F401; load libtorch symbols before the native extension
    from . import _C

    c = _C.choose(rows_per_matrix, K, n_matrices)
    return None if c is None else Choice(*c)
