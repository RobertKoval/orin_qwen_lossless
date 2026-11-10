"""MTP drafter capture sizes (draft_graphs.py, T-084): passes 2..N run one token per running request, so every request
count up to max_num_seqs must find a drafter graph of its own power-of-two size instead of the target model's sizes,
which vLLM rounds up to multiples of k + 1; the target's sizes stay captured for the drafter's first pass."""
import pytest

from vllm_bf16n.draft_graphs import drafter_capture_sizes


def padded(sizes, n):
    return min(s for s in sizes if s >= n)


@pytest.mark.parametrize("shared, seqs, max_size, want", [
    ([4, 8], 1, 8, [1, 4, 8]),               # production: k = 3, one sequence
    ([4, 8, 16], 2, 16, [1, 2, 4, 8, 16]),   # two sequences
    ([4, 8, 16], 3, 16, [1, 2, 4, 8, 16]),   # 3 requests pad to 4, as vLLM pads them today
    ([4, 8, 16], 4, 16, [1, 2, 4, 8, 16]),
    ([4, 8, 16], 5, 16, [1, 2, 4, 8, 16]),
    ([2], 4, 2, [1, 2]),                     # never above the largest capture size
    ([4, 8], 2, 1, [1, 4, 8]),
])
def test_drafter_sizes(shared, seqs, max_size, want):
    got = drafter_capture_sizes(shared, seqs, max_size)
    assert got == want
    assert set(shared) <= set(got)                                   # the first pass keeps the target's graphs
    for n in range(1, min(seqs, max_size) + 1):                      # passes 2..N: n running requests, n rows
        assert padded(got, n) < 2 * n                                # at most the next power of two


@pytest.mark.parametrize("seqs, max_size", [(0, 8), (1, 0)])
def test_drafter_sizes_reject_empty_bounds(seqs, max_size):
    with pytest.raises(ValueError):
        drafter_capture_sizes([4, 8], seqs, max_size)
