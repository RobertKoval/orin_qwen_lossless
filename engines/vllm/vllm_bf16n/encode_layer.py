"""Encode one loaded BF16 matrix into BF16N on its device: host encode (all cores) -> upload -> bitwise GPU round-trip
check (abort on mismatch, like llama.cpp) -> one log line."""
from __future__ import annotations

import torch
from vllm.logger import init_logger

from . import policy

logger = init_logger("vllm.bf16n")  # under vLLM's configured logger hierarchy
_first = True


class _Arena:
    """Bump allocator for the small encoded tensors (< SMALL bytes) in 256 MiB device slabs. Through the caching
    allocator every 1-10 MB tensor gets its own 20 MB segment, and the unused rest stays reserved (measured M2: 3 GiB
    reserved above the 43.4 GiB allocated after loading). Views keep their slab alive."""

    SLAB = 256 << 20
    SMALL = 16 << 20
    ALIGN = 256

    def __init__(self):
        self.slab = None
        self.used = 0

    def take(self, src: torch.Tensor, device) -> torch.Tensor:
        n = src.numel()
        if n >= self.SMALL:
            return src.to(device)
        if self.slab is None or self.slab.device != torch.device(device) or self.used + n > self.SLAB:
            self.slab = torch.empty(self.SLAB, dtype=torch.uint8, device=device)
            self.used = 0
        out = self.slab[self.used : self.used + n]
        out.copy_(src.reshape(-1))
        self.used += -(-n // self.ALIGN) * self.ALIGN
        return out.view(src.shape)


_arena = _Arena()
_pool = None


def _weights_pool():
    """A separate caching-allocator pool for the encoded weights. vLLM materialises each layer's BF16 staging from the
    default pool, often by splitting a block cached from the previous layer's staging; an encoded tensor allocated
    from the same pool can then land in the rest of that split block, and once the staging is freed the block can
    never be returned (M5: torch reserved stayed 2.8 GiB above allocated after loading, even after empty_cache)."""
    global _pool
    if _pool is None:
        _pool = torch.cuda.MemPool()
    return _pool


def encode_matrix(name: str, w: torch.Tensor, choice: policy.Choice):
    """w: (..., K) BF16 on the target device. Returns ops.Weight; raises RuntimeError on a round-trip mismatch."""
    from . import ops

    global _first
    if _first:
        _first = False
        from .sched import primary_ctx_state

        flags, _ = primary_ctx_state(w.device.index or 0)
        logger.info("bf16n: worker pid %d, CUDA primary context scheduling flags %#x (0 auto, 1 spin, 2 yield, 4 blocking)",
                    __import__("os").getpid(), flags & 7)
    ops.init()
    # vLLM drops the previous layer's BF16 staging only after its processing returns (the layerwise loader holds the
    # params until info.reset()), so it is still cached here: release it before allocating encoded tensors, or they
    # are carved out of those blocks and the remainders stay reserved for good (M2: 3-4 GiB above allocated)
    torch.cuda.empty_cache()
    device = w.device
    host = w.detach().to("cpu", copy=True).contiguous()
    plane, aux, meta = ops.encode(host, choice.format, choice.R, choice.chunk_major, choice.frag_order)
    with torch.cuda.use_mem_pool(_weights_pool()):
        enc = ops.Weight(_arena.take(plane, device), _arena.take(aux, device), meta)
    ok = ops.roundtrip_equal(enc, w)
    ratio = enc.nbytes / (2 * host.numel())
    logger.info("bf16n: %-40s %-7s K=%5d rows=%7d R=%2d ratio=%.4f roundtrip=%s", name,
                ops.FORMAT_NAMES[meta.format], meta.K, meta.rows, meta.R, ratio, "OK" if ok else "FAIL")
    if not ok:
        raise RuntimeError(f"BF16N round-trip mismatch for {name}")
    return enc


def attach(layer: torch.nn.Module, prefix: str, enc) -> None:
    """Register plane/aux as buffers (so vLLM's memory accounting and .to() see them) and keep the metadata."""
    layer.register_buffer(f"{prefix}_plane", enc.plane, persistent=False)
    layer.register_buffer(f"{prefix}_aux", enc.aux, persistent=False)
    setattr(layer, f"{prefix}_meta", enc.meta)


def weight_of(layer: torch.nn.Module, prefix: str):
    from . import ops

    return ops.Weight(getattr(layer, f"{prefix}_plane"), getattr(layer, f"{prefix}_aux"), getattr(layer, f"{prefix}_meta"))


_released = False


def release_after_loading() -> None:
    """Called from the methods' process_weights_after_loading when vLLM's final pass finds a layer already encoded:
    by then every layer's BF16 staging is dropped, so hand the cached blocks back once and log where memory stands."""
    global _released
    if _released or not torch.cuda.is_available():
        return
    _released = True
    torch.cuda.empty_cache()
    import psutil

    free, total = torch.cuda.mem_get_info()
    logger.info("bf16n: after loading: torch allocated %.2f GiB, reserved %.2f GiB, process RSS %.2f GiB, "
                "device free %.2f of %.2f GiB", torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30, psutil.Process().memory_info().rss / 2**30, free / 2**30,
                total / 2**30)
