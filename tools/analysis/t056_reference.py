"""PRIVATE T-056 CPU reference, inspired by arXiv:2610.38210v1 Eqs. (1)-(2).

NOT the GAAC implementation/wire format: group traversal, bit order, tags,
headers, calibration ties and slot policy below are explicit prototype choices.
All data paths operate on little-endian BF16 uint16 bits, never Python floats.
No torch/CUDA imports. This module is not a production attention backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

HEADER = struct.Struct("<4sBBHIIII")  # own 24-byte header, not specified by the paper
CAL_HEADER = struct.Struct("<4sIII")
TABLE_ENTRY_BYTES = 4
GROUP = 8
RAW, ADDITIVE = 0, 1
WIDTHS = (3, 4, 8)


def _integer(value, name, minimum=0, maximum=0xFFFFFFFF):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bits(data):
    if not isinstance(data, bytes) or len(data) % 2:
        raise ValueError("BF16 input must be bytes containing complete little-endian uint16 words")
    return struct.iter_unpack("<H", data)


def _pack(values, width):
    """Own LSB-first packing. Caller supplies only small integer bitfields."""
    out = bytearray()
    accumulator = occupied = 0
    for value in values:
        if type(value) is not int or not 0 <= value < (1 << width):
            raise ValueError("bitfield does not fit its declared width")
        accumulator |= value << occupied
        occupied += width
        while occupied >= 8:
            out.append(accumulator & 255)
            accumulator >>= 8
            occupied -= 8
    if occupied:
        out.append(accumulator)
    return bytes(out)


def _unpack(data, count, width):
    if len(data) != (count * width + 7) // 8:
        raise ValueError("packed bitfield length mismatch")
    accumulator = occupied = cursor = 0
    for _ in range(count):
        while occupied < width:
            accumulator |= data[cursor] << occupied
            occupied += 8
            cursor += 1
        yield accumulator & ((1 << width) - 1)
        accumulator >>= width
        occupied -= width
    if accumulator:
        raise ValueError("nonzero unused packed bits")


@dataclass(frozen=True)
class Calibration:
    """Two nonwrapping exponent windows per channel; shared across token blocks."""
    base3: tuple[int, ...]
    base4: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.base3, tuple) or not isinstance(self.base4, tuple):
            raise ValueError("calibration bases must be immutable tuples")
        if not self.base3 or len(self.base3) != len(self.base4) or len(self.base3) % GROUP:
            raise ValueError("channel count must be positive, equal for both bases, and divisible by eight")
        _integer(len(self.base3), "channels", 1)
        for value in self.base3:
            _integer(value, "3-bit base", maximum=248)
        for value in self.base4:
            _integer(value, "4-bit base", maximum=240)

    @property
    def channels(self):
        return len(self.base3)

    @classmethod
    def from_rows(cls, rows: Iterable[bytes], channels: int):
        _integer(channels, "channels", GROUP)
        if channels % GROUP:
            raise ValueError("channels must be divisible by eight")
        hist = [[0] * 256 for _ in range(channels)]
        count = 0
        for row in rows:
            if not isinstance(row, bytes) or len(row) != channels * 2:
                raise ValueError("calibration row shape mismatch")
            for channel, (word,) in enumerate(_bits(row)):
                hist[channel][(word >> 7) & 255] += 1
            count += 1
        if not count:
            raise ValueError("calibration requires real rows, not an empty sample")

        def bases(width):
            result = []
            for frequencies in hist:
                span = 1 << width
                covered = best = sum(frequencies[:span])
                chosen = 0
                for start in range(1, 257 - span):
                    covered += frequencies[start + span - 1] - frequencies[start - 1]
                    if covered > best:  # own deterministic tie policy: smallest base wins
                        best, chosen = covered, start
                result.append(chosen)
            return tuple(result)

        return cls(bases(3), bases(4))

    def to_bytes(self):
        return CAL_HEADER.pack(b"T56C", 1, self.channels, 0) + bytes(self.base3) + bytes(self.base4)

    @classmethod
    def from_bytes(cls, data):
        if not isinstance(data, bytes) or len(data) < CAL_HEADER.size:
            raise ValueError("truncated calibration")
        magic, version, channels, reserved = CAL_HEADER.unpack_from(data)
        if magic != b"T56C" or version != 1 or reserved or len(data) != CAL_HEADER.size + 2 * channels:
            raise ValueError("invalid calibration header/length")
        return cls(tuple(data[CAL_HEADER.size:CAL_HEADER.size + channels]), tuple(data[CAL_HEADER.size + channels:]))


@dataclass(frozen=True)
class StoredBlock:
    storage: bytes
    calibration: Calibration  # exact shared control identity, not a collision-prone per-block hash
    used_bytes: int
    raw_fallback: bool
    extra_bytes: int
    mode_counts: tuple[int, int, int]  # attempted groups; raw-block fallback discards their encoded streams
    attempted_bytes: int

    @property
    def slack_bytes(self):
        return len(self.storage) - self.used_bytes


class ReferenceCodec:
    """Own token-major/channel-major groups, 0/1/2 tags, separate streams.

    Slot capacity counts the WHOLE compressed record including our header.
    Default 3032 = inferred Appendix-D 3008-byte payload slot + our 24-byte header
    for a 16x128 tile. It is not an official calibrated GAAC capacity.
    S == C is compressed; S > C retains the complete raw uint16 block.
    """
    def __init__(self, calibration: Calibration, block_tokens=16, slot_capacity=3032):
        if not isinstance(calibration, Calibration):
            raise ValueError("a validated calibration is required")
        self.calibration = calibration
        self.block_tokens = _integer(block_tokens, "block_tokens", 1)
        self.slot_capacity = _integer(slot_capacity, "slot_capacity")
        self.elements = self.block_tokens * calibration.channels
        _integer(self.elements * 2, "raw block bytes", 1)
        self.raw_bytes = 2 * self.elements

    def encode_block(self, data: bytes):
        if not isinstance(data, bytes) or len(data) != self.raw_bytes:
            raise ValueError("only a complete block can be finalized")
        words = [word for (word,) in _bits(data)]
        sm = bytes(((word >> 8) & 128) | (word & 127) for word in words)
        tags, lows, extras = [], [], bytearray()
        counts = [0, 0, 0]
        channels = self.calibration.channels
        for first in range(0, self.elements, GROUP):
            exponents = [(word >> 7) & 255 for word in words[first:first + GROUP]]
            channel = first % channels
            offsets = [e - self.calibration.base3[channel + i] for i, e in enumerate(exponents)]
            tag = 0
            if not all(0 <= x < 8 for x in offsets):
                tag = 1
                offsets = [e - self.calibration.base4[channel + i] for i, e in enumerate(exponents)]
                if not all(0 <= x < 16 for x in offsets):
                    tag, offsets = 2, exponents  # whole group raw; no indexed per-value exceptions
            counts[tag] += 1
            tags.append(tag)
            lows.extend(value & 7 for value in offsets)
            if tag:
                extras.extend(_pack((value >> 3 for value in offsets), WIDTHS[tag] - 3))
        payload = sm + _pack(tags, 2) + _pack(lows, 3) + bytes(extras)
        attempted = HEADER.size + len(payload)
        if attempted <= self.slot_capacity:
            header = HEADER.pack(b"T056", 1, ADDITIVE, 0, self.block_tokens, channels, len(payload), len(extras))
            blob = header + payload
            storage = blob + bytes(self.slot_capacity - len(blob))
            return StoredBlock(storage, self.calibration, len(blob), False, len(extras), tuple(counts), attempted)
        header = HEADER.pack(b"T056", 1, RAW, 0, self.block_tokens, channels, len(data), 0)
        blob = header + data
        return StoredBlock(blob, self.calibration, len(blob), True, 0, tuple(counts), attempted)

    def decode_block(self, block: StoredBlock):
        if not isinstance(block, StoredBlock) or block.calibration != self.calibration:
            raise ValueError("block belongs to a different calibration")
        data = block.storage
        if not isinstance(data, bytes) or len(data) < HEADER.size:
            raise ValueError("truncated block header")
        magic, version, mode, flags, tokens, channels, payload_len, extra_len = HEADER.unpack_from(data)
        if magic != b"T056" or version != 1 or flags or tokens != self.block_tokens or channels != self.calibration.channels:
            raise ValueError("invalid block identity/shape")
        used = HEADER.size + payload_len
        if used != block.used_bytes or used > len(data) or any(data[used:]):
            raise ValueError("invalid block length or nonzero slot slack")
        payload = data[HEADER.size:used]
        if mode == RAW:
            if not block.raw_fallback or payload_len != self.raw_bytes or extra_len or len(data) != used:
                raise ValueError("invalid whole-block raw fallback")
            return payload
        if mode != ADDITIVE or block.raw_fallback or len(data) != self.slot_capacity:
            raise ValueError("invalid block storage mode/capacity")
        groups = self.elements // GROUP
        tag_bytes = (groups * 2 + 7) // 8
        low_bytes = self.elements * 3 // 8
        fixed = self.elements + tag_bytes + low_bytes
        if payload_len != fixed + extra_len or extra_len != block.extra_bytes:
            raise ValueError("invalid additive stream lengths")
        sm = payload[:self.elements]
        tags = list(_unpack(payload[self.elements:self.elements + tag_bytes], groups, 2))
        lows = list(_unpack(payload[self.elements + tag_bytes:fixed], self.elements, 3))
        extras = payload[fixed:]
        out = bytearray(self.raw_bytes)
        cursor = 0
        for group, tag in enumerate(tags):
            if tag not in (0, 1, 2):
                raise ValueError("reserved group mode")
            high = [0] * GROUP
            if tag:
                length = WIDTHS[tag] - 3  # eight values -> length bytes
                if cursor + length > len(extras):
                    raise ValueError("truncated wider-group stream")
                high = list(_unpack(extras[cursor:cursor + length], GROUP, length))
                cursor += length
            for i in range(GROUP):
                index = group * GROUP + i
                channel = index % channels
                code = lows[index] | (high[i] << 3)
                base = self.calibration.base3[channel] if tag == 0 else self.calibration.base4[channel] if tag == 1 else 0
                exponent = base + code
                if not 0 <= exponent <= 255:
                    raise ValueError("exponent overflow; never wrap or clamp")
                word = ((sm[index] & 128) << 8) | (exponent << 7) | (sm[index] & 127)
                struct.pack_into("<H", out, 2 * index, word)
        if cursor != len(extras):
            raise ValueError("unused wider-group bytes")
        return bytes(out)


class IncrementalCache:
    """CPU semantic model only: committed appends, immutable full blocks, raw tail.

    truncate/fork decode at most one boundary block, never the whole history.
    This does not implement the vLLM speculative-commit or GDN state protocol.
    """
    def __init__(self, codec: ReferenceCodec):
        self.codec = codec
        self.blocks: list[StoredBlock] = []
        self._tail = bytearray()
        self._tail_tokens = 0

    @property
    def token_count(self):
        return len(self.blocks) * self.codec.block_tokens + self._tail_tokens

    @property
    def tail(self):
        return bytes(self._tail[:self._tail_tokens * self.codec.calibration.channels * 2])

    def append(self, row: bytes):
        size = self.codec.calibration.channels * 2
        if not isinstance(row, bytes) or len(row) != size:
            raise ValueError("append requires one BF16 uint16 row")
        if not self._tail:
            self._tail = bytearray(self.codec.raw_bytes)
        start = self._tail_tokens * size
        self._tail[start:start + size] = row
        self._tail_tokens += 1
        if self._tail_tokens == self.codec.block_tokens:
            block = self.codec.encode_block(bytes(self._tail))
            self.blocks.append(block)
            self._tail = bytearray()
            self._tail_tokens = 0

    def truncate(self, tokens: int):
        _integer(tokens, "committed token count", maximum=self.token_count)
        full, partial = divmod(tokens, self.codec.block_tokens)
        boundary = self.codec.decode_block(self.blocks[full]) if partial and full < len(self.blocks) else self.tail
        self.blocks = self.blocks[:full]
        self._tail = bytearray(self.codec.raw_bytes) if partial else bytearray()
        self._tail_tokens = partial
        if partial:
            size = partial * self.codec.calibration.channels * 2
            self._tail[:size] = boundary[:size]

    def fork_prefix(self, tokens: int):
        _integer(tokens, "prefix token count", maximum=self.token_count)
        result = IncrementalCache(self.codec)
        result.blocks = self.blocks.copy()  # immutable block objects may be shared
        result._tail = self._tail.copy()
        result._tail_tokens = self._tail_tokens
        result.truncate(tokens)
        return result

    def read(self):
        """Small CPU test convenience only; NEVER the planned GPU attention path."""
        return b"".join(self.codec.decode_block(block) for block in self.blocks) + self.tail

    def accounting(self):
        """Format-byte accounting, NOT Python RSS or a claim about CUDA allocation."""
        tail_allocated = len(self._tail)
        entries = len(self.blocks) + bool(self._tail_tokens)
        return {"raw_value_bytes": self.token_count * self.codec.calibration.channels * 2,
                "stored_used_bytes": sum(b.used_bytes for b in self.blocks) + len(self.tail),
                "allocated_format_bytes": sum(len(b.storage) for b in self.blocks) + tail_allocated
                                          + entries * TABLE_ENTRY_BYTES + len(self.codec.calibration.to_bytes()),
                "slot_slack_bytes": sum(b.slack_bytes for b in self.blocks),
                "tail_slack_bytes": tail_allocated - len(self.tail),
                "raw_fallback_blocks": sum(b.raw_fallback for b in self.blocks),
                "header_bytes": len(self.blocks) * HEADER.size,
                "table_bytes": entries * TABLE_ENTRY_BYTES,
                "base_table_bytes": len(self.codec.calibration.to_bytes())}
