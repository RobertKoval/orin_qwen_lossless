"""Behavioral CPU tests of the T056 reference; no torch/CUDA or source-string tests."""
from dataclasses import replace
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/analysis"))
from t056_reference import Calibration, HEADER, IncrementalCache, ReferenceCodec


def words(exponents, mantissas=None, signs=None):
    mantissas = mantissas or [0] * len(exponents)
    signs = signs or [0] * len(exponents)
    return b"".join(struct.pack("<H", (s << 15) | (e << 7) | m) for e, m, s in zip(exponents, mantissas, signs))


def calibration(channels=8, narrow=120, wide=116):
    return Calibration((narrow,) * channels, (wide,) * channels)


class ReferenceTests(unittest.TestCase):
    def test_every_bf16_bit_pattern_through_actual_additive_payload(self):
        cal = calibration(256, 0, 0)
        codec = ReferenceCodec(cal, block_tokens=256, slot_capacity=140000)
        original = struct.pack("<65536H", *range(65536))
        block = codec.encode_block(original)
        self.assertFalse(block.raw_fallback)  # exercise payload, not a raw-copy tautology
        self.assertEqual(block.mode_counts, (256, 256, 7680))
        self.assertEqual(codec.decode_block(block), original)

    def test_special_values_keep_sign_and_nan_payloads_in_narrow_and_raw_groups(self):
        values = [0x0000, 0x8000, 0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80,
                  0x7F81, 0x7FC0, 0x7FFF, 0xFF81, 0xFFC0, 0xFFFF, 0x0080, 0x8080]
        raw = struct.pack("<16H", *values)
        for cap in (1000, 0):
            with self.subTest(cap=cap):
                codec = ReferenceCodec(calibration(8, 0, 0), block_tokens=2, slot_capacity=cap)
                self.assertEqual(codec.decode_block(codec.encode_block(raw)), raw)

    def test_group_promotion_checks_each_channel_and_rebases_the_whole_group(self):
        cal = Calibration(tuple(range(120, 128)), tuple(range(116, 124)))
        codec = ReferenceCodec(cal, block_tokens=1, slot_capacity=100)
        exponents = list(cal.base3)
        first = codec.encode_block(words(exponents))
        self.assertEqual(first.mode_counts, (1, 0, 0))
        exponents[7] += 8
        promoted = codec.encode_block(words(exponents, list(range(8)), [i % 2 for i in range(8)]))
        self.assertEqual(promoted.mode_counts, (0, 1, 0))
        self.assertEqual(promoted.extra_bytes, 1)
        self.assertEqual(codec.decode_block(promoted), words(exponents, list(range(8)), [i % 2 for i in range(8)]))
        exponents[7] = 255
        raw_group = codec.encode_block(words(exponents))
        self.assertEqual(raw_group.mode_counts, (0, 0, 1))
        self.assertEqual(raw_group.extra_bytes, 5)
        self.assertEqual(codec.decode_block(raw_group), words(exponents))

    def test_narrow_window_boundaries_and_wide_window_boundaries(self):
        # Every case changes one channel, leaving the other seven at exponent120.
        expected = {115: 2, 116: 1, 117: 1, 119: 1, 120: 0, 121: 0,
                    126: 0, 127: 0, 128: 1, 130: 1, 131: 1, 132: 2, 133: 2}
        codec = ReferenceCodec(calibration(), block_tokens=1, slot_capacity=100)
        for exponent, tag in expected.items():
            with self.subTest(exponent=exponent):
                raw = words([exponent] + [120] * 7)
                block = codec.encode_block(raw)
                self.assertEqual(block.mode_counts, tuple(int(i == tag) for i in range(3)))
                self.assertEqual(codec.decode_block(block), raw)

    def test_base_extremes_never_wrap_exponents(self):
        for narrow, wide, exponent in [(0, 0, 0), (0, 0, 255), (248, 240, 255), (248, 240, 0)]:
            codec = ReferenceCodec(calibration(8, narrow, wide), block_tokens=1, slot_capacity=100)
            raw = words([exponent] * 8, list(range(8)), [0, 1] * 4)
            self.assertEqual(codec.decode_block(codec.encode_block(raw)), raw)
        for narrow in (-1, 249):
            with self.assertRaises(ValueError):
                calibration(8, narrow, 0)
        for wide in (-1, 241):
            with self.assertRaises(ValueError):
                calibration(8, 0, wide)

    def test_exact_capacity_boundary_full_raw_fallback_and_all_overhead(self):
        raw = words([120] * 8)
        sample = ReferenceCodec(calibration(), block_tokens=1, slot_capacity=100).encode_block(raw)
        size = sample.used_bytes
        self.assertEqual(size, 24 + 8 + 1 + 3)  # actual own header + SM + tags + low codes
        for cap, fallback in [(size - 1, True), (size, False), (size + 1, False)]:
            with self.subTest(cap=cap):
                codec = ReferenceCodec(calibration(), block_tokens=1, slot_capacity=cap)
                block = codec.encode_block(raw)
                self.assertEqual(block.raw_fallback, fallback)
                self.assertEqual(len(block.storage), 24 + len(raw) if fallback else cap)
                self.assertEqual(block.slack_bytes, 0 if fallback else cap - size)
                self.assertEqual(codec.decode_block(block), raw)

    def test_empty_partial_odd_and_overfull_blocks_are_rejected(self):
        codec = ReferenceCodec(calibration(), block_tokens=2)
        for raw in (b"", bytes(15), bytes(16), bytes(31), bytes(33), bytes(48)):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                codec.encode_block(raw)
        codec.encode_block(bytes(32))  # exact full block is valid
        for tokens in (-1, 0, True):
            with self.assertRaises(ValueError):
                ReferenceCodec(calibration(), block_tokens=tokens)
        for cap in (-1, 0x100000000, True):
            with self.assertRaises(ValueError):
                ReferenceCodec(calibration(), slot_capacity=cap)

    def test_malformed_streams_rejected_before_accepting_corruption(self):
        codec = ReferenceCodec(calibration(), block_tokens=1, slot_capacity=100)
        block = codec.encode_block(words([120] * 8))
        for storage in (block.storage[:10], block.storage[:block.used_bytes - 1], block.storage + b"\x01"):
            with self.subTest(size=len(storage)), self.assertRaises(ValueError):
                codec.decode_block(replace(block, storage=storage))
        tag_at = HEADER.size + 8
        for tag in (3, 4):  # reserved mode, nonzero padding bits in the mode byte
            data = bytearray(block.storage)
            data[tag_at] = tag
            with self.assertRaises(ValueError):
                codec.decode_block(replace(block, storage=bytes(data)))
        data = bytearray(block.storage)
        data[-1] = 1
        with self.assertRaises(ValueError):
            codec.decode_block(replace(block, storage=bytes(data)))
        with self.assertRaises(ValueError):
            ReferenceCodec(calibration(8, 121, 116), block_tokens=1, slot_capacity=100).decode_block(block)

    def test_truncated_wider_groups_and_extra_bytes_rejected(self):
        codec = ReferenceCodec(calibration(), block_tokens=1, slot_capacity=100)
        block = codec.encode_block(words([255] * 8))
        data = bytearray(block.storage)
        values = list(HEADER.unpack_from(data))
        values[-2] -= 1
        values[-1] -= 1
        data[:HEADER.size] = HEADER.pack(*values)
        data[block.used_bytes - 1] = 0
        with self.assertRaises(ValueError):
            codec.decode_block(replace(block, storage=bytes(data), used_bytes=block.used_bytes - 1, extra_bytes=4))

    def test_calibration_chooses_maximum_coverage_not_minmax(self):
        rows = [words([120] * 8)] * 3 + [words([127] * 8)] * 2 + [words([0] * 8)]
        cal = Calibration.from_rows(iter(rows), channels=8)
        self.assertEqual(cal.base3, (120,) * 8)
        self.assertEqual(cal.base4, (112,) * 8)  # earliest width16 window covering120 and127
        self.assertEqual(Calibration.from_bytes(cal.to_bytes()), cal)
        # Equal-coverage ambiguity is a declared prototype choice, not paper provenance.
        self.assertEqual(Calibration.from_rows([words([255] * 8)], 8).base3, (248,) * 8)
        for data in (b"", cal.to_bytes()[:-1], cal.to_bytes() + b"\0"):
            with self.assertRaises(ValueError):
                Calibration.from_bytes(data)
        with self.assertRaises(ValueError):
            Calibration.from_rows([], 8)
        with self.assertRaises(ValueError):
            Calibration.from_rows([bytes(15)], 8)
        for channels in (0, 7, 9):
            with self.assertRaises(ValueError):
                Calibration.from_rows([], channels)


class AppendTests(unittest.TestCase):
    def test_incremental_partial_full_boundary_roundtrip_and_accounting(self):
        codec = ReferenceCodec(calibration(), block_tokens=2, slot_capacity=100)
        cache = IncrementalCache(codec)
        row = words([120] * 8)
        self.assertEqual(cache.read(), b"")
        self.assertEqual(cache.accounting()["allocated_format_bytes"], len(codec.calibration.to_bytes()))
        for count in (1, 2, 3):
            cache.append(row)
            self.assertEqual(cache.read(), row * count)
            accounting = cache.accounting()
            full, partial = divmod(count, 2)
            self.assertEqual(accounting["header_bytes"], 24 * full)
            self.assertEqual(accounting["table_bytes"], 4 * (full + bool(partial)))
            self.assertEqual(accounting["allocated_format_bytes"], full * 100 + bool(partial) * 32
                             + 4 * (full + bool(partial)) + len(codec.calibration.to_bytes()))
            self.assertEqual(accounting["tail_slack_bytes"], 16 if partial else 0)

    def test_whole_raw_fallback_and_partial_tail_are_both_accounted(self):
        codec = ReferenceCodec(calibration(), block_tokens=2, slot_capacity=0)
        cache = IncrementalCache(codec)
        row = words([255] * 8)
        for _ in range(3):
            cache.append(row)
        info = cache.accounting()
        self.assertEqual(cache.read(), row * 3)
        self.assertEqual(info["raw_fallback_blocks"], 1)
        self.assertEqual(info["stored_used_bytes"], 24 + 32 + 16)
        self.assertEqual(info["allocated_format_bytes"], 24 + 32 + 32 + 8 + len(codec.calibration.to_bytes()))

    def test_truncate_and_fork_preserve_shared_prefix_and_rejected_suffix(self):
        codec = ReferenceCodec(calibration(), block_tokens=2, slot_capacity=100)
        rows = [words([120 + i] * 8) for i in range(5)]
        original = IncrementalCache(codec)
        for row in rows:
            original.append(row)
        for keep in (0, 1, 2, 3, 4, 5):
            with self.subTest(keep=keep):
                child = original.fork_prefix(keep)
                self.assertEqual(child.read(), b"".join(rows[:keep]))
                replacement = words([255] * 8)
                child.append(replacement)
                self.assertEqual(child.read(), b"".join(rows[:keep]) + replacement)
                self.assertEqual(original.read(), b"".join(rows))
        for bad in (-1, 6):
            with self.assertRaises(ValueError):
                original.truncate(bad)
        self.assertEqual(original.read(), b"".join(rows))
        original.truncate(3)
        self.assertEqual(original.read(), b"".join(rows[:3]))
        original.append(rows[4])
        self.assertEqual(original.read(), b"".join(rows[:3]) + rows[4])

    def test_two_and_four_query_rollback_crossing_real_sixteen_token_boundary(self):
        rows = [words([(120 + i) % 256] * 8, [i % 128] * 8, [i % 2] * 8) for i in range(19)]
        for capacity in (0, 3032):
            for proposed in (2, 4):
                for accepted in range(proposed + 1):
                    with self.subTest(capacity=capacity, proposed=proposed, accepted=accepted):
                        cache = IncrementalCache(ReferenceCodec(calibration(), block_tokens=16, slot_capacity=capacity))
                        for row in rows[:15 + proposed]:
                            cache.append(row)
                        cache.truncate(15 + accepted)
                        self.assertEqual(cache.read(), b"".join(rows[:15 + accepted]))
                        cache.append(words([255] * 8))
                        self.assertEqual(cache.read(), b"".join(rows[:15 + accepted]) + words([255] * 8))

    def test_invalid_append_does_not_change_committed_cache(self):
        cache = IncrementalCache(ReferenceCodec(calibration(), block_tokens=2))
        row = words([120] * 8)
        cache.append(row)
        for invalid in (b"", row[:-1], row + b"\0\0"):
            with self.assertRaises(ValueError):
                cache.append(invalid)
            self.assertEqual(cache.read(), row)


if __name__ == "__main__":
    unittest.main()
