"""Stdlib CPU disk-writer contracts; synthetic already-copied BF16 bytes only."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ANALYSIS = Path(__file__).resolve().parents[2] / "tools/analysis"
sys.path.insert(0, str(ANALYSIS))
import t056_capture_io as capture


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw_rows(count, seed=0):
    return b"".join(struct.pack("<128H", *[((i + index + seed) % 2 << 15) | (100 << 7) | ((i + index + seed) & 127)
                                         for i in range(128)]) for index in range(count))


def descriptor(role="calibration", *, rows=17, position=0, **changes):
    request = "request-" + role
    sequence = "sequence-" + role
    result = {"layer": 3, "head": 0, "kind": "K", "channel_start": 0, "channels": 128,
              "capture_stage": "post_rope", "position_start": position, "rows": rows,
              "domain": "english" if role == "calibration" else "code", "role": role,
              "sequence_id": sequence, "request_id": request, "committed_tokens": 262144,
              "request_provenance": {"model_id": capture.MODEL_ID, "checkpoint_revision": capture.MODEL_REVISION,
                                     "request_id": request, "sequence_id": sequence, "source_id": "source-" + role,
                                     "source_file_sha256": sha(("file-" + role).encode()),
                                     "prompt_token_ids_sha256": sha(("prompt-" + role).encode()),
                                     "completed_step": 3, "runtime_verified": True,
                                     "source_dtype": "bfloat16", "capture_site": "after_cache_write"}}
    result.update(changes)
    result["request_provenance"].update(prompt_tokens=result["committed_tokens"], committed_tokens=result["committed_tokens"],
                                        before_computed=max(0, result["committed_tokens"] - 16),
                                        domain=result["domain"], role=result["role"])
    return result


def route(position, count, *, token=17, page_offset=0, page_tokens=256):
    ids = [token] * count
    return {"physical_page": position // page_tokens, "page_offset": page_offset, "kernel_page_tokens": page_tokens,
            "position_start": position, "rows": count, "token_ids": ids,
            "token_ids_sha256": sha(struct.pack("<" + "I" * count, *ids))}


class CaptureWriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t056-capture-io-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.index = 0

    def writer(self, **kwargs):
        self.index += 1
        return capture.CaptureWriter(self.root / f"capture-{self.index}", **kwargs)

    def consume(self, manifest, ok=True):
        result = subprocess.run([sys.executable, "-B", str(ANALYSIS / "t056_measure.py"), str(manifest)],
                                capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        return result

    def raw_bytes(self, writer):
        return {p.name: p.read_bytes() for p in writer.output_dir.glob("*.raw")}

    def complete_one(self, writer, *, rows=2):
        identity = writer.start_shard(descriptor(rows=rows))
        writer.add_chunk(identity, position_start=0, rows=rows, data=raw_rows(rows), route=route(0, rows))
        return writer.finalize()

    def test_streamed_order_hashes_routes_and_private_output_are_consumed_by_measure(self):
        caller = {"requests": [{"id": "request-calibration"}, {"id": "request-evaluation"}],
                  "source_api_sha256": {"cache-write": sha(b"synthetic API provenance")}}
        writer = self.writer(capture_provenance=caller)
        expected = {}
        for role in ("calibration", "evaluation"):
            desc = descriptor(role)
            identity = writer.start_shard(desc)
            expected[identity + ".raw"] = raw_rows(17)
            writer.add_chunk(identity, position_start=0, rows=16, data=raw_rows(16), route=route(0, 16))
            before = writer.bytes_written
            with self.assertRaises(ValueError):
                writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=route(0, 1))
            self.assertEqual(writer.bytes_written, before)
            writer.add_chunk(identity, position_start=16, rows=1, data=raw_rows(1, seed=16), route=route(16, 1, page_offset=16))
        manifest = writer.finalize()
        measured = self.consume(manifest)
        self.assertEqual(measured["streams"][0]["calibration_domain"], "english")
        self.assertEqual(measured["streams"][0]["domain"], "code")
        self.assertEqual(measured["streams"][0]["views"][0]["totals"]["raw_bytes"], 17 * 256)
        self.assertEqual(measured["streams"][0]["views"][0]["totals"]["roundtrip_verified_blocks"], 1)
        document = json.loads(manifest.read_text())
        self.assertEqual(document["capture_provenance"]["status"], "complete")
        self.assertEqual(document["capture_provenance"]["requests"], caller["requests"])
        for identity, metadata in document["capture_provenance"]["shards"].items():
            data = expected[metadata["path"]]
            self.assertEqual((writer.output_dir / metadata["path"]).read_bytes(), data)
            self.assertEqual(metadata["sha256"], sha(data))
            self.assertEqual([part["position_start"] for part in metadata["chunks"]], [0, 16])
            self.assertEqual([part["rows"] for part in metadata["chunks"]], [16, 1])
            self.assertEqual(metadata["chunks"][0]["route"]["token_ids"], [17] * 16)
        actual = sum(p.stat().st_size for p in writer.output_dir.iterdir())
        self.assertEqual(writer.bytes_written, actual)
        self.assertEqual(writer.report["metadata_bytes"], manifest.stat().st_size)
        self.assertLessEqual(actual, writer.max_output_bytes)
        self.assertEqual(stat.S_IMODE(writer.output_dir.stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in writer.output_dir.iterdir()))
        before = self.raw_bytes(writer)
        with self.assertRaises(ValueError):
            writer.finalize()
        with self.assertRaises(FileExistsError):
            capture.CaptureWriter(writer.output_dir)
        self.assertEqual(self.raw_bytes(writer), before)

    def test_partial_final_chunk_and_incomplete_finalize_are_retryable_not_complete(self):
        writer = self.writer(max_chunk_rows=3)
        identity = writer.start_shard(descriptor(rows=5))
        writer.add_chunk(identity, position_start=0, rows=3, data=raw_rows(3))
        before = self.raw_bytes(writer)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            writer.finalize()
        self.assertEqual(self.raw_bytes(writer), before)
        self.assertEqual(writer.report["status"], "open")
        self.consume(writer.output_dir / "manifest.json", ok=False)
        writer.add_chunk(identity, position_start=3, rows=2, data=raw_rows(2, seed=3))
        manifest = writer.finalize()
        document = json.loads(manifest.read_text())
        self.assertEqual((writer.output_dir / document["shards"][0]["path"]).read_bytes(), raw_rows(5))
        self.assertEqual(document["capture_provenance"]["shards"][identity]["written_rows"], 5)

    def test_descriptor_dtype_identity_stage_and_committed_bounds_reject_before_files_change(self):
        writer = self.writer()
        good = descriptor(rows=2, position=10, committed_tokens=12)
        original = self.raw_bytes(writer)
        mutations = (("layer", 4), ("layer", 0), ("layer", 40), ("head", 2), ("channels", 256),
                     ("channel_start", 127), ("channel_start", 129), ("capture_stage", "pre_rope"),
                     ("kind", "GDN"), ("rows", 0), ("committed_tokens", 11), ("committed_tokens", 262145),
                     ("position_start", -1), ("rows", True), ("domain", "unknown"))
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                altered = copy.deepcopy(good)
                altered[field] = value
                with self.assertRaises(ValueError):
                    writer.start_shard(altered)
                self.assertEqual(self.raw_bytes(writer), original)
                self.assertEqual(writer.bytes_written, 0)
        for field, value in (("checkpoint_revision", "f" * 40), ("sequence_id", "other-seq"),
                             ("runtime_verified", False), ("runtime_verified", 1),
                             ("source_dtype", "float16"), ("capture_site", "pre_cache"),
                             ("prompt_tokens", 10), ("committed_tokens", 11), ("before_computed", 12), ("completed_step", 0)):
            altered = copy.deepcopy(good)
            altered["request_provenance"][field] = value
            with self.assertRaises(ValueError):
                writer.start_shard(altered)
            self.assertEqual(self.raw_bytes(writer), original)
        for bound in (12, 13):
            accepted = self.writer()
            accepted.start_shard(descriptor(rows=2, position=10, committed_tokens=bound))

    def test_chunk_dtype_order_shape_limit_and_route_errors_do_not_write_extra_bytes(self):
        writer = self.writer()
        identity = writer.start_shard(descriptor(rows=257))
        writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1))
        before = self.raw_bytes(writer)
        bad_calls = [dict(position_start=1, rows=1, data=raw_rows(1), source_dtype="float16"),
                     dict(position_start=0, rows=1, data=raw_rows(1)),
                     dict(position_start=2, rows=1, data=raw_rows(1)),
                     dict(position_start=1, rows=0, data=b""),
                     dict(position_start=1, rows=257, data=raw_rows(257)),
                     dict(position_start=1, rows=1, data=raw_rows(1)[:-1]),
                     dict(position_start=1, rows=1, data=bytearray(raw_rows(1)))]
        for arguments in bad_calls:
            with self.subTest(arguments={key: value for key, value in arguments.items() if key != "data"}):
                with self.assertRaises(ValueError):
                    writer.add_chunk(identity, **arguments)
                self.assertEqual(self.raw_bytes(writer), before)
                self.assertEqual(writer.bytes_written, 256)
        writer.add_chunk(identity, position_start=1, rows=256, data=raw_rows(256, seed=1), route=route(1, 256))
        self.assertEqual(writer.bytes_written, 257 * 256)
        with self.assertRaises(ValueError):
            writer.add_chunk(identity, position_start=257, rows=1, data=raw_rows(1))

    def test_route_token_count_digest_logical_and_page_boundaries(self):
        original = route(0, 2)
        mutations = (("token_ids", [17]), ("token_ids_sha256", "f" * 64), ("position_start", 1), ("rows", 1),
                     ("page_offset", -1), ("physical_page", -1), ("kernel_page_tokens", 0), ("token_ids", [-1, 17]),
                     ("token_ids", [2**32, 17]), ("token_ids", [True, 17]))
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                writer = self.writer()
                identity = writer.start_shard(descriptor(rows=2))
                altered = copy.deepcopy(original)
                altered[field] = value
                with self.assertRaises(ValueError):
                    writer.add_chunk(identity, position_start=0, rows=2, data=raw_rows(2), route=altered)
                self.assertEqual(writer.bytes_written, 0)
                self.assertEqual(self.raw_bytes(writer)[identity + ".raw"], b"")
        for offset, valid in ((5, True), (6, True), (7, False)):
            writer = self.writer()
            identity = writer.start_shard(descriptor(rows=2))
            arguments = dict(position_start=0, rows=2, data=raw_rows(2), route=route(0, 2, page_offset=offset, page_tokens=8))
            if valid:
                writer.add_chunk(identity, **arguments)
                self.assertEqual(writer.bytes_written, 512)
            else:
                with self.assertRaises(ValueError):
                    writer.add_chunk(identity, **arguments)
                self.assertEqual(writer.bytes_written, 0)
        ids = [0, 2**32 - 1, 17, 17]
        self.assertEqual(capture.token_ids_sha256(iter(ids)), sha(struct.pack("<4I", *ids)))

    def test_worker_allocation_deltas_are_bounded_audit_metadata_not_kv_size(self):
        for delta, valid in ((-1, True), (0, True), (1, True),
                             (64 * 1024**2 - 1, True), (64 * 1024**2, True), (64 * 1024**2 + 1, False), (True, False)):
            with self.subTest(delta=delta):
                writer = self.writer()
                identity = writer.start_shard(descriptor(rows=1))
                page = {**route(0, 1), "device_allocated_delta": delta, "device_reserved_delta": 0}
                if valid:
                    writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=page)
                    manifest = json.loads(writer.finalize().read_text())
                    recorded = manifest["capture_provenance"]["shards"][identity]["chunks"][0]["route"]
                    self.assertEqual(recorded["device_allocated_delta"], delta)
                    self.assertEqual(writer.report["data_bytes"], 256)
                else:
                    with self.assertRaises(ValueError):
                        writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=page)
                    self.assertEqual(writer.bytes_written, 0)
        writer = self.writer()
        identity = writer.start_shard(descriptor(rows=1))
        with self.assertRaises(ValueError):
            writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1),
                             route={**route(0, 1), "device_allocated_delta": 0})
        self.assertEqual(writer.bytes_written, 0)

    def test_same_sequence_position_token_identity_is_consistent_across_kv_shards(self):
        writer = self.writer()
        first = writer.start_shard(descriptor(rows=2))
        second = writer.start_shard(descriptor(rows=2, kind="V", capture_stage="value_cache"))
        writer.add_chunk(first, position_start=0, rows=2, data=raw_rows(2), route=route(0, 2))
        before = self.raw_bytes(writer)
        with self.assertRaisesRegex(ValueError, "conflicting token"):
            writer.add_chunk(second, position_start=0, rows=2, data=raw_rows(2), route=route(0, 2, token=18))
        self.assertEqual(self.raw_bytes(writer), before)
        writer.add_chunk(second, position_start=0, rows=2, data=raw_rows(2), route=route(0, 2))
        writer.finalize()

    def test_dataset_aliases_and_overlapping_positions_are_rejected(self):
        for reused in ("sequence_id", "request_id", "source_id", "source_file_sha256", "prompt_token_ids_sha256"):
            with self.subTest(reused=reused):
                writer = self.writer()
                cal = descriptor(rows=2)
                writer.start_shard(cal)
                evaluation = descriptor("evaluation", rows=2, layer=39)
                if reused in ("sequence_id", "request_id"):
                    evaluation[reused] = cal[reused]
                    evaluation["request_provenance"][reused] = cal[reused]
                else:
                    evaluation["request_provenance"][reused] = cal["request_provenance"][reused]
                with self.assertRaises(ValueError):
                    writer.start_shard(evaluation)
                self.assertEqual(writer.bytes_written, 0)
                self.assertEqual(writer.report["shards"], 1)
        for start, valid in ((1, False), (2, True), (3, True)):
            writer = self.writer()
            writer.start_shard(descriptor(rows=2))
            if valid:
                writer.start_shard(descriptor(rows=2, position=start))
            else:
                with self.assertRaisesRegex(ValueError, "overlapping"):
                    writer.start_shard(descriptor(rows=2, position=start))
        writer = self.writer()
        writer.start_shard(descriptor(rows=2, committed_tokens=16))
        with self.assertRaises(ValueError):
            writer.start_shard(descriptor(rows=2, layer=39, committed_tokens=17))

    def test_raw_quota_boundary_and_reserved_metadata_before_writes(self):
        reserve = 4096
        for allowance, valid in ((255, False), (256, True), (257, True)):
            writer = self.writer(max_output_bytes=reserve + allowance, metadata_reserve_bytes=reserve)
            if valid:
                self.complete_one(writer, rows=1)
                self.assertLessEqual(writer.bytes_written, reserve + allowance)
                with self.assertRaises(ValueError):
                    writer.start_shard(descriptor(rows=1, layer=39))
            else:
                with self.assertRaisesRegex(ValueError, "quota"):
                    writer.start_shard(descriptor(rows=1))
                self.assertEqual(writer.bytes_written, 0)
        writer = self.writer(max_output_bytes=reserve + 512, metadata_reserve_bytes=reserve)
        identity = writer.start_shard(descriptor(rows=2))
        writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1))
        before = self.raw_bytes(writer)
        with self.assertRaisesRegex(ValueError, "quota"):
            writer.start_shard(descriptor(rows=1, layer=39))
        self.assertEqual(self.raw_bytes(writer), before)
        self.assertEqual(writer.bytes_written, 256)

    def test_actual_metadata_quota_boundary_preflights_route_before_raw_write(self):
        probe = self.writer(metadata_reserve_bytes=4096)
        manifest = self.complete_one(probe, rows=1)
        size = manifest.stat().st_size
        self.assertEqual(len(str(size)), len(str(4096)))
        for budget, valid in ((size - 1, False), (size, True), (size + 1, True)):
            with self.subTest(budget=budget):
                writer = self.writer(metadata_reserve_bytes=budget)
                identity = writer.start_shard(descriptor(rows=1))
                if valid:
                    writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=route(0, 1))
                    writer.finalize()
                    self.assertEqual(writer.report["metadata_bytes"], size)
                else:
                    with self.assertRaisesRegex(ValueError, "metadata reserve"):
                        writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=route(0, 1))
                    self.assertEqual(writer.bytes_written, 0)
                    self.assertEqual(self.raw_bytes(writer)[identity + ".raw"], b"")

    def test_host_budget_bounds_chunk_plus_actual_serialized_metadata(self):
        probe = self.writer(metadata_reserve_bytes=4096, max_host_bytes=8192)
        manifest = self.complete_one(probe, rows=1)
        size = manifest.stat().st_size
        for budget, valid in ((size + 255, False), (size + 256, True), (size + 257, True)):
            writer = self.writer(metadata_reserve_bytes=4096, max_host_bytes=budget)
            identity = writer.start_shard(descriptor(rows=1))
            if valid:
                writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=route(0, 1))
                writer.finalize()
                self.assertEqual(writer.report["status"], "complete")
            else:
                with self.assertRaisesRegex(ValueError, "max_host_bytes"):
                    writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=route(0, 1))
                self.assertEqual(writer.bytes_written, 0)

    def test_metadata_is_bounded_and_copied_not_mutable_caller_state(self):
        for value in ({"status": "complete"}, {"source_api_sha256": {"api": "not-a-hash"}},
                      {"notes": "x" * 5000}, {"notes": float("nan")}, {"notes": object()}):
            with self.assertRaises(ValueError):
                self.writer(metadata_reserve_bytes=4096, capture_provenance=value)
        caller = {"source_api_sha256": {"api": sha(b"original")}, "requests": [{"id": "source"}]}
        writer = self.writer(capture_provenance=caller)
        desc = descriptor(rows=1)
        identity = writer.start_shard(desc)
        desc["request_provenance"]["source_id"] = "mutated"
        caller["source_api_sha256"]["api"] = sha(b"mutated")
        supplied_route = route(0, 1)
        writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1), route=supplied_route)
        supplied_route["token_ids"][0] = 999
        manifest = json.loads(writer.finalize().read_text())["capture_provenance"]
        self.assertEqual(manifest["source_api_sha256"]["api"], sha(b"original"))
        self.assertEqual(manifest["shards"][identity]["request_provenance"]["source_id"], "source-calibration")
        self.assertEqual(manifest["shards"][identity]["chunks"][0]["route"]["token_ids"], [17])

    def test_partial_disk_write_failure_poisoned_without_complete_manifest(self):
        writer = self.writer()
        identity = writer.start_shard(descriptor(rows=2))
        original = capture.os.write
        calls = 0

        def fail_after_prefix(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(fd, data[:17])
            raise OSError("synthetic disk full")
        with mock.patch.object(capture.os, "write", side_effect=fail_after_prefix):
            with self.assertRaises(OSError):
                writer.add_chunk(identity, position_start=0, rows=2, data=raw_rows(2))
        self.assertEqual(writer.bytes_written, 17)
        self.assertEqual(self.raw_bytes(writer)[identity + ".raw"], raw_rows(2)[:17])
        self.assertEqual(writer.report["status"], "failed")
        with self.assertRaises(ValueError):
            writer.finalize()
        self.consume(writer.output_dir / "manifest.json", ok=False)

    def test_tampered_data_and_manifest_publication_failure_never_publish_success(self):
        for failure in ("tampered", "metadata_disk_full", "existing_manifest"):
            with self.subTest(failure=failure):
                writer = self.writer()
                identity = writer.start_shard(descriptor(rows=1))
                writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1))
                if failure == "tampered":
                    path = writer.output_dir / (identity + ".raw")
                    path.write_bytes(b"X" + path.read_bytes()[1:])
                    with self.assertRaisesRegex(ValueError, "streamed hash"):
                        writer.finalize()
                elif failure == "metadata_disk_full":
                    with mock.patch.object(capture.os, "write", side_effect=OSError("synthetic metadata disk full")):
                        with self.assertRaises(OSError):
                            writer.finalize()
                else:
                    destination = writer.output_dir / "manifest.json"
                    destination.write_text("operator-owned file, do not overwrite")
                    with self.assertRaises(FileExistsError):
                        writer.finalize()
                    self.assertEqual(destination.read_text(), "operator-owned file, do not overwrite")
                self.assertEqual(writer.report["status"], "failed")
                self.assertEqual(writer.report["metadata_bytes"], 0)
                self.consume(writer.output_dir / "manifest.json", ok=False)

    def test_foreign_or_changed_files_cannot_be_ignored_by_quota_preflight(self):
        for failure in ("foreign", "grown", "missing", "symlink"):
            with self.subTest(failure=failure):
                writer = self.writer()
                identity = writer.start_shard(descriptor(rows=2))
                writer.add_chunk(identity, position_start=0, rows=1, data=raw_rows(1))
                path = writer.output_dir / (identity + ".raw")
                if failure == "foreign":
                    (writer.output_dir / "foreign.bin").write_bytes(b"operator data")
                elif failure == "grown":
                    path.write_bytes(path.read_bytes() + b"unexpected")
                elif failure == "missing":
                    path.unlink()
                else:
                    target = self.root / "outside.raw"
                    target.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(target)
                before = {p.name: p.read_bytes() for p in writer.output_dir.iterdir()}
                with self.assertRaises((ValueError, FileExistsError)):
                    writer.add_chunk(identity, position_start=1, rows=1, data=raw_rows(1, seed=1))
                self.assertEqual({p.name: p.read_bytes() for p in writer.output_dir.iterdir()}, before)
                self.assertEqual(writer.bytes_written, 256)
                self.assertEqual(writer.report["status"], "failed")

    def test_finalization_deadline_cannot_publish_after_verification_or_metadata_write(self):
        def ready():
            writer = self.writer()
            for role in ("calibration", "evaluation"):
                identity = writer.start_shard(descriptor(role, rows=16))
                writer.add_chunk(identity, position_start=0, rows=16, data=raw_rows(16), route=route(0, 16))
            return writer

        # This exact dataset is consumable if publication succeeds; negative
        # consumption checks below cannot pass merely for missing calibration.
        control = ready()
        measured = self.consume(control.finalize())
        self.assertEqual(measured["streams"][0]["views"][0]["totals"]["roundtrip_verified_blocks"], 1)
        for phase in ("before_work", "verification", "metadata_write"):
            with self.subTest(phase=phase):
                writer = ready()
                before = self.raw_bytes(writer)
                expired = phase == "before_work"

                def deadline():
                    if expired:
                        raise RuntimeError("synthetic deadline exceeded")

                verify = writer._verify_file
                write = capture.os.write

                def verify_then_expire(*args):
                    nonlocal expired
                    verify(*args)
                    if phase == "verification":
                        expired = True

                def write_then_expire(*args):
                    nonlocal expired
                    count = write(*args)
                    if phase == "metadata_write":
                        expired = True
                    return count

                with mock.patch.object(writer, "_verify_file", side_effect=verify_then_expire), \
                     mock.patch.object(capture.os, "write", side_effect=write_then_expire):
                    with self.assertRaisesRegex(RuntimeError, "deadline exceeded"):
                        writer.finalize(check_deadline=deadline)
                self.assertEqual(writer.report["status"], "failed")
                self.assertEqual(writer.report["metadata_bytes"], 0)
                self.assertEqual(self.raw_bytes(writer), before)
                self.assertEqual(sum(path.stat().st_size for path in writer.output_dir.iterdir()), writer.bytes_written)
                self.consume(writer.output_dir / "manifest.json", ok=False)
                writer.close()

    def test_pilot_geometry_eighty_shards_fits_five_mib_raw_and_is_measure_consumable(self):
        writer = self.writer(capture_provenance={"source_api_sha256": {"fixture": sha(b"original CPU source API")}})
        windows = (("calibration", 0), ("calibration", 65280),
                   ("evaluation", 0), ("evaluation", 65280), ("evaluation", 204544))
        data = raw_rows(256)
        for role, position in windows:
            for layer, head, kind, start in itertools.product((3, 39), (0, 1), ("K", "V"), (0, 128)):
                desc = descriptor(role, rows=256, position=position, layer=layer, head=head, kind=kind,
                                  channel_start=start, capture_stage="post_rope" if kind == "K" else "value_cache",
                                  committed_tokens=65536 if role == "calibration" else 204800)
                identity = writer.start_shard(desc)
                writer.add_chunk(identity, position_start=position, rows=256, data=data,
                                 route=route(position, 256, token=17 if role == "calibration" else 23,
                                             page_offset=position % 4096, page_tokens=4096))
        self.assertEqual(writer.bytes_written, 5 * 1024**2)
        manifest = writer.finalize()
        self.assertEqual(writer.report["shards"], 80)
        self.assertLessEqual(writer.bytes_written, 8 * 1024**2)
        self.assertEqual(writer.bytes_written, sum(p.stat().st_size for p in writer.output_dir.iterdir()))
        report = self.consume(manifest)
        self.assertEqual(len(report["streams"]), 16)
        self.assertEqual(sum(stream["views"][0]["totals"]["raw_bytes"] for stream in report["streams"]), 3 * 1024**2)
        self.assertTrue(all(stream["coverage"]["sampling"] == "gapped_samples" for stream in report["streams"]))


if __name__ == "__main__":
    unittest.main()
