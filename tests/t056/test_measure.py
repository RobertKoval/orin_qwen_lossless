"""CPU-only CLI/measurement contracts; no runtime, model, torch or CUDA imports."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ANALYSIS = Path(__file__).resolve().parents[2] / "tools/analysis"
sys.path.insert(0, str(ANALYSIS))
import t056_measure as measurement
from t056_reference import Calibration, HEADER, ReferenceCodec, TABLE_ENTRY_BYTES

REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


def row(channels=128, exponent=100, seed=0):
    words = [((i + seed) % 2 << 15) | exponent << 7 | ((i * 13 + seed) & 127) for i in range(channels)]
    return struct.pack("<" + "H" * channels, *words)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t056-measure-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.manifest = self.root / "shards.json"
        self.document = {
            "schema_version": 1,
            "model": {"id": "Qwen/Qwen3.6-35B-A3B", "revision": REVISION},
            "model_role": "target", "source_dtype": "bfloat16", "format": "uint16-le",
            "state_kind": "attention_kv", "capture_site": "after_cache_write", "head_channels": 256,
            "capture_provenance": {"status": "synthetic CPU fixture, not a runtime capture"}, "shards": []}
        self.add_shard("calibration", rows=8, sequence="cal-seq")
        self.add_shard("evaluation", rows=17, sequence="eval-seq")
        self.save()

    def add_shard(self, role, *, rows=16, sequence=None, position=0, exponent=100, **partition):
        channels = partition.get("channels", 128)
        name = f"shard-{len(self.document['shards'])}.raw"
        data = b"".join(row(channels, exponent, index) for index in range(rows))
        (self.root / name).write_bytes(data)
        entry = {"path": name, "kind": "K", "capture_stage": "post_rope", "layer": 3, "head": 0,
                 "channel_start": 0, "channels": channels, "domain": "English", "role": role,
                 "sequence_id": sequence or role + "-seq", "position_start": position, "rows": rows}
        entry.update(partition)
        if entry["kind"] == "V" and "capture_stage" not in partition:
            entry["capture_stage"] = "value_cache"
        self.document["shards"].append(entry)
        return entry

    def save(self, document=None, path=None):
        path = path or self.manifest
        path.write_text(json.dumps(self.document if document is None else document))
        return path

    def cli(self, *arguments, ok=True, manifests=None):
        result = subprocess.run([sys.executable, "-B", str(ANALYSIS / "t056_measure.py"),
                                 *map(str, manifests or [self.manifest]), *map(str, arguments)],
                                text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "", "invalid input must not produce a partial report")
        return result

    def report(self, *args):
        return json.loads(self.cli(*args).stdout)

    def table_bytes(self):
        return len(Calibration.from_rows((row(seed=i) for i in range(8)), channels=128).to_bytes())

    def test_default_accounting_matches_independent_full_and_tail_costs(self):
        report = self.report()
        self.assertEqual(report["identity"]["model"]["revision"], REVISION)
        self.assertEqual(len(report["streams"]), 1)
        views = report["streams"][0]["views"]
        self.assertEqual([v["window_tokens"] for v in views], [65536, 204800])
        for view in views:
            totals = view["totals"]
            # 2048 sign/mantissa bytes + 3 bits/value + 2 bits/group + OUR 24B header.
            used = 2048 + 2048 * 3 // 8 + (2048 // 8) * 2 // 8 + HEADER.size
            base = self.table_bytes()
            self.assertEqual(totals["raw_bytes"], 17 * 128 * 2)
            self.assertEqual(totals["full_blocks"], 1)
            self.assertEqual(totals["roundtrip_verified_blocks"], 1)
            self.assertEqual(totals["compressed_slot_used_bytes"], used)
            self.assertEqual(totals["compressed_slot_capacity_bytes"], 3032)
            self.assertEqual(totals["compressed_slot_slack_bytes"], 3032 - used)
            self.assertEqual(totals["headers_bytes"], HEADER.size)
            self.assertEqual(totals["block_table_bytes"], 2 * TABLE_ENTRY_BYTES)
            self.assertEqual(totals["base_table_bytes"], base)
            self.assertEqual(totals["partial_tail_used_bytes"], 256)
            self.assertEqual(totals["partial_tail_capacity_bytes"], 4096)
            self.assertEqual(totals["partial_tail_slack_bytes"], 4096 - 256)
            self.assertEqual(totals["allocated_bytes"], 3032 + 4096 + 2 * TABLE_ENTRY_BYTES + base)
            self.assertEqual(totals["total_used_bytes"], used + 256 + 2 * TABLE_ENTRY_BYTES + base)
            self.assertEqual(totals["mode_counts"], [256, 0, 0])
            self.assertEqual(totals["extra_bytes"], 0)
            self.assertEqual(totals["attempted_bytes"], used)
            self.assertEqual(sum(w["accounting"]["base_table_bytes"] for w in view["windows"]), 0)

    def test_whole_raw_fallback_includes_header_and_table_no_slot_slack(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=16, exponent=200)
        self.save()
        totals = self.report()["streams"][0]["views"][0]["totals"]
        self.assertEqual(totals["raw_fallback_blocks"], 1)
        self.assertEqual(totals["raw_fallback_payload_bytes"], 4096)
        self.assertEqual(totals["raw_fallback_capacity_bytes"], 4096 + HEADER.size)
        self.assertEqual(totals["stored_used_bytes"], 4096 + HEADER.size)
        self.assertEqual(totals["allocated_bytes"], 4096 + HEADER.size + TABLE_ENTRY_BYTES + self.table_bytes())
        self.assertEqual(totals["headers_bytes"], HEADER.size)
        self.assertEqual(totals["compressed_slot_slack_bytes"], 0)
        self.assertEqual(totals["mode_counts"], [0, 0, 256])
        self.assertEqual(totals["roundtrip_verified_blocks"], 1)

    def test_actual_slot_capacity_exact_boundary(self):
        self.document["shards"][1]["rows"] = 16
        path = self.root / self.document["shards"][1]["path"]
        path.write_bytes(path.read_bytes()[:4096])
        self.save()
        attempted = 2048 + 768 + 64 + HEADER.size
        for capacity, raw in ((attempted - 1, True), (attempted, False), (attempted + 1, False), (0, True)):
            with self.subTest(capacity=capacity):
                totals = self.report("--slot-capacity", capacity)["streams"][0]["views"][0]["totals"]
                self.assertEqual(totals["raw_fallback_blocks"], int(raw))
                self.assertEqual(totals["compressed_slot_slack_bytes"], 0 if raw else capacity - attempted)

    def test_partial_tail_boundary_below_at_above_block(self):
        original = copy.deepcopy(self.document)
        for count in (15, 16, 17):
            with self.subTest(rows=count):
                self.document = copy.deepcopy(original)
                self.document["shards"].pop()
                self.add_shard("evaluation", rows=count)
                self.save()
                totals = self.report()["streams"][0]["views"][0]["totals"]
                self.assertEqual(totals["full_blocks"], count // 16)
                self.assertEqual(totals["partial_tail_blocks"], int(count % 16 != 0))
                self.assertEqual(totals["partial_tail_rows"], count % 16)
                self.assertEqual(totals["block_table_bytes"], TABLE_ENTRY_BYTES * ((count + 15) // 16))
                self.assertEqual(totals["headers_bytes"], HEADER.size * (count // 16))

    def test_absolute_window_boundaries_and_base_table_only_once(self):
        for boundary in (65536, 204800):
            with self.subTest(boundary=boundary):
                self.document["shards"][1]["position_start"] = boundary - 1
                self.document["shards"][1]["rows"] = 3
                path = self.root / self.document["shards"][1]["path"]
                path.write_bytes(row() * 3)
                self.save()
                view = next(v for v in self.report()["streams"][0]["views"] if v["window_tokens"] == boundary)
                self.assertEqual([w["position_start"] for w in view["windows"]], [0, boundary])
                self.assertEqual([w["accounting"]["rows"] for w in view["windows"]], [1, 2])
                totals = view["totals"]
                self.assertEqual(totals["partial_tail_blocks"], 2)
                self.assertEqual(totals["partial_tail_capacity_bytes"], 2 * 4096)
                self.assertEqual(totals["base_table_bytes"], self.table_bytes())
                self.assertEqual(totals["allocated_bytes"], 8192 + 2 * TABLE_ENTRY_BYTES + self.table_bytes())

    def test_adjacent_shards_merge_but_position_gaps_and_sequences_do_not(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=8, sequence="eval-a", position=0)
        continuation = self.add_shard("evaluation", rows=8, sequence="eval-a", position=8)
        self.save()
        totals = self.report()["streams"][0]["views"][0]["totals"]
        self.assertEqual((totals["full_blocks"], totals["partial_tail_blocks"]), (1, 0))
        continuation["position_start"] = 9
        self.save()
        totals = self.report()["streams"][0]["views"][0]["totals"]
        # The gap and absolute token-16 boundary make three incomplete fragments.
        self.assertEqual((totals["full_blocks"], totals["partial_tail_blocks"]), (0, 3))
        continuation["sequence_id"] = "eval-b"
        continuation["position_start"] = 0
        self.save()
        streams = self.report()["streams"]
        self.assertEqual({s["sequence_id"] for s in streams}, {"eval-a", "eval-b"})
        for stream in streams:
            totals = stream["views"][0]["totals"]
            self.assertEqual(totals["full_blocks"], 0)
            self.assertEqual(totals["partial_tail_capacity_bytes"], 4096)
            self.assertEqual(totals["base_table_bytes"], self.table_bytes())

    def test_partition_domain_kv_head_and_layer_reports_do_not_mix(self):
        self.document["shards"] = []
        partitions = [
            {"layer": 3, "head": 0, "kind": "K", "channel_start": 0, "channels": 128, "domain": "English"},
            {"layer": 7, "head": 1, "kind": "V", "channel_start": 128, "channels": 128, "domain": "code"},
            {"layer": 39, "head": 0, "kind": "K", "channel_start": 0, "channels": 256, "domain": "Ukrainian"}]
        for index, partition in enumerate(partitions):
            for role in ("calibration", "evaluation"):
                source = {**partition, "domain": "English"} if role == "calibration" else partition
                self.add_shard(role, sequence=f"{role}-{index}", **source)
        self.save()
        streams = self.report()["streams"]
        actual = {(s["layer"], s["head"], s["kind"], tuple(s["channel_slice"]), s["domain"]) for s in streams}
        expected = {(p["layer"], p["head"], p["kind"], (p["channel_start"], p["channel_start"] + p["channels"]), p["domain"].lower())
                    for p in partitions}
        self.assertEqual(actual, expected)
        self.assertEqual(len(streams), len(expected))

    def test_frozen_english_calibration_transfers_to_separate_code_and_ukrainian_streams(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", sequence="held-out-code", rows=16, domain="code", exponent=200)
        self.add_shard("evaluation", sequence="held-out-ukrainian", rows=17, domain="Ukrainian")
        self.save()
        report = self.report()
        self.assertEqual(report["calibration_domain"], "english")
        self.assertEqual(len(report["calibrations"]), 1)
        self.assertEqual(report["calibrations"][0]["rows"], 8)
        streams = {stream["domain"]: stream for stream in report["streams"]}
        self.assertEqual(set(streams), {"code", "ukrainian"})
        self.assertEqual({stream["sequence_id"] for stream in streams.values()}, {"held-out-code", "held-out-ukrainian"})
        for stream in streams.values():
            self.assertEqual(stream["calibration_domain"], "english")
            self.assertEqual(stream["views"][0]["totals"]["base_table_bytes"], self.table_bytes())
        code = streams["code"]["views"][0]["totals"]
        # Held-out exponent200 cannot use an oracle table fitted to evaluation:
        # English exponent100 calibration forces the actual whole raw fallback.
        self.assertEqual(code["raw_bytes"], 4096)
        self.assertEqual(code["raw_fallback_blocks"], 1)
        self.assertEqual(code["allocated_bytes"], 4096 + HEADER.size + TABLE_ENTRY_BYTES + self.table_bytes())
        ukrainian = streams["ukrainian"]["views"][0]["totals"]
        self.assertEqual(ukrainian["raw_bytes"], 17 * 256)
        self.assertEqual(ukrainian["compressed_blocks"], 1)
        self.assertEqual(ukrainian["partial_tail_rows"], 1)
        self.assertEqual(ukrainian["allocated_bytes"], 3032 + 4096 + 2 * TABLE_ENTRY_BYTES + self.table_bytes())

    def test_calibration_domain_is_explicit_with_no_extra_domain_or_missing_partition_fallback(self):
        self.document["shards"][0]["domain"] = "code"
        self.save()
        self.assertIn("unexpected calibration domain", self.cli(ok=False).stderr)
        report = self.report("--calibration-domain", "code")
        self.assertEqual(report["streams"][0]["calibration_domain"], "code")
        self.assertEqual(report["streams"][0]["domain"], "english")
        self.document["shards"][0]["domain"] = "Ukrainian"
        self.save()
        self.assertEqual(self.report("--calibration-domain", "ukrainian")["streams"][0]["calibration_domain"], "ukrainian")
        self.document["shards"][0]["domain"] = "English"
        self.add_shard("calibration", sequence="extra-code-cal", domain="code", rows=8)
        self.save()
        self.assertIn("unexpected calibration domain", self.cli(ok=False).stderr)
        self.document["shards"].pop()
        self.document["shards"][1]["head"] = 1
        self.save()
        self.assertIn("matching calibration", self.cli(ok=False).stderr)
        self.document["shards"][1]["head"] = 0
        self.add_shard("calibration", sequence="unused-cal", rows=8, head=1)
        self.save()
        self.assertIn("unused structural calibration", self.cli(ok=False).stderr)

    def test_sparse_near_200k_is_explicitly_a_sixteen_row_sample_not_full_context_allocation(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=16, position=204800)
        self.save()
        stream = self.report()["streams"][0]
        coverage = stream["coverage"]
        self.assertEqual(coverage["observed_rows"], 16)
        self.assertEqual(coverage["observed_range"], [204800, 204816])
        self.assertEqual(coverage["observed_span_tokens"], 16)
        self.assertEqual(coverage["gap_tokens_in_observed_span"], 0)
        self.assertEqual(coverage["sampling"], "contiguous_nonprefix_sample")
        self.assertFalse(coverage["prefix_from_zero"])
        for view in stream["views"]:
            totals = view["totals"]
            self.assertEqual(totals["raw_bytes"], 16 * 256)
            self.assertEqual(totals["allocated_bytes"], 3032 + TABLE_ENTRY_BYTES + self.table_bytes())
            window = view["windows"][0]["coverage"]
            self.assertFalse(window["complete_reporting_window"])
            self.assertEqual(window["observed_fraction_of_window"], 16 / view["window_tokens"])

    def test_prefix_gapped_coverage_and_complete_reporting_window_are_distinct(self):
        stream = self.report()["streams"][0]
        self.assertEqual(stream["coverage"]["sampling"], "contiguous_prefix")
        self.assertEqual(stream["coverage"]["observed_intervals"], [[0, 17]])
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=2, position=0)
        self.add_shard("evaluation", rows=2, position=32)
        self.save()
        coverage = self.report()["streams"][0]["coverage"]
        self.assertEqual(coverage["sampling"], "gapped_samples")
        self.assertFalse(coverage["contiguous"])
        self.assertFalse(coverage["prefix_from_zero"])
        self.assertEqual(coverage["observed_rows"], 4)
        self.assertEqual(coverage["observed_span_tokens"], 34)
        self.assertEqual(coverage["gap_tokens_in_observed_span"], 30)
        self.assertEqual(coverage["observed_intervals"], [[0, 2], [32, 34]])
        self.document["shards"] = self.document["shards"][:1]
        self.add_shard("evaluation", rows=16, position=0)
        self.save()
        views = self.report("--position-windows", "16", "32")["streams"][0]["views"]
        self.assertTrue(views[0]["windows"][0]["coverage"]["complete_reporting_window"])
        self.assertEqual(views[0]["windows"][0]["coverage"]["observed_fraction_of_window"], 1)
        self.assertFalse(views[1]["windows"][0]["coverage"]["complete_reporting_window"])
        self.assertEqual(views[1]["windows"][0]["coverage"]["observed_fraction_of_window"], 0.5)

    def test_absolute_block_grouping_does_not_change_with_reporting_window(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=16, position=65535)
        self.save()
        views = self.report()["streams"][0]["views"]
        self.assertEqual(views[0]["totals"], views[1]["totals"])
        self.assertEqual(views[0]["totals"]["partial_tail_blocks"], 2)
        self.assertEqual(views[0]["totals"]["full_blocks"], 0)
        self.assertEqual(len(views[0]["windows"]), 2)
        self.assertEqual(len(views[1]["windows"]), 1)
        for width, valid in ((15, False), (16, True), (17, False)):
            self.cli("--position-windows", width, ok=valid)

    def test_full_channel_geometry_warns_and_never_silently_scales_slot(self):
        self.document["shards"] = []
        self.add_shard("calibration", rows=8, channels=256)
        self.add_shard("evaluation", rows=16, channels=256)
        self.save()
        stream = self.report()["streams"][0]
        self.assertTrue(stream["geometry_warnings"])
        self.assertEqual(stream["views"][0]["totals"]["raw_fallback_blocks"], 1)
        self.assertEqual(stream["views"][0]["totals"]["raw_fallback_capacity_bytes"], 8192 + HEADER.size)
        scaled = self.report("--slot-capacity", "6040")["streams"][0]
        self.assertEqual(scaled["geometry_warnings"], [])
        self.assertEqual(scaled["views"][0]["totals"]["compressed_slot_capacity_bytes"], 6040)
        self.assertEqual(scaled["views"][0]["totals"]["compressed_slot_used_bytes"], 4096 + 1536 + 128 + HEADER.size)
        other_geometry = self.report("--block-tokens", "3", "--position-windows", "6")["streams"][0]
        self.assertTrue(other_geometry["geometry_warnings"])

    def test_leakage_uses_sequence_and_absolute_position_not_shard_names(self):
        self.document["shards"][0]["rows"] = 8
        self.document["shards"][1]["sequence_id"] = "cal-seq"
        for start, valid in ((7, False), (8, True), (9, True)):
            with self.subTest(start=start):
                self.document["shards"][1]["position_start"] = start
                self.save()
                result = self.cli(ok=valid)
                if not valid:
                    self.assertIn("leakage", result.stderr)
        # Disjoint layer/head/channel labels do not excuse reuse of sequence positions.
        self.document["shards"][1].update(position_start=0, layer=7, head=1, channel_start=128)
        self.save()
        self.assertIn("leakage", self.cli(ok=False).stderr)

    def test_overlapping_evaluation_partitions_rejected_but_tiling_allowed(self):
        first = self.document["shards"][1]
        self.add_shard("evaluation", rows=16, sequence="eval-seq", position=16)
        self.save()
        self.assertIn("overlapping", self.cli(ok=False).stderr)
        self.document["shards"].pop()
        self.add_shard("calibration", rows=8, sequence="cal-seq", channel_start=128)
        second = self.add_shard("evaluation", rows=17, sequence="eval-seq", channel_start=128)
        self.save()
        streams = self.report()["streams"]
        self.assertEqual({tuple(s["channel_slice"]) for s in streams}, {(0, 128), (128, 256)})
        second["channel_start"] = first["channel_start"]
        self.save()
        self.assertIn("overlapping", self.cli(ok=False).stderr)

    def test_dtype_stage_gdn_and_model_identity_cannot_be_inferred_or_overridden(self):
        original = copy.deepcopy(self.document)
        root_mutations = (("source_dtype", "float16"), ("source_dtype", "F16"), ("format", "uint16-be"),
                          ("state_kind", "gdn"), ("model_role", "draft"), ("capture_site", "before_cache_write"),
                          ("head_channels", 128), ("schema_version", True))
        for field, value in root_mutations:
            with self.subTest(field=field, value=value):
                self.document = copy.deepcopy(original)
                self.document[field] = value
                self.save()
                self.cli(ok=False)
        shard_mutations = (("source_dtype", "float16"), ("state_kind", "gdn"), ("capture_stage", "pre_rope"),
                           ("model_role", "draft"), ("capture_site", "before_cache_write"), ("kind", "GDN"),
                           ("model", {"id": "another/model", "revision": REVISION}))
        for field, value in shard_mutations:
            with self.subTest(shard_field=field, value=value):
                self.document = copy.deepcopy(original)
                self.document["shards"][0][field] = value
                self.save()
                self.cli(ok=False)
        for field, value in (("id", "another/model"), ("revision", "a" * 40), ("revision", "main")):
            self.document = copy.deepcopy(original)
            self.document["model"][field] = value
            self.save()
            self.cli(ok=False)

    def test_verified_layer_head_channel_and_position_boundaries(self):
        original = copy.deepcopy(self.document)
        invalid = (("layer", 2), ("layer", 4), ("layer", 38), ("layer", 40),
                   ("head", -1), ("head", 2), ("channel_start", 127), ("channel_start", 129),
                   ("channels", 127), ("channels", 129), ("channels", 0),
                   ("position_start", -1), ("position_start", True), ("rows", 0))
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                self.document = copy.deepcopy(original)
                self.document["shards"][1][field] = value
                self.save()
                self.cli(ok=False)
        self.document = copy.deepcopy(original)
        self.document["shards"][1]["rows"] = 1
        (self.root / self.document["shards"][1]["path"]).write_bytes(row())
        for position, valid in ((262142, True), (262143, True), (262144, False)):
            self.document["shards"][1]["position_start"] = position
            self.save()
            self.cli(ok=valid)

    def test_odd_truncated_extra_and_shape_invalid_files(self):
        path = self.root / self.document["shards"][1]["path"]
        original = path.read_bytes()
        for data in (original[:-1], original[:-2], original + b"\0\0", b""):
            with self.subTest(bytes=len(data)):
                path.write_bytes(data)
                self.cli(ok=False)
        path.write_bytes(original)
        self.document["shards"][1]["rows"] += 1
        self.save()
        self.cli(ok=False)

    def test_conflicting_domains_missing_calibration_and_reused_files_fail(self):
        original = copy.deepcopy(self.document)
        self.document["shards"][1].update(sequence_id="cal-seq", domain="code", position_start=100)
        self.save()
        self.assertIn("conflicting domains", self.cli(ok=False).stderr)
        self.document = copy.deepcopy(original)
        self.document["shards"][1].update(domain="code", head=1)
        self.save()
        self.assertIn("matching calibration", self.cli(ok=False).stderr)
        self.document = copy.deepcopy(original)
        self.document["shards"][1].update(path=self.document["shards"][0]["path"], rows=8)
        self.save()
        self.assertIn("same physical shard", self.cli(ok=False).stderr)
        self.document = copy.deepcopy(original)
        self.document["shards"][1]["capture_stage"] = "value_cache"
        self.save()
        self.assertIn("capture_stage", self.cli(ok=False).stderr)

    def test_multiple_manifests_combine_without_hiding_conflicts(self):
        calibration = copy.deepcopy(self.document)
        evaluation = copy.deepcopy(self.document)
        calibration["shards"] = calibration["shards"][:1]
        evaluation["shards"] = evaluation["shards"][1:]
        self.save(calibration)
        second = self.save(evaluation, self.root / "evaluation.json")
        report = json.loads(self.cli(manifests=[self.manifest, second]).stdout)
        self.assertEqual(len(report["input_manifests"]), 2)
        self.assertEqual(report["streams"][0]["views"][0]["totals"]["rows"], 17)
        evaluation["model"]["revision"] = "f" * 40
        self.save(evaluation, second)
        self.cli(manifests=[self.manifest, second], ok=False)

    def test_no_partial_report_overwrite_or_duplicate_json_keys(self):
        output = self.root / "report.json"
        self.cli("--output", output)
        previous = output.read_bytes()
        self.cli("--output", output, ok=False)
        self.assertEqual(output.read_bytes(), previous)
        raw = self.root / self.document["shards"][1]["path"]
        previous = raw.read_bytes()
        self.cli("--output", raw, ok=False)
        self.assertEqual(raw.read_bytes(), previous)
        self.manifest.write_text('{"schema_version": 1, "schema_version": 1}')
        self.assertIn("duplicate manifest key", self.cli(ok=False).stderr)

    def test_cli_rejects_invalid_limits_and_buffer_just_below_at_above_block(self):
        for args in (("--block-tokens", "0"), ("--slot-capacity", "-1"), ("--position-windows", "0"),
                     ("--position-windows", "65536", "65536"), ("--max-buffer-bytes", "0"),
                     ("--position-windows", "65535"), ("--position-windows", "65537"),
                     ("--calibration-domain", "unknown")):
            self.cli(*args, ok=False)
        for size, valid in ((4095, False), (4096, True), (4097, True)):
            with self.subTest(buffer=size):
                self.cli("--max-buffer-bytes", size, ok=valid)

    def test_raw_reads_are_bounded_even_when_file_is_larger_than_buffer(self):
        self.document["shards"].pop()
        self.add_shard("evaluation", rows=129)
        self.save()
        raw_paths = {self.root / shard["path"] for shard in self.document["shards"]}
        original_open = Path.open
        limit = 4096
        reads = []

        class GuardedReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def read(self, size=-1):
                if not 0 <= size <= limit:
                    raise AssertionError(f"unbounded raw read: {size}")
                reads.append(size)
                return self.stream.read(size)

        def guarded_open(path, *args, **kwargs):
            result = original_open(path, *args, **kwargs)
            return GuardedReader(result) if path in raw_paths else result

        with mock.patch.object(Path, "open", guarded_open):
            report = measurement.measure([self.manifest], buffer_bytes=limit)
        self.assertEqual(report["streams"][0]["views"][0]["totals"]["rows"], 129)
        self.assertTrue(reads)
        self.assertLessEqual(max(reads), limit)

    def test_changed_source_and_failed_bit_roundtrip_abort_measurement(self):
        _, shards, _ = measurement.load_inputs([self.manifest], block_tokens=16, buffer_bytes=4096)
        shard = shards[0]
        before = shard.path.stat()
        data = bytearray(shard.path.read_bytes())
        data[0] ^= 1
        shard.path.write_bytes(data)
        os.utime(shard.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(measurement.MeasurementError, "changed during"):
            list(shard.rows_iter())
        # Rehash on a fresh run, then ensure even plausible storage accounting is
        # never reported when the actual codec's decoded bits differ.
        with mock.patch.object(ReferenceCodec, "decode_block", return_value=b"wrong decoded bits"):
            with self.assertRaisesRegex(measurement.MeasurementError, "bit roundtrip"):
                measurement.measure([self.manifest])

    def test_extreme_bf16_bit_patterns_remain_bits_not_f16_values(self):
        words = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0x0001, 0xFFFF) * 16
        raw_row = struct.pack("<128H", *words)
        evaluation = self.document["shards"][1]
        evaluation["rows"] = 16
        raw = raw_row * 16
        (self.root / evaluation["path"]).write_bytes(raw)
        self.save()
        report = self.report()
        source = next(s for s in report["input_shards"] if s["role"] == "evaluation")
        self.assertEqual(source["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["streams"][0]["views"][0]["totals"]["roundtrip_verified_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
