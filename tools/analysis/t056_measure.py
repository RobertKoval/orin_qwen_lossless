#!/usr/bin/env python3
"""T056 paper-inspired reference measurement, NOT a GAAC reproduction.

CPU/stdlib only. No official GAAC wire layout has been established here. This measures the
local reference codec's actual storage plus explicitly modeled table/tail costs,
not hypothetical GPU allocations inferred from Python RSS.

Example manifest (each raw file contains contiguous token-major uint16-LE rows):
{
  "schema_version": 1,
  "model": {"id": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0"},
  "model_role": "target", "source_dtype": "bfloat16", "format": "uint16-le",
  "state_kind": "attention_kv", "capture_site": "after_cache_write", "head_channels": 256,
  "shards": [
    {"path": "cal-k.raw", "kind": "K", "capture_stage": "post_rope",
     "layer": 3, "head": 0, "channel_start": 0, "channels": 128,
     "domain": "English", "role": "calibration", "sequence_id": "cal-1",
     "position_start": 0, "rows": 32},
    {"path": "eval-k.raw", "kind": "K", "capture_stage": "post_rope",
     "layer": 3, "head": 0, "channel_start": 0, "channels": 128,
     "domain": "English", "role": "evaluation", "sequence_id": "eval-1",
     "position_start": 65536, "rows": 32}
  ]
}

Only target full-attention layers 3,7,...39, KV heads 0/1 and channel slices
0:128, 128:256 or 0:256 are supported. Draft/MTP and GDN are rejected. Positions
must lie in [0,262144). capture_site declares capture AFTER the actual cache write;
post-RoPE does not specify or infer the model's partial/interleaved rotation map.
V shards MUST declare capture_stage="value_cache". Domains are English, code,
and Ukrainian (case-insensitive). Paths resolve relative to their manifest.
Separate shard files can continue a stream. Blocks follow absolute token-block
boundaries, never crossing sequence gaps; window widths must be multiples of
block_tokens so reporting windows cannot change grouping. Incomplete observed
block fragments use raw full-block capacity. Window-size views are alternative,
NON-ADDITIVE presentations of the same samples, not full-context projections.
A stream is one layer/head/K-or-V/channel-slice/evaluation-domain/sequence, with
one base table charged per stream/view. Calibration uses ONLY the explicit
--calibration-domain (default english), matched by structural partition, never
evaluation rows. English calibration can evaluate code/Ukrainian transfer.
Positions are absolute, zero-based. Coverage distinguishes complete prefixes,
non-prefix samples and gaps; a 16-row sample near 200K is NOT a 200K allocation.

Default slot3032 is OUR 24B header + inferred 3008B payload for 16x128 ONLY.
A full 16x256 block needs a chosen slot6040 (24 + 2*3008), or separate 128-channel
shards; scaling is NEVER automatic. Other token geometries require an explicit
capacity decision. Geometry warnings are included in each stream report.

Dtype and capture stage CANNOT be inferred from 16-bit bytes. These declarations
require independent runtime provenance (including true BF16 dtype, RoPE/cache
placement, model revision, sequence identities and capture completeness). Optional
capture_provenance metadata is carried through, not authenticated. Renamed
sequences or false capture labels cannot be detected from raw shard bytes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

# The local stdlib codec owns component sizes; storage/used sizes below always
# come from each encoded block, not a duplicated estimate of the wire format.
if __package__:
    from .t056_reference import Calibration, HEADER, ReferenceCodec, TABLE_ENTRY_BYTES
else:
    from t056_reference import Calibration, HEADER, ReferenceCodec, TABLE_ENTRY_BYTES

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
FULL_ATTENTION_LAYERS = frozenset(range(3, 40, 4))
MAX_POSITION = 262144
HEADER_BYTES = HEADER.size
DEFAULT_WINDOWS = (65536, 204800)


class MeasurementError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MeasurementError(message)


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate manifest key: {key}")
        result[key] = value
    return result


def read_manifest(path):
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read manifest {path}: {exc}") from exc
    require(isinstance(value, dict), "manifest must be an object")
    return value


def snapshot(path):
    info = path.stat()
    require(path.is_file(), f"shard is not a regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def hash_file(path, buffer_bytes):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(buffer_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Shard:
    path: Path
    kind: str
    layer: int
    head: int
    channel_start: int
    channels: int
    domain: str
    role: str
    sequence_id: str
    position_start: int
    rows: int
    file_snapshot: tuple
    sha256: str

    @property
    def end(self):
        return self.position_start + self.rows

    @property
    def partition(self):
        return (self.layer, self.head, self.kind, self.channel_start, self.channels, self.domain)

    @property
    def structural_partition(self):
        return self.partition[:5]

    @property
    def stream_key(self):
        return (*self.partition, self.sequence_id)

    def rows_iter(self):
        require(snapshot(self.path) == self.file_snapshot, f"shard changed: {self.path}")
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            info = source.fileno()
            # Verify the opened inode too, not only a pathname checked before open.
            opened = os.fstat(info)
            require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) == self.file_snapshot,
                    f"shard changed while opening: {self.path}")
            for _ in range(self.rows):
                row = source.read(self.channels * 2)
                require(len(row) == self.channels * 2, f"truncated shard: {self.path}")
                digest.update(row)
                yield row
            require(not source.read(1), f"shape-invalid shard: {self.path}")
        require(snapshot(self.path) == self.file_snapshot and digest.hexdigest() == self.sha256,
                f"shard changed during measurement: {self.path}")


def partition_fields(key):
    layer, head, kind, start, channels, domain = key[:6]
    result = {"layer": layer, "head": head, "kind": kind,
              "channel_slice": [start, start + channels], "domain": domain}
    if len(key) == 7:
        result["sequence_id"] = key[6]
    return result


def load_inputs(paths, *, block_tokens, buffer_bytes, calibration_domain="english"):
    identity = None
    shards = []
    manifests = []
    seen_files = set()
    root_keys = {"schema_version", "model", "model_role", "source_dtype", "format", "state_kind", "capture_site",
                 "head_channels", "shards", "capture_provenance"}
    shard_keys = {"path", "kind", "capture_stage", "layer", "head", "channel_start", "channels", "domain", "role",
                  "sequence_id", "position_start", "rows", "model", "model_role", "source_dtype", "format", "state_kind",
                  "capture_site", "head_channels"}
    for manifest_path in paths:
        manifest_path = manifest_path.resolve(strict=True)
        document = read_manifest(manifest_path)
        require(not document.keys() - root_keys, f"unknown manifest fields: {sorted(document.keys() - root_keys)}")
        require(type(document.get("schema_version")) is int and document["schema_version"] == 1, "schema_version must be 1")
        model = document.get("model")
        require(isinstance(model, dict) and set(model) == {"id", "revision"}, "model must declare id and immutable revision")
        require(model["id"] == MODEL_ID, f"model.id must be {MODEL_ID}")
        require(model["revision"] == MODEL_REVISION, f"model.revision must be the verified checkpoint {MODEL_REVISION}")
        require(document.get("model_role") == "target", "only model_role=target is supported; draft/MTP must remain separate")
        require(document.get("capture_site") == "after_cache_write", "capture_site must declare after_cache_write")
        require(document.get("source_dtype") == "bfloat16", "source_dtype must be bfloat16; F16 is not BF16")
        require(document.get("format") == "uint16-le", "format must be uint16-le")
        require(document.get("state_kind") == "attention_kv", "state_kind must be attention_kv, never GDN/recurrent state")
        head_channels = integer(document.get("head_channels"), "head_channels", minimum=1)
        require(head_channels == 256, "verified target head_channels must be 256")
        current = {key: document[key] for key in ("model", "model_role", "source_dtype", "format", "state_kind", "capture_site", "head_channels")}
        require(identity is None or identity == current, "conflicting model/source identities across manifests")
        identity = current
        provenance = document.get("capture_provenance", {})
        require(isinstance(provenance, dict), "capture_provenance must be an object")
        manifests.append({"path": str(manifest_path), "capture_provenance": provenance})
        entries = document.get("shards")
        require(isinstance(entries, list) and entries, "manifest needs a nonempty shards list")
        for entry in entries:
            require(isinstance(entry, dict), "shard must be an object")
            require(not entry.keys() - shard_keys, f"unknown shard fields: {sorted(entry.keys() - shard_keys)}")
            for key, value in current.items():
                require(key not in entry or (type(entry[key]) is type(value) and entry[key] == value),
                        f"conflicting shard {key}")
            kind = entry.get("kind")
            require(kind in ("K", "V"), "kind must be K or V")
            require(entry.get("capture_stage") == ("post_rope" if kind == "K" else "value_cache"),
                    f"invalid capture_stage for {kind}")
            layer = integer(entry.get("layer"), "layer")
            head = integer(entry.get("head"), "head")
            require(layer in FULL_ATTENTION_LAYERS, "layer must be target full-attention layer 3,7,...39; GDN/MTP excluded")
            require(head in (0, 1), "target KV head must be 0 or 1, not a query head")
            start = integer(entry.get("channel_start"), "channel_start")
            channels = integer(entry.get("channels"), "channels", minimum=1)
            require((start, channels) in ((0, 128), (128, 128), (0, 256)),
                    "channel slice must be 0:128, 128:256 or full 0:256")
            require(start + channels <= head_channels, "channel slice exceeds head_channels")
            require(channels * 2 * block_tokens <= buffer_bytes, "full block exceeds --max-buffer-bytes")
            domain = entry.get("domain")
            require(isinstance(domain, str) and domain.lower() in ("english", "code", "ukrainian"), "unsupported domain")
            domain = domain.lower()
            role = entry.get("role")
            require(role in ("calibration", "evaluation"), "role must be calibration or evaluation")
            sequence = entry.get("sequence_id")
            require(isinstance(sequence, str) and bool(sequence.strip()), "sequence_id must be nonempty")
            position = integer(entry.get("position_start"), "position_start")
            rows = integer(entry.get("rows"), "rows", minimum=1)
            require(position + rows <= MAX_POSITION, "absolute positions exceed verified max_position 262144")
            raw_path = entry.get("path")
            require(isinstance(raw_path, str) and bool(raw_path), "shard path must be explicit")
            path = (manifest_path.parent / raw_path).resolve(strict=True)
            stamp = snapshot(path)
            require(stamp[2] % 2 == 0, f"odd byte count, not uint16-LE: {path}")
            require(stamp[2] == rows * channels * 2, f"shape-invalid/truncated shard: {path}")
            require(stamp[:2] not in seen_files, "same physical shard reused with duplicate/conflicting identities")
            seen_files.add(stamp[:2])
            digest = hash_file(path, buffer_bytes)
            require(snapshot(path) == stamp, f"shard changed while hashing: {path}")
            shards.append(Shard(path, kind, layer, head, start, channels, domain, role, sequence, position, rows, stamp, digest))
    require(shards, "no shards")
    validate_partitions(shards, calibration_domain=calibration_domain)
    return identity, shards, manifests


def validate_partitions(shards, *, calibration_domain):
    require(calibration_domain in ("english", "code", "ukrainian"), "unsupported calibration domain")
    by_sequence = defaultdict(list)
    for shard in shards:
        by_sequence[shard.sequence_id].append(shard)
    for sequence, entries in by_sequence.items():
        require(len({s.domain for s in entries}) == 1, f"conflicting domains for sequence {sequence}")
        calibration = sorted((s.position_start, s.end) for s in entries if s.role == "calibration")
        evaluation = sorted((s.position_start, s.end) for s in entries if s.role == "evaluation")
        i = j = 0
        while i < len(calibration) and j < len(evaluation):
            a, b = calibration[i], evaluation[j]
            require(not (a[0] < b[1] and b[0] < a[1]),
                    f"calibration/evaluation leakage at sequence {sequence}, position {max(a[0], b[0])}")
            if a[1] <= b[0]:
                i += 1
            else:
                j += 1
        # A sequence can appear across heads/layers/K/V, but the same state
        # elements cannot be assigned twice, even under different slice labels.
        grouped = defaultdict(list)
        for shard in entries:
            grouped[(shard.role, shard.layer, shard.head, shard.kind)].append(shard)
        for group in grouped.values():
            active = []
            for shard in sorted(group, key=lambda s: s.position_start):
                active = [s for s in active if s.end > shard.position_start]
                for other in active:
                    require(not (shard.channel_start < other.channel_start + other.channels
                                 and other.channel_start < shard.channel_start + shard.channels),
                            f"overlapping channel/position partition for sequence {sequence}")
                active.append(shard)
    calibration = [s for s in shards if s.role == "calibration"]
    require(all(s.domain == calibration_domain for s in calibration),
            f"unexpected calibration domain: all calibration shards must use {calibration_domain}")
    calibration_keys = {s.structural_partition for s in calibration}
    evaluation_keys = {s.structural_partition for s in shards if s.role == "evaluation"}
    require(calibration_keys and evaluation_keys, "both calibration and evaluation data are required")
    require(evaluation_keys <= calibration_keys, "each evaluated structural partition needs matching calibration shards")
    require(calibration_keys <= evaluation_keys, "unused structural calibration partition; no corresponding evaluation")


def empty_accounting():
    return {key: 0 for key in (
        "rows", "raw_bytes", "full_blocks", "stored_used_bytes", "stored_capacity_bytes", "headers_bytes",
        "block_table_bytes", "compressed_blocks", "compressed_slot_used_bytes", "compressed_slot_capacity_bytes",
        "compressed_slot_slack_bytes", "raw_fallback_blocks", "raw_fallback_payload_bytes", "raw_fallback_capacity_bytes",
        "partial_tail_blocks", "partial_tail_rows", "partial_tail_used_bytes", "partial_tail_capacity_bytes",
        "partial_tail_slack_bytes", "extra_bytes", "attempted_bytes", "roundtrip_verified_blocks") } | {"mode_counts": [0, 0, 0]}


def add_metrics(target, source):
    for key, value in source.items():
        if key == "mode_counts":
            target[key] = [a + b for a, b in zip(target[key], value)]
        else:
            target[key] += value


def full_block_metrics(codec, raw, *, block_tokens, channels, slot_capacity):
    block = codec.encode_block(raw)
    storage = block.storage
    require(isinstance(storage, bytes), "codec storage must be bytes")
    used = integer(block.used_bytes, "codec used_bytes", minimum=HEADER_BYTES)
    require(used <= len(storage), "codec used bytes exceed storage capacity")
    require(type(block.raw_fallback) is bool, "codec raw_fallback must be bool")
    expected_capacity = HEADER_BYTES + len(raw) if block.raw_fallback else slot_capacity
    require(len(storage) == expected_capacity, "codec storage violates declared slot/raw capacity")
    require(not block.raw_fallback or used == expected_capacity, "raw fallback must use exact header+raw storage")
    require(codec.decode_block(block) == raw, "codec bit roundtrip failed")
    modes = block.mode_counts
    require(isinstance(modes, tuple) and len(modes) == 3, "codec mode_counts must be a three-tuple")
    for count in modes:
        integer(count, "codec mode count")
    metrics = empty_accounting()
    metrics.update(rows=block_tokens, raw_bytes=len(raw), full_blocks=1, stored_used_bytes=used,
                   stored_capacity_bytes=len(storage), headers_bytes=HEADER_BYTES, block_table_bytes=TABLE_ENTRY_BYTES,
                   extra_bytes=integer(block.extra_bytes, "codec extra_bytes"),
                   attempted_bytes=integer(block.attempted_bytes, "codec attempted_bytes"),
                   roundtrip_verified_blocks=1, mode_counts=list(modes))
    if block.raw_fallback:
        metrics.update(raw_fallback_blocks=1, raw_fallback_payload_bytes=len(raw), raw_fallback_capacity_bytes=len(storage))
    else:
        metrics.update(compressed_blocks=1, compressed_slot_used_bytes=used,
                       compressed_slot_capacity_bytes=len(storage), compressed_slot_slack_bytes=len(storage) - used)
    return metrics


def tail_metrics(rows, *, block_tokens, channels):
    used = rows * channels * 2
    capacity = block_tokens * channels * 2
    result = empty_accounting()
    result.update(rows=rows, raw_bytes=used, stored_used_bytes=used, stored_capacity_bytes=capacity,
                  block_table_bytes=TABLE_ENTRY_BYTES, partial_tail_blocks=1, partial_tail_rows=rows,
                  partial_tail_used_bytes=used, partial_tail_capacity_bytes=capacity,
                  partial_tail_slack_bytes=capacity - used)
    return result


def finish_accounting(metrics, base_table_bytes=0):
    result = dict(metrics)
    result["base_table_bytes"] = base_table_bytes
    result["total_used_bytes"] = metrics["stored_used_bytes"] + metrics["block_table_bytes"] + base_table_bytes
    result["allocated_bytes"] = metrics["stored_capacity_bytes"] + metrics["block_table_bytes"] + base_table_bytes
    result["raw_to_allocated_ratio"] = metrics["raw_bytes"] / result["allocated_bytes"] if result["allocated_bytes"] else None
    return result


def observed_coverage(shards, *, window=None):
    """Describe only observed positions, never extrapolate a context allocation."""
    intervals = []
    for shard in shards:
        start, end = shard.position_start, shard.end
        if window is not None:
            start, end = max(start, window[0]), min(end, window[1])
        if start < end:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    require(merged, "coverage requires observed positions")
    rows = sum(end - start for start, end in merged)
    start, end = merged[0][0], merged[-1][1]
    contiguous = len(merged) == 1
    prefix = contiguous and start == 0
    result = {"observed_rows": rows, "observed_range": [start, end], "observed_span_tokens": end - start,
              "gap_tokens_in_observed_span": end - start - rows,
              "observed_intervals": merged, "contiguous": contiguous, "prefix_from_zero": prefix,
              "sampling": "contiguous_prefix" if prefix else "contiguous_nonprefix_sample" if contiguous else "gapped_samples",
              "allocation_scope": "observed_samples_only_not_full_context_or_gpu_kv"}
    if window is not None:
        result.update(reporting_window_range=list(window), reporting_window_tokens=window[1] - window[0],
                      observed_fraction_of_window=rows / (window[1] - window[0]),
                      complete_reporting_window=contiguous and [start, end] == list(window))
    return result


def geometry_warnings(channels, block_tokens, slot_capacity):
    if slot_capacity != 3032 or (block_tokens, channels) == (16, 128):
        return []
    if (block_tokens, channels) == (16, 256):
        return ["slot3032 is the 16x128 default, not a 16x256 slot; explicitly choose slot6040 (24+2*3008) or separate 128-channel shards; no automatic scaling"]
    return ["slot3032 is the 16x128 default; this token/channel geometry needs an explicit capacity decision; no automatic scaling"]


def measure_view(shards, codec, *, window_tokens, block_tokens, slot_capacity, channels, base_table_bytes):
    windows = {}
    pending = bytearray()
    pending_rows = 0
    current_window = None
    current_block = None
    previous_position = None

    def flush_tail():
        nonlocal pending_rows
        if pending_rows:
            add_metrics(windows[current_window], tail_metrics(pending_rows, block_tokens=block_tokens, channels=channels))
            pending.clear()
            pending_rows = 0

    for shard in sorted(shards, key=lambda s: s.position_start):
        for index, row in enumerate(shard.rows_iter()):
            position = shard.position_start + index
            window = position // window_tokens * window_tokens
            block_index = position // block_tokens
            if current_block != block_index or (previous_position is not None and position != previous_position + 1):
                flush_tail()
            current_block = block_index
            current_window = window
            previous_position = position
            if window not in windows:
                windows[window] = empty_accounting()
            pending.extend(row)
            pending_rows += 1
            if pending_rows == block_tokens:
                add_metrics(windows[window], full_block_metrics(codec, bytes(pending), block_tokens=block_tokens,
                                                               channels=channels, slot_capacity=slot_capacity))
                pending.clear()
                pending_rows = 0
    flush_tail()
    total = empty_accounting()
    reports = []
    for start, metrics in sorted(windows.items()):
        add_metrics(total, metrics)
        reports.append({"position_start": start, "position_end_exclusive": start + window_tokens,
                        "coverage": observed_coverage(shards, window=(start, start + window_tokens)),
                        "accounting": finish_accounting(metrics)})
    return {"window_tokens": window_tokens, "base_table_bytes": base_table_bytes,
            "coverage": observed_coverage(shards),
            "totals": finish_accounting(total, base_table_bytes), "windows": reports}


def measure(paths, *, block_tokens=16, slot_capacity=3032, windows=DEFAULT_WINDOWS, buffer_bytes=1024 * 1024,
            calibration_domain="english"):
    integer(block_tokens, "block_tokens", minimum=1)
    integer(slot_capacity, "slot_capacity")
    integer(buffer_bytes, "max-buffer-bytes", minimum=1)
    require(slot_capacity <= buffer_bytes, "slot_capacity exceeds --max-buffer-bytes")
    require(windows and len(windows) == len(set(windows)), "position windows must be nonempty and unique")
    for window in windows:
        integer(window, "position window", minimum=1)
        require(window % block_tokens == 0, "position windows must be multiples of block_tokens")
    identity, shards, manifests = load_inputs(paths, block_tokens=block_tokens, buffer_bytes=buffer_bytes,
                                            calibration_domain=calibration_domain)
    calibration_shards = defaultdict(list)
    evaluation_shards = defaultdict(list)
    for shard in shards:
        (calibration_shards[shard.structural_partition] if shard.role == "calibration" else evaluation_shards[shard.stream_key]).append(shard)
    calibrations = {}
    summaries = []
    for key, entries in sorted(calibration_shards.items()):
        count = 0

        def rows_iter():
            nonlocal count
            for shard in sorted(entries, key=lambda s: (s.sequence_id, s.position_start)):
                for row in shard.rows_iter():
                    count += 1
                    yield row
        calibration = Calibration.from_rows(rows_iter(), channels=key[4])
        require(count == sum(s.rows for s in entries), "calibration did not consume all declared rows")
        require(calibration.channels == key[4], "calibration channel count mismatch")
        table = calibration.to_bytes()
        require(isinstance(table, bytes), "calibration table must serialize to bytes")
        calibrations[key] = (calibration, len(table))
        sequences = defaultdict(list)
        for shard in entries:
            sequences[shard.sequence_id].append(shard)
        summaries.append({**partition_fields((*key, calibration_domain)), "role": "calibration", "rows": count,
                          "calibration_domain": calibration_domain,
                          "sequences": [{"sequence_id": sequence, "coverage": observed_coverage(parts)}
                                        for sequence, parts in sorted(sequences.items())],
                          "base_table_bytes": len(table), "base_table_sha256": hashlib.sha256(table).hexdigest()})
    reports = []
    for key, entries in sorted(evaluation_shards.items()):
        calibration, table_bytes = calibrations[key[:5]]
        views = []
        for window in windows:
            codec = ReferenceCodec(calibration, block_tokens=block_tokens, slot_capacity=slot_capacity)
            views.append(measure_view(entries, codec, window_tokens=window, block_tokens=block_tokens,
                                      slot_capacity=slot_capacity, channels=key[4], base_table_bytes=table_bytes))
        reports.append({**partition_fields(key), "role": "evaluation", "calibration_domain": calibration_domain,
                        "coverage": observed_coverage(entries),
                        "geometry_warnings": geometry_warnings(key[4], block_tokens, slot_capacity), "views": views})
    return {"schema_version": 1, "measurement": "T056 paper-inspired reference, NOT GAAC reproduction",
            "identity": identity, "capture_provenance_status": "declarations_only_runtime_provenance_required",
            "calibration_domain": calibration_domain,
            "input_manifests": manifests,
            "input_shards": [{**partition_fields(s.stream_key), "role": s.role, "path": str(s.path),
                              "position_start": s.position_start, "rows": s.rows, "raw_bytes": s.file_snapshot[2],
                              "sha256": s.sha256} for s in shards],
            "accounting_contract": {"header_bytes": HEADER_BYTES, "block_table_entry_bytes": TABLE_ENTRY_BYTES,
                                    "block_tokens": block_tokens, "compressed_slot_capacity_bytes": slot_capacity,
                                    "compressed_slot_includes_header": True,
                                    "paper_payload_slot_inference_bytes": 3008,
                                    "default_slot_geometry": {"block_tokens": 16, "channels": 128, "slot_capacity": 3032},
                                    "full_256_slot_at_16_tokens": 6040,
                                    "slot_scaling": "never automatic; 6040 is OUR 24B header + 2*inferred3008 payload",
                                    "partial_tail": "raw full-block capacity per incomplete observed block fragment, including unaligned starts/gaps; actual raw used rows, no header, one table entry",
                                    "base_table": "once per sequence stream per window-size view, excluded from individual windows",
                                    "window_views": "non-additive views of identical absolute-token-block accounting; widths must be multiples of block_tokens",
                                    "allocation_scope": "observed samples only, never extrapolated to a full 64K/200K context or GPU KV allocation",
                                    "extra_bytes": "actual compressed group extras already included in storage; zero on raw fallback; not added again",
                                    "mode_counts": "attempted 3/4/raw whole-group modes, including attempts that caused full raw fallback",
                                    "allocation_basis": "reference storage capacity plus explicit table/tail model, NOT Python RSS"},
            "calibrations": summaries, "streams": reports}


def nonnegative(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return result


def positive(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+", help="explicit JSON shard manifests")
    parser.add_argument("--output", type=Path, help="create a new JSON report (default stdout; never overwrites)")
    parser.add_argument("--block-tokens", type=positive, default=16)
    parser.add_argument("--slot-capacity", type=nonnegative, default=3032, help="OUR padded compressed slot including 24-byte header")
    parser.add_argument("--position-windows", type=positive, nargs="+", default=list(DEFAULT_WINDOWS),
                        help="absolute-position bin widths; each must be a multiple of block-tokens")
    parser.add_argument("--calibration-domain", choices=("english", "code", "ukrainian"), default="english",
                        help="use ONLY this calibration domain for every evaluation domain; no oracle fallback")
    parser.add_argument("--max-buffer-bytes", type=positive, default=1024 * 1024, help="bound raw block, read chunk and slot size")
    args = parser.parse_args(argv)
    try:
        report = measure(args.manifests, block_tokens=args.block_tokens, slot_capacity=args.slot_capacity,
                         windows=args.position_windows, buffer_bytes=args.max_buffer_bytes,
                         calibration_domain=args.calibration_domain)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as target:
                target.write(text)
        else:
            print(text, end="")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"t056 measurement: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
