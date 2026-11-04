"""Bounded dev-only disk sink for ALREADY COPIED BF16 uint16-LE chunks.

No tensor, model, CUDA, RPC, or package code is imported. Bytes cannot establish
BF16 dtype, post-RoPE placement, or a physical page mapping: the dev caller must
supply honest runtime provenance. This writer validates declarations, logical
order, page bounds, quotas and actual file hashes, not the runtime itself.

API: CaptureWriter(...).start_shard(descriptor) -> str;
add_chunk(id, position_start=..., rows=..., data=bytes, source_dtype="bfloat16",
route=None|{physical_page, page_offset, kernel_page_tokens, position_start, rows,
            token_ids, token_ids_sha256}); finalize(check_deadline=lambda: None) -> Path.
The optional deadline callback is cooperative: checked before finalization,
after each data-file verification, and immediately before atomic publication.
Calls must be serialized. No raw chunk is retained after add_chunk returns.

Descriptor fields: layer (3/39), head (0/1), kind (K/V), channel_start (0/128),
channels (128), capture_stage (post_rope/value_cache), position_start, rows,
domain (english/code/ukrainian), role (calibration/evaluation), sequence_id,
request_id, committed_tokens and request_provenance. request_provenance is:
{model_id, checkpoint_revision, request_id, sequence_id, source_id,
 prompt_token_ids_sha256, source_file_sha256, prompt_tokens, committed_tokens,
 before_computed, completed_step, role, domain, source_dtype: "bfloat16",
 capture_site: "after_cache_write", runtime_verified: true, ...bounded extras}.
The committed cursor must equal the full prompt length: this pilot never captures
sampled/generated tokens. Hashes/IDs are caller receipts, not inferred from KV
bytes. token_ids_sha256(ids) hashes uint32-LE tokens
incrementally; per-chunk route digests use this same convention. Constructor
capture_provenance may add bounded JSON fields such as requests/source_api_sha256
but cannot override writer-owned status, hashes, limits or routing declarations.

Generated manifest.json uses the measurement schema. Status, per-file SHA-256,
request provenance and logical/page routes live in capture_provenance, leaving
its measure-compatible shard entries unchanged. No manifest is published before
ALL declared rows are written, closed, and hash-verified. Validation errors leave
the writer retryable; disk failures or corruption poison it. Incomplete outputs
are deliberately retained WITHOUT a complete manifest for operator inspection.

metadata_reserve_bytes is a hard serialized-metadata budget reserved from the
disk quota before any raw data is written. max_host_bytes bounds one incoming
chunk plus serialized metadata, NOT Python RSS, object overhead, or GPU memory.
Private directory/file modes are 0700/0600; paths are generated, never caller IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import math

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MAX_POSITION = 262144
ROW_BYTES = 128 * 2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    _require(type(value) is int and minimum <= value <= maximum,
             f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _identifier(value, name):
    _require(isinstance(value, str) and bool(value.strip()) and value == value.strip() and len(value) <= 1024,
             f"{name} must be a nonempty, trimmed identifier of at most 1024 characters")
    return value


def _sha256(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{name} must be lowercase SHA-256")
    return value


def token_ids_sha256(token_ids):
    """SHA-256 over uint32-LE token IDs; repeated values are meaningful and valid."""
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(struct.pack("<I", _integer(token, "token_id", maximum=2**32 - 1)))
    return digest.hexdigest()


def _metadata_copy(value, limit):
    """Validate/copy bounded JSON before serialization; no arbitrary object hooks."""
    remaining = limit

    def charge(count):
        nonlocal remaining
        remaining -= count
        _require(remaining >= 0, "caller capture_provenance exceeds metadata/host budget")

    def string(value):
        charge(2)
        for char in value:
            number = ord(char)
            charge(2 if number in (34, 92, 8, 12, 10, 13, 9)
                   else 6 if number < 32 or 127 <= number <= 65535 else 12 if number > 65535 else 1)
        return value

    def copy(value, depth=0):
        _require(depth <= 20, "capture_provenance nesting exceeds 20 levels")
        if value is None:
            charge(4)
        elif type(value) is bool:
            charge(4 if value else 5)
        elif type(value) is int:
            _require(abs(value) <= 2**63 - 1, "capture_provenance integer out of range")
            charge(len(str(value)))
        elif type(value) is float:
            _require(math.isfinite(value), "nonfinite capture_provenance number")
            charge(len(str(value)))
        elif type(value) is str:
            return string(value)
        elif type(value) is list:
            charge(2 + max(0, len(value) - 1))
            _require(len(value) <= remaining, "capture_provenance list exceeds budget")
            return [copy(item, depth + 1) for item in value]
        elif type(value) is dict:
            charge(2 + max(0, len(value) - 1))
            _require(len(value) <= remaining, "capture_provenance object exceeds budget")
            result = {}
            for key, item in value.items():
                _require(type(key) is str, "capture_provenance keys must be strings")
                string(key)
                charge(1)
                result[key] = copy(item, depth + 1)
            return result
        else:
            raise ValueError("capture_provenance must contain only JSON data")
        return value
    return copy(value)


def _request_provenance(value, measured, request_id, committed, metadata_limit):
    keys = {"model_id", "checkpoint_revision", "request_id", "sequence_id", "source_id", "prompt_token_ids_sha256",
            "source_file_sha256", "prompt_tokens", "committed_tokens", "before_computed", "completed_step", "role", "domain",
            "source_dtype", "capture_site", "runtime_verified"}
    _require(type(value) is dict and keys <= value.keys(), "request_provenance is missing scheduler receipt fields")
    _require(value["model_id"] == MODEL_ID and value["checkpoint_revision"] == MODEL_REVISION, "conflicting model/checkpoint provenance")
    _require(value["request_id"] == request_id, "request_id conflicts with request provenance")
    for key in ("sequence_id", "role", "domain"):
        _require(value[key] == measured[key], f"{key} conflicts with request provenance")
    _identifier(value["source_id"], "source_id")
    _sha256(value["prompt_token_ids_sha256"], "prompt_token_ids_sha256")
    _sha256(value["source_file_sha256"], "source_file_sha256")
    prompt = _integer(value["prompt_tokens"], "prompt_tokens", minimum=1, maximum=MAX_POSITION)
    cursor = _integer(value["committed_tokens"], "provenance committed_tokens", minimum=1, maximum=MAX_POSITION)
    _require(cursor == committed == prompt, "pilot requires the completed prompt only; committed bounds conflict")
    before = _integer(value["before_computed"], "before_computed", maximum=MAX_POSITION)
    _require(before < committed, "completed-step receipt must advance the committed cursor")
    _integer(value["completed_step"], "completed_step", minimum=1, maximum=MAX_POSITION)
    _require(value["runtime_verified"] is True and value["source_dtype"] == "bfloat16"
             and value["capture_site"] == "after_cache_write", "runtime receipt must verify BF16 after_cache_write")
    return _metadata_copy(value, metadata_limit)


def _descriptor(value, metadata_limit):
    keys = {"layer", "head", "kind", "channel_start", "channels", "capture_stage", "position_start", "rows",
            "domain", "role", "sequence_id", "request_id", "committed_tokens", "request_provenance"}
    _require(isinstance(value, dict) and value.keys() == keys, "descriptor fields do not match the capture contract")
    layer = _integer(value["layer"], "layer")
    head = _integer(value["head"], "head")
    channel = _integer(value["channel_start"], "channel_start")
    channels = _integer(value["channels"], "channels")
    _require(layer in range(3, 40, 4), "pilot layer must be a full-attention layer 3,7,...,39, never GDN/MTP")
    _require(head in (0, 1), "KV head must be 0 or 1")
    _require(channel in (0, 128) and channels == 128, "pilot channel slice must be 0:128 or 128:256")
    kind = value["kind"]
    _require(kind in ("K", "V"), "kind must be K or V")
    _require(value["capture_stage"] == ("post_rope" if kind == "K" else "value_cache"), "wrong K/V capture_stage")
    start = _integer(value["position_start"], "position_start", maximum=MAX_POSITION - 1)
    rows = _integer(value["rows"], "rows", minimum=1, maximum=MAX_POSITION)
    committed = _integer(value["committed_tokens"], "committed_tokens", minimum=1, maximum=MAX_POSITION)
    _require(start + rows <= committed, "declared shard exceeds committed_tokens upper bound")
    _require(value["domain"] in ("english", "code", "ukrainian"), "unsupported domain")
    _require(value["role"] in ("calibration", "evaluation"), "role must be calibration or evaluation")
    sequence = _identifier(value["sequence_id"], "sequence_id")
    request = _identifier(value["request_id"], "request_id")
    measured = {key: value[key] for key in ("layer", "head", "kind", "channel_start", "channels", "capture_stage",
                                            "position_start", "rows", "domain", "role")}
    measured["sequence_id"] = sequence
    provenance = _request_provenance(value["request_provenance"], measured, request, committed, metadata_limit)
    return measured, request, committed, provenance


def _route(value, position, rows):
    if value is None:
        return None
    required = {"physical_page", "page_offset", "kernel_page_tokens", "position_start", "rows", "token_ids", "token_ids_sha256"}
    deltas = {"device_allocated_delta", "device_reserved_delta"}
    _require(type(value) is dict and required <= value.keys() <= required | deltas,
             "route must declare page bounds, logical position/count, token_ids and token_ids_sha256")
    _require(not value.keys() & deltas or deltas <= value.keys(), "both device allocation deltas are required when supplied")
    allocation = {name: _integer(value[name], name, minimum=-(2**63), maximum=64 * 1024**2)
                  for name in deltas if name in value}
    _require(_integer(value["position_start"], "route position_start") == position
             and _integer(value["rows"], "route rows", minimum=1) == rows, "route position/count conflicts with chunk")
    ids = value["token_ids"]
    _require(type(ids) is list and len(ids) == rows, "route token_ids count must equal rows")
    digest = _sha256(value["token_ids_sha256"], "token_ids_sha256")
    _require(token_ids_sha256(ids) == digest, "route token_ids_sha256 mismatch")
    page = _integer(value["physical_page"], "physical_page")
    offset = _integer(value["page_offset"], "page_offset", maximum=MAX_POSITION - 1)
    tokens = _integer(value["kernel_page_tokens"], "kernel_page_tokens", minimum=1, maximum=MAX_POSITION)
    _require(offset + rows <= tokens, "chunk crosses physical page bounds")
    return {"physical_page": page, "page_offset": offset, "kernel_page_tokens": tokens,
            "position_start": position, "rows": rows, "token_ids": list(ids), "token_ids_sha256": digest, **allocation}


@dataclass
class _Shard:
    id: str
    descriptor: dict
    request_id: str
    committed_tokens: int
    provenance: dict
    inode: tuple | None = None
    written_rows: int = 0
    chunks: list = field(default_factory=list)
    digest: object = field(default_factory=hashlib.sha256)

    @property
    def filename(self):
        return self.id + ".raw"


class CaptureWriter:
    def __init__(self, output_dir, *, max_output_bytes=8 * 1024**2, metadata_reserve_bytes=1024**2,
                 max_chunk_rows=256, max_host_bytes=16 * 1024**2, capture_provenance=None):
        self.max_output_bytes = _integer(max_output_bytes, "max_output_bytes", minimum=1)
        self.metadata_reserve_bytes = _integer(metadata_reserve_bytes, "metadata_reserve_bytes", minimum=1)
        self.max_chunk_rows = _integer(max_chunk_rows, "max_chunk_rows", minimum=1, maximum=256)
        self.max_host_bytes = _integer(max_host_bytes, "max_host_bytes", minimum=1)
        _require(metadata_reserve_bytes < max_output_bytes, "metadata reserve must leave room for raw data")
        self._shards = {}
        self._data_bytes = 0
        self._metadata_bytes = 0
        self._status = "open"
        self._failure = None
        caller = {} if capture_provenance is None else capture_provenance
        _require(type(caller) is dict, "capture_provenance must be an object")
        reserved = {"status", "writer", "runtime_verification", "route_verification", "data_bytes", "limits", "shards"}
        _require(not caller.keys() & reserved, "capture_provenance cannot override writer-owned fields")
        self._capture_provenance = _metadata_copy(caller, min(metadata_reserve_bytes, max_host_bytes))
        if "source_api_sha256" in self._capture_provenance:
            hashes = self._capture_provenance["source_api_sha256"]
            _require(type(hashes) is dict, "source_api_sha256 must be a mapping")
            for digest in hashes.values():
                _sha256(digest, "source API digest")
        path = Path(output_dir).absolute()
        # Check the requested name BEFORE resolving ancestors: a dangling output
        # symlink must never turn into permission to create its target.
        if os.path.lexists(path):
            raise FileExistsError(f"output directory already exists: {path}")
        self.output_dir = path.parent.resolve(strict=True) / path.name
        self._preflight(list(self._shards.values()))
        self.output_dir.mkdir(mode=0o700)
        self._directory_inode = self._inode(self.output_dir.stat())

    @staticmethod
    def _inode(info):
        return info.st_dev, info.st_ino

    @property
    def bytes_written(self):
        return self._data_bytes + self._metadata_bytes

    @property
    def report(self):
        return {"status": self._status, "data_bytes": self._data_bytes, "metadata_bytes": self._metadata_bytes,
                "bytes_written": self.bytes_written, "shards": len(self._shards),
                "manifest": str(self.output_dir / "manifest.json") if self._status == "complete" else None,
                "failure": self._failure}

    def _open(self):
        _require(self._status == "open", f"writer is {self._status}; cannot modify it")
        info = self.output_dir.lstat()
        _require(stat.S_ISDIR(info.st_mode) and self._inode(info) == self._directory_inode,
                 "capture output directory identity changed")
        # A private, exclusive directory has exactly the files we created. Never
        # omit foreign bytes from the quota or append after a declared file grew.
        expected = {shard.filename: shard for shard in self._shards.values()}
        for path in self.output_dir.iterdir():
            if path.name not in expected:
                error = FileExistsError(f"unexpected capture file; refusing to overwrite or ignore it: {path.name}")
                self._fail(error)
                raise error
            shard = expected.pop(path.name)
            info = path.lstat()
            if not (stat.S_ISREG(info.st_mode) and self._inode(info) == shard.inode
                    and info.st_size == shard.written_rows * ROW_BYTES):
                self._fail("capture file identity/size changed")
                raise ValueError(self._failure)
        if expected:
            self._fail("declared capture file is missing")
            raise ValueError(self._failure)

    def _manifest_bytes(self, shards, *, pending=None):
        entries = []
        provenance = {}
        for shard in shards:
            entries.append({"path": shard.filename, **shard.descriptor})
            chunks = shard.chunks
            written = shard.written_rows
            if pending is not None and pending[0] == shard.id:
                chunks = [*chunks, pending[1]]
                written += pending[1]["rows"]
            provenance[shard.id] = {"path": shard.filename, "sha256": shard.digest.hexdigest(),
                                    "request_id": shard.request_id, "committed_tokens": shard.committed_tokens,
                                    "request_provenance": shard.provenance, "written_rows": written, "chunks": chunks}
        document = {"schema_version": 1, "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                    "model_role": "target", "source_dtype": "bfloat16", "format": "uint16-le",
                    "state_kind": "attention_kv", "capture_site": "after_cache_write", "head_channels": 256,
                    "shards": entries,
                    "capture_provenance": {**self._capture_provenance,
                                           "status": "complete", "writer": "T056 dev-only copied-chunk disk writer",
                                           "runtime_verification": "caller declarations, not independently attested by this byte writer",
                                           "route_verification": "logical order and page bounds only; caller supplies physical mapping",
                                           "data_bytes": sum(s.descriptor["rows"] * ROW_BYTES for s in shards),
                                           "limits": {"max_output_bytes": self.max_output_bytes,
                                                      "metadata_reserve_bytes": self.metadata_reserve_bytes,
                                                      "max_chunk_rows": self.max_chunk_rows,
                                                      "max_host_bytes": self.max_host_bytes}, "shards": provenance}}
        return (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

    def _preflight(self, shards, *, pending=None, chunk_bytes=0):
        planned = sum(s.descriptor["rows"] * ROW_BYTES for s in shards)
        _require(planned <= self.max_output_bytes - self.metadata_reserve_bytes, "raw data quota exceeds reserved output budget")
        metadata = self._manifest_bytes(shards, pending=pending)
        _require(len(metadata) <= self.metadata_reserve_bytes, "metadata reserve quota exceeded")
        _require(len(metadata) + chunk_bytes <= self.max_host_bytes, "chunk plus serialized metadata exceeds max_host_bytes")
        _require(planned + len(metadata) <= self.max_output_bytes, "actual data plus metadata exceeds output quota")
        return metadata

    def _validate_identity(self, candidate):
        d, p = candidate.descriptor, candidate.provenance
        identity = (candidate.request_id, p["source_id"], p["prompt_token_ids_sha256"], p["source_file_sha256"],
                    p["prompt_tokens"], d["domain"], d["role"])
        for other in self._shards.values():
            old, q = other.descriptor, other.provenance
            previous_identity = (other.request_id, q["source_id"], q["prompt_token_ids_sha256"], q["source_file_sha256"],
                                 q["prompt_tokens"], old["domain"], old["role"])
            if old["sequence_id"] == d["sequence_id"]:
                _require(identity == previous_identity, "conflicting sequence/request identity or calibration/evaluation reuse")
            if candidate.request_id == other.request_id:
                _require(d["sequence_id"] == old["sequence_id"] and identity == previous_identity, "conflicting request identity")
                if p["completed_step"] == q["completed_step"]:
                    _require(candidate.committed_tokens == other.committed_tokens, "conflicting committed_tokens at one completed step")
            if d["role"] != old["role"]:
                reused = (candidate.request_id == other.request_id or p["prompt_token_ids_sha256"] == q["prompt_token_ids_sha256"]
                          or p["source_id"] == q["source_id"] or p["source_file_sha256"] == q["source_file_sha256"])
                _require(not reused, "calibration/evaluation dataset identity leakage, even with renamed sequence IDs")
            same_partition = all(d[key] == old[key] for key in ("sequence_id", "layer", "head", "kind", "channel_start"))
            if same_partition:
                _require(d["position_start"] + d["rows"] <= old["position_start"]
                         or old["position_start"] + old["rows"] <= d["position_start"], "overlapping logical shard positions")

    def start_shard(self, descriptor):
        self._open()
        measured, request, committed, provenance = _descriptor(descriptor, min(self.metadata_reserve_bytes, self.max_host_bytes))
        shard = _Shard(f"shard-{len(self._shards):06d}", measured, request, committed, provenance)
        self._validate_identity(shard)
        self._preflight([*self._shards.values(), shard])
        path = self.output_dir / shard.filename
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                shard.inode = self._inode(os.fstat(fd))
            finally:
                os.close(fd)
        except OSError as exc:
            self._fail(exc)
            raise
        self._shards[shard.id] = shard
        return shard.id

    def _fail(self, error):
        self._status = "failed"
        self._failure = str(error)

    def add_chunk(self, shard_id, *, position_start, rows, data: bytes, source_dtype="bfloat16", route=None):
        self._open()
        _require(isinstance(shard_id, str) and shard_id in self._shards, "unknown shard_id")
        shard = self._shards[shard_id]
        _require(source_dtype == "bfloat16", "source_dtype must be bfloat16; byte width cannot prove dtype")
        count = _integer(rows, "chunk rows", minimum=1, maximum=self.max_chunk_rows)
        position = _integer(position_start, "chunk position_start", maximum=MAX_POSITION - 1)
        _require(position == shard.descriptor["position_start"] + shard.written_rows, "chunk order must be contiguous with no gaps/duplicates")
        _require(shard.written_rows + count <= shard.descriptor["rows"], "chunk exceeds declared shard rows/committed bound")
        _require(isinstance(data, bytes) and len(data) == count * ROW_BYTES, "chunk must contain exactly rows*128*2 bytes")
        page_route = _route(route, position, count)
        if page_route is not None:
            for other in self._shards.values():
                if other.descriptor["sequence_id"] != shard.descriptor["sequence_id"]:
                    continue
                for chunk in other.chunks:
                    previous = chunk["route"]
                    if previous is None:
                        continue
                    start = max(position, chunk["position_start"])
                    end = min(position + count, chunk["position_start"] + chunk["rows"])
                    if start < end:
                        _require(page_route["token_ids"][start - position:end - position]
                                 == previous["token_ids"][start - chunk["position_start"]:end - chunk["position_start"]],
                                 "conflicting token IDs at the same sequence positions")
        record = {"position_start": position, "rows": count, "route": page_route}
        self._preflight(list(self._shards.values()), pending=(shard_id, record), chunk_bytes=len(data))
        path = self.output_dir / shard.filename
        try:
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                info = os.fstat(fd)
                _require(stat.S_ISREG(info.st_mode) and self._inode(info) == shard.inode
                         and info.st_size == shard.written_rows * ROW_BYTES, "shard file changed before append")
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short/zero capture write")
                    shard.digest.update(view[:written])
                    self._data_bytes += written
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, ValueError) as exc:
            self._fail(exc)
            raise
        shard.written_rows += count
        shard.chunks.append(record)

    def _verify_file(self, shard, read_size):
        path = self.output_dir / shard.filename
        _require(not path.is_symlink(), "shard file replaced by symlink")
        with path.open("rb", buffering=0) as stream:
            info = os.fstat(stream.fileno())
            _require(stat.S_ISREG(info.st_mode) and self._inode(info) == shard.inode
                     and info.st_size == shard.descriptor["rows"] * ROW_BYTES, "shard identity/length changed before finalize")
            digest = hashlib.sha256()
            buffer = bytearray(read_size)
            view = memoryview(buffer)
            while count := stream.readinto(buffer):
                digest.update(view[:count])
            _require(digest.digest() == shard.digest.digest(), "shard content differs from streamed hash")

    def finalize(self, *, check_deadline=lambda: None):
        _require(self._status == "open", f"writer is {self._status}; cannot finalize it")
        try:
            check_deadline()
        except BaseException as exc:
            self._fail(exc)
            raise
        self._open()
        _require(self._shards, "capture needs at least one shard")
        _require(all(s.written_rows == s.descriptor["rows"] for s in self._shards.values()), "incomplete declared shard lengths")
        metadata = self._preflight(list(self._shards.values()))
        partial = self.output_dir / ".manifest.partial"
        final = self.output_dir / "manifest.json"
        linked = False
        partial_created = False
        try:
            read_size = min(self.max_chunk_rows * ROW_BYTES, self.max_host_bytes - len(metadata))
            _require(read_size > 0, "metadata leaves no host budget for verification")
            for shard in self._shards.values():
                self._verify_file(shard, read_size)
                check_deadline()
            fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            partial_created = True
            try:
                view = memoryview(metadata)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short/zero manifest write")
                    self._metadata_bytes += written
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            # Atomic no-replace publication, including on macOS. Both names refer
            # briefly to the SAME inode, not a second metadata payload allocation.
            check_deadline()
            os.link(partial, final)
            linked = True
            partial.unlink()
        except BaseException as exc:
            self._fail(exc)
            if linked:
                final.unlink()
            if partial_created and partial.exists():
                partial.unlink()
            self._metadata_bytes = 0
            raise
        self._status = "complete"
        return final

    def close(self):
        if self._status == "open":
            self._fail("closed without complete finalize")

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
