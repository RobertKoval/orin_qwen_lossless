#!/usr/bin/env python3
"""PRIVATE DEV-ONLY BF16 KV pilot. Dry-run by default; never contacts a server.

No torch/vLLM imports until explicit --execute --parent-lease. Execution creates
one isolated synchronous in-process engine; callable worker RPCs read cache only
BETWEEN completed engine steps. No import hooks, monkeypatches, extensions or
production package/service changes. CUDA capture is NOT locally validated.

--short-pilot uses independent4096/8192-token requests and samples12MiB of KV
at layers3/39 (1024 calibration and2048 held-out rows per partition), not full caches.
The existing long-context profile remains explicit and is not the first pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import sysconfig
import time
import uuid

# -I omits the script directory; permit only these sibling dev helpers, not repo
# root/engines/vllm. They cannot shadow the installed vLLM/plugin packages.
ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
LAYERS = tuple(range(3, 40, 4))  # all ten full-attention layers (T-057 gate; the 11-01 pilot read 3 and 39)
MAX_DEVICE = 64 * 1024**2
MAX_HOST = 16 * 1024**2
MAX_CHUNK_ROWS = 256
DOMAIN_CHOICES = ("english", "code", "ukrainian")
PRIVACY_ENV = ("VLLM_NO_USAGE_STATS", "DO_NOT_TRACK", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
CACHE_KEYS = ("CUDA_CACHE_PATH", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "TORCH_EXTENSIONS_DIR", "FLASHINFER_WORKSPACE_BASE", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TMPDIR")
# Exact installed sources reviewed read-only. Fail closed on another local patch
# stack, even if its vLLM version label is unchanged. These are hashes, not code.
INSPECTED = {
    "__init__.py": "fd3708e5a13abe98c566afd79a6b1987cef70c0548bb19c1218ff6a7f9d43346",
    "engine/arg_utils.py": "f6fe438718d8d0bb6a23486d6e2dc50ee11dbdd5f5a4549e8719d1242e047422",
    "v1/engine/llm_engine.py": "67630b541b49d79108e7acf14741450ba328c9d469dec5613278421e625d60b6",
    "v1/engine/core_client.py": "b350e95b61799c1f3f0876274ac5ea3cfc697fd2b263955199739334b010b91d",
    "v1/engine/core.py": "e1486de709675fa64109bb79f2b35b19c974b38a8bc2616b5e177ea3ff96c421",
    "v1/core/sched/scheduler.py": "14da3b89e347738c4e27209dbf8d9d24ec42c2a2009fb85c8782e1ee4e52bae7",
    "v1/worker/gpu_model_runner.py": "8b088ffdc5206091f505ead5e8a3f9c6eabf3314cffe9629d3649b42388c511b",
    "v1/worker/block_table.py": "5e85b99c15181cdb9793299924c80563ea5b3d770d8234239f2bf0e69cea4223",
    "v1/worker/utils.py": "3d7f9d5ba87f454c6b58dd9e237a99579477a26429540b970a8aef88197e4e95",
    "v1/executor/uniproc_executor.py": "b4783f43e65ac2a6b0c4faa93e5d65a18f93a8a21fe9d886c221bd0b39811c6a",
    "v1/worker/worker_base.py": "e6017acf80e442405f09f99413c0227515ae097d3e1dc79005b0b5c266932f12",
    "v1/attention/backends/triton_attn.py": "15c1da30d64cbc882b96aa6388c4affe8c79030bb86c0023c72b2f81b97bcd11",
    "model_executor/models/qwen3_next.py": "e11614eb71f86ab217dd671c377bcdf4f808b5b392cc9b9d62d32da5bc6c2e45",
    "model_executor/models/qwen3_5.py": "95013496de9ca5357acd1e72f35347ed8d8d6b2b4af7ea1cb907a1afeebe05e5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_digest(tokens):
    digest = hashlib.sha256()
    for first in range(0, len(tokens), 4096):
        values = tokens[first:first + 4096]
        require(all(type(x) is int and 0 <= x <= 0xFFFFFFFF for x in values), "invalid token ID")
        digest.update(struct.pack(f"<{len(values)}I", *values))
    return digest.hexdigest()


def contains_sequence(whole, part):
    """Linear-time containment guard against reusing calibration as eval text."""
    require(bool(part), "empty token sequence")
    prefix = [0] * len(part)
    matched = 0
    for i in range(1, len(part)):
        while matched and part[i] != part[matched]:
            matched = prefix[matched - 1]
        if part[i] == part[matched]:
            matched += 1
        prefix[i] = matched
    matched = 0
    for value in whole:
        while matched and value != part[matched]:
            matched = prefix[matched - 1]
        if value == part[matched]:
            matched += 1
        if matched == len(part):
            return True
    return False


def validate_architecture(config):
    text = config.get("text_config", config)
    expected = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(40)]
    require(text.get("num_hidden_layers") == 40 and text.get("layer_types") == expected,
            "checkpoint must have verified full-attention layers3,7,...39; no GDN capture")
    require(text.get("num_key_value_heads") == 2 and text.get("num_attention_heads") == 16
            and text.get("head_dim") == 256, "unsupported head geometry")
    require(text.get("dtype", text.get("torch_dtype")) == "bfloat16", "checkpoint dtype must be bfloat16")
    require(text.get("max_position_embeddings") == 262144, "unsupported context configuration")
    return text


def sample_windows(tokens):
    require(tokens in (65536, 204800), "pilot contexts are exactly65536 and204800")
    return (0, tokens - 256) if tokens == 65536 else (0, 65536 - 256, tokens - 256)


def validate_prompt_file(path, expected_tokens, vocab_size):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    require(path.is_file() and before.st_size <= 8 * 1024**2, "token file must be a regular JSON array <=8MiB")
    with path.open("rb") as stream:
        data = stream.read(8 * 1024**2 + 1)
    after = path.stat()
    require(len(data) <= 8 * 1024**2 and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "token source changed/exceeded8MiB while reading")
    values = json.loads(data)
    require(isinstance(values, list) and len(values) == expected_tokens, f"expected exactly{expected_tokens} token IDs")
    require(all(type(v) is int and 0 <= v < vocab_size for v in values), "token ID outside checkpoint vocabulary")
    return values, {"path": str(path), "source_file_sha256": hashlib.sha256(data).hexdigest(),
                    "prompt_token_ids_sha256": token_digest(values), "prompt_tokens": len(values)}


def make_plan(checkpoint, calibration_tokens, evaluation_tokens, output, *, calibration_source, evaluation_source,
              evaluation_domain="english", max_output_bytes=None, short_pilot=False):
    checkpoint = Path(checkpoint).resolve(strict=True)
    require(checkpoint.is_dir() and checkpoint.name == REVISION, "use the exact approved checkpoint snapshot directory")
    data = (checkpoint / "config.json").read_bytes()
    text = validate_architecture(json.loads(data))
    require(evaluation_domain in DOMAIN_CHOICES, "unsupported evaluation domain")
    require(all(isinstance(source, str) and source and source == source.strip() and len(source) <= 1024
                for source in (calibration_source, evaluation_source)) and calibration_source != evaluation_source,
            "calibration/evaluation source IDs must be trimmed, bounded and independent")
    require(type(short_pilot) is bool, "invalid pilot profile")
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "output must be a NEW private directory")
    require(output.parent.is_dir(), "output parent directory must already exist")
    cal, cal_meta = validate_prompt_file(calibration_tokens, 4096 if short_pilot else 65536, text["vocab_size"])
    evaluation, eval_meta = validate_prompt_file(evaluation_tokens, 8192 if short_pilot else 204800, text["vocab_size"])
    require(cal_meta["path"] != eval_meta["path"] and not contains_sequence(evaluation, cal),
            "calibration sequence is contained in evaluation; choose independent source requests")
    row_counts = (1024, 2048) if short_pilot else (256, 256)
    windows = [(len(values) - rows,) if short_pilot else sample_windows(len(values))
               for values, rows in zip((cal, evaluation), row_counts)]
    # every layer: 2 KV heads x K/V x two 128-channel halves, 256 bytes per row of 128 BF16 values
    raw_bytes = len(LAYERS) * 8 * 256 * sum(len(w) * rows for w, rows in zip(windows, row_counts))
    limit = raw_bytes + 8 * 1024**2
    if max_output_bytes is None:
        max_output_bytes = limit
    require(type(max_output_bytes) is int and raw_bytes + 1024**2 <= max_output_bytes <= limit,
            "pilot quota must cover raw data plus1MiB metadata within the profile limit")
    selected = [{first: token_digest(values[first:first + rows]) for first in starts}
                for values, rows, starts in zip((cal, evaluation), row_counts, windows)]
    if short_pilot:
        require(not contains_sequence(evaluation, cal[-row_counts[0]:]),
                "selected calibration text is contained in evaluation")
    require(not set(selected[0].values()) & set(selected[1].values()),
            "calibration/evaluation selected token windows are duplicated; remove shared source/header windows")
    requests = []
    for role, domain, source_id, meta, selected_windows, rows, starts in [
            ("calibration", "english", calibration_source, cal_meta, selected[0], row_counts[0], windows[0]),
            ("evaluation", evaluation_domain, evaluation_source, eval_meta, selected[1], row_counts[1], windows[1])]:

        requests.append({**meta, "role": role, "domain": domain, "source_id": source_id,
                         "sequence_id": f"t056-{role}-{meta['prompt_token_ids_sha256'][:16]}",
                         "windows": list(starts), "window_rows": rows,
                         "selected_token_window_sha256": {str(first): digest for first, digest in selected_windows.items()}})
    plan = {"status": "dry_run", "dev_only": True, "cuda_tested": False,
            "model": {"id": MODEL_ID, "revision": REVISION}, "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": hashlib.sha256(data).hexdigest(), "output": str(output),
            "architecture": {name: text[name] for name in ("num_hidden_layers", "layer_types", "head_dim",
                              "num_key_value_heads", "num_attention_heads", "rope_parameters")},
            "max_output_bytes": max_output_bytes, "max_device_staging_bytes": MAX_DEVICE,
            "max_host_staging_bytes": MAX_HOST, "max_chunk_rows": MAX_CHUNK_ROWS,
            "raw_pilot_bytes": raw_bytes, "short_pilot": short_pilot, "requests": requests, "layers": list(LAYERS),
            "inspected_runtime_hashes": INSPECTED,
            "profile": {"multiprocess_mode": False, "executor": "uni", "async_scheduling": False,
                        "enforce_eager": True, "compilation_mode": 0, "speculation": False,
                        "prefix_caching": False, "max_num_seqs": 1,
                        "max_model_len": 8194 if short_pilot else 262144,
                        "max_num_batched_tokens": 2112 if short_pilot else 4288,
                        "kv_cache_bytes": (1 if short_pilot else 6) * 1024**3}}
    return plan, [cal, evaluation]


def route_chunk(position, rows, page_tokens, page_ids, num_physical_pages):
    require(type(position) is int and position >= 0 and type(rows) is int and 1 <= rows <= MAX_CHUNK_ROWS,
            "invalid logical chunk range")
    require(type(page_tokens) is int and page_tokens > 0, "invalid kernel page size")
    page_index, offset = divmod(position, page_tokens)
    require(page_index < len(page_ids), "logical position has no allocated kernel page")
    page = page_ids[page_index]
    require(type(page) is int and 0 <= page < num_physical_pages, "physical page outside cache")
    return {"physical_page": page, "page_offset": offset, "kernel_page_tokens": page_tokens,
            "position_start": position, "rows": min(rows, page_tokens - offset)}


def validate_page_order(pages, allocation_pages, factor, page_tokens, committed_tokens, physical_pages):
    require(type(factor) is int and factor >= 1, "invalid allocation/kernel subdivision")
    expected = [int(p) * factor + j for p in allocation_pages for j in range(factor)]
    require(pages == expected and len(pages) * page_tokens >= committed_tokens,
            "kernel table does not match allocated committed-token order")
    require(all(type(p) is int and 0 <= p < physical_pages for p in pages), "invalid physical kernel page")
    require(len(pages) == len(set(pages)), "pilot rejects aliased/repeated logical pages")


def validate_progress(before, after, prompt_tokens, *, output_tokens, preemptions, step):
    require(all(type(x) is int for x in (before, after, prompt_tokens, output_tokens, preemptions, step)),
            "invalid execution receipt")
    require(step >= 1 and preemptions == 0, "pilot rejects preempted/replayed request histories")
    require(step != 1 or before == 0, "pilot request must begin at token0 without cached/remote prefix")
    require(0 <= before < after <= prompt_tokens, "computed prompt cursor did not advance monotonically")
    require(output_tokens == (1 if after == prompt_tokens else 0),
            "unexpected decode/finish state; capture only the completed prompt, not sampled unwritten tokens")
    return {"before_computed": before, "committed_tokens": after, "completed_step": step}


def validate_configuration(cfg):
    require(cfg.speculative_config is None and not cfg.cache_config.enable_prefix_caching,
            "pilot requires speculation and prefix caching disabled")
    require(cfg.scheduler_config.async_scheduling is False and cfg.model_config.enforce_eager,
            "pilot requires synchronous eager execution")
    parallel = cfg.parallel_config
    require(all(getattr(parallel, field) == 1 for field in ("tensor_parallel_size", "pipeline_parallel_size",
            "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size", "world_size")),
            "pilot supports one device/rank and no context parallelism")
    require(cfg.scheduler_config.max_num_seqs == 1 and cfg.compilation_config.mode == 0 and cfg.lora_config is None
            and cfg.kv_transfer_config is None, "pilot requires single-sequence/no-compilation/no-LoRA/no-external-KV configuration")


def validate_worker_policy(runner, receipt):
    validate_configuration(runner.vllm_config)
    require(runner.use_async_scheduling is False and runner.execute_model_state is None,
            "worker execution/sampling is incomplete or asynchronous")
    require(list(runner.input_batch.req_ids) == [receipt["request_id"]]
            and runner.input_batch.req_id_to_index == {receipt["request_id"]: 0},
            "worker batch must contain only the pilot request in row0")
    request = runner.requests[receipt["request_id"]]
    require(not request.mm_features and request.lora_request is None and request.prompt_embeds is None,
            "multimodal/LoRA/embedding requests are outside this token-only pilot")
    require(request.num_computed_tokens == receipt["before_computed"],
            "worker pre-step cursor differs from scheduler receipt")
    require(len(request.prompt_token_ids) == receipt["committed_tokens"]
            and len(request.output_token_ids) == 1, "worker is not frozen immediately after prompt completion")
    require(token_digest(request.prompt_token_ids) == receipt["prompt_token_ids_sha256"],
            "worker prompt order differs from the source token IDs")
    return request


def resolve_cache(runner, layer):
    require(layer in LAYERS, "pilot captures the full-attention layers 3,7,...,39 only")
    context = runner.compilation_config.static_forward_context
    names = [name for name in context if re.search(rf"(?:^|\.)layers\.{layer}\.self_attn\.attn$", name)
             and "mtp" not in name.lower()]
    require(len(names) == 1, "cannot identify exactly one target attention layer by name")
    name = names[0]
    groups = runner.kv_cache_config.kv_cache_groups
    ids = [i for i, group in enumerate(groups) if name in group.layer_names]
    require(len(ids) == 1, "attention layer must belong to exactly one cache group")
    group = ids[0]
    table = runner.input_batch.block_table.block_tables[group]
    require(type(context[name].impl).__name__ == "TritonAttentionImpl", "pilot requires the inspected Triton attention backend")
    cache = context[name].kv_cache
    shape = tuple(cache.shape)
    require(len(shape) == 5 and shape[1] == 2 and shape[3:] == (2, 256), "unsupported logical paged KV layout")
    require(shape[2] == table.block_size and cache.stride(-1) == 1, "cache and kernel block table geometry disagree")
    valid = int(table.num_blocks_per_row[0])
    require(0 < valid <= table.get_cpu_tensor().shape[1], "invalid populated kernel page count")
    pages = [int(x) for x in table.get_cpu_tensor()[0, :valid].tolist()]
    return cache, table, pages, name, group


def validate_staging_delta(allocated, reserved, baseline_allocated, baseline_reserved):
    require(all(type(x) is int and x >= 0 for x in (allocated, reserved, baseline_allocated, baseline_reserved)),
            "invalid CUDA allocation counters")
    delta = allocated - baseline_allocated, reserved - baseline_reserved
    require(max(delta) <= MAX_DEVICE, "cumulative capture staging/allocator growth exceeded64MiB")
    return delta


def worker_pilot_rpc(worker, *, receipt, operation, layer=None, head=None, kind=None, channel_start=None,
                     position_start=None, rows=None, expected_binding=None):
    """Callable vendor RPC; NO class/module monkeypatch and NO persistent worker state."""
    import torch  # only future explicit --execute calls reach this function
    require(sys.byteorder == "little", "pilot currently requires little-endian host bits")
    runner = worker.model_runner
    request = validate_worker_policy(runner, receipt)
    require(not torch.cuda.is_current_stream_capturing(), "capture RPC cannot run inside a CUDA graph")
    torch.cuda.synchronize(worker.device)  # outside engine step/capture; all cache writes complete
    if operation == "inspect":
        baseline_allocated = torch.cuda.memory_allocated(worker.device)
        baseline_reserved = torch.cuda.memory_reserved(worker.device)
        last = receipt["committed_tokens"] - receipt["before_computed"]
        require(1 <= last <= 4288 and int(runner.num_scheduled_tokens.np[0]) == last,
                "worker scheduled count differs from completed-step receipt")
        positions = runner._get_positions(last).to(device="cpu").tolist()
        expected = list(range(receipt["before_computed"], receipt["committed_tokens"]))
        require(positions == expected or positions == [expected, expected, expected],
                "actual RoPE input positions do not match text-only committed order")
        actual_ids = runner.input_ids.gpu[:last].to(device="cpu").tolist()
        require(actual_ids == request.prompt_token_ids[receipt["before_computed"]:receipt["committed_tokens"]],
                "actual last-step input token order differs from request")
        bindings = {}
        for index in LAYERS:
            cache, table, pages, name, group = resolve_cache(runner, index)
            require(cache.dtype == torch.bfloat16 and cache.is_cuda, "actual cache tensor is not CUDA BF16")
            factor = int(table.blocks_per_kv_block)
            validate_page_order(pages, request.block_ids[group], factor, table.block_size,
                                receipt["committed_tokens"], cache.shape[0])
            bindings[str(index)] = {"layer_name": name, "shape": list(cache.shape), "strides": list(cache.stride()),
                                    "cache_pointer": cache.data_ptr(), "kernel_page_tokens": table.block_size,
                                    "kernel_page_table_sha256": token_digest(pages), "group_index": group,
                                    "source_dtype": "bfloat16"}
        validate_staging_delta(torch.cuda.memory_allocated(worker.device), torch.cuda.memory_reserved(worker.device),
                               baseline_allocated, baseline_reserved)
        return {"bindings": bindings, "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                "actual_last_positions": [expected[0], expected[-1] + 1],
                "actual_last_input_ids_sha256": token_digest(actual_ids), "outside_cuda_graph": True,
                "gpu_write_completion_synchronized": True, "dtype_verified_from_tensor": True,
                "transport_bitcast": "int16; unchanged two-byte words serialized as uint16-le"}
    require(operation == "read", "unknown capture RPC operation")
    require(type(head) is int and head in (0, 1) and kind in ("K", "V")
            and type(channel_start) is int and channel_start in (0, 128), "invalid capture partition")
    require(position_start + rows <= receipt["committed_tokens"], "capture would read unwritten/uncommitted tokens")
    cache, table, pages, name, group = resolve_cache(runner, layer)
    binding = {"layer_name": name, "shape": list(cache.shape), "strides": list(cache.stride()),
               "cache_pointer": cache.data_ptr(), "kernel_page_tokens": table.block_size,
               "kernel_page_table_sha256": token_digest(pages), "group_index": group, "source_dtype": "bfloat16"}
    require(binding == expected_binding and cache.dtype == torch.bfloat16 and cache.is_cuda,
            "cache binding/page order/dtype changed during the frozen capture")
    route = route_chunk(position_start, rows, table.block_size, pages, cache.shape[0])
    count = route["rows"]
    # Reinterpret BEFORE copying; the ordinary signed16-bit carrier avoids
    # relying on shell-uint16 noncontiguous-copy support. No numeric conversion.
    view = cache[route["physical_page"], 0 if kind == "K" else 1,
                 route["page_offset"]:route["page_offset"] + count, head, channel_start:channel_start + 128]
    device_bits = view.view(torch.int16).contiguous()  # at most64KiB, never a full page/cache copy
    allocated, reserved = validate_staging_delta(torch.cuda.memory_allocated(worker.device),
        torch.cuda.memory_reserved(worker.device), receipt["baseline_allocated_bytes"], receipt["baseline_reserved_bytes"])
    host_bits = device_bits.to(device="cpu", non_blocking=False)
    data = host_bits.numpy().tobytes(order="C")
    require(len(data) == count * 128 * 2 and 4 * len(data) + 65536 <= MAX_HOST, "host staging shape/budget violation")
    tokens = request.prompt_token_ids[position_start:position_start + count]
    route.update(token_ids=tokens, token_ids_sha256=token_digest(tokens), device_allocated_delta=allocated,
                 device_reserved_delta=reserved)
    return {"position_start": position_start, "rows": count, "source_dtype": "bfloat16", "data": data, "route": route}


def validate_runtime_environment(environment):
    require(all(environment.get(name) == "1" for name in PRIVACY_ENV),
            "disable usage reporting and network model access before starting this DEV process")
    root = Path(environment.get("T056_RUNTIME_ROOT", ""))
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "explicit private T056_RUNTIME_ROOT directory required")
    root = root.resolve()
    info = root.stat()
    require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0
            and stat.S_IMODE(info.st_mode) & 0o700 == 0o700,
            "T056_RUNTIME_ROOT must be owner-controlled0700, not shared/world-writable")
    for name in CACHE_KEYS:
        value = Path(environment.get(name, ""))
        require(value.is_absolute() and value.resolve().is_relative_to(root) and value.resolve() != root
                and (not value.exists() or value.is_dir()),
                f"{name} must be an isolated child directory of T056_RUNTIME_ROOT, not a production cache")
    return {name: environment[name] for name in ("T056_RUNTIME_ROOT", *CACHE_KEYS, *PRIVACY_ENV)}


def verify_runtime():
    isolated = validate_runtime_environment(os.environ)
    require(os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") == "0", "pilot supports the inspected V1 runner only")
    require(os.environ.get("VLLM_MTP_LOOP_BREAKER_ENABLED", "0") == "0",
            "disable the known startup hook in this DEV process before starting Python")
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    for package in ("vllm", "vllm_bf16n", "torch"):
        spec = importlib.util.find_spec(package)
        require(spec and spec.origin and Path(spec.origin).resolve() == purelib / package / "__init__.py",
                "pilot requires canonical installed packages, not source/vendor/editable shadowing")
        dist = importlib.metadata.distribution(package.replace("_", "-"))
        url = dist.read_text("direct_url.json")
        require(not url or not json.loads(url).get("dir_info", {}).get("editable"), "editable production dependency rejected")
    require(importlib.metadata.version("vllm") == "0.20.0+cu126"
            and importlib.metadata.version("torch") == "2.11.0", "unsupported vLLM/torch version")
    root = purelib / "vllm"
    for name, expected in INSPECTED.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f"installed API changed since review: {name}; reinspect before capture")
    plugin_root = purelib / "vllm_bf16n"
    require(importlib.metadata.version("vllm-bf16n") == "0.1.0+g09b28236a300", "pilot requires the production wheel (11-01)")
    extension = plugin_root / "_C.cpython-310-aarch64-linux-gnu.so"
    require(hashlib.sha256(extension.read_bytes()).hexdigest() == "f8fccb746eac745d6551f848961893d1745ba071672e9707455bbfe4e24e1f0f",
            "production extension payload differs")
    plugin_sources = {name: hashlib.sha256((plugin_root / name).read_bytes()).hexdigest()
                      for name in ("model.py", "triton_spec.py", "async_align.py")}
    return {"vllm": importlib.metadata.version("vllm"), "torch": importlib.metadata.version("torch"),
            "plugin": importlib.metadata.version("vllm-bf16n"), "source_sha256": INSPECTED,
            "plugin_source_sha256": plugin_sources, "isolated_runtime_environment": isolated}


def deadline_check(start, max_seconds, now):
    require(now - start <= max_seconds, "pilot wall-time cap reached; stop and coordinate")


def stream_request(writer, rpc, request_plan, receipt, inspected, checkpoint, *, check_deadline=lambda: None):
    """Same orchestration for real callable RPC and CPU fake-source tests."""
    provenance = {**request_plan, **receipt, "runtime_inspection": inspected, "checkpoint": checkpoint,
                  "checkpoint_revision": REVISION, "model_id": MODEL_ID, "source_dtype": "bfloat16",
                  "capture_site": "after_cache_write", "runtime_verified": True}
    for layer in LAYERS:
        for head in (0, 1):
            for kind in ("K", "V"):
                for channel in (0, 128):
                    for first in request_plan["windows"]:
                        desc = {"layer": layer, "head": head, "kind": kind, "channel_start": channel, "channels": 128,
                                "capture_stage": "post_rope" if kind == "K" else "value_cache",
                                "position_start": first, "rows": request_plan.get("window_rows", 256), "domain": request_plan["domain"],
                                "role": request_plan["role"], "sequence_id": request_plan["sequence_id"],
                                "request_id": receipt["request_id"], "committed_tokens": receipt["committed_tokens"],
                                "request_provenance": provenance}
                        shard = writer.start_shard(desc)
                        position = first
                        while position < first + request_plan.get("window_rows", 256):
                            check_deadline()
                            chunk = rpc(receipt=receipt, operation="read", layer=layer, head=head, kind=kind,
                                        channel_start=channel, position_start=position, rows=min(MAX_CHUNK_ROWS, first + request_plan.get("window_rows", 256) - position),
                                        expected_binding=inspected["bindings"][str(layer)])
                            check_deadline()  # an inline RPC may itself have consumed the remaining lease
                            # Validate advancement independently from the writer to prevent RPC stalls/overreads.
                            require(chunk["position_start"] == position and type(chunk["rows"]) is int
                                    and 1 <= chunk["rows"] <= min(MAX_CHUNK_ROWS, first + request_plan.get("window_rows", 256) - position), "RPC returned an invalid logical chunk")
                            writer.add_chunk(shard, **chunk)
                            check_deadline()
                            position += chunk["rows"]


def release_captured_request(engine, core, internal_id):
    """add_request returns an INTERNAL ID; the vendor abort default is external."""
    engine.abort_request([internal_id], internal=True)
    require(internal_id not in core.scheduler.requests, "captured request was not released")


def run_pilot(plan, token_lists, *, parent_lease, max_seconds=1500):
    require(bool(parent_lease.strip()), "explicit parent lease ID required; a CLI flag does not grant permission")
    require("torch" not in sys.modules and "vllm" not in sys.modules,
            "run --execute in a fresh DEV process, never inside an existing model/server interpreter")
    require(os.environ.get("T056_EXCLUSIVE_LEASE") == parent_lease,
            "capture requires the scoped controller's exclusive stopped-production lease")
    start = time.monotonic()
    runtime = verify_runtime()  # no torch import before API/path checks
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor.abstract import Executor
    import torch
    from t056_capture_io import CaptureWriter

    require(torch.cuda.get_device_capability(0) == (8, 7), "pilot profile requires SM87")
    free, total = torch.cuda.mem_get_info(0)
    runtime["initial_cuda_free_total"] = [free, total]  # telemetry, not a UMA startup watermark
    args = EngineArgs(model=plan["checkpoint"], tokenizer=plan["checkpoint"], skip_tokenizer_init=True,
                      quantization="bf16n", dtype="bfloat16", kv_cache_dtype="auto", language_model_only=True,
                      max_model_len=plan["profile"]["max_model_len"], max_num_seqs=1,
                      max_num_batched_tokens=plan["profile"]["max_num_batched_tokens"],
                      kv_cache_memory_bytes=plan["profile"]["kv_cache_bytes"], gpu_memory_utilization=0.55,
                      enable_prefix_caching=False, enable_chunked_prefill=True, enforce_eager=True,
                      compilation_config={"mode": 0}, async_scheduling=False, speculative_config=None,
                      distributed_executor_backend="uni", attention_backend="TRITON_ATTN")
    cfg = args.create_engine_config()
    validate_configuration(cfg)
    engine = LLMEngine(cfg, Executor.get_class(cfg), log_stats=False, multiprocess_mode=False)
    writer = None
    try:
        core = engine.engine_core.engine_core
        require(type(engine.engine_core).__name__ == "InprocClient"
                and type(core.model_executor).__name__ == "UniProcExecutor"
                and core.model_executor.max_concurrent_batches == 1 and not core.async_scheduling,
                "capture cannot run with background engine/async batches")
        writer = CaptureWriter(Path(plan["output"]), max_output_bytes=plan["max_output_bytes"],
                               metadata_reserve_bytes=plan["max_output_bytes"] - plan["raw_pilot_bytes"],
                               capture_provenance={"parent_lease": parent_lease, "runtime": runtime,
                                                   "plan": plan, "cuda_tested_before_this_run": False})
        for request_plan, tokens in zip(plan["requests"], token_lists, strict=True):
            require(not core.scheduler.requests, "pilot engine unexpectedly contains another request")
            requested = f"{request_plan['sequence_id']}-{uuid.uuid4().hex[:8]}"
            req_id = engine.add_request(requested, {"prompt_token_ids": tokens},
                                        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, detokenize=False))
            step = 0
            while True:
                deadline_check(start, max_seconds, time.monotonic())
                state = core.scheduler.requests[req_id]
                before = int(state.num_computed_tokens)
                engine.step()  # returns only after synchronous execution/sampling; no next step during RPCs
                deadline_check(start, max_seconds, time.monotonic())
                step += 1
                state = core.scheduler.requests[req_id]
                receipt = validate_progress(before, int(state.num_computed_tokens), len(tokens),
                                            output_tokens=state.num_output_tokens, preemptions=state.num_preemptions, step=step)
                if receipt["committed_tokens"] == len(tokens):
                    break
            receipt.update(request_id=req_id, prompt_token_ids_sha256=request_plan["prompt_token_ids_sha256"])
            inspected = engine.collective_rpc(worker_pilot_rpc, kwargs={"receipt": receipt, "operation": "inspect"})
            require(len(inspected) == 1, "capture requires exactly one worker result")
            receipt.update(baseline_allocated_bytes=inspected[0]["baseline_allocated_bytes"],
                           baseline_reserved_bytes=inspected[0]["baseline_reserved_bytes"])
            def rpc(**kwargs):
                result = engine.collective_rpc(worker_pilot_rpc, kwargs=kwargs)
                require(len(result) == 1, "unexpected worker result count")
                return result[0]

            def deadline():
                deadline_check(start, max_seconds, time.monotonic())

            stream_request(writer, rpc, request_plan, receipt, inspected[0], plan["checkpoint"], check_deadline=deadline)
            release_captured_request(engine, core, req_id)
        deadline_check(start, max_seconds, time.monotonic())
        result = writer.finalize(check_deadline=lambda: deadline_check(start, max_seconds, time.monotonic()))
        return {"status": "captured", "manifest": str(result), "bytes_written": writer.bytes_written,
                "cuda_capture_executed": True, "production_modified": False}
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            engine.engine_core.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-tokens", required=True)
    parser.add_argument("--evaluation-tokens", required=True)
    parser.add_argument("--calibration-source", required=True, help="reviewed dataset partition/source ID")
    parser.add_argument("--evaluation-source", required=True, help="independent reviewed partition/source ID")
    parser.add_argument("--evaluation-domain", choices=DOMAIN_CHOICES, default="english")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-output-bytes", type=int, default=None)
    parser.add_argument("--short-pilot", action="store_true", help="4096/8192 tokens,1GiB KV; first actual-data pilot")
    parser.add_argument("--execute", action="store_true", help="load isolated dev model ONLY AFTER parent clearance")
    parser.add_argument("--parent-lease", default="")
    args = parser.parse_args(argv)
    try:
        plan, tokens = make_plan(args.checkpoint, args.calibration_tokens, args.evaluation_tokens, args.output,
                                calibration_source=args.calibration_source, evaluation_source=args.evaluation_source,
                                evaluation_domain=args.evaluation_domain, max_output_bytes=args.max_output_bytes,
                                short_pilot=args.short_pilot)
        result = run_pilot(plan, tokens, parent_lease=args.parent_lease) if args.execute else plan
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(f"t056 capture: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
