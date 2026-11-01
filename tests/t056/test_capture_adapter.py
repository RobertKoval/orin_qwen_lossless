"""CPU protocol/planning tests. These DO NOT execute the CUDA callback or a model."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ANALYSIS = Path(__file__).resolve().parents[2] / "tools/analysis"
sys.path.insert(0, str(ANALYSIS))
import t056_capture as capture


def config():
    return {"text_config": {"num_hidden_layers": 40, "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(40)],
                            "num_key_value_heads": 2, "num_attention_heads": 16, "head_dim": 256,
                            "dtype": "bfloat16", "max_position_embeddings": 262144, "vocab_size": 248320,
                            "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_interleaved": True}}}


def policy():
    return NS(speculative_config=None, cache_config=NS(enable_prefix_caching=False),
              scheduler_config=NS(async_scheduling=False, max_num_seqs=1), model_config=NS(enforce_eager=True),
              compilation_config=NS(mode=0), lora_config=None, kv_transfer_config=None,
              parallel_config=NS(**{field: 1 for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                                      "decode_context_parallel_size", "prefill_context_parallel_size", "world_size")}))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="t056-capture-plan-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.checkpoint = self.root / capture.REVISION
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps(config()))
        self.cal = self.root / "cal.json"
        self.evaluation = self.root / "eval.json"
        self.cal.write_text(json.dumps([7] * 65536))
        self.evaluation.write_text(json.dumps([8] * 204800))
        self.output = self.root / "pilot"

    def plan(self, **kwargs):
        return capture.make_plan(self.checkpoint, self.cal, self.evaluation, self.output,
                                 calibration_source="train-a", evaluation_source="test-b", **kwargs)

    def cli(self, extra=(), ok=True):
        # A module-import trap would fail on ANY torch/vLLM import, including CPU
        # imports; success therefore exercises the dry-run's no-runtime boundary.
        trap = '''import importlib.abc, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in ('torch', 'vllm', 'vllm_bf16n'):
   raise RuntimeError('forbidden runtime import during CPU test: '+name)
sys.meta_path.insert(0, NoRuntime())
p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name='__main__')
'''
        argv = [sys.executable, "-I", "-B", "-c", trap, str(ANALYSIS / "t056_capture.py"),
                "--checkpoint", str(self.checkpoint), "--calibration-tokens", str(self.cal),
                "--evaluation-tokens", str(self.evaluation), "--calibration-source", "train-a",
                "--evaluation-source", "test-b", "--output", str(self.output), *extra]
        result = subprocess.run(argv, text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def test_dry_run_does_not_import_runtime_and_declares_bounded_pilot(self):
        result = self.cli()
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "dry_run")
        self.assertFalse(plan["cuda_tested"])
        self.assertEqual(plan["raw_pilot_bytes"], 5 * 1024**2)
        self.assertEqual(plan["layers"], [3, 39])
        self.assertEqual([x["windows"] for x in plan["requests"]], [[0, 65280], [0, 65280, 204544]])
        self.assertEqual([x["role"] for x in plan["requests"]], ["calibration", "evaluation"])
        self.assertEqual(self.cli().returncode, 0)  # dry-run did not consume/create its exclusive output

    def test_short_pilot_is_small_and_never_imports_runtime(self):
        self.cal.write_text(json.dumps([7] * 4096))
        self.evaluation.write_text(json.dumps([8] * 8192))
        result = self.cli(("--short-pilot",))
        plan = json.loads(result.stdout)
        self.assertEqual(plan["profile"]["max_model_len"], 8192 + 2)
        self.assertEqual(plan["profile"]["max_num_batched_tokens"], 2112)
        self.assertEqual(plan["profile"]["kv_cache_bytes"], 1024**3)
        self.assertEqual([(r["windows"], r["window_rows"]) for r in plan["requests"]], [([3072], 1024), ([6144], 2048)])
        self.assertEqual(plan["raw_pilot_bytes"], 12 * 1024**2)
        for count in (4095, 4097):
            self.cal.write_text(json.dumps([7] * count))
            self.cli(("--short-pilot",), ok=False)
        self.cal.write_text(json.dumps([7] * 4096))
        for count in (8191, 8193):
            self.evaluation.write_text(json.dumps([8] * count))
            self.cli(("--short-pilot",), ok=False)
        self.evaluation.write_text(json.dumps([8] * 8192))
        for quota in (13 * 1024**2 - 1, 13 * 1024**2, 13 * 1024**2 + 1):
            self.cli(("--short-pilot", "--max-output-bytes", str(quota)), ok=quota >= 13 * 1024**2)

    def test_execute_without_parent_lease_fails_before_any_runtime_import(self):
        result = self.cli(("--execute",), ok=False)
        self.assertIn("parent lease", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_exclusive_lease_binding_precedes_runtime_import(self):
        for value in (None, "wrong-window", "CPU_ONLY_WINDOW"):
            env = {} if value is None else {"T056_EXCLUSIVE_LEASE": value}
            with self.subTest(value=value), patch.dict(capture.os.environ, env, clear=True), \
                    patch.object(capture, "verify_runtime", side_effect=ValueError("CPU contract reached")) as verify:
                with self.assertRaisesRegex(ValueError, "CPU contract reached" if value == "CPU_ONLY_WINDOW" else "exclusive"):
                    capture.run_pilot({}, [], parent_lease="CPU_ONLY_WINDOW")
                self.assertEqual(verify.call_count, int(value == "CPU_ONLY_WINDOW"))

    def test_existing_model_interpreter_is_rejected_before_loading_another_engine(self):
        for module in ("torch", "vllm"):
            with self.subTest(module=module), patch.dict(sys.modules, {module: NS()}):
                with self.assertRaisesRegex(ValueError, "fresh DEV process"):
                    capture.run_pilot({}, [], parent_lease="CPU_TEST_DO_NOT_EXECUTE")

    def test_wrong_architecture_dtype_and_gdn_layers_fail_before_runtime(self):
        for name, value in (("head_dim", 128), ("num_key_value_heads", 4), ("dtype", "float16"),
                            ("num_hidden_layers", 39), ("max_position_embeddings", 65536)):
            with self.subTest(name=name):
                bad = config()
                bad["text_config"][name] = value
                (self.checkpoint / "config.json").write_text(json.dumps(bad))
                self.cli(ok=False)
        bad = config()
        bad["text_config"]["layer_types"][39] = "linear_attention"
        (self.checkpoint / "config.json").write_text(json.dumps(bad))
        self.cli(ok=False)

    def test_token_length_type_vocabulary_and_calibration_leakage_rejected(self):
        for tokens in ([7] * 65535, [7] * 65537, [True] * 65536, [-1] * 65536, [248320] * 65536):
            with self.subTest(first=tokens[0], length=len(tokens)):
                self.cal.write_text(json.dumps(tokens))
                self.cli(ok=False)
        self.cal.write_text(json.dumps([7] * 65536))
        for first in (0, 60000, 139264):
            values = [8] * 204800
            values[first:first + 65536] = [7] * 65536
            self.evaluation.write_text(json.dumps(values))
            result = self.cli(ok=False)
            self.assertIn("contained", result.stderr)

    def test_reused_selected_window_is_rejected_even_when_complete_prompts_differ(self):
        for first in (0, 65280, 204544):
            values = [8] * 204800
            values[first:first + 256] = [7] * 256
            self.evaluation.write_text(json.dumps(values))
            with self.subTest(first=first):
                result = self.cli(ok=False)
                self.assertIn("selected token windows are duplicated", result.stderr)

    def test_output_quota_boundary_and_existing_output_fail(self):
        for quota, ok in ((6 * 1024**2 - 1, False), (6 * 1024**2, True),
                          (8 * 1024**2, True), (8 * 1024**2 + 1, False)):
            with self.subTest(quota=quota):
                self.cli(("--max-output-bytes", str(quota)), ok=ok)
        self.output.mkdir()
        sentinel = self.output / "unrelated"
        sentinel.write_bytes(b"keep existing output untouched")
        self.cli(ok=False)
        self.assertEqual(sentinel.read_bytes(), b"keep existing output untouched")

    def test_containment_handles_overlap_and_repeated_prefixes(self):
        self.assertTrue(capture.contains_sequence([1, 1, 1, 2, 1, 2], [1, 1, 2]))
        self.assertFalse(capture.contains_sequence([1, 1, 1, 2, 1, 2], [1, 2, 2]))
        self.assertTrue(capture.contains_sequence([1, 2], [1, 2]))
        with self.assertRaises(ValueError):
            capture.contains_sequence([1], [])


class EnvironmentTests(unittest.TestCase):
    def test_runtime_caches_must_be_private_not_shared_production_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            environment = {"T056_RUNTIME_ROOT": str(root)}
            environment.update({name: str(root / name.lower()) for name in capture.CACHE_KEYS})
            environment.update({name: "1" for name in capture.PRIVACY_ENV})
            self.assertEqual(capture.validate_runtime_environment(environment), environment)
            for key in capture.CACHE_KEYS:
                bad = dict(environment)
                bad[key] = "/srv/example-cache"
                with self.subTest(key=key), self.assertRaises(ValueError):
                    capture.validate_runtime_environment(bad)
            for key in capture.PRIVACY_ENV:
                bad = dict(environment)
                bad[key] = "0"
                with self.subTest(privacy=key), self.assertRaises(ValueError):
                    capture.validate_runtime_environment(bad)
            for mode in (0o600, 0o701, 0o770, 0o777):
                root.chmod(mode)
                with self.subTest(mode=oct(mode)), self.assertRaises(ValueError):
                    capture.validate_runtime_environment(environment)
            root.chmod(0o700)
            self.assertEqual(capture.validate_runtime_environment(environment), environment)
            with self.assertRaises(ValueError):
                capture.validate_runtime_environment({})


class MappingPolicyTests(unittest.TestCase):
    def test_kernel_pages_not_allocation_pages_define_position_routes(self):
        # Two allocation pages, each subdivided into two16-token kernel pages.
        pages = [6, 7, 2, 3]
        capture.validate_page_order(pages, [3, 1], 2, 16, 64, 8)
        self.assertEqual(capture.route_chunk(31, 3, 16, pages, 8),
                         {"physical_page": 7, "page_offset": 15, "kernel_page_tokens": 16, "position_start": 31, "rows": 1})
        self.assertEqual(capture.route_chunk(32, 2, 16, pages, 8)["physical_page"], 2)
        for wrong in ([3, 1], [7, 6, 2, 3], [6, 7, 2, 2]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                capture.validate_page_order(wrong, [3, 1], 2, 16, 64, 8)

    def test_real_page_boundary_and_chunk_limit_below_at_above(self):
        pages = list(range(200))
        self.assertEqual(capture.route_chunk(65280, 256, 1072, pages, 200)["rows"], 112)
        self.assertEqual(capture.route_chunk(65392, 144, 1072, pages, 200)["rows"], 144)
        self.assertEqual(capture.route_chunk(204544, 256, 1072, pages, 200)["rows"], 208)
        self.assertEqual(capture.route_chunk(204752, 48, 1072, pages, 200)["rows"], 48)
        for size, ok in ((0, False), (1, True), (255, True), (256, True), (257, False)):
            if ok:
                self.assertEqual(capture.route_chunk(0, size, 1072, pages, 200)["rows"], size)
            else:
                with self.assertRaises(ValueError):
                    capture.route_chunk(0, size, 1072, pages, 200)
        for position in (-1, 214400, True):
            with self.assertRaises(ValueError):
                capture.route_chunk(position, 1, 1072, pages, 200)

    def test_hybrid_layers_resolve_named_attention_cache_and_its_kernel_group(self):
        class TritonAttentionImpl: pass
        class Cache:
            shape = (20, 2, 16, 2, 256)
            def __init__(self, marker): self.marker = marker
            def stride(self, dimension): return 1
        class Vector(list):
            def tolist(self): return list(self)
        class Matrix:
            shape = (1, 4)
            def __getitem__(self, indexes): return Vector([6, 7, 2, 3][indexes[1]])
        table = NS(block_size=16, blocks_per_kv_block=2, num_blocks_per_row=[4], get_cpu_tensor=lambda: Matrix())
        names = {layer: f"language_model.model.layers.{layer}.self_attn.attn" for layer in (3, 39)}
        context = {name: NS(impl=TritonAttentionImpl(), kv_cache=Cache(layer)) for layer, name in names.items()}
        context["language_model.model.layers.2.linear_attn"] = NS(kv_cache="NOT_ATTENTION_KV")
        runner = NS(compilation_config=NS(static_forward_context=context),
                    kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["language_model.model.layers.2.linear_attn"]),
                                                        NS(layer_names=list(names.values()))]),
                    input_batch=NS(block_table=NS(block_tables=["GDN_GROUP", table])))
        for layer in (3, 39):
            cache, resolved_table, pages, name, group = capture.resolve_cache(runner, layer)
            self.assertEqual((cache.marker, pages, name, group), (layer, [6, 7, 2, 3], names[layer], 1))
            self.assertIs(resolved_table, table)
        with self.assertRaises(ValueError):
            capture.resolve_cache(runner, 2)
        context[names[3]].kv_cache.shape = (20, 2, 16, 4, 128)
        with self.assertRaises(ValueError):
            capture.resolve_cache(runner, 3)

    def test_cumulative_staging_limit_boundary_not_only_per_chunk_growth(self):
        baseline = 10 * 1024**3
        for delta in (capture.MAX_DEVICE - 1, capture.MAX_DEVICE):
            self.assertEqual(capture.validate_staging_delta(baseline + delta, baseline + delta, baseline, baseline), (delta, delta))
        for allocated, reserved in ((baseline + capture.MAX_DEVICE + 1, baseline), (baseline, baseline + capture.MAX_DEVICE + 1)):
            with self.assertRaises(ValueError):
                capture.validate_staging_delta(allocated, reserved, baseline, baseline)
        # Freed model scratch can lower a counter; it does not excuse excessive growth of the other counter.
        self.assertEqual(capture.validate_staging_delta(baseline - 1, baseline, baseline, baseline), (-1, 0))

    def test_completed_prompt_not_sampled_unwritten_token_defines_committed_range(self):
        self.assertEqual(capture.validate_progress(0, 8, 10, output_tokens=0, preemptions=0, step=1)["committed_tokens"], 8)
        self.assertEqual(capture.validate_progress(8, 10, 10, output_tokens=1, preemptions=0, step=2)["before_computed"], 8)
        with self.assertRaises(ValueError):
            capture.validate_progress(8, 10, 10, output_tokens=1, preemptions=0, step=1)
        for before, after, count, preempted in ((8, 8, 0, 0), (8, 11, 1, 0), (8, 10, 0, 0),
                                                 (8, 10, 2, 0), (0, 8, 1, 0), (8, 10, 1, 1)):
            with self.subTest(before=before, after=after, count=count, preempted=preempted), self.assertRaises(ValueError):
                capture.validate_progress(before, after, 10, output_tokens=count, preemptions=preempted, step=2)

    def test_worker_policy_rejects_inflight_stale_speculative_and_async_states(self):
        tokens = list(range(10))
        request = NS(prompt_token_ids=tokens, output_token_ids=[99], num_computed_tokens=8,
                     mm_features=[], lora_request=None, prompt_embeds=None)
        runner = NS(vllm_config=policy(), execute_model_state=None, use_async_scheduling=False, input_batch=NS(req_ids=["r"], req_id_to_index={"r": 0}), requests={"r": request})
        receipt = {"request_id": "r", "before_computed": 8, "committed_tokens": 10,
                   "prompt_token_ids_sha256": capture.token_digest(tokens)}
        self.assertIs(capture.validate_worker_policy(runner, receipt), request)
        for path, value in (("execute_model_state", object()), ("use_async_scheduling", True), ("vllm_config.speculative_config", object()),
                            ("vllm_config.cache_config.enable_prefix_caching", True),
                            ("vllm_config.scheduler_config.async_scheduling", True),
                            ("vllm_config.parallel_config.decode_context_parallel_size", 2),
                            ("vllm_config.kv_transfer_config", object()), ("vllm_config.compilation_config.mode", 1)):
            fake = copy.deepcopy(runner)
            owner = fake
            bits = path.split(".")
            for bit in bits[:-1]:
                owner = getattr(owner, bit)
            setattr(owner, bits[-1], value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                capture.validate_worker_policy(fake, receipt)
        for before, output in ((10, [99]), (8, []), (8, [99, 100])):
            fake = copy.deepcopy(runner)
            fake.requests["r"].num_computed_tokens = before
            fake.requests["r"].output_token_ids = output
            with self.assertRaises(ValueError):
                capture.validate_worker_policy(fake, receipt)
        fake = copy.deepcopy(runner)
        fake.requests["r"].prompt_token_ids.reverse()
        with self.assertRaises(ValueError):
            capture.validate_worker_policy(fake, receipt)


class RequestLifecycleTests(unittest.TestCase):
    def test_internal_abort_releases_request_for_next_independent_text(self):
        requests = {"cal-randomid": object()}
        core = NS(scheduler=NS(requests=requests))
        class Engine:
            def abort_request(self, ids, internal=False):
                if internal:
                    for rid in ids:
                        requests.pop(rid, None)
        capture.release_captured_request(Engine(), core, "cal-randomid")
        self.assertEqual(requests, {})
        requests["eval-randomid"] = object()
        capture.release_captured_request(Engine(), core, "eval-randomid")
        self.assertEqual(requests, {})


class DeadlineTests(unittest.TestCase):
    def test_exact_deadline_boundary(self):
        for now in (math.nextafter(10.0, 0), 10.0):
            capture.deadline_check(0, 10, now)
        with self.assertRaises(ValueError):
            capture.deadline_check(0, 10, math.nextafter(10.0, math.inf))

    def test_last_rpc_or_disk_write_cannot_overrun_then_publish_completed_capture(self):
        for blocking in ("rpc", "disk"):
            with self.subTest(blocking=blocking):
                now = [0]
                class Writer:
                    chunks = 0
                    published = False
                    def start_shard(self, descriptor): return descriptor["kind"]
                    def add_chunk(self, shard, **data):
                        self.chunks += 1
                        if blocking == "disk" and self.chunks == 16:
                            now[0] = 11
                    def finalize(self): self.published = True
                writer = Writer()
                calls = []
                def deadline(): capture.deadline_check(0, 10, now[0])
                def rpc(**kwargs):
                    calls.append(kwargs)
                    if blocking == "rpc" and len(calls) == 16:
                        now[0] = 11
                    return {"position_start": kwargs["position_start"], "rows": kwargs["rows"], "data": b""}
                request = {"windows": [0], "role": "calibration", "domain": "english", "sequence_id": "cal"}
                receipt = {"request_id": "r", "committed_tokens": 65536}
                inspected = {"bindings": {str(layer): {} for layer in capture.LAYERS}}
                with self.assertRaises(ValueError):
                    capture.stream_request(writer, rpc, request, receipt, inspected, "/private/checkpoint", check_deadline=deadline)
                    writer.finalize()
                self.assertEqual(len(calls), 16)
                self.assertEqual(writer.chunks, 15 if blocking == "rpc" else 16)
                self.assertFalse(writer.published)


class StreamingProtocolTests(unittest.TestCase):
    def test_short_capture_writes_2048_heldout_rows_per_partition_with_bounded_page_chunks(self):
        from t056_capture_io import CaptureWriter
        with tempfile.TemporaryDirectory() as temporary:
            writer = CaptureWriter(Path(temporary).resolve() / "short", max_output_bytes=16 * 1024**2,
                                   capture_provenance={"CPU_FAKE_SOURCE_TEST": True})
            total = 0
            try:
                for role, size, rows, value in (("calibration", 4096, 1024, 7), ("evaluation", 8192, 2048, 8)):
                    tokens = [value] * size
                    request = {"role": role, "domain": "english", "source_id": role, "sequence_id": role,
                               "path": "/private/fake-tokens", "source_file_sha256": str(value) * 64,
                               "prompt_tokens": size, "prompt_token_ids_sha256": capture.token_digest(tokens),
                               "windows": [size - rows], "window_rows": rows}
                    receipt = {"request_id": role, "before_computed": size - 16, "committed_tokens": size,
                               "completed_step": 4, "prompt_token_ids_sha256": request["prompt_token_ids_sha256"]}
                    inspected = {"bindings": {str(layer): {} for layer in capture.LAYERS}}
                    def rpc(**kwargs):
                        nonlocal total
                        self.assertLessEqual(kwargs["rows"], 256)
                        route = capture.route_chunk(kwargs["position_start"], kwargs["rows"], 1056, list(range(8)), 8)
                        count = route["rows"]; first = kwargs["position_start"]
                        data = b"".join(struct.pack("<H", p & 65535) * 128 for p in range(first, first + count))
                        self.assertLessEqual(len(data), 65536); total += len(data)
                        chunk_tokens = tokens[first:first + count]
                        route.update(token_ids=chunk_tokens, token_ids_sha256=capture.token_digest(chunk_tokens),
                                     device_allocated_delta=0, device_reserved_delta=0)
                        return {"position_start": first, "rows": count, "data": data, "source_dtype": "bfloat16", "route": route}
                    capture.stream_request(writer, rpc, request, receipt, inspected, "/private/fake-checkpoint")
                manifest_path = writer.finalize()
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(total, 12 * 1024**2)
                for shard in manifest["shards"]:
                    self.assertEqual(shard["rows"], 2048 if shard["role"] == "evaluation" else 1024)
                    expected = b"".join(struct.pack("<H", p & 65535) * 128
                                        for p in range(shard["position_start"], shard["position_start"] + shard["rows"]))
                    self.assertEqual((manifest_path.parent / shard["path"]).read_bytes(), expected)
            finally:
                writer.close()

    def test_actual_pilot_orchestration_with_cpu_fake_pages_retains_all_partitions_and_order(self):
        from t056_capture_io import CaptureWriter
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            writer = CaptureWriter(root / "pilot", capture_provenance={"CPU_FAKE_SOURCE_TEST": True})
            try:
                expected_bytes = 0
                calls = 0
                for role, size, value in (("calibration", 65536, 7), ("evaluation", 204800, 8)):
                    tokens = [value] * size
                    request = {"role": role, "domain": "english", "source_id": role + "-source", "sequence_id": role,
                               "path": "/private/fixture-tokens.json", "source_file_sha256": str(value) * 64,
                               "prompt_tokens": size, "prompt_token_ids_sha256": capture.token_digest(tokens),
                               "windows": list(capture.sample_windows(size))}
                    receipt = {"request_id": "r-" + role, "before_computed": size - 16, "committed_tokens": size,
                               "completed_step": 100, "prompt_token_ids_sha256": request["prompt_token_ids_sha256"]}
                    inspected = {"bindings": {str(layer): {"CPU_FAKE_LAYER": layer} for layer in capture.LAYERS}}
                    page_tokens = 16  # worst-case1280 bounded RPC chunks, not presumed1072-token pages
                    pages = list(reversed(range(size // page_tokens)))

                    def bits(position, layer, head, kind, channel, rows):
                        return b"".join(struct.pack("<128H", *[((p * 131 + layer * 17 + head * 1000
                                       + (9000 if kind == "V" else 0) + channel + c) & 65535) for c in range(128)])
                                        for p in range(position, position + rows))

                    def rpc(**kwargs):
                        nonlocal calls
                        calls += 1
                        self.assertEqual(kwargs["operation"], "read")
                        route = capture.route_chunk(kwargs["position_start"], kwargs["rows"], page_tokens, pages, len(pages))
                        count = route["rows"]
                        # CPU source is physically reverse-paged; a wrong route changes actual data, not just metadata.
                        source_position = (len(pages) - 1 - route["physical_page"]) * page_tokens + route["page_offset"]
                        raw = bits(source_position, kwargs["layer"], kwargs["head"], kwargs["kind"], kwargs["channel_start"], count)
                        chunk_tokens = tokens[kwargs["position_start"]:kwargs["position_start"] + count]
                        route.update(token_ids=chunk_tokens, token_ids_sha256=capture.token_digest(chunk_tokens),
                                     device_allocated_delta=0, device_reserved_delta=0)
                        return {"position_start": kwargs["position_start"], "rows": count, "data": raw,
                                "source_dtype": "bfloat16", "route": route}

                    capture.stream_request(writer, rpc, request, receipt, inspected, "/private/approved-checkpoint")
                    expected_bytes += len(request["windows"]) * 16 * 256 * 128 * 2
                manifest_path = writer.finalize()
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(calls, 1280)
                self.assertEqual(len(manifest["shards"]), 80)
                self.assertEqual(expected_bytes, 5 * 1024**2)
                observed = 0
                for shard in manifest["shards"]:
                    raw = (manifest_path.parent / shard["path"]).read_bytes()
                    observed += len(raw)
                    self.assertEqual(raw, bits(shard["position_start"], shard["layer"], shard["head"], shard["kind"],
                                               shard["channel_start"], shard["rows"]))
                self.assertEqual(observed, expected_bytes)
                self.assertLessEqual(writer.bytes_written, capture.MAX_OUTPUT)
            finally:
                writer.close()

    def test_invalid_rpc_advancement_fails_before_another_iteration(self):
        from t056_capture_io import CaptureWriter
        with tempfile.TemporaryDirectory() as temporary:
            writer = CaptureWriter(Path(temporary).resolve() / "pilot")
            request = {"role": "calibration", "domain": "english", "source_id": "train", "sequence_id": "cal",
                       "path": "/private/cal", "source_file_sha256": "a" * 64, "prompt_tokens": 65536,
                       "prompt_token_ids_sha256": "b" * 64, "windows": [0]}
            receipt = {"request_id": "r", "before_computed": 64000, "committed_tokens": 65536, "completed_step": 16,
                       "prompt_token_ids_sha256": "b" * 64}
            inspected = {"bindings": {str(layer): {} for layer in capture.LAYERS}}
            calls = []

            def invalid(**kwargs):
                calls.append(kwargs)
                return {"position_start": kwargs["position_start"], "rows": 0}
            try:
                with self.assertRaises(ValueError):
                    capture.stream_request(writer, invalid, request, receipt, inspected, "/private/checkpoint")
                self.assertEqual(len(calls), 1)
                with self.assertRaises(ValueError):
                    writer.finalize()
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
