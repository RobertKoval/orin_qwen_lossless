"""T-053: regressions must fail both the comparison and its CLI, never just print differences."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.xengine.collect_outputs import PROMPT_SHA256  # noqa: E402
from bench.xengine.compare_outputs import Mismatch, compare, load  # noqa: E402


def recording(tokens=3):
    def probability(i):
        return {"id": i, "token": str(i), "bytes": [48 + i], "logprob": -float(i)}
    prompts = []
    for _ in range(5):
        probs = [{**probability(0), "top_logprobs": [probability(k) for k in range(5)]} for _ in range(tokens)]
        prompts.append(dict(content="000", tokens_predicted=tokens, finish_reason="length" if tokens == 96 else "eos",
                            truncated=False, prompt_ids=[100, 101],
                            ids=[p["id"] for p in probs], probs=probs,
                            top=[[[p["id"], p["logprob"]] for p in c["top_logprobs"]] for c in probs]))
    return dict(schema=1, engine="llama", model="qwen3-a3b-30b", mtp="off", n_predict=96, n_probs=5,
                prompt_sha256=PROMPT_SHA256, request_extra={}, server_config=None, prompts=prompts)


def refresh_aliases(run):
    for p in run["prompts"]:
        p["ids"] = [v["id"] for v in p["probs"]]
        p["top"] = [[[v["id"], v["logprob"]] for v in c["top_logprobs"]] for c in p["probs"]]


def test_complete_equal_and_timing_does_not_affect_gate():
    a = recording()
    b = deepcopy(a)
    b.update(url="different port")
    b["prompts"][0]["timings"] = {"predicted_per_second": 1.0}
    assert compare(a, b) == 15


@pytest.mark.parametrize("n", [0, 1, 95, 96, 97])
def test_token_count_semantic_boundaries(n):
    run = recording(n)
    if 1 <= n <= 96:
        assert compare(run, deepcopy(run)) == 5 * n
    else:
        with pytest.raises(Mismatch):
            compare(run, deepcopy(run))


@pytest.mark.parametrize("kind", [
    "empty", "missing_prompt", "extra_prompt", "missing_token", "extra_token", "empty_tokens", "missing_top",
    "extra_top", "empty_top", "missing_id", "wrong_id", "wrong_order", "missing_logprob", "changed_logprob",
    "nan", "infinity", "null", "signed_zero", "changed_selected_logprob", "changed_content", "changed_prompt_ids",
    "empty_prompt_ids", "lossy_pairs", "duplicate_top_ids", "different_mtp", "different_request", "different_compile",
    "different_frozen_prompt", "alias_only_difference",
])
def test_every_incomplete_or_different_record_fails_library_and_cli(kind, tmp_path):
    a = recording()
    b = deepcopy(a)
    p = b["prompts"][0]
    top = p["probs"][0]["top_logprobs"]
    if kind == "empty": b["prompts"] = []
    elif kind == "missing_prompt": b["prompts"].pop()
    elif kind == "extra_prompt": b["prompts"].append(deepcopy(p))
    elif kind == "missing_token": p["probs"].pop(); p["tokens_predicted"] -= 1
    elif kind == "extra_token": p["probs"].append(deepcopy(p["probs"][0])); p["tokens_predicted"] += 1
    elif kind == "empty_tokens": p["probs"] = []; p["tokens_predicted"] = 0
    elif kind == "missing_top": top.pop()
    elif kind == "extra_top": top.append(deepcopy(top[0]))
    elif kind == "empty_top": top.clear()
    elif kind == "missing_id": del top[0]["id"]
    elif kind == "wrong_id": top[0]["id"] = 123
    elif kind == "wrong_order": top[0], top[1] = top[1], top[0]
    elif kind == "missing_logprob": del top[0]["logprob"]
    elif kind == "changed_logprob": top[1]["logprob"] = math.nextafter(top[1]["logprob"], -math.inf)
    elif kind == "nan": top[0]["logprob"] = float("nan")
    elif kind == "infinity": top[0]["logprob"] = float("-inf")
    elif kind == "null": top[0]["logprob"] = None
    elif kind == "signed_zero": top[0]["logprob"] = 0.0  # original is -0.0
    elif kind == "changed_selected_logprob": p["probs"][0]["logprob"] = -0.01
    elif kind == "changed_content": p["content"] = "other"
    elif kind == "changed_prompt_ids": p["prompt_ids"][0] += 1
    elif kind == "empty_prompt_ids": p["prompt_ids"] = []
    elif kind == "lossy_pairs": p["probs"][0] = [["0", 0.0]]
    elif kind == "duplicate_top_ids": top[1]["id"] = top[0]["id"]
    elif kind == "different_mtp": b["mtp"] = "on"
    elif kind == "different_request": b["request_extra"] = {"repetition_penalty": 1.1}
    elif kind == "different_compile": b["server_config"] = {"compile_mode": "other"}
    elif kind == "different_frozen_prompt": b["prompt_sha256"] = ["bad"] * 5
    elif kind == "alias_only_difference": p["ids"][0] += 1
    # Keep redundant aliases consistent when feasible, so failures exercise the actual probability contract too.
    if kind not in ("missing_id", "missing_logprob", "lossy_pairs", "alias_only_difference"):
        refresh_aliases(b)
    with pytest.raises((Mismatch, ValueError)):
        compare(a, b)
    left, right = tmp_path / "a.json", tmp_path / "b.json"
    left.write_text(json.dumps(a))
    right.write_text(json.dumps(b))
    result = subprocess.run([sys.executable, str(ROOT / "bench/xengine/compare_outputs.py"), str(left), str(right)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "FAIL:" in result.stderr


@pytest.mark.parametrize("top_size", [4, 5, 6])
def test_exact_top5_cardinality(top_size):
    run = recording()
    top = run["prompts"][0]["probs"][0]["top_logprobs"]
    if top_size == 4:
        top.pop()
    elif top_size == 6:
        top.append({"id": 10, "token": "x", "logprob": -10.0})
    refresh_aliases(run)
    if top_size == 5:
        assert compare(run, deepcopy(run)) == 15
    else:
        with pytest.raises(Mismatch): compare(run, deepcopy(run))


def test_cli_equal_success_and_missing_file_failure(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(recording()))
    cmd = [sys.executable, str(ROOT / "bench/xengine/compare_outputs.py"), str(path)]
    ok = subprocess.run(cmd + [str(path)], capture_output=True, text=True)
    assert ok.returncode == 0 and "EXACT:" in ok.stdout
    bad = subprocess.run(cmd + [str(tmp_path / "missing.json")], capture_output=True, text=True)
    assert bad.returncode == 1


@pytest.mark.parametrize("reason,truncated,accept", [("length", False, False), ("eos", False, True),
                                                      ("none", False, False), ("word", False, False),
                                                      ("eos", True, False), (None, False, False)])
def test_both_sides_incomplete_cannot_pass(reason, truncated, accept):
    run = recording(95)
    for p in run["prompts"]:
        p["finish_reason"] = reason
        p["truncated"] = truncated
    if accept:
        assert compare(run, deepcopy(run)) == 475
    else:
        with pytest.raises(Mismatch): compare(run, deepcopy(run))


@pytest.mark.parametrize("bad", [[], {}, {"schema": 1, "prompts": []}])
def test_equal_empty_or_incomplete_captures_still_fail(bad):
    with pytest.raises(Mismatch):
        compare(bad, deepcopy(bad))


def test_duplicate_json_fields_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":1}')
    with pytest.raises(Mismatch, match="duplicate"):
        load(path)
