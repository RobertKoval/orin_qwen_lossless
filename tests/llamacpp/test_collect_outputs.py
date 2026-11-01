"""Exercise HTTP request/response behavior without a server, GPU, model, or source-string assertions."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.prompts.prompts import PROMPTS, chat, serving_long_doc  # noqa: E402
from bench.xengine.collect_outputs import collect  # noqa: E402
from bench.xengine.compare_outputs import Mismatch, compare  # noqa: E402


@pytest.mark.parametrize("engine", ["llama", "vllm"])
@pytest.mark.parametrize("top_count", [4, 5, 6])
def test_complete_frozen_requests_and_full_numeric_response(engine, top_count):
    requests = []
    expected = list(PROMPTS) + [chat("Summarize this document in three sentences.\n\n" + serving_long_doc())]
    top = [{"id": i, "token": str(i), "bytes": [48 + i], "logprob": -float(i)} for i in range(top_count)]
    llama = dict(content="00", tokens_predicted=2, stop_type="eos", truncated=False,
                 completion_probabilities=[{**top[0], "top_logprobs": top} for _ in range(2)],
                 provider_extra="retained in raw")
    vllm = dict(usage={"completion_tokens": 2}, choices=[dict(text="00", finish_reason="stop", stop_reason=None,
                 logprobs=dict(tokens=["token_id:0"] * 2, token_logprobs=[0.0] * 2,
                 top_logprobs=[{f"token_id:{i}": -float(i) for i in range(top_count)} for _ in range(2)]))])

    def http(request, timeout):
        body = json.loads(request.data)
        requests.append((request.full_url, body))
        response = {"tokens": [100, 101]} if request.full_url.endswith("/tokenize") else llama if engine == "llama" else vllm
        return io.BytesIO(json.dumps(response).encode())

    with patch("urllib.request.urlopen", side_effect=http):
        if top_count != 5:
            with pytest.raises(Mismatch, match="five"):
                collect(engine, "http://isolated-test", extra={"repetition_penalty": 1.1})
            return
        out = collect(engine, "http://isolated-test", extra={"repetition_penalty": 1.1})
    assert compare(out, deepcopy(out)) == 10
    completions = [body for url, body in requests if not url.endswith("/tokenize")]
    assert [r["prompt"] for r in completions] == expected
    for req in completions:
        assert req["temperature"] == 0 and req["seed"] == 1234
        assert req["repetition_penalty"] == 1.1
        assert req["n_predict" if engine == "llama" else "max_tokens"] == 96
        assert req["n_probs" if engine == "llama" else "logprobs"] == 5
    assert out["prompts"][0]["raw"] == (llama if engine == "llama" else vllm)
    assert out["prompts"][0]["prompt_ids"] == [100, 101]
    assert out["prompts"][0]["ids"] == [0, 0]


def test_frozen_fields_cannot_be_overridden():
    def http(request, timeout):
        return io.BytesIO(b'{"tokens":[100]}')
    with patch("urllib.request.urlopen", side_effect=http):
        with pytest.raises(ValueError, match="override"):
            collect("llama", "http://isolated-test", extra={"n_predict": 1})
