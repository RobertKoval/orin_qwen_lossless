#!/usr/bin/env python3
"""Frozen S5 serving capture, no server lifecycle effects. Five prompts by default.

--request-json records extra request options (e.g. a loop breaker); use identical options for A/B.
--server-config records the operator's launch/compile configuration receipt, without interpreting it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.prompts.prompts import PROMPTS, chat, serving_long_doc  # noqa: E402
from bench.xengine.compare_outputs import validate  # noqa: E402

S5_PROMPTS = tuple(PROMPTS) + (chat("Summarize this document in three sentences.\n\n" + serving_long_doc()),)
PROMPT_SHA256 = [hashlib.sha256(p.encode()).hexdigest() for p in S5_PROMPTS]


def collect(engine, url, model="qwen3-a3b-30b", mtp="off", include_long=True, extra=None, server_config=None, timeout=600):
    prompts = S5_PROMPTS if include_long else S5_PROMPTS[:4]
    out = {"schema": 1, "engine": engine, "model": model, "mtp": mtp, "n_predict": 96, "n_probs": 5,
           "prompt_sha256": PROMPT_SHA256[:len(prompts)], "url": url, "request_extra": extra or {},
           "server_config": server_config, "prompts": []}

    def post(path, body):
        request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body, allow_nan=False).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def request(frozen):
        if set(frozen).intersection(extra or {}):
            raise ValueError("--request-json must not override frozen request fields")
        return {**(extra or {}), **frozen}

    def token_id(token):
        prefix = "token_id:"
        if not isinstance(token, str) or not token.startswith(prefix):
            raise ValueError(f"vLLM omitted requested token IDs: {token!r}")
        return int(token[len(prefix):])

    for index, prompt in enumerate(prompts):
        if engine == "llama":
            prompt_ids = post("/tokenize", {"content": prompt, "parse_special": True})["tokens"]
            r = post("/completion", request({"prompt": prompt, "n_predict": 96, "temperature": 0, "n_probs": 5,
                     "cache_prompt": False, "seed": 1234, "post_sampling_probs": False}))
            probs = r["completion_probabilities"]
            item = {"content": r["content"], "tokens_predicted": r["tokens_predicted"], "probs": probs,
                    "finish_reason": {"eos": "eos", "limit": "length"}.get(r.get("stop_type"), "invalid"),
                    "truncated": r.get("truncated", True)}
        else:
            prompt_ids = post("/tokenize", {"model": model, "prompt": prompt})["tokens"]
            r = post("/v1/completions", request({"model": model, "prompt": prompt, "max_tokens": 96,
                     "temperature": 0, "logprobs": 5, "seed": 1234, "return_tokens_as_token_ids": True}))
            choice = r["choices"][0]
            lp = choice["logprobs"]
            if not (len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"])):
                raise ValueError("vLLM probability arrays have unequal lengths")
            probs = []
            for j, token in enumerate(lp["tokens"]):
                # vLLM exposes a mapping. Stable descending sort preserves provider order on exact ties.
                top = sorted(lp["top_logprobs"][j].items(), key=lambda pair: -pair[1])
                probs.append({"id": token_id(token), "token": token, "logprob": lp["token_logprobs"][j],
                              "top_logprobs": [{"id": token_id(t), "token": t, "logprob": v} for t, v in top]})
            finish = choice.get("finish_reason")
            if finish == "stop" and choice.get("stop_reason") is None:
                finish = "eos"  # No requested stop strings: OpenAI stop means the model ended naturally.
            item = {"content": choice["text"], "tokens_predicted": r["usage"]["completion_tokens"], "probs": probs,
                    "finish_reason": finish, "truncated": False}
        item.update(prompt_ids=prompt_ids, ids=[p["id"] for p in probs],
                    top=[[[p["id"], p["logprob"]] for p in c["top_logprobs"]] for c in probs], raw=r)
        out["prompts"].append(item)
        print(f"captured {index + 1}/{len(prompts)}: {item['tokens_predicted']} tokens", file=sys.stderr, flush=True)
    validate(out, len(prompts))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("llama", "vllm"), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3-a3b-30b")
    p.add_argument("--mtp", choices=("off", "on"), default="off", help="operator declaration of launch mode")
    p.add_argument("--include-long", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--request-json", type=Path)
    p.add_argument("--server-config", type=Path)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    try:
        if args.out.exists():
            raise ValueError(f"refusing to overwrite {args.out}")
        extra = json.loads(args.request_json.read_text()) if args.request_json else {}
        if not isinstance(extra, dict):
            raise ValueError("--request-json must contain an object")
        config = json.loads(args.server_config.read_text()) if args.server_config else None
        result = collect(args.engine, args.url, args.model, args.mtp, args.include_long, extra, config, args.timeout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            json.dump(result, f, allow_nan=False, indent=2)
            f.write("\n")
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(f"collect-outputs: FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
