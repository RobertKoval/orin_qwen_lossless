#!/usr/bin/env python3
"""Fail-closed comparison of complete run_prompts/collect_prompts outputs (T-053).

No tolerance, token-set comparison, prefix truncation or timing-based acceptance.
Logprob comparison uses IEEE binary64 bytes of the parsed JSON numbers, including signed zero.
Old [token, logprob] pairs are deliberately rejected: they discarded token IDs and selected-token data.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Mismatch(ValueError):
    pass


def _require(ok, where, message):
    if not ok:
        raise Mismatch(f"{where}: {message}")


def _prob(prob, where):
    _require(isinstance(prob, dict), where, "expected complete probability object, not a lossy pair")
    _require({"id", "token", "logprob"} <= prob.keys(), where, "missing id/token/logprob")
    _require(type(prob["id"]) is int and prob["id"] >= 0, where, "invalid token id")
    _require(isinstance(prob["token"], str), where, "invalid token text")
    if "bytes" in prob:
        _require(isinstance(prob["bytes"], list) and all(type(b) is int and 0 <= b <= 255 for b in prob["bytes"]),
                 where, "invalid token bytes")
    value = prob["logprob"]
    _require(type(value) in (int, float) and math.isfinite(value), where, "logprob must be finite")


def validate(data, expected_prompts=5):
    _require(isinstance(data, dict) and data.get("schema") == 1, "run", "expected complete schema-1 capture")
    _require(data.get("mtp") in ("off", "on"), "run", "missing MTP mode")
    _require(data.get("n_predict") == 96 and data.get("n_probs") == 5, "run", "expected 96 tokens/top-5 request")
    prompts = data.get("prompts")
    _require(isinstance(prompts, list) and len(prompts) == expected_prompts, "run",
             f"expected exactly {expected_prompts} prompts")
    for i, prompt in enumerate(prompts):
        where = f"prompt[{i}]"
        _require(isinstance(prompt, dict) and isinstance(prompt.get("content"), str), where, "missing content")
        probs = prompt.get("probs")
        _require(isinstance(probs, list) and 0 < len(probs) <= 96, where, "expected 1..96 generated tokens")
        _require(type(prompt.get("tokens_predicted")) is int and prompt["tokens_predicted"] == len(probs),
                 where, "tokens_predicted must equal the complete probability stream length")
        _require(prompt.get("truncated") is False, where, "missing termination receipt or context truncated")
        _require(prompt.get("finish_reason") in ("length", "eos"), where, "missing/invalid completion termination")
        _require(prompt["finish_reason"] == "eos" or len(probs) == 96, where, "early length stop without EOS")
        for j, prob in enumerate(probs):
            at = f"{where}.token[{j}]"
            _prob(prob, at)
            top = prob.get("top_logprobs")
            _require(isinstance(top, list) and len(top) == 5, at, "expected exactly five ordered top_logprobs")
            for k, p in enumerate(top):
                _prob(p, f"{at}.top[{k}]")
            _require(len({p["id"] for p in top}) == 5, at, "duplicate top-5 token IDs")
        prompt_ids = prompt.get("prompt_ids")
        _require(isinstance(prompt_ids, list) and len(prompt_ids) > 0 and
                 all(type(t) is int and t >= 0 for t in prompt_ids), where, "missing/invalid prompt token IDs")
        _exact(prompt.get("ids"), [p["id"] for p in probs], where + ".ids")
        _exact(prompt.get("top"), [[[p["id"], p["logprob"]] for p in c["top_logprobs"]] for c in probs], where + ".top")
    return prompts


def _exact(a, b, where):
    if isinstance(a, dict) and isinstance(b, dict):
        _require(a.keys() == b.keys(), where, "object fields differ")
        for key in a:
            _exact(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        _require(len(a) == len(b), where, f"length differs: {len(a)} vs {len(b)}")
        for i in range(len(a)):
            _exact(a[i], b[i], f"{where}[{i}]")
    elif where.endswith(".logprob") or (type(a) is float and type(b) in (int, float)) or (type(b) is float and type(a) in (int, float)):
        _require(struct.pack(">d", float(a)) == struct.pack(">d", float(b)), where,
                 f"IEEE bits differ: {a!r} vs {b!r}")
    else:
        _require(type(a) is type(b) and a == b, where, f"differs: {a!r} vs {b!r}")


def compare(a, b, expected_prompts=5):
    pa, pb = validate(a, expected_prompts), validate(b, expected_prompts)
    _require(isinstance(a, dict) == isinstance(b, dict), "run", "different recording schemas")
    if isinstance(a, dict):
        # URL, server properties and timing receipts differ legitimately across builds.
        for key in ("schema", "engine", "model", "mtp", "n_predict", "n_probs", "prompt_sha256", "request_extra", "server_config"):
            _require(key in a and key in b, "run", f"missing {key}")
            _exact(a[key], b[key], "run." + key)
        _require(isinstance(a["prompt_sha256"], list) and len(a["prompt_sha256"]) == expected_prompts,
                 "run", "missing frozen prompt identities")
        from bench.xengine.collect_outputs import PROMPT_SHA256
        _require(a["prompt_sha256"] == PROMPT_SHA256[:expected_prompts], "run", "not the frozen S5 prompt suite")
    for i in range(expected_prompts):
        for key in ("content", "tokens_predicted", "finish_reason", "truncated", "probs", "prompt_ids", "ids", "top"):
            _require(key in pa[i] and key in pb[i], f"prompt[{i}]", f"missing {key}")
            _exact(pa[i][key], pb[i][key], f"prompt[{i}].{key}")
    return sum(len(p["probs"]) for p in pa)


def load(path):
    # Duplicate object fields make recordings ambiguous; reject rather than last-value-wins.
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise Mismatch(f"duplicate JSON field {key!r}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--prompts", type=int, default=5, help="expected prompt count (default five frozen S5 prompts)")
    args = parser.parse_args(argv)
    if args.prompts < 1:
        parser.error("--prompts must be positive")
    try:
        tokens = compare(load(args.old), load(args.new), args.prompts)
    except (ValueError, OSError, TypeError, OverflowError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"EXACT: {args.prompts} prompts, {tokens} tokens, full ordered top-5 probability objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
