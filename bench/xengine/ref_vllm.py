#!/usr/bin/env python3
"""A vLLM server's greedy outputs in ref_llama.py's format (prompt ids, generated ids, top-5 by id), e.g. the
BF16N_REFERENCE server (exact dequant + reference kernels) as the reference for compare_llama.py.
  ref_vllm.py <out.json> [port=8010]"""
import json
import pathlib
import sys
import urllib.request

root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import N_PREDICT, PROMPTS, TOP_K  # noqa: E402

out_path, port = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "8010")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


out = []
for p in PROMPTS:
    c = post("/v1/completions", {"model": "qwen3-a3b-30b", "prompt": p, "max_tokens": N_PREDICT, "temperature": 0,
                                 "logprobs": TOP_K, "return_tokens_as_token_ids": True})["choices"][0]
    out.append(dict(prompt_ids=post("/tokenize", {"model": "qwen3-a3b-30b", "prompt": p})["tokens"], content=c["text"],
                    ids=[int(t.split(":")[1]) for t in c["logprobs"]["tokens"]],
                    top=[sorted(((int(k.split(":")[1]), lp) for k, lp in d.items()), key=lambda kv: -kv[1])
                         for d in c["logprobs"]["top_logprobs"]]))
json.dump(out, open(root / out_path, "w"))
print("wrote", out_path, [len(o["ids"]) for o in out])
