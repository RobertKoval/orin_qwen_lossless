#!/usr/bin/env python3
"""llama.cpp reference for M2: greedy completions of the run_prompts.sh prompts with top-5 logprobs, by token id.
  ref_llama.py [port=8099] -> out/m2-llama-ref.json"""
import json
import pathlib
import sys
import urllib.request

root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import N_PREDICT, PROMPTS, TOP_K  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else "8099"


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


out = []
for p in PROMPTS:
    r = post("/completion", {"prompt": p, "n_predict": N_PREDICT, "temperature": 0, "n_probs": TOP_K,
                             "cache_prompt": False})
    probs = r["completion_probabilities"]
    out.append(dict(
        prompt_ids=post("/tokenize", {"content": p, "parse_special": True})["tokens"],
        content=r["content"],
        ids=[c["id"] for c in probs],
        top=[[(t["id"], t["logprob"]) for t in c["top_logprobs"]] for c in probs],
        tok_s=r["timings"]["predicted_per_second"],
    ))
json.dump(out, open(root / "out/m2-llama-ref.json", "w"))
print("llama ref tok/s:", [round(o["tok_s"], 2) for o in out], "tokens:", [len(o["ids"]) for o in out])
