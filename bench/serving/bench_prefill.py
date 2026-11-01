#!/usr/bin/env python3
"""Repeated prefill measurement (T-038): prompts of given lengths built from the frozen corpus (bench/prompts), each with a unique random
prefix (no prefix-cache hits), max_tokens=1, TTFT from a streamed request. Prints every run and median / min / max.
  bench_prefill.py [port=8010] [reps=5] [lengths=2048,8192,23000] -> out/prefill-<tag>.json (tag from $TAG)"""
import json, os, pathlib, statistics, sys, time, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8010"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
lengths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2048,8192,23000").split(",")]
root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import chat, prefill_doc  # noqa: E402

doc = prefill_doc()
model = os.environ.get("MODEL_NAME", "qwen3-a3b-30b")


def post(path, body, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def ntok(text):
    try:
        return len(json.load(post("/tokenize", {"model": model, "prompt": text}))["tokens"])
    except Exception:  # llama-server
        return len(json.load(post("/tokenize", {"content": text}))["tokens"])


def prompt_of(n):
    lo, hi = 100, len(doc)  # bisect the document length to n tokens
    while hi - lo > 64:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ntok(chat("[00000000]\n" + doc[:mid])) < n else (lo, mid)
    return doc[:lo]


res = {}
for n in lengths:
    body = prompt_of(n)
    runs = []
    for _ in range(reps):
        p = chat(f"[{os.urandom(4).hex()}]\n" + body)
        t0 = time.time()
        first = None
        usage = None
        for line in post("/v1/completions", {"model": model, "prompt": p, "max_tokens": 1, "temperature": 0,
                                             "stream": True, "stream_options": {"include_usage": True}}):
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                j = json.loads(line[6:])
                if j.get("usage"):
                    usage = j["usage"]
                if first is None and any(c.get("text") for c in j.get("choices", [])):
                    first = time.time()
        pt = usage["prompt_tokens"] if usage else ntok(p)
        runs.append(pt / ((first or time.time()) - t0))
        print(f"{pt:6d} tokens: {runs[-1]:7.1f} tok/s", flush=True)
    res[n] = runs
    print(f"== {n}: median {statistics.median(runs):.0f}, min {min(runs):.0f}, max {max(runs):.0f} tok/s", flush=True)
json.dump(res, open(root / f"out/prefill-{os.environ.get('TAG', 'run')}.json", "w"))
