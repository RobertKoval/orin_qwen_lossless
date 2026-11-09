#!/usr/bin/env python3
"""Concurrent decode streams against an OpenAI-compatible server (T-077): S client threads, each with its own prompts
(distinct texts, so the streams share no prefix and route experts independently), greedy, N tokens per request with
ignore_eos. Per stream: decode tok/s = (tokens - 1) / (last chunk - first chunk) of each streamed request; the run also
reports the aggregate rate while all streams decode.
  bench_streams.py <port> [streams=2] [tokens=512] [requests per stream=2]   (MODEL_NAME env, default qwen3-a3b-30b;
  TAG env: the texts go to out/streams-<TAG>.json; COMPARE=<other TAG>: greedy texts per prompt must match that run)"""
import json, os, pathlib, sys, threading, time, urllib.request

port = sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
R = int(sys.argv[4]) if len(sys.argv) > 4 else 2
model = os.environ.get("MODEL_NAME", "qwen3-a3b-30b")
PROMPTS = [
    "Write a detailed history of the bicycle, chapter by chapter, with dates and inventors.",
    "Explain, step by step and with code examples, how a compiler turns C source code into machine code.",
    "Describe a long journey across the mountains of Norway in the voice of an old sailor.",
    "Give a thorough tutorial on linear algebra for machine learning, from vectors to singular value decomposition.",
    "Tell the story of a small coastal town over three centuries, generation by generation.",
    "Explain how a modern CPU pipeline works, including branch prediction, caches and out-of-order execution.",
]


def run(prompt):
    body = json.dumps(dict(model=model, prompt=prompt, max_tokens=N, temperature=0, ignore_eos=True, stream=True,
                           stream_options=dict(include_usage=True))).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                text.append(d["choices"][0]["text"])
                now = time.time()
                first = first or now
                last = now
    n = usage["completion_tokens"] if usage else N
    return dict(ttft=first - t0, tok_s=(n - 1) / (last - first), start=first, end=last, n=n, prompt=prompt, text="".join(text))


results = [[] for _ in range(S)]


def stream(i):
    for k in range(R):
        results[i].append(run(PROMPTS[(i + S * k) % len(PROMPTS)]))


threads = [threading.Thread(target=stream, args=(i,)) for i in range(S)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0
for i, rs in enumerate(results):
    print(f"stream {i}: decode tok/s {[round(r['tok_s'], 2) for r in rs]} mean {sum(r['tok_s'] for r in rs) / len(rs):.2f}, "
          f"TTFT {[round(r['ttft'], 2) for r in rs]}", flush=True)
both_start = max(rs[0]["start"] for rs in results)
both_end = min(rs[-1]["end"] for rs in results)
tokens = sum(r["n"] for rs in results for r in rs)
print(f"{S} streams x {R} requests x {N} tokens: wall {wall:.1f} s, aggregate {tokens / wall:.2f} tok/s incl. prefill; "
      f"all streams decoding together for {max(0.0, both_end - both_start):.1f} s")

root = pathlib.Path(__file__).resolve().parents[2]
(root / "out").mkdir(exist_ok=True)
tag = os.environ.get("TAG", "run")
texts = {r["prompt"]: r["text"] for rs in results for r in rs}
json.dump(dict(streams=S, tokens=N, results=results), open(root / f"out/streams-{tag}.json", "w"))
other = os.environ.get("COMPARE")
if other:
    ref = {r["prompt"]: r["text"] for rs in json.load(open(root / f"out/streams-{other}.json"))["results"] for r in rs}
    common = [p for p in texts if p in ref]
    same = [p for p in common if texts[p] == ref[p]]
    print(f"greedy texts equal to {other}: {len(same)} of {len(common)} prompts")
    for p in common:
        if texts[p] != ref[p]:
            a, b = texts[p], ref[p]
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(f"  differs at char {k}: {p[:40]!r}")
