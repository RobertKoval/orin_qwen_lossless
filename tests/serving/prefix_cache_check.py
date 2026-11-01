#!/usr/bin/env python3
"""Prefix-cache state check for MTP + GDN align mode (M6, async_align.py): a greedy multi-turn conversation whose
second turn reuses blocks completed during the first turn's decode (GDN states copied into prefix-cache blocks by
the accepted-token bookkeeping). Writes the turn-2 tokens / top logprobs and the cached-token count to
out/prefix-<tag>.json so that two server configurations can be compared exactly (compare: same file format).
  prefix_cache_check.py <tag> [port=8010]          -> out/prefix-<tag>.json
  prefix_cache_check.py --compare <tagA> <tagB>"""
import json, os, pathlib, sys, urllib.request

root = pathlib.Path(__file__).resolve().parents[2]  # repo root
if sys.argv[1] == "--compare":
    a, b = (json.load(open(root / f"out/prefix-{t}.json")) for t in sys.argv[2:4])
    for k in a:
        same = a[k]["tokens"] == b[k]["tokens"]
        n = next((i for i, (x, y) in enumerate(zip(a[k]["tokens"], b[k]["tokens"])) if x != y), None)
        print(f"{k}: (cached, prompt) {a[k]['cached']} / {b[k]['cached']}, tokens identical: {same}" + ("" if same else f" (first diff at {n})"))
    sys.exit(0)

tag = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "8010"
hdr = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    hdr["Authorization"] = "Bearer " + os.environ["API_KEY"]


def complete(prompt, n):
    body = dict(model="qwen3-a3b-30b", prompt=prompt, max_tokens=n, temperature=0, logprobs=1)
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                                                json.dumps(body).encode(), hdr), timeout=900))
    c = r["choices"][0]
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return c["text"], c["logprobs"]["tokens"], (cached, r["usage"]["prompt_tokens"])


sys.path.insert(0, str(root))
from bench.prompts.prompts import chat, prefix_cache_doc  # noqa: E402

doc = prefix_cache_doc()
out = {}
for name, q in (("en", "Explain how a hash map works, step by step."), ("uk", "Розкажи детально, як працює хеш-таблиця.")):
    p1 = chat(doc + "\n\n" + q)
    t1, _, (_, n1) = complete(p1, 1200)  # decode crosses a 1072-token block boundary: state copied by bookkeeping
    p2 = p1 + t1 + "<|im_end|>\n" + chat("Now summarise your answer in two sentences.")
    t2, toks2, cached = complete(p2, 120)
    out[name] = dict(turn1=t1, turn2=t2, tokens=toks2, cached=cached)
    print(name, f"turn-1 prompt {n1} + 1200 generated; turn-2 prompt {cached[1]}, cached {cached[0]}", "| turn-2:", t2[:80].replace("\n", " "))
json.dump(out, open(root / f"out/prefix-{tag}.json", "w"), ensure_ascii=False, indent=1)
