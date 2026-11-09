#!/usr/bin/env python3
"""GDN state checks against a running server (T-066): no request may see another request's recurrent state, and a
prefix-cache hit must give the same next-token distribution as computing the prompt from scratch.

  state_checks.py <port> [--fresh]      exit 0 = all pass

--fresh: the server has just started; the first generation is the 4-token probe '2+2=' (the shape-aliased prefill that
returned NaN, '!!!!!!!!', on vLLM 0.20). Checks:
  [1] (--fresh) first generation finite, not all token 0
  [2] short prompts (1-5 tokens; 4 = 1 + MTP 3 aliases the verify shape) give the same greedy text after two different
      preceding requests, with finite logprobs
  [3] prefix-cache hits: a filling prompt of 1072 / 2144 / 2184 tokens (block boundary, two blocks, block + tail) under a
      fresh cache_salt, then a query sharing 1 or 2 blocks; the query's first-token top-5 logprobs must match a
      no-cache run (own salt) within 0.05 (a wrong state was off by ~91). Hit sizes from /metrics are printed.
Block size 1072 is the production layout (deploy/orin/run_qwen3_a3b_vllm_bf16n.sh)."""
import json, math, random, re, sys, urllib.request

port = sys.argv[1]
U = f"http://127.0.0.1:{port}"
fails = []


def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(U + path, json.dumps(body).encode(),
                     {"Content-Type": "application/json"}), timeout=600))


def gen(prompt, n=8, salt=None, lp=5):
    body = {"model": "qwen3-a3b-30b", "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": lp}
    if salt:
        body["cache_salt"] = salt
    try:
        c = post("/v1/completions", body)["choices"][0]
    except urllib.error.HTTPError as e:  # vLLM answers 400 when logprobs are NaN/inf (not JSON-serialisable)
        return None, None, e.read().decode()[:100]
    return c["text"], c["logprobs"]["top_logprobs"][0] if lp else None, None


def hits():
    m = urllib.request.urlopen(U + "/metrics", timeout=30).read().decode()
    return sum(float(x) for x in re.findall(r"^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)", m, re.M))


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


if "--fresh" in sys.argv:
    t, lp, err = gen("2+2=")
    check(err is None and t.strip("!") != "", f"[1] first generation '2+2=' -> {t!r} {err or ''}")

pre = ["The capital of France is", "Write a long story about a dragon who lives in the mountains of Norway and"]
for p in ["Hi", "2+2", "2+2=", "One two three four", "the quick brown fox jumps"]:
    outs = []
    for q in pre:
        gen(q, 16, lp=None)
        outs.append(gen(p))
    finite = all(o[2] is None and all(math.isfinite(v) for v in o[1].values()) for o in outs)
    check(finite and outs[0][0] == outs[1][0], f"[2] {p!r}: {outs[0][0]!r} | {outs[1][0]!r}")

tok = lambda s: post("/tokenize", {"model": "qwen3-a3b-30b", "prompt": s})["tokens"]  # noqa: E731
text = " ".join(f"Section {i}: the committee reviewed item {i * 7 % 101} and noted that the budget for project "
                f"{i * 13 % 97} was revised to {i * 31 % 1000} units after the audit of quarter {i % 4 + 1}." for i in range(400))
A = tok(text)[:2400]
for P in (1072, 2144, 2184):
    for Q in (1080, 2152):
        salt = f"fill-{P}-{Q}-{random.random()}"
        gen(A[:P], 1, salt=salt, lp=None)
        h0 = hits()
        t, lp, err = gen(A[:Q], 10, salt=salt)
        hit = hits() - h0
        rt, rlp, _ = gen(A[:Q], 10, salt=f"ref-{random.random()}")
        d = max(abs(lp.get(k, -99) - v) for k, v in rlp.items()) if err is None else math.inf
        check(d <= 0.05, f"[3] fill {P}, query {Q}: hit {hit:.0f} tokens, max|dlogp| {d:.3f} {t!r} | ref {rt!r}")

print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
