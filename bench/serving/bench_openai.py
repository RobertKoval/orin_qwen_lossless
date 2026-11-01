#!/usr/bin/env python3
"""Greedy decode / prefill benchmark against an OpenAI-compatible server (vLLM or llama-server).
  bench_openai.py <tag> [port] [n_tokens]  ->  out/vllm-<tag>.json
Decode tok/s = (n-1) / (t_last - t_first) over the stream, n counted in tokens (logprobs) rather than chunks; prefill tok/s = prompt_tokens / TTFT on a ~12k-token prompt
built from the frozen corpus (bench/prompts; cache disabled by a unique prefix).
LOGPROBS=0: no logprobs requested (as real clients); decode = (completion_tokens from usage - 1) / (last - first chunk)."""
import json, sys, time, urllib.request, pathlib, os

tag = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "8010"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = "qwen3-a3b-30b"
root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import chat, serving_long_doc, serving_prompts  # noqa: E402

doc = serving_long_doc()
prompts = serving_prompts()

LP = int(os.environ.get("LOGPROBS", "5"))


def spec_counters():
    """(draft, accepted) totals from vLLM's /metrics (None without speculative decoding or on llama-server)."""
    try:
        txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    except Exception:
        return None
    v = {}
    for line in txt.splitlines():
        for k in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k):
                v[k] = v.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return (v["vllm:spec_decode_num_draft_tokens_total"], v["vllm:spec_decode_num_accepted_tokens_total"]) if len(v) == 2 else None


def run(prompt, n, logprobs=LP or None):
    body = json.dumps(dict(model=model, prompt=prompt, max_tokens=n, temperature=0, logprobs=logprobs,
                           stream=True, stream_options={"include_usage": True})).encode()
    hdr = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        hdr["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, hdr)
    t0 = time.time(); times = []; text = ""; probs = []; usage = None
    for line in urllib.request.urlopen(req):
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"): usage = j["usage"]
        for c in j["choices"]:
            if c["text"]:
                # tokens in this chunk: with speculative decoding one chunk carries every token accepted in a step,
                # so counting chunks undercounts by the mean acceptance length (the M0 MTP figure did)
                k = len((c.get("logprobs") or {}).get("tokens") or [None])
                times += [time.time()] * k; text += c["text"]
                if (c.get("logprobs") or {}).get("top_logprobs"): probs += [sorted(d.items(), key=lambda kv: -kv[1])[:5] for d in c["logprobs"]["top_logprobs"]]
    dec = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
    if logprobs is None and usage and len(times) > 1:  # chunk count != token count: use the reported total
        dec = (usage["completion_tokens"] - 1) / (times[-1] - times[0])
    gaps = [b - a for a, b in zip(times, times[1:])]
    g = max(gaps) if gaps else 0.0  # T-046: the longest pause between stream chunks and when it happened (epoch s)
    return dict(text=text, ttft=times[0] - t0, n=len(times), usage=usage, dec=dec, probs=probs, max_gap=g,
                gap_abs=times[gaps.index(g)] if gaps else 0.0)

run(prompts[0], 16)  # warm-up
for _ in range(int(os.environ.get("REPEAT", "1"))):  # REPEAT=2: report the second pass (first-pass stalls excluded)
    res = []
    for p in prompts:
        c0 = spec_counters()
        r = run(p, ntok)
        c1 = spec_counters()
        r["draft_acceptance"] = (c1[1] - c0[1]) / (c1[0] - c0[0]) if c0 and c1 and c1[0] > c0[0] else None
        res.append(r)
long_prompt = chat(f"[{os.urandom(4).hex()}]\n" + doc + "\nSummarize in one sentence.")
lp = run(long_prompt, 8, None)
pl = lp["usage"]["prompt_tokens"]
print(tag, "decode tok/s:", [round(r["dec"], 2) for r in res], "mean %.2f" % (sum(r["dec"] for r in res) / len(res)))
if any(r["draft_acceptance"] is not None for r in res):
    print(tag, "draft acceptance per prompt:", [round(r["draft_acceptance"], 3) for r in res])
print(tag, "tokens counted / reported:", [(r["n"], (r["usage"] or {}).get("completion_tokens")) for r in res])
print(tag, f"prefill: {pl} tokens, TTFT {lp['ttft']:.2f} s -> {pl/lp['ttft']:.0f} tok/s")
pathlib.Path(root / "out").mkdir(exist_ok=True)
json.dump(dict(prompts=res, prefill=dict(tokens=pl, ttft=lp["ttft"])), open(root / f"out/vllm-{tag}.json", "w"))
