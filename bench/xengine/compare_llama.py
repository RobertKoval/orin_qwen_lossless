#!/usr/bin/env python3
"""M2 cross-engine check: vLLM-BF16N against the llama.cpp BF16N reference (out/m2-llama-ref.json, bench/xengine/ref_llama.sh)
on the run_prompts.sh prompts, in the conventions of bench/xengine/compare_outputs.py.

1. Greedy: vLLM generates 96 tokens per prompt (temperature 0, top-5 logprobs, token ids); text identity, first top-1
   divergence, and whether it is a near-tie (top-2 gap < NEAR_TIE in either engine at that position).
2. Teacher-forced: vLLM scores llama.cpp's own tokens (prompt_logprobs over prompt + llama continuation), so every
   position compares the two engines on identical context: top-1 agreement, |dlogprob| of llama's token, and a
   top-5 KL estimate KL(llama || vLLM) (each engine's top-5 plus one tail bucket; a token missing from vLLM's top-5
   gets vLLM's 5th logprob as an upper bound of its probability).
   compare_llama.py [port=8010] [tag=m2] [ref=out/m2-llama-ref.json] -> out/<tag>-compare.json
   (ref may also be a vLLM run written by ref_vllm.py, e.g. the BF16N_REFERENCE server)
"""
import json
import math
import pathlib
import statistics
import sys
import urllib.request

root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import N_PREDICT, PROMPTS, TOP_K  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else "8010"
tag = sys.argv[2] if len(sys.argv) > 2 else "m2"
NEAR_TIE = 0.05
ref = json.load(open(root / (sys.argv[3] if len(sys.argv) > 3 else "out/m2-llama-ref.json")))


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def gap(top):
    s = sorted((lp for _, lp in top), reverse=True)
    return s[0] - s[1] if len(s) > 1 else math.inf


def kl_top5(p_top, q_top):
    """KL(P || Q) over P's top-5 + tail bucket; Q's probability of a token outside its top-5 is bounded by its 5th."""
    q = dict(q_top)
    q_floor = min(q.values())
    pt = sum(math.exp(lp) for _, lp in p_top)
    qt = sum(math.exp(lp) for lp in q.values())
    kl = 0.0
    q_used = 0.0
    for tok, lp in p_top:
        lq = q.get(tok, q_floor)
        q_used += math.exp(lq)
        kl += math.exp(lp) * (lp - lq)
    p_tail, q_tail = max(1.0 - pt, 1e-12), max(1.0 - max(qt, q_used), 1e-12)
    return kl + p_tail * math.log(p_tail / q_tail)


report = []
dlp, agree, kls = [], [], []
for i, (p, r) in enumerate(zip(PROMPTS, ref)):
    # 1. greedy generation
    g = post("/v1/completions", {"model": "qwen3-a3b-30b", "prompt": p, "max_tokens": N_PREDICT, "temperature": 0,
                                 "logprobs": TOP_K, "return_tokens_as_token_ids": True})["choices"][0]
    v_ids = [int(t.split(":")[1]) for t in g["logprobs"]["tokens"]]
    v_top = [[(int(k.split(":")[1]), lp) for k, lp in d.items()] for d in g["logprobs"]["top_logprobs"]]
    n = min(len(v_ids), len(r["ids"]))
    div = next((k for k in range(n) if v_ids[k] != r["ids"][k]), None)
    near = None if div is None else min(gap(r["top"][div]), gap(v_top[div])) < NEAR_TIE
    # 2. teacher-forced on llama's tokens
    prompt_ids = post("/tokenize", {"model": "qwen3-a3b-30b", "prompt": p})["tokens"]
    tf = post("/v1/completions", {"model": "qwen3-a3b-30b", "prompt": prompt_ids + r["ids"], "max_tokens": 1,
                                  "temperature": 0, "prompt_logprobs": TOP_K})["choices"][0]["prompt_logprobs"]
    P = len(prompt_ids)
    p_agree, p_dlp, p_kl = [], [], []
    for k, tok in enumerate(r["ids"]):
        d = tf[P + k]
        q_top = sorted(((int(t), v["logprob"]) for t, v in d.items()), key=lambda kv: -kv[1])
        q_top5 = [kv for kv in q_top if d[str(kv[0])]["rank"] <= TOP_K]
        l_top = r["top"][k]
        p_agree.append(q_top5[0][0] == l_top[0][0])
        l_lp = dict(l_top).get(tok)
        if l_lp is not None:
            p_dlp.append(abs(d[str(tok)]["logprob"] - l_lp))
        p_kl.append(kl_top5(l_top, q_top5))
    agree += p_agree; dlp += p_dlp; kls += p_kl
    item = dict(prompt=i, prompt_ids_equal=prompt_ids == r["prompt_ids"], text_identical=g["text"] == r["content"],
                first_divergence=div, near_tie=near, n_tokens=n,
                llama_gap=None if div is None else gap(r["top"][div]),
                vllm_gap=None if div is None else gap(v_top[div]),
                tf_top1_agree=sum(p_agree) / len(p_agree), tf_kl_mean=statistics.mean(p_kl),
                tf_dlogprob_median=statistics.median(p_dlp), tf_dlogprob_max=max(p_dlp), vllm_text=g["text"])
    report.append(item)
    print("prompt%d: prompt_ids_equal=%s text_identical=%s first_top1_divergence=%s near_tie=%s | teacher-forced: "
          "top1 agree %.3f, KL %.2e, |dlogprob| median %.2e max %.2e" % (
              i, item["prompt_ids_equal"], item["text_identical"], div, near, item["tf_top1_agree"],
              item["tf_kl_mean"], item["tf_dlogprob_median"], item["tf_dlogprob_max"]))
s = sorted(dlp)
print("teacher-forced over %d tokens: top-1 agreement %.4f, mean KL(top-5 est.) %.2e (p99 %.2e, max %.2e), "
      "|dlogprob| median %.2e p99 %.2e max %.2e" % (
          len(agree), sum(agree) / len(agree), statistics.mean(kls), sorted(kls)[int(0.99 * (len(kls) - 1))],
          max(kls), statistics.median(s), s[int(0.99 * (len(s) - 1))], s[-1]))
json.dump(dict(prompts=report, top1_agree=sum(agree) / len(agree), kl_mean=statistics.mean(kls),
               dlogprob_median=statistics.median(s), dlogprob_p99=s[int(0.99 * (len(s) - 1))], dlogprob_max=s[-1]),
          open(root / f"out/{tag}-compare.json", "w"), indent=1)
