#!/usr/bin/env python3
"""Keep the server decoding (greedy, 256-token requests) for N seconds, e.g. while nsys captures a window.
  decode_loop.py [port=8010] [seconds=150]"""
import json, sys, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8010"
t_end = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 150)
p = "<|im_start|>user\nWrite a long story about a lighthouse keeper.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
n = 0
while time.time() < t_end:
    body = json.dumps(dict(model="qwen3-a3b-30b", prompt=p, max_tokens=256, temperature=0, ignore_eos=True)).encode()
    t0 = time.time()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                                  {"Content-Type": "application/json"})).read()
    n += 1
    print(f"request {n}: {256 / (time.time() - t0):.2f} tok/s (incl. prefill)", flush=True)
