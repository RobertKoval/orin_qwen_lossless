#!/usr/bin/env python3
"""Profile decode at long context under a server started with NSYS=<name>: an N-token haystack prompt once (prefill),
then POST /start_profile, the same prompt again (prefix-cache hit, so the request is almost all decode, 120 tokens),
/stop_profile, wait for out/<name>*.sqlite.   longctx_profile.py <name> <port> <tokens>"""
import glob, json, os, pathlib, sys, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/serving"))
sys.argv, (name, port, n) = [sys.argv[0], sys.argv[2], "ctx"], sys.argv[1:4]
import long_ctx_test as L  # noqa: E402  (reads port from sys.argv[1])

hay = L.haystack(int(n))
p = f"<|im_start|>user\n{hay}\n\nSummarize the collection in about 300 words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
body = lambda m: {"model": L.model, "prompt": p, "max_tokens": m, "temperature": 0, "ignore_eos": True}  # noqa: E731
t0 = time.time(); json.load(L.post("/v1/completions", body(4))); print(f"prefill done in {time.time() - t0:.0f} s", flush=True)
post = lambda path: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", b"", {})).read()  # noqa: E731
post("/start_profile")
t0 = time.time(); r = json.load(L.post("/v1/completions", body(120)))
print(f"decode request {time.time() - t0:.1f} s, usage {r['usage']}", flush=True)
post("/stop_profile")
out = str(ROOT / "out") + "/"
for _ in range(120):
    time.sleep(5)
    if glob.glob(out + name + "*.sqlite"):
        time.sleep(10)
        break
print(glob.glob(out + name + "*"))
