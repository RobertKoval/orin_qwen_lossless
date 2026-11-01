#!/usr/bin/env python3
"""T-046 probe: N sequential greedy requests (the 4 bench_openai prompts in turn, 256 tokens, no logprobs, streamed);
per request: decode tok/s, the longest gap between stream chunks (one chunk = one engine step), and the major page
faults (swap-ins / file reads) and CPU time of the API server and engine core during it, plus system swap-in pages.
LOGPROBS=k: request top-k logprobs (bench_openai's default mode is 5).
LONG_EVERY=k: after every k requests one ~23k-token prefill with a unique prefix (as bench_openai's prefill test).
  stall_probe.py <port> <n> -> out/stall-<tag>.json (TAG env), summary on stdout"""
import json, os, pathlib, subprocess, sys, time, urllib.request

port, n = sys.argv[1], int(sys.argv[2])
root = pathlib.Path(__file__).resolve().parents[2]  # repo root
src = (root / "bench/serving/bench_openai.py").read_text()
ns = {"__file__": str(root / "bench/serving/bench_openai.py"), "sys": sys}
sys.argv = [sys.argv[0], "probe", port]
exec(src[:src.index("LP = int(")], ns)  # prompts and chat template only
prompts = ns["prompts"]


def pids():
    out = {}
    for name in ("VLLM::EngineCor",):
        r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout.split()
        if r:
            out["engine"] = int(r[0])
    r = subprocess.run(["pgrep", "-x", "python3"], capture_output=True, text=True).stdout.split()
    for p in r:
        try:
            if b"vllm.entrypoints.openai.api_server" in open(f"/proc/{p}/cmdline", "rb").read():
                out["api"] = int(p)
        except OSError:
            pass
    return out


def stat(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[9]), int(f[11]) + int(f[12])  # majflt, utime+stime (ticks)


def pswpin():
    for line in open("/proc/vmstat"):
        if line.startswith("pswpin "):
            return int(line.split()[1])


P = pids()
res = []
long_every = int(os.environ.get("LONG_EVERY", "0"))


def long_prefill():
    p = ns["chat"](f"[{os.urandom(4).hex()}]\n" + ns["doc"] + "\nSummarize in one sentence.")
    body = json.dumps(dict(model="qwen3-a3b-30b", prompt=p, max_tokens=8, temperature=0)).encode()
    t = time.time()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                                  {"Content-Type": "application/json"})).read()
    print(f"    long prefill {time.time() - t:.1f} s", flush=True)


for i in range(n):
    if long_every and i and i % long_every == 0:
        long_prefill()
    prompt = prompts[i % len(prompts)]
    s0 = {k: stat(v) for k, v in P.items()}; sw0 = pswpin()
    req = dict(model="qwen3-a3b-30b", prompt=prompt, max_tokens=256, temperature=0, stream=True,
               stream_options={"include_usage": True})
    if int(os.environ.get("LOGPROBS", "0")):
        req["logprobs"] = int(os.environ["LOGPROBS"])
    body = json.dumps(req).encode()
    t0 = time.time(); times = []; usage = None
    for line in urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                                              {"Content-Type": "application/json"})):
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        if any(c.get("text") for c in j.get("choices", [])):
            times.append(time.time())
    s1 = {k: stat(v) for k, v in P.items()}; sw1 = pswpin()
    gaps = [b - a for a, b in zip(times, times[1:])]
    g = max(gaps) if gaps else 0
    r = dict(i=i, prompt=i % len(prompts), t0=t0, dec=(usage["completion_tokens"] - 1) / (times[-1] - times[0]),
             max_gap=g, gap_at=times[gaps.index(g)] - t0 if gaps else 0, gap_abs=times[gaps.index(g)] if gaps else 0,
             steps=len(times),
             majflt={k: s1[k][0] - s0[k][0] for k in P}, cpu_s={k: (s1[k][1] - s0[k][1]) / 100 for k in P},
             pswpin=sw1 - sw0)
    res.append(r)
    print(f"{i:3d} p{r['prompt']} {r['dec']:5.1f} tok/s  max gap {g * 1e3:6.0f} ms  majflt {r['majflt']}  "
          f"swapin {r['pswpin']}  cpu {r['cpu_s']}", flush=True)
json.dump(res, open(root / f"out/stall-{os.environ.get('TAG', 'probe')}.json", "w"), indent=1)
by = {}
for r in res:
    by.setdefault(r["prompt"], []).append(r["dec"])
med = {k: sorted(v)[len(v) // 2] for k, v in by.items()}
slow = [r for r in res if r["dec"] < 0.85 * med[r["prompt"]]]
print(f"{len(slow)} slow of {n} (< 85 % of the prompt's median {med}):",
      [(r["i"], round(r["dec"], 1), round(r["max_gap"], 2), r["majflt"], r["pswpin"]) for r in slow])
