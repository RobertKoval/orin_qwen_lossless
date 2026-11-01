#!/usr/bin/env python3
"""Profile one warmed-up prefill under a server started with NSYS=<name>: an N-token prompt twice (warm-up), then
POST /start_profile, the same length once more, /stop_profile, wait for out/<name>*.sqlite.
  prefill_profile.py <name> [port=8010] [tokens=8192]"""
import glob, pathlib, subprocess, sys, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[4]
name = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else "8010"; n = sys.argv[3] if len(sys.argv) > 3 else "8192"
post = lambda path: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", b"", {})).read()
subprocess.run([sys.executable, str(ROOT / "bench/serving/bench_prefill.py"), port, "2", n], check=True)
post("/start_profile")
subprocess.run([sys.executable, str(ROOT / "bench/serving/bench_prefill.py"), port, "1", n], check=True)
post("/stop_profile")
out = str(ROOT / "out") + "/"
for _ in range(120):
    time.sleep(5)
    if glob.glob(out + name + "*.sqlite"):
        time.sleep(10)
        break
print(glob.glob(out + name + "*"))
