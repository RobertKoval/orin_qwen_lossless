#!/usr/bin/env python3
"""Profile steady decode under a server started with NSYS=<name> (engines/vllm/tools/run_dev.sh): warm up, POST /start_profile,
decode for N seconds, POST /stop_profile, wait for nsys to write out/<name>*.sqlite.
  decode_profile.py <name> [port=8010] [seconds=15]"""
import glob, json, pathlib, subprocess, sys, time, urllib.request
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
name = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else "8010"; secs = sys.argv[3] if len(sys.argv) > 3 else "15"
post = lambda path: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", b"", {})).read()
subprocess.run([sys.executable, str(HERE / "decode_loop.py"), port, "20"], check=True)  # warm-up
post("/start_profile")
subprocess.run([sys.executable, str(HERE / "decode_loop.py"), port, secs], check=True)
post("/stop_profile")
out = str(ROOT / "out") + "/"
for _ in range(120):
    time.sleep(5)
    if glob.glob(out + name + "*.sqlite"):
        time.sleep(10)
        break
print(glob.glob(out + name + "*"))
