"""Speculative-decoding counters of a running vLLM server."""
import urllib.request


def spec_counters(port):
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
