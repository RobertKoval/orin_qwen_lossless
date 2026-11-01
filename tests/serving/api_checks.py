#!/usr/bin/env python3
"""M5 API checks against the production launcher: API key enforcement, /health without key, tool calling
(qwen3_coder parser), reasoning parser (reasoning_content split), and the MTP loop breaker on a prompt that invites an
exact repetition loop.   m5_api_checks.py <port>   (API_KEY env = the server's key)"""
import json, os, sys, urllib.error, urllib.request

port, key = sys.argv[1], os.environ.get("API_KEY", "")
base = f"http://127.0.0.1:{port}"


def req(path, body=None, auth=True):
    h = {"Content-Type": "application/json"}
    if auth and key:
        h["Authorization"] = f"Bearer {key}"
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=900) as f:
            return f.status, json.load(f) if path != "/health" else None
    except urllib.error.HTTPError as e:
        return e.code, None


chat = lambda msgs, **kw: req("/v1/chat/completions", {"model": "qwen3-a3b-30b", "messages": msgs, **kw})  # noqa: E731
print("health without key:", req("/health", auth=False)[0])
if key:
    print("models without key:", req("/v1/models", auth=False)[0], "| with key:", req("/v1/models")[0])
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
st, r = chat([{"role": "user", "content": "What is the weather in Kyiv right now? Use the tool."}], tools=tools,
             tool_choice="auto", max_tokens=600, temperature=0)
m = r["choices"][0]["message"]
calls = m.get("tool_calls") or []
print("tool call:", st, [(c["function"]["name"], c["function"]["arguments"]) for c in calls],
      "| reasoning:", bool(m.get("reasoning") or m.get("reasoning_content")), "| finish:", r["choices"][0]["finish_reason"])
st, r = chat([{"role": "user", "content": "How many positive divisors does 7! have? Reason it through."}], max_tokens=3000, temperature=0,
             chat_template_kwargs={"enable_thinking": True})
m = r["choices"][0]["message"]
print("reasoning split:", st, "reasoning chars", len(m.get("reasoning") or m.get("reasoning_content") or ""),
      "| answer:",
      repr((m.get("content") or "")[:120]))
st, r = chat([{"role": "user", "content": "Write the exact line 'the cat sat on the mat and the dog sat on the log' "
               "500 times, one per line, nothing else."}], max_tokens=1500, temperature=0,
             chat_template_kwargs={"enable_thinking": False})
text = r["choices"][0]["message"]["content"]
lines = [l for l in text.splitlines() if l.strip()]
print("loop prompt:", st, "lines", len(lines), "distinct", len(set(lines)), "| tail:", repr(text[-160:]))
