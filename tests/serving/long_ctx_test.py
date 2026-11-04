#!/usr/bin/env python3
"""M5 long-context / vision checks against a running server (vLLM or llama-server, OpenAI completions API):
  image  - a 3840x2160 test image with text drawn on it (chat completions, base64); reports the answer
  ctx N  - an N-token haystack (frozen docs + llama.cpp sources) with a needle at 50 % depth; one request that asks for
           the needle and then a ~300-word summary: prefill tok/s (TTFT), decode tok/s (streamed tokens after the first,
           counted by logprobs), needle found or not
After every check prints MemFree + Cached of this machine (the Orin), i.e. the memory margin. CTX_SALT=<text> fixes
           the prompt's leading salt (default random: no prefix-cache hit), so two servers get identical prompts.
  plp N  - an N-token prompt with prompt_logprobs=5 (T-045: must not kill the engine); reports time and entry count
  images K - K 3840x2160 images in one chat request
  long_ctx_test.py <port> image | images <K> | ctx <tokens> [<tokens> ...] | plp <tokens> [...]   (MODEL_NAME, API_KEY env)"""
import base64, io, json, os, pathlib, random, sys, time, urllib.request

port, what = sys.argv[1], sys.argv[2]
model = os.environ.get("MODEL_NAME", "qwen3-a3b-30b")
key = os.environ.get("API_KEY", "")
root = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(root))
from bench.prompts.prompts import haystack_docs  # noqa: E402
from bench.serving.spec_metrics import spec_counters  # noqa: E402
NEEDLE = "The secret passphrase for the Orin vault is TANGERINE-4471-OBSIDIAN."


def headers():
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                                         headers()), timeout=3600)


def margin():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemFree", "Cached")}
    return f"MemFree+Cached {(m['MemFree'] + m['Cached']) / 2**20:.2f} GiB"


def corpus():
    """Frozen docs (bench/prompts) + the sources of the llama.cpp tree LLAMACPP_SRC (pin + BF16N series, as built by
    engines/llamacpp/build.sh; the published runs used that tree with the patch applied)."""
    src = pathlib.Path(os.environ.get("LLAMACPP_SRC", root / "engines/llamacpp/src"))
    if not (src / "src").is_dir():
        sys.exit(f"long_ctx_test: LLAMACPP_SRC={src} is not a llama.cpp tree (set LLAMACPP_SRC)")
    files = sorted(src.glob("docs/**/*.md")) + sorted(src.glob("src/*.cpp")) + sorted(src.glob("ggml/src/*.c")) + \
        sorted(src.glob("tools/**/*.cpp"))
    return "\n\n".join([f"=== {n} ===\n" + t for n, t in haystack_docs()] +
                       [f"=== {f.name} ===\n" + f.read_text(errors="ignore") for f in files])


def ntok(text):
    try:
        return len(json.load(post("/tokenize", {"model": model, "prompt": text}))["tokens"])
    except Exception:
        return len(json.load(post("/tokenize", {"content": text}))["tokens"])


def haystack(n):
    text = corpus()
    ratio = 400000 / ntok(text[:400000])  # chars per token on a sample
    body = text[: int(n * ratio)]
    while len(body) < n * ratio:
        body += text
    body = body[: int(n * ratio)]
    mid = len(body) // 2
    mid = body.rfind("\n", 0, mid) + 1
    return body[:mid] + "\n" + NEEDLE + "\n" + body[mid:]


def ctx_test(n):
    hay = haystack(n)
    q = ("Above is a long collection of documents and source files. Somewhere in it there is a secret passphrase for "
         "the Orin vault. First state the passphrase exactly. Then summarize the collection in about 300 words.")
    p = f"<|im_start|>user\n[{os.environ.get('CTX_SALT') or os.urandom(4).hex()}]\n{hay}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    c0 = spec_counters(port)
    t0 = time.time(); times = []; text = ""; usage = None
    for line in post("/v1/completions", {"model": model, "prompt": p, "max_tokens": 400, "temperature": 0,
                                         "logprobs": 1, "stream": True, "stream_options": {"include_usage": True}}):
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            if c.get("text"):
                k = len((c.get("logprobs") or {}).get("tokens") or [None])
                times += [time.time()] * k
                text += c["text"]
    pt = usage["prompt_tokens"] if usage else n
    dec = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
    found = "TANGERINE-4471-OBSIDIAN" in text
    c1 = spec_counters(port)
    acc = (c1[1] - c0[1]) / (c1[0] - c0[0]) if c0 and c1 and c1[0] > c0[0] else None
    print(f"ctx {pt} tokens: prefill {pt / (times[0] - t0):.0f} tok/s (TTFT {times[0] - t0:.1f} s), decode {dec:.2f} "
          f"tok/s over {len(times)} tokens, draft acceptance {acc if acc is None else round(acc, 3)}, "
          f"needle {'FOUND' if found else 'MISSING'}; {margin()}", flush=True)
    print("  answer start:", repr(text[:160]), flush=True)
    return dict(prompt_tokens=pt, ttft=times[0] - t0, decode=dec, draft_acceptance=acc, needle=found, text=text)


def plp_test(n):
    p = f"[{os.urandom(4).hex()}]\n" + haystack(n)
    t0 = time.time()
    r = json.load(post("/v1/completions", {"model": model, "prompt": p, "max_tokens": 4, "temperature": 0,
                                         "prompt_logprobs": 5}))
    plp = r["choices"][0].get("prompt_logprobs") or []
    ok = sum(1 for e in plp if e)
    print(f"prompt_logprobs over {r['usage']['prompt_tokens']} tokens: {time.time() - t0:.1f} s, {ok} entries; "
          f"{margin()}", flush=True)
    return dict(prompt_tokens=r["usage"]["prompt_tokens"], entries=ok, seconds=time.time() - t0)


def image_test(k=1):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (3840, 2160), (235, 240, 250))
    d = ImageDraw.Draw(img)
    for i in range(0, 3840, 160):
        d.line([(i, 0), (i, 2160)], fill=(200, 210, 230), width=3)
    d.rectangle([400, 500, 1400, 1500], fill=(220, 60, 60))
    d.ellipse([2200, 600, 3200, 1600], fill=(60, 120, 220))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 180)
    except OSError:
        font = ImageFont.load_default()
    d.text((900, 1750), "ORIN 4K TEST 7731", fill=(20, 20, 20), font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    t0 = time.time()
    r = json.load(post("/v1/chat/completions", {
        "model": model, "max_tokens": 120, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            *[{"type": "image_url", "image_url": {"url": url}}] * k,
            {"type": "text", "text": "Describe the shapes and their colours, and read the text in the image exactly."}]}]}))
    ans = r["choices"][0]["message"]["content"]
    print(f"{k} x image 3840x2160: {time.time() - t0:.1f} s, prompt tokens {r['usage']['prompt_tokens']}, "
          f"text read {'OK' if '7731' in ans else 'NO'}; {margin()}\n  answer: {ans[:300]!r}", flush=True)
    return dict(seconds=time.time() - t0, prompt_tokens=r["usage"]["prompt_tokens"], answer=ans)


res = {}
if what == "image":
    res["image"] = image_test()
elif what == "images":
    res["images"] = image_test(int(sys.argv[3]))
elif what == "plp":
    for n in sys.argv[3:]:
        res[n] = plp_test(int(n))
else:
    for n in sys.argv[3:]:
        res[n] = ctx_test(int(n))
tag = os.environ.get("TAG", "m5")
json.dump(res, open(root / f"out/{tag}-{what}.json", "w"), indent=1)
