#!/usr/bin/env bash
# Start llama-server with the given environment, run fixed greedy prompts (three short, one of ~1500 tokens;
# 96 generated tokens, top-5 logprobs), stop it.
#   MODEL=... scripts/run_prompts.sh <tag> [VAR=value ...]   -> out/prompts-<tag>.json
# Compare two runs with: python3 scripts/compare_outputs.py out/prompts-a.json out/prompts-b.json
#
# Used in docs/RESULTS.md:
#   run_prompts.sh fast LLAMA_BF16N=hybrid
#   run_prompts.sh ref  LLAMA_BF16N=hybrid GGML_BF16N_REFERENCE=1   # exact dequant + stock BF16 cuBLAS path
set -euo pipefail
: "${MODEL:?set MODEL to the BF16 GGUF}"
LLAMA="${LLAMA:-./llama.cpp/build/bin}"
PORT="${PORT:-8099}"
tag=$1
shift
mkdir -p out

env "$@" "$LLAMA/llama-server" -m "$MODEL" -ngl 99 -fa on -c 8192 -np 1 -t 8 \
    --host 127.0.0.1 --port "$PORT" --no-webui > "out/server-$tag.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
    sleep 5
    curl -s "127.0.0.1:$PORT/health" | grep -q ok && break
    kill -0 $pid 2>/dev/null || { echo "server exited, see out/server-$tag.log"; exit 1; }
done

python3 - "$tag" "$PORT" "$(dirname "$0")/../docs/FORMAT.md" <<'PY'
import json, sys, urllib.request
tag, port, doc = sys.argv[1], sys.argv[2], open(sys.argv[3]).read()
prompts = [
    "<|im_start|>user\nWrite a Python function that checks whether a number is prime, then explain its complexity in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nПоясни коротко, чому небо блакитне.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nA train leaves at 14:35 and arrives at 18:10. How long is the trip in minutes? Answer with reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    # ~1500 tokens: exercises the batched (prefill) kernels with full micro-batches
    "<|im_start|>user\nSummarize this document in three sentences.\n\n" + doc + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
out = []
for p in prompts:
    req = json.dumps({"prompt": p, "n_predict": 96, "temperature": 0, "n_probs": 5, "cache_prompt": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", req, {"Content-Type": "application/json"})))
    out.append(dict(content=r["content"], timings=r["timings"],
                    probs=[[(t["token"], t["logprob"]) for t in c["top_logprobs"]] for c in r["completion_probabilities"]]))
json.dump(out, open(f"out/prompts-{tag}.json", "w"))
print(tag, "tok/s:", [round(o["timings"]["predicted_per_second"], 2) for o in out])
PY
