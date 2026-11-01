#!/usr/bin/env bash
# Start an isolated llama-server, collect five frozen S5 prompts (up to 96 tokens, full top-5), stop only this child.
# MTP-on launches belong to the operator; use collect_outputs.py against that isolated server.
#   MODEL=... bench/xengine/run_prompts.sh <tag> [VAR=value ...]   -> out/prompts-<tag>.json
# Compare two runs with: python3 bench/xengine/compare_outputs.py out/prompts-a.json out/prompts-b.json
#
# Used in docs/llamacpp/RESULTS.md:
#   run_prompts.sh fast LLAMA_BF16N=hybrid
#   run_prompts.sh ref  LLAMA_BF16N=hybrid GGML_BF16N_REFERENCE=1   # exact dequant + stock BF16 cuBLAS path
set -euo pipefail
: "${MODEL:?set MODEL to the BF16 GGUF}"
LLAMA="${LLAMA:-./engines/llamacpp/src/build/bin}"
PORT="${PORT:-8099}"
tag=$1
shift
mkdir -p out
# Refuse an occupied port instead of accidentally collecting from an unrelated/production server.
python3 - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.bind(("127.0.0.1", int(sys.argv[1])))
PY

env "$@" "$LLAMA/llama-server" -m "$MODEL" -ngl 99 -fa on -c 32768 -ub 2048 -np 1 -t 8 \
    --host 127.0.0.1 --port "$PORT" --no-webui > "out/server-$tag.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null || true' EXIT
ready=0
for _ in $(seq 1 120); do
    sleep 5
    kill -0 $pid 2>/dev/null || { echo "server exited, see out/server-$tag.log"; exit 1; }
    if curl -fsS "127.0.0.1:$PORT/health" | grep -q ok; then
        kill -0 $pid 2>/dev/null || { echo "server exited, see out/server-$tag.log"; exit 1; }
        ready=1
        break
    fi
done
[ "$ready" = 1 ] || { echo "server readiness timed out, see out/server-$tag.log"; exit 1; }

python3 "$(dirname "$0")/collect_outputs.py" --engine llama --url "http://127.0.0.1:$PORT" \
    --mtp off --out "out/prompts-$tag.json"
