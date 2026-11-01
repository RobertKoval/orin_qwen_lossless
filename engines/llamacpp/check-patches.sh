#!/usr/bin/env bash
# The series applies with `git am` to a fresh checkout of LLAMACPP_COMMIT; prints the resulting tree id.
#   engines/llamacpp/check-patches.sh        (LLAMACPP_MIRROR=<local clone>: fetch without network)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
g init -q "$tmp"
fetch_pin "$tmp"
g -C "$tmp" checkout -q -B bf16n "$LLAMACPP_COMMIT"
g -C "$tmp" am -q "${series[@]}"
echo "check-patches: ok, ${#series[@]} patches, tree $(g -C "$tmp" rev-parse 'HEAD^{tree}')"
