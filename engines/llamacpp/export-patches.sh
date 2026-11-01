#!/usr/bin/env bash
# Regenerate patches/ from branch bf16n of the work tree (one commit per topic on top of LLAMACPP_COMMIT).
#   engines/llamacpp/export-patches.sh [workdir=engines/llamacpp/src]
# Deterministic: unnumbered, zero commit ids, no signature, full blob ids, Myers diff, no user git config.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
dir="${1:-$here/src}"
tmp="$(mktemp -d)"
g -C "$dir" format-patch -q -N --zero-commit --no-signature --full-index --diff-algorithm=myers --no-renames \
    -o "$tmp" "$LLAMACPP_COMMIT..bf16n"
rm -f "$here"/patches/*.patch
mv "$tmp"/*.patch "$here/patches/"
rmdir "$tmp"
ls "$here/patches"
