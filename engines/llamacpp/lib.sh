# Shared by build.sh, export-patches.sh and check-patches.sh (sourced). Sets here, root, the pins and the series.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
# shellcheck source=../../versions.env
source "$root/versions.env"
shopt -s nullglob
series=("$here"/patches/*.patch)
shopt -u nullglob
[ ${#series[@]} -gt 0 ] || { echo "no patches in $here/patches" >&2; exit 1; }
# git without the user's config (diff algorithm, prefixes, abbrev, signing): the series must be byte-reproducible
g() { GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git -c user.name=bf16n -c user.email=bf16n@localhost "$@"; }
# fetch the pinned commit into repo $1 (from LLAMACPP_MIRROR, a local llama.cpp clone, if set: no network)
fetch_pin() { g -C "$1" fetch -q --depth 1 "${LLAMACPP_MIRROR:-$LLAMACPP_REPO}" "$LLAMACPP_COMMIT"; }
# tree id of the pin with the series applied, computed in a scratch index of repo $1 (the work tree is not touched)
series_tree() {
    local idx; idx="$(mktemp)"
    GIT_INDEX_FILE="$idx" g -C "$1" read-tree "$LLAMACPP_COMMIT"
    for p in "${series[@]}"; do GIT_INDEX_FILE="$idx" g -C "$1" apply --cached "$p"; done
    GIT_INDEX_FILE="$idx" g -C "$1" write-tree
    rm -f "$idx"
}
