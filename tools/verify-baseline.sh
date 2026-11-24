#!/usr/bin/env bash
# Verify the baseline is exactly reproducible: every patch in zig/patches applies cleanly,
# in order, on a clean upstream pin.
#
# This is the geodetic check behind docs/CURRENT-BASELINE.md §1. It does NOT build anything.
#
#   tools/verify-baseline.sh              # verify and clean up
#   KEEP=1 tools/verify-baseline.sh       # keep the worktree for inspection (path printed)
#   TF_UPSTREAM=/path/to/tf tools/verify-baseline.sh
#
# Discipline (docs/DEVELOPMENT-HANDBOOK.md §5): apply with --3way, and `git add -A` after EACH
# patch before applying the next. Applying the whole series first and resolving later fails
# with "does not exist in index".
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PIN=${TF_PIN:-fbbc97132cd004237d9887b8d3185896efaedc5c}
UPSTREAM=${TF_UPSTREAM:-$REPO/../_upstream/tf}

[[ -d "$UPSTREAM/.git" || -f "$UPSTREAM/.git" ]] || {
  echo "FAIL: upstream clone not found at $UPSTREAM" >&2
  echo "      clone https://github.com/bertholomus/TensorFold.git there, or set TF_UPSTREAM" >&2
  exit 1
}
UPSTREAM=$(cd "$UPSTREAM" && pwd)

git -C "$UPSTREAM" cat-file -e "$PIN^{commit}" 2>/dev/null || {
  echo "FAIL: pin $PIN not present in $UPSTREAM (fetch it first)" >&2
  exit 1
}

WT=$(mktemp -d "${TMPDIR:-/tmp}/tf-verify-XXXXXX")
cleanup() {
  if [[ "${KEEP:-0}" == 1 ]]; then
    echo "KEEP=1: worktree left at $WT"
  else
    git -C "$UPSTREAM" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
  fi
}
trap cleanup EXIT

git -C "$UPSTREAM" worktree add --detach "$WT" "$PIN" >/dev/null 2>&1 || {
  echo "FAIL: could not create worktree at pin" >&2
  exit 1
}

total=0; ok=0
for p in "$REPO"/zig/patches/*.patch; do
  total=$((total + 1))
  name=$(basename "$p")
  if ! git -C "$WT" apply --3way "$p" >/dev/null 2>&1; then
    echo "FAIL at $name" >&2
    echo "  conflicts:" >&2
    git -C "$WT" diff --name-only --diff-filter=U | sed 's/^/    /' >&2
    echo "  the baseline is NOT reproducible; see handbook §5" >&2
    KEEP=1 exit 1
  fi
  if [[ -n "$(git -C "$WT" diff --name-only --diff-filter=U)" ]]; then
    echo "FAIL: $name left conflicts" >&2
    KEEP=1 exit 1
  fi
  git -C "$WT" add -A
  ok=$((ok + 1))
done

echo "RESULT: $ok/$total patches applied cleanly on pin ${PIN:0:10}"
if [[ "$ok" != "$total" ]]; then exit 1; fi
echo "Baseline is reproducible. See docs/CURRENT-BASELINE.md."