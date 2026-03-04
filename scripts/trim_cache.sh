#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
GLOBAL_CARGO=0
FEROS_STATE=0
NO_LOCAL=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
FEROS_STATE_ROOT="${DATA_HOME}/fer-os"

print_help() {
  cat <<'USAGE'
Trim rebuildable caches for feR-os (feRcuda).

Usage:
  trim_cache.sh [options]

Default behavior:
  - removes local repo build caches only:
    - <repo>/build
    - <repo>/rust/fercuda-ffi/target
    - <repo>/rust/deps/*/target

Options:
  --dry-run          Show what would be removed and reclaim estimate.
  --global-cargo     Also remove global Cargo cache dirs under ~/.cargo:
                     - registry/cache
                     - registry/src
                     - git/db
                     - git/checkouts
  --feros-state      Also remove installer-managed build state:
                     - ~/.local/share/fer-os/build
                     - ~/.local/share/fer-os/build-local
  --no-local         Skip local repo cache cleanup.
  -h, --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --global-cargo)
      GLOBAL_CARGO=1
      shift
      ;;
    --feros-state)
      FEROS_STATE=1
      shift
      ;;
    --no-local)
      NO_LOCAL=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

add_if_exists() {
  local p="$1"
  if [[ -e "$p" ]]; then
    TARGETS+=("$p")
  fi
}

human_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec --suffix=B "$bytes"
  else
    echo "${bytes}B"
  fi
}

size_bytes() {
  local p="$1"
  du -sb "$p" 2>/dev/null | awk '{print $1}'
}

TARGETS=()

if [[ "$NO_LOCAL" -eq 0 ]]; then
  add_if_exists "$REPO_ROOT/build"
  add_if_exists "$REPO_ROOT/rust/fercuda-ffi/target"
  if [[ -d "$REPO_ROOT/rust/deps" ]]; then
    while IFS= read -r dep_target; do
      add_if_exists "$dep_target"
    done < <(find "$REPO_ROOT/rust/deps" -mindepth 2 -maxdepth 2 -type d -name target 2>/dev/null)
  fi
fi

if [[ "$FEROS_STATE" -eq 1 ]]; then
  add_if_exists "$FEROS_STATE_ROOT/build"
  add_if_exists "$FEROS_STATE_ROOT/build-local"
fi

if [[ "$GLOBAL_CARGO" -eq 1 ]]; then
  add_if_exists "$HOME/.cargo/registry/cache"
  add_if_exists "$HOME/.cargo/registry/src"
  add_if_exists "$HOME/.cargo/git/db"
  add_if_exists "$HOME/.cargo/git/checkouts"
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "[trim-cache] nothing to remove."
  exit 0
fi

echo "[trim-cache] targets:"
TOTAL=0
for p in "${TARGETS[@]}"; do
  b="$(size_bytes "$p" || echo 0)"
  TOTAL=$((TOTAL + b))
  echo "  - $p ($(human_size "$b"))"
done

echo "[trim-cache] estimated reclaimable: $(human_size "$TOTAL")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[trim-cache] dry-run enabled; no files removed."
  exit 0
fi

for p in "${TARGETS[@]}"; do
  rm -rf "$p"
done

echo "[trim-cache] done."
