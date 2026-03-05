#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.local"
BUILD_TYPE="Release"
ARCH_LIST=""
WITH_TESTS=0
JOBS="${JOBS:-$(nproc)}"
REPO_URL="${FERCUDA_REPO_URL:-https://github.com/DaronPopov/feRcuda.git}"
BRANCH="${FERCUDA_BRANCH:-main}"
FERRITE_MCP_REPO_URL="${FERRITE_MCP_REPO_URL:-https://github.com/DaronPopov/ferrite-mcp.git}"
FERRITE_MCP_BRANCH="${FERRITE_MCP_BRANCH:-main}"
UPDATE=1
USE_LOCAL_SOURCE=0
RUST_BOOTSTRAP=1
WITH_FERRITE_MCP=1

DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
STATE_ROOT="${DATA_HOME}/fer-os"
SOURCE_DIR="${STATE_ROOT}/src/feRcuda"
FERRITE_MCP_SOURCE_DIR="${STATE_ROOT}/src/ferrite-mcp"
BUILD_DIR="${STATE_ROOT}/build"

print_help() {
  cat <<'USAGE'
feR-os installer (feRcuda)

Usage:
  install.sh [options]

Default behavior:
  - Uses managed source at ~/.local/share/fer-os/src/feRcuda
  - Uses managed source at ~/.local/share/fer-os/src/ferrite-mcp
  - On rerun, updates source to latest origin/main
  - Rebuilds and reinstalls feRcuda + ferrite-mcp into --prefix

Options:
  --prefix <path>         Install prefix (default: ~/.local)
  --build-type <type>     CMake build type (default: Release)
  --arch-list <list>      CUDA arch list (e.g. "75;86;89")
  --with-tests            Build and run tests after compile
  --jobs <n>              Parallel build jobs (default: nproc)
  --repo-url <url>        Git URL (default: https://github.com/DaronPopov/feRcuda.git)
  --branch <name>         Git branch/tag to track (default: main)
  --ferrite-mcp-repo-url <url>  Git URL (default: https://github.com/DaronPopov/ferrite-mcp.git)
  --ferrite-mcp-branch <name>   Git branch/tag to track (default: main)
  --source-dir <path>     Managed source path (default: ~/.local/share/fer-os/src/feRcuda)
  --ferrite-mcp-source-dir <path> Managed source path (default: ~/.local/share/fer-os/src/ferrite-mcp)
  --no-ferrite-mcp        Skip ferrite-mcp clone/build/install
  --no-update             Reuse existing checkout without fetching latest
  --use-local-source      Build from current directory instead of managed checkout
  --no-rust-bootstrap     Skip Rust toolchain/dependency bootstrap
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --arch-list)
      ARCH_LIST="$2"
      shift 2
      ;;
    --with-tests)
      WITH_TESTS=1
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --ferrite-mcp-repo-url)
      FERRITE_MCP_REPO_URL="$2"
      shift 2
      ;;
    --ferrite-mcp-branch)
      FERRITE_MCP_BRANCH="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      BUILD_DIR="$(dirname "$SOURCE_DIR")/build"
      shift 2
      ;;
    --ferrite-mcp-source-dir)
      FERRITE_MCP_SOURCE_DIR="$2"
      shift 2
      ;;
    --no-ferrite-mcp)
      WITH_FERRITE_MCP=0
      shift
      ;;
    --no-update)
      UPDATE=0
      shift
      ;;
    --use-local-source)
      USE_LOCAL_SOURCE=1
      shift
      ;;
    --no-rust-bootstrap)
      RUST_BOOTSTRAP=0
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

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd cmake
need_cmd git

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH. Install CUDA toolkit before running installer." >&2
  exit 1
fi

ensure_rust_toolchain() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  echo "[feR-os] cargo not found, installing rustup toolchain (minimal profile)"
  need_cmd curl
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo still not found after rustup install." >&2
    exit 1
  fi
}

bootstrap_rust_crates() {
  ensure_rust_toolchain
  local manifests=(
    "$ROOT_DIR/rust/fercuda-ffi/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-ml-lite/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-math/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-vision-lite/Cargo.toml"
  )
  for manifest in "${manifests[@]}"; do
    if [[ -f "$manifest" ]]; then
      echo "[feR-os] cargo fetch -> $manifest"
      cargo fetch --manifest-path "$manifest"
    fi
  done
}

ROOT_DIR=""

validate_repo_layout() {
  local p="$1"
  [[ -f "${p}/CMakeLists.txt" && -d "${p}/src" && -d "${p}/include" ]]
}

prepare_managed_source() {
  mkdir -p "$(dirname "$SOURCE_DIR")"

  if [[ -d "$SOURCE_DIR/.git" ]]; then
    echo "[feR-os] using managed checkout: $SOURCE_DIR"
    git -C "$SOURCE_DIR" remote set-url origin "$REPO_URL" || true
    if [[ "$UPDATE" -eq 1 ]]; then
      echo "[feR-os] updating source: origin/${BRANCH}"
      git -C "$SOURCE_DIR" fetch --depth 1 origin "$BRANCH"
      git -C "$SOURCE_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
      git -C "$SOURCE_DIR" reset --hard "origin/$BRANCH"
    fi
  else
    if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
      local backup="${SOURCE_DIR}.backup.$(date +%s)"
      echo "[feR-os] non-git directory at source path, moving to: $backup"
      mv "$SOURCE_DIR" "$backup"
    fi
    echo "[feR-os] cloning ${REPO_URL} (${BRANCH}) -> $SOURCE_DIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SOURCE_DIR"
  fi

  if ! validate_repo_layout "$SOURCE_DIR"; then
    echo "Managed source is missing required project files: $SOURCE_DIR" >&2
    exit 1
  fi

  ROOT_DIR="$SOURCE_DIR"
}

validate_ferrite_mcp_layout() {
  local p="$1"
  [[ -f "${p}/Cargo.toml" ]]
}

prepare_managed_ferrite_mcp_source() {
  if [[ "$WITH_FERRITE_MCP" -eq 0 ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$FERRITE_MCP_SOURCE_DIR")"

  if [[ -d "$FERRITE_MCP_SOURCE_DIR/.git" ]]; then
    echo "[feR-os] using ferrite-mcp checkout: $FERRITE_MCP_SOURCE_DIR"
    git -C "$FERRITE_MCP_SOURCE_DIR" remote set-url origin "$FERRITE_MCP_REPO_URL" || true
    if [[ "$UPDATE" -eq 1 ]]; then
      if [[ -n "$(git -C "$FERRITE_MCP_SOURCE_DIR" status --porcelain)" ]]; then
        echo "[feR-os] ferrite-mcp has local changes; skipping update/reset to preserve custom checkout"
      else
        echo "[feR-os] updating ferrite-mcp: origin/${FERRITE_MCP_BRANCH}"
        git -C "$FERRITE_MCP_SOURCE_DIR" fetch --depth 1 origin "$FERRITE_MCP_BRANCH"
        git -C "$FERRITE_MCP_SOURCE_DIR" checkout -B "$FERRITE_MCP_BRANCH" "origin/$FERRITE_MCP_BRANCH"
        git -C "$FERRITE_MCP_SOURCE_DIR" reset --hard "origin/$FERRITE_MCP_BRANCH"
      fi
    fi
  else
    if [[ -e "$FERRITE_MCP_SOURCE_DIR" && ! -d "$FERRITE_MCP_SOURCE_DIR/.git" ]]; then
      local backup="${FERRITE_MCP_SOURCE_DIR}.backup.$(date +%s)"
      echo "[feR-os] non-git directory at ferrite-mcp path, moving to: $backup"
      mv "$FERRITE_MCP_SOURCE_DIR" "$backup"
    fi
    echo "[feR-os] cloning ${FERRITE_MCP_REPO_URL} (${FERRITE_MCP_BRANCH}) -> $FERRITE_MCP_SOURCE_DIR"
    git clone --depth 1 --branch "$FERRITE_MCP_BRANCH" "$FERRITE_MCP_REPO_URL" "$FERRITE_MCP_SOURCE_DIR"
  fi

  if ! validate_ferrite_mcp_layout "$FERRITE_MCP_SOURCE_DIR"; then
    echo "ferrite-mcp source is missing required files: $FERRITE_MCP_SOURCE_DIR" >&2
    exit 1
  fi
}

prepare_local_source() {
  local cwd
  cwd="$(pwd)"
  if ! validate_repo_layout "$cwd"; then
    echo "--use-local-source requires running inside the feRcuda repo root." >&2
    exit 1
  fi
  ROOT_DIR="$cwd"
  mkdir -p "$STATE_ROOT"
  BUILD_DIR="${STATE_ROOT}/build-local"
  echo "[feR-os] using local source: $ROOT_DIR"
}

if [[ "$USE_LOCAL_SOURCE" -eq 1 ]]; then
  prepare_local_source
else
  prepare_managed_source
fi
prepare_managed_ferrite_mcp_source

mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ -n "$ARCH_LIST" ]]; then
  CMAKE_ARGS+=( -DCMAKE_CUDA_ARCHITECTURES="$ARCH_LIST" )
fi

echo "[feR-os] configure"
cmake "${CMAKE_ARGS[@]}"

echo "[feR-os] build"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[feR-os] install -> $PREFIX"
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

if [[ "$WITH_TESTS" -eq 1 ]]; then
  echo "[feR-os] test"
  (cd "$BUILD_DIR" && ctest --output-on-failure)
fi

if [[ "$RUST_BOOTSTRAP" -eq 1 ]]; then
  echo "[feR-os] rust bootstrap"
  bootstrap_rust_crates
fi

if [[ "$WITH_FERRITE_MCP" -eq 1 ]]; then
  ensure_rust_toolchain
  echo "[feR-os] install ferrite-mcp -> $PREFIX"
  cargo install \
    --path "$FERRITE_MCP_SOURCE_DIR/crates/shell-bin" \
    --bin ferrite \
    --locked \
    --force \
    --root "$PREFIX"
fi

mkdir -p "$STATE_ROOT"
if [[ -d "$ROOT_DIR/.git" ]]; then
  COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD || true)"
  BRANCH_NOW="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD || true)"
  FERRITE_MCP_COMMIT=""
  FERRITE_MCP_BRANCH_NOW=""
  if [[ "$WITH_FERRITE_MCP" -eq 1 && -d "$FERRITE_MCP_SOURCE_DIR/.git" ]]; then
    FERRITE_MCP_COMMIT="$(git -C "$FERRITE_MCP_SOURCE_DIR" rev-parse --short HEAD || true)"
    FERRITE_MCP_BRANCH_NOW="$(git -C "$FERRITE_MCP_SOURCE_DIR" rev-parse --abbrev-ref HEAD || true)"
  fi
  cat > "$STATE_ROOT/install-meta.txt" <<META
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
repo_url=$REPO_URL
branch=$BRANCH_NOW
commit=$COMMIT
source_dir=$ROOT_DIR
build_dir=$BUILD_DIR
prefix=$PREFIX
rust_bootstrap=$RUST_BOOTSTRAP
ferrite_mcp_enabled=$WITH_FERRITE_MCP
ferrite_mcp_repo_url=$FERRITE_MCP_REPO_URL
ferrite_mcp_branch=$FERRITE_MCP_BRANCH_NOW
ferrite_mcp_commit=$FERRITE_MCP_COMMIT
ferrite_mcp_source_dir=$FERRITE_MCP_SOURCE_DIR
META
fi

echo "[feR-os] done"
if [[ -f "$STATE_ROOT/install-meta.txt" ]]; then
  echo "[feR-os] install metadata: $STATE_ROOT/install-meta.txt"
fi
echo "Export runtime path if needed:"
echo "  export LD_LIBRARY_PATH=\"$PREFIX/lib:${LD_LIBRARY_PATH:-}\""
