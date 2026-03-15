#!/usr/bin/env bash
# ============================================================================
# feRcuda one-line bootstrap installer
#
#   curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install.sh | bash
#
# Supports: x86_64, aarch64 (ARM64/DGX Spark/Jetson)
# Requires: CUDA toolkit (nvcc), CMake 3.20+, C/C++ compiler
# Auto-installs: build deps, Rust toolchain, Cargo crates
# ============================================================================
set -euo pipefail

# -- Configuration (override via env) ----------------------------------------
PREFIX="${FERCUDA_PREFIX:-${HOME}/.local}"
BUILD_TYPE="${FERCUDA_BUILD_TYPE:-Release}"
ARCH_LIST="${FERCUDA_ARCH_LIST:-}"
WITH_TESTS="${FERCUDA_WITH_TESTS:-0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
REPO_URL="${FERCUDA_REPO_URL:-https://github.com/DaronPopov/feRcuda.git}"
BRANCH="${FERCUDA_BRANCH:-main}"
UPDATE=1
USE_LOCAL_SOURCE=0
RUST_BOOTSTRAP=1
FERRITE_MCP_REPO_URL="${FERRITE_MCP_REPO_URL:-https://github.com/DaronPopov/ferrite-mcp.git}"
FERRITE_MCP_BRANCH="${FERRITE_MCP_BRANCH:-main}"
WITH_FERRITE_MCP=1

DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
STATE_ROOT="${DATA_HOME}/fer-os"
SOURCE_DIR="${STATE_ROOT}/src/feRcuda"
FERRITE_MCP_SOURCE_DIR="${STATE_ROOT}/src/ferrite-mcp"
BUILD_DIR="${STATE_ROOT}/build"

# -- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[feR-os]${NC} $*"; }
ok()   { echo -e "${GREEN}[feR-os]${NC} $*"; }
warn() { echo -e "${YELLOW}[feR-os]${NC} $*"; }
err()  { echo -e "${RED}[feR-os]${NC} $*" >&2; }

# -- Help --------------------------------------------------------------------
print_help() {
  cat <<'USAGE'
feRcuda portable bootstrap installer

One-line install:
  curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install.sh | bash

Usage:
  install.sh [options]

Options:
  --prefix <path>             Install prefix (default: ~/.local)
  --build-type <type>         CMake build type (default: Release)
  --arch-list <list>          CUDA arch list (e.g. "75;86;89;90")
  --with-tests                Build and run ctest after compile
  --jobs <n>                  Parallel build jobs (default: nproc)
  --repo-url <url>            feRcuda git URL
  --branch <name>             Git branch/tag (default: main)
  --no-ferrite-mcp            Skip ferrite-mcp
  --no-update                 Reuse existing checkout
  --use-local-source          Build from current directory
  --no-rust-bootstrap         Skip Rust toolchain install
  -h, --help                  Show this help

Environment overrides:
  FERCUDA_PREFIX, FERCUDA_BUILD_TYPE, FERCUDA_ARCH_LIST, FERCUDA_WITH_TESTS,
  FERCUDA_REPO_URL, FERCUDA_BRANCH, JOBS
USAGE
}

# -- Arg parse ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)           PREFIX="$2"; shift 2;;
    --build-type)       BUILD_TYPE="$2"; shift 2;;
    --arch-list)        ARCH_LIST="$2"; shift 2;;
    --with-tests)       WITH_TESTS=1; shift;;
    --jobs)             JOBS="$2"; shift 2;;
    --repo-url)         REPO_URL="$2"; shift 2;;
    --branch)           BRANCH="$2"; shift 2;;
    --no-ferrite-mcp)   WITH_FERRITE_MCP=0; shift;;
    --no-update)        UPDATE=0; shift;;
    --use-local-source) USE_LOCAL_SOURCE=1; shift;;
    --no-rust-bootstrap) RUST_BOOTSTRAP=0; shift;;
    --source-dir)       SOURCE_DIR="$2"; BUILD_DIR="$(dirname "$SOURCE_DIR")/build"; shift 2;;
    -h|--help)          print_help; exit 0;;
    *)                  err "Unknown arg: $1"; print_help; exit 1;;
  esac
done

# -- Platform detection ------------------------------------------------------
detect_platform() {
  ARCH="$(uname -m)"
  OS="$(uname -s)"
  DISTRO="unknown"

  case "$OS" in
    Linux)
      if   [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        DISTRO="${ID:-unknown}"
      elif [[ -f /etc/redhat-release ]]; then
        DISTRO="rhel"
      fi
      ;;
    Darwin) DISTRO="macos";;
    *)      DISTRO="unknown";;
  esac

  log "Platform: ${OS} ${ARCH} (${DISTRO})"

  case "$ARCH" in
    x86_64|amd64) ARCH_FAMILY="x86_64";;
    aarch64|arm64) ARCH_FAMILY="aarch64";;
    *) err "Unsupported architecture: $ARCH"; exit 1;;
  esac
}

# -- Dependency installer (cross-distro) -------------------------------------
install_build_deps() {
  local missing=()

  command -v cmake   >/dev/null 2>&1 || missing+=(cmake)
  command -v git     >/dev/null 2>&1 || missing+=(git)
  command -v make    >/dev/null 2>&1 || missing+=(make)
  command -v cc      >/dev/null 2>&1 || missing+=(gcc)
  command -v g++     >/dev/null 2>&1 || missing+=(g++)

  [[ ${#missing[@]} -eq 0 ]] && return 0

  log "Installing missing build deps: ${missing[*]}"

  case "$DISTRO" in
    ubuntu|debian|pop|linuxmint|raspbian)
      sudo apt-get update -qq
      sudo apt-get install -y -qq cmake git build-essential
      ;;
    fedora|rhel|centos|rocky|almalinux|amzn)
      if command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y cmake git gcc gcc-c++ make
      else
        sudo yum install -y cmake3 git gcc gcc-c++ make
        if command -v cmake3 >/dev/null 2>&1 && ! command -v cmake >/dev/null 2>&1; then
          sudo ln -sf "$(command -v cmake3)" /usr/local/bin/cmake
        fi
      fi
      ;;
    arch|manjaro|endeavouros)
      sudo pacman -Sy --noconfirm cmake git base-devel
      ;;
    opensuse*|sles)
      sudo zypper install -y cmake git gcc gcc-c++ make
      ;;
    alpine)
      sudo apk add cmake git build-base linux-headers
      ;;
    *)
      warn "Unknown distro '${DISTRO}' — please install cmake, git, gcc, g++, make manually"
      for cmd in "${missing[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
          err "Required command not found: $cmd"
          exit 1
        fi
      done
      ;;
  esac
}

# -- CUDA detection ----------------------------------------------------------
detect_cuda() {
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
    log "CUDA ${CUDA_VERSION} at ${CUDA_PATH}"
    return 0
  fi

  # Try standard paths
  for try_path in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
    if [[ -x "${try_path}/bin/nvcc" ]]; then
      export PATH="${try_path}/bin:${PATH}"
      export LD_LIBRARY_PATH="${try_path}/lib64:${LD_LIBRARY_PATH:-}"
      CUDA_VERSION="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
      CUDA_PATH="${try_path}"
      log "CUDA ${CUDA_VERSION} at ${CUDA_PATH}"
      return 0
    fi
  done

  # Jetson/ARM: CUDA is often in /usr/local/cuda but PATH not set
  if [[ "$ARCH_FAMILY" == "aarch64" && -d /usr/local/cuda ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_VERSION="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
      CUDA_PATH="/usr/local/cuda"
      log "CUDA ${CUDA_VERSION} at ${CUDA_PATH} (Jetson/ARM)"
      return 0
    fi
  fi

  err "CUDA toolkit not found. Install CUDA before running installer."
  err "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
  err "  Fedora/RHEL:   sudo dnf install cuda"
  err "  Arch:          sudo pacman -S cuda"
  err "  Jetson:        JetPack includes CUDA"
  exit 1
}

# -- GPU architecture auto-detect --------------------------------------------
detect_gpu_arch() {
  if [[ -n "$ARCH_LIST" ]]; then
    log "Using specified CUDA architectures: ${ARCH_LIST}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local raw
    raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$raw" ]]; then
      ARCH_LIST="$(echo "$raw" | tr -d '.' | tr '\n' ';' | sed 's/;$//')"
      log "Auto-detected GPU architectures: ${ARCH_LIST}"
      return 0
    fi
  fi

  # Fallback: build for common architectures
  case "$ARCH_FAMILY" in
    x86_64)  ARCH_LIST="52;60;70;75;80;86;89;90";;
    aarch64) ARCH_LIST="53;62;72;75;80;86;87;90";;
  esac
  log "Using portable GPU architectures: ${ARCH_LIST}"
}

# -- Rust toolchain ----------------------------------------------------------
ensure_rust_toolchain() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  log "Installing Rust toolchain (minimal profile)"
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
  if ! command -v cargo >/dev/null 2>&1; then
    err "cargo not found after rustup install"
    exit 1
  fi
}

# -- Source management -------------------------------------------------------
validate_repo_layout() {
  [[ -f "${1}/CMakeLists.txt" && -d "${1}/native" && -d "${1}/include" ]]
}

prepare_source() {
  if [[ "$USE_LOCAL_SOURCE" -eq 1 ]]; then
    ROOT_DIR="$(pwd)"
    if ! validate_repo_layout "$ROOT_DIR"; then
      err "--use-local-source requires running inside the feRcuda repo root"
      exit 1
    fi
    BUILD_DIR="${STATE_ROOT}/build-local"
    log "Using local source: ${ROOT_DIR}"
    return
  fi

  mkdir -p "$(dirname "$SOURCE_DIR")"

  if [[ -d "$SOURCE_DIR/.git" ]]; then
    log "Using managed checkout: ${SOURCE_DIR}"
    git -C "$SOURCE_DIR" remote set-url origin "$REPO_URL" 2>/dev/null || true
    if [[ "$UPDATE" -eq 1 ]]; then
      log "Updating to origin/${BRANCH}"
      git -C "$SOURCE_DIR" fetch --depth 1 origin "$BRANCH"
      git -C "$SOURCE_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
      git -C "$SOURCE_DIR" reset --hard "origin/$BRANCH"
    fi
  else
    [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]] && \
      mv "$SOURCE_DIR" "${SOURCE_DIR}.backup.$(date +%s)"
    log "Cloning ${REPO_URL} (${BRANCH})"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SOURCE_DIR"
  fi

  if ! validate_repo_layout "$SOURCE_DIR"; then
    err "Source is missing required project files"
    exit 1
  fi
  ROOT_DIR="$SOURCE_DIR"
}

# -- Build -------------------------------------------------------------------
build_fercuda() {
  mkdir -p "$BUILD_DIR"

  local cmake_args=(
    -S "$ROOT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
  )

  if [[ -n "$ARCH_LIST" ]]; then
    cmake_args+=( -DCMAKE_CUDA_ARCHITECTURES="$ARCH_LIST" )
  fi

  log "CMake configure"
  cmake "${cmake_args[@]}"

  log "Building with ${JOBS} jobs"
  cmake --build "$BUILD_DIR" -j"$JOBS"

  log "Installing to ${PREFIX}"
  cmake --install "$BUILD_DIR" --prefix "$PREFIX"
}

run_tests() {
  if [[ "$WITH_TESTS" -eq 1 ]]; then
    log "Running tests"
    cd "$BUILD_DIR" && ctest --output-on-failure --timeout 300
  fi
}

bootstrap_rust() {
  if [[ "$RUST_BOOTSTRAP" -eq 0 ]]; then return 0; fi
  ensure_rust_toolchain

  local manifests=(
    "$ROOT_DIR/rust/fercuda-ffi/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-math/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-ml-lite/Cargo.toml"
    "$ROOT_DIR/rust/deps/fercuda-vision-lite/Cargo.toml"
    "$ROOT_DIR/rust/deps/ptx-os/Cargo.toml"
    "$ROOT_DIR/rust/deps/candle-ptx-os/Cargo.toml"
  )
  for manifest in "${manifests[@]}"; do
    if [[ -f "$manifest" ]]; then
      log "cargo fetch -> $(basename "$(dirname "$manifest")")"
      FERCUDA_BUILD_DIR="$BUILD_DIR" cargo fetch --manifest-path "$manifest" 2>/dev/null || true
    fi
  done
}

build_ferrite_mcp() {
  if [[ "$WITH_FERRITE_MCP" -eq 0 ]]; then return 0; fi
  ensure_rust_toolchain

  mkdir -p "$(dirname "$FERRITE_MCP_SOURCE_DIR")"
  if [[ -d "$FERRITE_MCP_SOURCE_DIR/.git" ]]; then
    [[ "$UPDATE" -eq 1 ]] && {
      git -C "$FERRITE_MCP_SOURCE_DIR" fetch --depth 1 origin "$FERRITE_MCP_BRANCH" 2>/dev/null || true
      git -C "$FERRITE_MCP_SOURCE_DIR" checkout -B "$FERRITE_MCP_BRANCH" "origin/$FERRITE_MCP_BRANCH" 2>/dev/null || true
    }
  else
    log "Cloning ferrite-mcp"
    git clone --depth 1 --branch "$FERRITE_MCP_BRANCH" "$FERRITE_MCP_REPO_URL" "$FERRITE_MCP_SOURCE_DIR" 2>/dev/null || {
      warn "ferrite-mcp clone failed; skipping"
      return 0
    }
  fi

  if [[ -f "$FERRITE_MCP_SOURCE_DIR/Cargo.toml" ]]; then
    log "Installing ferrite-mcp"
    cargo install \
      --path "$FERRITE_MCP_SOURCE_DIR/crates/shell-bin" \
      --bin ferrite \
      --locked --force --root "$PREFIX" 2>/dev/null || \
      warn "ferrite-mcp build failed; skipping"
  fi
}

install_remote_access_tools() {
  local autostart_src="$ROOT_DIR/scripts/ferrite-autostart"
  local ferrite_up_src="$ROOT_DIR/scripts/ferrite-up.sh"
  local autostart_dst="$PREFIX/bin/ferrite-autostart"
  local ferrite_up_dst="$PREFIX/bin/ferrite-up"

  mkdir -p "$PREFIX/bin"

  if [[ -f "$autostart_src" ]]; then
    install -m 0755 "$autostart_src" "$autostart_dst"
    ok "  Remote:     ${autostart_dst}"
  else
    warn "Remote autostart script missing at $autostart_src"
  fi

  if [[ -f "$ferrite_up_src" ]]; then
    install -m 0755 "$ferrite_up_src" "$ferrite_up_dst"
    ok "  Remote:     ${ferrite_up_dst}"
  else
    warn "ferrite-up helper missing at $ferrite_up_src"
  fi
}

# -- Install metadata --------------------------------------------------------
write_metadata() {
  mkdir -p "$STATE_ROOT"
  local commit="" branch_now=""
  if [[ -d "$ROOT_DIR/.git" ]]; then
    commit="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
    branch_now="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  fi
  cat > "$STATE_ROOT/install-meta.txt" <<META
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
platform=${OS} ${ARCH} (${DISTRO})
cuda_version=${CUDA_VERSION:-unknown}
prefix=${PREFIX}
build_dir=${BUILD_DIR}
source_dir=${ROOT_DIR}
arch_list=${ARCH_LIST}
commit=${commit}
branch=${branch_now}
META
}

# -- Main --------------------------------------------------------------------
main() {
  echo ""
  echo "  ╔═══════════════════════════════════════════════╗"
  echo "  ║  feRcuda — Deterministic CUDA Runtime         ║"
  echo "  ║  Portable Bootstrap Installer                 ║"
  echo "  ╚═══════════════════════════════════════════════╝"
  echo ""

  detect_platform
  install_build_deps
  detect_cuda
  detect_gpu_arch
  prepare_source
  build_fercuda
  run_tests
  bootstrap_rust
  build_ferrite_mcp
  install_remote_access_tools
  write_metadata

  echo ""
  ok "Installation complete!"
  ok ""
  ok "  Libraries:  ${PREFIX}/lib/libptx_core.so"
  ok "              ${PREFIX}/lib/libptx_kernels.so"
  ok "              ${PREFIX}/lib/libptx_hook.so (LD_PRELOAD intercept)"
  ok "              ${PREFIX}/lib/libfercuda_capi.so"
  ok "  Headers:    ${PREFIX}/include/fercuda/"
  ok "              ${PREFIX}/include/ptx/"
  ok "  Metadata:   ${STATE_ROOT}/install-meta.txt"
  ok ""
  ok "Add to your environment:"
  ok "  export LD_LIBRARY_PATH=\"${PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
  ok "  export PATH=\"${PREFIX}/bin:\${PATH}\""
  echo ""
}

ROOT_DIR=""
CUDA_VERSION=""
CUDA_PATH=""
ARCH=""
OS=""
DISTRO=""
ARCH_FAMILY=""

main "$@"
