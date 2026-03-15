#!/usr/bin/env bash
# =============================================================================
# One-line install + run feRcuda MEGA TEST on B200
#
#   curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install_and_run_b200.sh | bash
#
# Clones feRcuda, builds for B200 (arch 100) + CUDA 13.1, runs mega_stress_b200.
# Requires: CUDA 13.1, cmake, rustc, pip install torch (for PyTorch phases)
# =============================================================================
set -euo pipefail

REPO_URL="${FERCUDA_REPO_URL:-https://github.com/DaronPopov/feRcuda.git}"
BRANCH="${FERCUDA_BRANCH:-main}"
DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
SOURCE_DIR="${DATA_HOME}/fer-os/src/feRcuda"
BUILD_DIR="${DATA_HOME}/fer-os/build"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
B200_ARCH="100"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  feRcuda B200: One-line install + MEGA TEST                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# 1. Clone
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$SOURCE_DIR")"
if [[ -d "$SOURCE_DIR/.git" ]]; then
  echo "[1/4] Updating existing checkout..."
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$BRANCH" 2>/dev/null || true
  git -C "$SOURCE_DIR" checkout -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || true
  git -C "$SOURCE_DIR" reset --hard "origin/$BRANCH" 2>/dev/null || true
else
  echo "[1/4] Cloning feRcuda..."
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SOURCE_DIR"
fi

ROOT="$SOURCE_DIR"
cd "$ROOT"


# -----------------------------------------------------------------------------
# Ensure cmake available (bootstrap via pip if missing, e.g. on Slurm clusters)
# -----------------------------------------------------------------------------
ensure_cmake() {
  if command -v cmake &>/dev/null; then return 0; fi
  if command -v module &>/dev/null; then
    module load cmake 2>/dev/null && command -v cmake &>/dev/null && return 0
  fi
  echo "cmake not found, installing via pip..."
  pip install --user cmake 2>/dev/null || pip3 install --user cmake 2>/dev/null || true
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v cmake &>/dev/null
}
ensure_cmake || { echo "ERROR: cmake required. Install: pip install cmake  OR  module load cmake"; exit 1; }
# -----------------------------------------------------------------------------
# 2. Build native layer (B200 arch 100)
# -----------------------------------------------------------------------------
echo "[2/4] Building native layer (arch $B200_ARCH)..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$B200_ARCH"
cmake --build "$BUILD_DIR" -j"$JOBS"

if [[ ! -f "$BUILD_DIR/libptx_core.so" ]]; then
  echo "ERROR: Native build failed (libptx_core.so not found)"
  exit 1
fi

# -----------------------------------------------------------------------------
# 3. Build Rust mega test
# -----------------------------------------------------------------------------
echo "[3/4] Building mega_stress_b200..."
export FERCUDA_BUILD_DIR="$BUILD_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

# CUDA 13.1 when cross-compiling
[[ -n "${CUDARC_CUDA_VERSION:-}" ]] && export CUDARC_CUDA_VERSION

if [[ -f "$ROOT/rust/deps/ferrite-torch/run_torch_example.sh" ]]; then
  cd "$ROOT/rust/deps/ferrite-torch"
  cargo build --example mega_stress_b200 --release 2>&1 | tail -5
else
  echo "ERROR: ferrite-torch not found"
  exit 1
fi

# -----------------------------------------------------------------------------
# 4. Run mega test (uses run_torch_example.sh for LIBTORCH/LD_LIBRARY_PATH)
# -----------------------------------------------------------------------------
echo "[4/4] Running MEGA TEST..."
echo ""

export FERCUDA_BUILD_DIR="$BUILD_DIR"
exec "$ROOT/rust/deps/ferrite-torch/run_torch_example.sh" mega_stress_b200
