#!/usr/bin/env bash
# Build feRcuda mega test as a Python wheel (cross-compile for B200).
#
# Run on ANY machine with cmake, nvcc, gcc, rust - GPU does NOT need to be B200.
# Target is B200 (arch 100) by default, but auto-falls back to local GPU if nvcc
# doesn't support arch 100 (requires CUDA 12.8+).
#
#   ./scripts/build_wheel.sh
#
# Override: FERCUDA_WHEEL_ARCH=89 FERCUDA_WHEEL_CUDA=12080 ./scripts/build_wheel.sh
#
# Output: python/dist/feRcuda_mega_test-0.1.0-*.whl

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: load modules only if 'module' exists (e.g. HPC clusters)
if [[ -n "${FERCUDA_MODULES:-}" ]] && command -v module &>/dev/null; then
  echo "Loading modules: $FERCUDA_MODULES"
  for m in $FERCUDA_MODULES; do module load "$m"; done
fi

# Target arch/cuda. Arch 100 (B200) requires nvcc 12.8+ (CUDA 12.8).
TARGET_ARCH="${FERCUDA_WHEEL_ARCH:-}"
TARGET_CUDA="${FERCUDA_WHEEL_CUDA:-}"

if [[ -z "$TARGET_ARCH" ]]; then
  # Probe: does nvcc actually support arch 100 (B200)? CUDA 12.8+ required.
  NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1)
  if echo '__global__ void k(){}' | nvcc -arch=compute_100 -x cu -c - -o /dev/null 2>/dev/null; then
    TARGET_ARCH=100
    TARGET_CUDA="${TARGET_CUDA:-13010}"
  else
    # Fall back to local GPU (nvcc doesn't support B200 - need CUDA 12.8+)
    LOCAL_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ -n "$LOCAL_CC" && "$LOCAL_CC" =~ ^[0-9]+$ ]]; then
      TARGET_ARCH="$LOCAL_CC"
      echo "Note: nvcc $NVCC_VER doesn't support B200 (arch 100). Using local GPU arch $TARGET_ARCH."
      echo "      Install CUDA 12.8+ to build for B200, or set FERCUDA_WHEEL_ARCH=100 on a build node that has it."
      [[ -z "$TARGET_CUDA" ]] && TARGET_CUDA=12060
    else
      TARGET_ARCH=89
      TARGET_CUDA="${TARGET_CUDA:-12080}"
      echo "Note: Could not detect GPU. Using arch 89 (Ada). Set FERCUDA_WHEEL_ARCH for B200."
    fi
  fi
fi
[[ -z "$TARGET_CUDA" ]] && TARGET_CUDA=$([[ "$TARGET_ARCH" == "100" ]] && echo 13010 || echo 12060)
CUDARC_FEATURE="cuda-${TARGET_CUDA}"

PKG_LIBS="$ROOT/python/feRcuda_mega_test/libs"
PKG_BIN="$ROOT/python/feRcuda_mega_test/bin"
mkdir -p "$PKG_LIBS" "$PKG_BIN"

echo "=== Building feRcuda mega test wheel ==="
echo "  Target arch: $TARGET_ARCH (100=B200, 89=Ada, 90=Hopper)"
echo "  Target CUDA: $TARGET_CUDA -> feature $CUDARC_FEATURE"
echo ""

# 1. Native layer - build only ptx_core and ptx_kernels (skip bench/tests)
echo "[1/4] Building native layer..."
CMAKE_EXTRA=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$TARGET_ARCH")
# Force nvcc when targeting B200 so CMake doesn't use cached CUDA 12.6 (lacks compute_100)
[[ "$TARGET_ARCH" == "100" ]] && CMAKE_EXTRA+=(-DCMAKE_CUDA_COMPILER="$(which nvcc)")
cmake -S . -B build "${CMAKE_EXTRA[@]}"
cmake --build build -j"$(nproc 2>/dev/null || echo 8)" --target ptx_core ptx_kernels

cp build/libptx_core.so build/libptx_kernels.so "$PKG_LIBS/"
echo "  -> $PKG_LIBS/"

# 2. Rust benchmarks (aten-ptx needs libtorch from pip)
echo "[2/4] Building mega_stress_b200 + mega_stress_long + mega_stress_extreme + torch_candle_demo..."
export FERCUDA_BUILD_DIR="$ROOT/build"
export LD_LIBRARY_PATH="$ROOT/build:${LD_LIBRARY_PATH:-}"
export CUDARC_CUDA_VERSION="$TARGET_CUDA"
# Resolve libtorch: use LIBTORCH only if valid (has lib/); else auto-detect from pip
if [[ -z "${LIBTORCH:-}" || ! -d "${LIBTORCH}/lib" ]]; then
  unset LIBTORCH
  for py in "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python3}" python3 python "$HOME/.venv/bin/python3" "$ROOT/.venv/bin/python3"; do
    [[ -z "$py" ]] && continue
    LIBTORCH=$("$py" -c "import torch; print(torch.__path__[0])" 2>/dev/null) || continue
    [[ -n "$LIBTORCH" && -d "$LIBTORCH/lib" ]] && break
    LIBTORCH=""
  done
  [[ -n "$LIBTORCH" && -d "$LIBTORCH/lib" ]] && export LIBTORCH
fi
if [[ -z "${LIBTORCH:-}" || ! -d "${LIBTORCH}/lib" ]]; then
  echo "Error: libtorch required for aten-ptx. Activate venv and: pip install torch"
  echo "  Or set LIBTORCH=/path/to/libtorch (must point to a real torch install with lib/)"
  exit 1
fi

cd "$ROOT/rust/deps/ferrite-torch"
cargo build --example mega_stress_b200 --example mega_stress_long --example mega_stress_extreme --example torch_candle_demo --release --no-default-features --features "${CUDARC_FEATURE},candle-cohab"

cp target/release/examples/mega_stress_b200 target/release/examples/mega_stress_long target/release/examples/mega_stress_extreme target/release/examples/torch_candle_demo "$PKG_BIN/"
echo "  -> $PKG_BIN/"

# 3. Build wheel
echo "[3/4] Building wheel..."
cd "$ROOT/python"
# Need a python with pip+build; venv may lack pip, so try python3 (often has build in ~/.local)
for py in python3 python /usr/bin/python3; do
  if "$py" -c "import build" 2>/dev/null; then
    PY="$py"; break
  fi
  "$py" -m pip install build --user -q 2>/dev/null && "$py" -c "import build" 2>/dev/null && PY="$py" && break
done
if [[ -z "${PY:-}" ]]; then
  echo "Error: No python with 'build' module. Install with: pip install build"
  exit 1
fi
"$PY" -m build --wheel

echo ""
echo "[4/4] Done. Wheel:"
ls -la dist/*.whl 2>/dev/null || true
echo ""
echo "Install: pip install dist/feRcuda_mega_test-*.whl"
echo "Run: feRcuda-mega-test"
