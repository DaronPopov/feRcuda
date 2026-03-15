#!/usr/bin/env bash
# Build feRcuda for NVIDIA B200 (Blackwell) + Driver 590 / CUDA 13.1
#
# Usage:
#   ./scripts/build_b200.sh              # Build on B200 machine (auto-detect)
#   ./scripts/build_b200.sh --cross      # Cross-compile (no B200 present)
#
# Requirements:
#   - CUDA 13.1 toolkit (nvcc 13.1)
#   - For B200: compute capability 10.0 (arch 100)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# B200 compute capability 10.0 -> arch 100
B200_ARCH="100"

CROSS=0
for arg in "$@"; do
    [[ "$arg" == "--cross" ]] && CROSS=1
done

echo "=== feRcuda build for B200 + CUDA 13.1 ==="
echo ""

# Native layer (CMake)
echo "[1/2] Building native layer..."
if [[ "$CROSS" -eq 1 ]]; then
    echo "  Cross-compile: forcing -DCMAKE_CUDA_ARCHITECTURES=$B200_ARCH"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$B200_ARCH"
else
    # Auto-detect from nvidia-smi (will get 100 on B200)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j"$(nproc 2>/dev/null || echo 8)"
echo ""

# Rust (cudarc-ptx uses cuda-version-from-build-system; nvcc 13.1 -> cuda-13010)
echo "[2/2] Building Rust crates..."
export FERCUDA_BUILD_DIR="$ROOT/build"

# Explicit CUDA 13.1 when cross-compiling (no nvcc 13.1 on build host)
# Format: major*1000 + minor*10 (13.1 -> 13010)
if [[ "$CROSS" -eq 1 ]]; then
    export CUDARC_CUDA_VERSION=13010
    echo "  Cross-compile: CUDARC_CUDA_VERSION=13010 (CUDA 13.1)"
fi

# Fetch/build Rust deps
cargo fetch --manifest-path rust/deps/ptx-os/Cargo.toml 2>/dev/null || true
cargo fetch --manifest-path rust/deps/candle-ptx-os/Cargo.toml 2>/dev/null || true
cargo fetch --manifest-path rust/deps/ferrite-torch/Cargo.toml 2>/dev/null || true

echo ""
echo "=== Build complete ==="
echo "Native libs: $ROOT/build/libptx_*.so"
echo ""
echo "To run ferrite-torch examples:"
echo "  ./rust/deps/ferrite-torch/run_torch_example.sh torch_basic"
echo ""
