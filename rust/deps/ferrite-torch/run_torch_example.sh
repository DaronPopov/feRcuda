#!/usr/bin/env bash
# Run ferrite-torch examples with correct LIBTORCH and LD_LIBRARY_PATH.
# Fixes CUDA version mismatch (pip torch vs system) by using torch's bundled libs.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE="${1:-torch_basic}"

# Resolve torch path: LIBTORCH env, or from pip
if [ -n "$LIBTORCH" ] && [ -d "$LIBTORCH/lib" ]; then
    TORCH_ROOT="$LIBTORCH"
else
    TORCH_ROOT=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)
fi

if [ -z "$TORCH_ROOT" ] || [ ! -d "$TORCH_ROOT/lib" ]; then
    echo "Error: Could not find libtorch. Set LIBTORCH or install: pip install torch"
    exit 1
fi

# feRcuda native libs (ptx_core, ptx_kernels)
FERCUDA_BUILD="${FERCUDA_BUILD_DIR:-$SCRIPT_DIR/../../../build}"
if [ ! -f "$FERCUDA_BUILD/libptx_core.so" ]; then
    echo "Error: Build native layer first: cmake -S . -B build && cmake --build build"
    exit 1
fi

export LD_LIBRARY_PATH="$FERCUDA_BUILD:$TORCH_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LIBTORCH="$TORCH_ROOT"

cd "$SCRIPT_DIR"
exec cargo run --example "$EXAMPLE" --release -- "${@:2}"
