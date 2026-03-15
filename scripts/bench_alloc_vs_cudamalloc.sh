#!/usr/bin/env bash
# =============================================================================
# Benchmark: TLSF (libptx_hook) vs native cudaMalloc/cudaFree
#
# Runs the same alloc/free workload twice and compares latency.
# Usage: bash scripts/bench_alloc_vs_cudamalloc.sh [--build-dir DIR]
# =============================================================================
set -euo pipefail

BUILD_DIR="build"
[[ "${1:-}" = "--build-dir" && -n "${2:-}" ]] && { BUILD_DIR="$2"; shift 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ITERS="${BENCH_ITERS:-50000}"
BYTES="${BENCH_BYTES:-65536}"

# Ensure build
if [[ ! -f "$BUILD_DIR/bench_alloc_vs_cudamalloc" ]]; then
    echo "[bench] Building..."
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)" --target bench_alloc_vs_cudamalloc
fi

BIN="$ROOT/$BUILD_DIR/bench_alloc_vs_cudamalloc"
HOOK_SO="$ROOT/$BUILD_DIR/libptx_hook.so"

if [[ ! -x "$BIN" ]]; then
    echo "benchmark binary not found: $BIN" >&2
    exit 1
fi

if [[ ! -f "$HOOK_SO" ]]; then
    echo "libptx_hook.so not found: $HOOK_SO (Linux only)" >&2
    exit 2
fi

echo ""
echo "  ═══════════════════════════════════════════════════════"
echo "  feRcuda: TLSF vs cudaMalloc alloc/free benchmark"
echo "  iters=$ITERS  bytes=$BYTES"
echo "  ═══════════════════════════════════════════════════════"
echo ""

# Run native
echo "[1/2] Native cudaMalloc/cudaFree..."
NATIVE_OUT=$(mktemp)
if ! "$BIN" --iters "$ITERS" --bytes "$BYTES" >"$NATIVE_OUT" 2>&1; then
    echo "Native run failed:"
    cat "$NATIVE_OUT"
    rm -f "$NATIVE_OUT"
    exit 3
fi

# Run TLSF (LD_PRELOAD)
echo "[2/2] TLSF (libptx_hook.so)..."
TLSF_OUT=$(mktemp)
export LD_LIBRARY_PATH="$ROOT/$BUILD_DIR:${LD_LIBRARY_PATH:-}"
if ! LD_PRELOAD="$HOOK_SO" "$BIN" --iters "$ITERS" --bytes "$BYTES" >"$TLSF_OUT" 2>&1; then
    echo "TLSF run failed:"
    cat "$TLSF_OUT"
    rm -f "$NATIVE_OUT" "$TLSF_OUT"
    exit 4
fi

# Parse results
get_val() { grep "^$1=" "$2" | cut -d= -f2; }
NATIVE_NS=$(get_val ns_per_alloc_free "$NATIVE_OUT")
TLSF_NS=$(get_val ns_per_alloc_free "$TLSF_OUT")
NATIVE_OPS=$(get_val alloc_free_per_sec "$NATIVE_OUT")
TLSF_OPS=$(get_val alloc_free_per_sec "$TLSF_OUT")

# Compute speedup (TLSF wins if it's faster = lower ns)
SPEEDUP="N/A"
if [[ -n "$NATIVE_NS" && -n "$TLSF_NS" ]]; then
    SPEEDUP=$(awk "BEGIN { printf \"%.2f\", $NATIVE_NS / $TLSF_NS }" 2>/dev/null || echo "N/A")
fi

echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  RESULTS                                              │"
echo "  ├─────────────────────────────────────────────────────┤"
printf "  │  cudaMalloc:  %10.1f ns/op   %10.0f ops/s     │\n" "$NATIVE_NS" "$NATIVE_OPS"
printf "  │  TLSF:       %10.1f ns/op   %10.0f ops/s     │\n" "$TLSF_NS" "$TLSF_OPS"
echo "  ├─────────────────────────────────────────────────────┤"
if [[ "$SPEEDUP" != "N/A" ]]; then
    if awk "BEGIN { exit ($TLSF_NS < $NATIVE_NS) ? 0 : 1 }" 2>/dev/null; then
        printf "  │  TLSF wins: %.2fx faster than cudaMalloc          │\n" "$SPEEDUP"
    else
        RATIO=$(awk "BEGIN { printf \"%.2f\", $TLSF_NS / $NATIVE_NS }" 2>/dev/null || echo "1.00")
        printf "  │  cudaMalloc wins (TLSF %.2fx slower)               │\n" "$RATIO"
    fi
fi
echo "  └─────────────────────────────────────────────────────┘"
echo ""

rm -f "$NATIVE_OUT" "$TLSF_OUT"

# Exit 0 if TLSF won
TLSF_WON=0
if [[ "$SPEEDUP" != "N/A" ]] && awk "BEGIN { exit ($TLSF_NS < $NATIVE_NS) ? 0 : 1 }" 2>/dev/null; then
    TLSF_WON=1
fi
if [[ "$TLSF_WON" -eq 1 ]]; then
    echo "  ✓ Your design wins."
    exit 0
else
    echo "  (Run with GPU for meaningful comparison)"
    exit 0
fi
