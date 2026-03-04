#!/usr/bin/env bash
set -euo pipefail

ITERS=20000
BYTES=$((64 * 1024))
ASYNC=0
ASYNC_TLSF=0
REGIME="tlsf"
GPU_HOT_SO="${FERCUDA_GPU_HOT_SO:-}"
BUILD_DIR="build"

print_help() {
  cat <<'USAGE'
Benchmark external CUDA alloc/free workload in two modes:
  1) native CUDA runtime
  2) feR-os intercept/runtime path (LD_PRELOAD)

Usage:
  benchmark_external_runtime.sh [options]

Options:
  --iters N         Number of alloc/free iterations (default: 20000)
  --bytes N         Bytes per allocation (default: 65536)
  --async           Use cudaMallocAsync/cudaFreeAsync path
  --async-tlsf      Set FERCUDA_INTERCEPT_ASYNC_TLSF=1 for intercept run
  --regime NAME     Intercept regime: tlsf|sizeclass (aliases: regime2|segv2)
  --gpu-hot-so P    Explicit libptx_os_shared.so path for intercept mode
  --build-dir DIR   CMake build dir (default: build)
  -h, --help        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters)
      ITERS="$2"; shift 2 ;;
    --bytes)
      BYTES="$2"; shift 2 ;;
    --async)
      ASYNC=1; shift ;;
    --async-tlsf)
      ASYNC_TLSF=1; shift ;;
    --regime)
      REGIME="$2"; shift 2 ;;
    --gpu-hot-so)
      GPU_HOT_SO="$2"; shift 2 ;;
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      print_help
      exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[bench] configure/build"
cmake -S . -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" --target bench_external_alloc fercuda_intercept >/dev/null

BENCH_BIN="$ROOT/$BUILD_DIR/bench_external_alloc"
INTERCEPT_SO="$ROOT/$BUILD_DIR/libfercuda_intercept.so"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "benchmark binary not found: $BENCH_BIN" >&2
  exit 2
fi
if [[ ! -f "$INTERCEPT_SO" ]]; then
  echo "intercept library not found: $INTERCEPT_SO" >&2
  exit 3
fi

ARGS=(--iters "$ITERS" --bytes "$BYTES")
if [[ "$ASYNC" -eq 1 ]]; then
  ARGS+=(--async)
fi

NATIVE_OUT="$(mktemp /tmp/feros-bench-native.XXXXXX)"
INTERCEPT_OUT="$(mktemp /tmp/feros-bench-intercept.XXXXXX)"
trap 'rm -f "$NATIVE_OUT" "$INTERCEPT_OUT"' EXIT

echo "[bench] run native"
set +e
"$BENCH_BIN" --label native "${ARGS[@]}" | tee "$NATIVE_OUT"
NATIVE_RC=$?
set -e
if [[ "$NATIVE_RC" -ne 0 ]]; then
  echo "[bench] native run failed (exit=$NATIVE_RC)."
  echo "[bench] This usually means CUDA runtime/device is unavailable on this host."
  echo "[bench] benchmark output:"
  cat "$NATIVE_OUT"
  exit "$NATIVE_RC"
fi

echo "[bench] run intercept"
INTERCEPT_ENV=(
  "LD_PRELOAD=$INTERCEPT_SO"
  "FERCUDA_INTERCEPT_ENABLE=1"
  "FERCUDA_INTERCEPT_MODE=permissive"
  "FERCUDA_INTERCEPT_LOG=0"
  "FERCUDA_INTERCEPT_REGIME=$REGIME"
)
if [[ "$ASYNC_TLSF" -eq 1 ]]; then
  INTERCEPT_ENV+=("FERCUDA_INTERCEPT_ASYNC_TLSF=1")
fi
if [[ -n "$GPU_HOT_SO" ]]; then
  INTERCEPT_ENV+=("FERCUDA_GPU_HOT_SO=$GPU_HOT_SO")
fi
set +e
env "${INTERCEPT_ENV[@]}" "$BENCH_BIN" --label intercept "${ARGS[@]}" | tee "$INTERCEPT_OUT"
INTERCEPT_RC=$?
set -e
if [[ "$INTERCEPT_RC" -ne 0 ]]; then
  echo "[bench] intercept run failed (exit=$INTERCEPT_RC)."
  echo "[bench] benchmark output:"
  cat "$INTERCEPT_OUT"
  exit "$INTERCEPT_RC"
fi

get_key() {
  local file="$1"
  local key="$2"
  awk -F= -v k="$key" '$1==k {print $2}' "$file" | tail -n 1
}

NATIVE_OPS="$(get_key "$NATIVE_OUT" bench.alloc_ops_per_s)"
INTERCEPT_OPS="$(get_key "$INTERCEPT_OUT" bench.alloc_ops_per_s)"
NATIVE_GBS="$(get_key "$NATIVE_OUT" bench.gb_per_s)"
INTERCEPT_GBS="$(get_key "$INTERCEPT_OUT" bench.gb_per_s)"
TLSF_OK="$(get_key "$INTERCEPT_OUT" intercept.tlsf_alloc_success)"
SIZECLASS_OK="$(get_key "$INTERCEPT_OUT" intercept.sizeclass_alloc_success)"
FALLBACKS="$(get_key "$INTERCEPT_OUT" intercept.fallback_alloc_calls)"
INIT_OK="$(get_key "$INTERCEPT_OUT" intercept.init_success)"

python3 - <<PY
native_ops = float("${NATIVE_OPS:-0}")
intercept_ops = float("${INTERCEPT_OPS:-0}")
native_gbs = float("${NATIVE_GBS:-0}")
intercept_gbs = float("${INTERCEPT_GBS:-0}")

def ratio(a, b):
    return (a / b) if b > 0 else 0.0

print("[bench] summary")
print(f"native.alloc_ops_per_s={native_ops:.2f}")
print(f"intercept.alloc_ops_per_s={intercept_ops:.2f}")
print(f"alloc_ops_speedup_vs_native={ratio(intercept_ops, native_ops):.4f}x")
print(f"native.gb_per_s={native_gbs:.4f}")
print(f"intercept.gb_per_s={intercept_gbs:.4f}")
print(f"gbps_speedup_vs_native={ratio(intercept_gbs, native_gbs):.4f}x")
PY

echo "[bench] intercept.init_success=${INIT_OK:-0}"
echo "[bench] intercept.regime=${REGIME}"
echo "[bench] intercept.tlsf_alloc_success=${TLSF_OK:-0}"
echo "[bench] intercept.sizeclass_alloc_success=${SIZECLASS_OK:-0}"
echo "[bench] intercept.fallback_alloc_calls=${FALLBACKS:-0}"

echo "[bench] done"
