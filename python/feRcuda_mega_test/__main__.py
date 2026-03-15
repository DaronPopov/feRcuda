"""Run the feRcuda mega stress test (bundled binary + libs)."""
import os
import subprocess
import sys
from pathlib import Path

def _get_package_dir() -> Path:
    return Path(__file__).resolve().parent

def _get_libs_dir() -> Path:
    return _get_package_dir() / "libs"

def _get_bin_dir() -> Path:
    return _get_package_dir() / "bin"

def _run_benchmark(exe_name: str) -> int:
    libs = _get_libs_dir()
    bin_dir = _get_bin_dir()
    exe = bin_dir / exe_name

    if not exe.exists():
        print(f"Error: {exe_name} binary not found. Rebuild the wheel.", file=sys.stderr)
        return 1

    torch_lib = ""
    try:
        import torch
        torch_lib = str(Path(torch.__path__[0]) / "lib")
    except ImportError:
        pass

    if not torch_lib or not Path(torch_lib).exists():
        print("Error: PyTorch required. Install: pip install torch", file=sys.stderr)
        return 1

    ld_path = os.pathsep.join(filter(None, [str(libs), torch_lib, os.environ.get("LD_LIBRARY_PATH", "")]))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_path
    env["FERCUDA_BUILD_DIR"] = str(libs)
    env["LIBTORCH"] = str(Path(torch_lib).parent)

    return subprocess.run([str(exe)], env=env).returncode

def main() -> int:
    """Run the standard mega stress test (~10k cycles)."""
    return _run_benchmark("mega_stress_b200")

def main_long() -> int:
    """Run the extended benchmark with numeric results (~50k cycles, longer)."""
    return _run_benchmark("mega_stress_long")

def main_extreme() -> int:
    """Run the compute-heavy plus allocation-heavy extended stress demo."""
    return _run_benchmark("mega_stress_extreme")

def main_demo() -> int:
    """Run the Torch+Candle demo with printed tensor values."""
    return _run_benchmark("torch_candle_demo")

def _get_val(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return ""

def main_bench_alloc() -> int:
    """Run cudaMalloc vs TLSF alloc/free benchmark (proves TLSF speedup)."""
    libs = _get_libs_dir()
    bin_dir = _get_bin_dir()
    bench = bin_dir / "bench_alloc_vs_cudamalloc"
    hook_so = libs / "libptx_hook.so"

    if not bench.exists():
        print("Error: bench_alloc_vs_cudamalloc not found. Rebuild the wheel.", file=sys.stderr)
        return 1
    if not hook_so.exists():
        print("Error: libptx_hook.so not found. Rebuild the wheel on Linux.", file=sys.stderr)
        return 1

    ld_path = os.pathsep.join(filter(None, [str(libs), os.environ.get("LD_LIBRARY_PATH", "")]))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_path

    iters = os.environ.get("BENCH_ITERS", "50000")
    bytes_arg = os.environ.get("BENCH_BYTES", "65536")
    args = [str(bench), "--iters", iters, "--bytes", bytes_arg]

    print("\n  ═══════════════════════════════════════════════════════")
    print("  feRcuda: cudaMalloc vs TLSF alloc/free benchmark")
    print(f"  iters={iters}  bytes={bytes_arg}")
    print("  ═══════════════════════════════════════════════════════\n")

    # Run 1: native cudaMalloc
    print("[1/2] Native cudaMalloc/cudaFree...")
    r1 = subprocess.run(args, env=env, capture_output=True, text=True)
    if r1.returncode != 0:
        print(r1.stderr or r1.stdout, file=sys.stderr)
        return 1
    native_lines = r1.stdout.strip().split("\n")
    native_ns = float(_get_val(native_lines, "ns_per_alloc_free") or "0")
    native_ops = float(_get_val(native_lines, "alloc_free_per_sec") or "0")

    # Run 2: TLSF (LD_PRELOAD)
    print("[2/2] TLSF (libptx_hook.so)...")
    env["LD_PRELOAD"] = str(hook_so)
    r2 = subprocess.run(args, env=env, capture_output=True, text=True)
    if r2.returncode != 0:
        print(r2.stderr or r2.stdout, file=sys.stderr)
        return 1
    tlsf_lines = r2.stdout.strip().split("\n")
    tlsf_ns = float(_get_val(tlsf_lines, "ns_per_alloc_free") or "0")
    tlsf_ops = float(_get_val(tlsf_lines, "alloc_free_per_sec") or "0")

    # Results
    speedup = native_ns / tlsf_ns if tlsf_ns > 0 else 0
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  RESULTS                                              │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  cudaMalloc:  {native_ns:10.1f} ns/op   {native_ops:10.0f} ops/s     │")
    print(f"  │  TLSF:       {tlsf_ns:10.1f} ns/op   {tlsf_ops:10.0f} ops/s     │")
    print("  ├─────────────────────────────────────────────────────┤")
    if tlsf_ns < native_ns:
        print(f"  │  TLSF wins: {speedup:.2f}x faster than cudaMalloc          │")
    else:
        print(f"  │  cudaMalloc wins (TLSF {native_ns/tlsf_ns:.2f}x slower)               │")
    print("  └─────────────────────────────────────────────────────┘\n")
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "long":
            sys.exit(main_long())
        if sys.argv[1] == "extreme":
            sys.exit(main_extreme())
        if sys.argv[1] == "demo":
            sys.exit(main_demo())
        if sys.argv[1] == "bench":
            sys.exit(main_bench_alloc())
    sys.exit(main())
