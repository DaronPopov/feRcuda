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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "long":
            sys.exit(main_long())
        if sys.argv[1] == "extreme":
            sys.exit(main_extreme())
        if sys.argv[1] == "demo":
            sys.exit(main_demo())
    sys.exit(main())
