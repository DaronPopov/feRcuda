# feR-os (feRcuda)

> **Copyright Daron Popov. All rights reserved.**  \
> This source is viewable for reference only.  \
> No license is granted for use, copying, modification, redistribution, sublicensing, or commercial use without prior written permission.

`feR-os` is a deterministic CUDA runtime substrate for GPU-managed execution.
It gives you a stable control plane to run memory-managed GPU workloads and bridge existing Rust/C/CUDA code onto your regime.

## What It Is

`feR-os` combines:
- deterministic numeric behavior (no implicit fast-math/fusion drift)
- O(1) TLSF pool allocation (native PTX-OS layer)
- ElasticPool delegation to native GPU runtime
- C API surface for language/runtime adapters
- optional CUDA allocation interception via `libptx_hook.so` (LD_PRELOAD)

This repo is the implementation and installer for that runtime layer.

## What It Enables

- predictable execution paths for long-running GPU services
- lower allocator jitter than per-op `cudaMalloc/cudaFree`
- integration of external libraries through an adapter/intercept path
- one runtime substrate usable from CUDA C++, C ABI, and Rust wrappers

## One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install.sh | bash
```

Installer defaults:
- managed checkout: `~/.local/share/fer-os/src/feRcuda`
- managed checkout: `~/.local/share/fer-os/src/ferrite-mcp`
- managed build dir: `~/.local/share/fer-os/build`
- install prefix: `~/.local`
- tracked branch: `main`
- Rust bootstrap: enabled (auto-fetches crate dependencies)
- ferrite-mcp install: enabled (installed via `cargo install --path ... --root ~/.local`)

Re-running the installer updates to the latest `origin/main`, rebuilds, and reinstalls.

## Installer Behavior (Managed)

`scripts/install.sh` is stateful and self-managed:
1. keeps managed git checkouts for `feRcuda` and `ferrite-mcp`
2. fetches latest remote commits on rerun
3. hard-resets managed checkouts to tracked branch heads
4. configures/builds/installs `feRcuda` from managed source
5. installs `ferrite-mcp` into the same prefix
6. writes install metadata to `~/.local/share/fer-os/install-meta.txt`
7. bootstraps Rust deps via `cargo fetch` across runtime crates

## Installer Options

```bash
bash scripts/install.sh --help
```

Common examples:

```bash
# Default managed install + update behavior
bash scripts/install.sh

# Pin GPU architectures
bash scripts/install.sh --arch-list "75;86;89"

# Run tests during install
bash scripts/install.sh --with-tests

# Install to custom prefix
bash scripts/install.sh --prefix /opt/fer-os

# Build from current working tree (no managed source update)
bash scripts/install.sh --use-local-source

# Skip Rust bootstrap
bash scripts/install.sh --no-rust-bootstrap

# Skip ferrite-mcp install (not recommended for full feR-os setup)
bash scripts/install.sh --no-ferrite-mcp
```

## Build / Runtime Notes

Default CUDA architectures are set in `CMakeLists.txt` and can be overridden with `--arch-list`.

### NVIDIA B200 + CUDA 13.1 (Driver 590)

```bash
# On B200 machine (auto-detects arch 100, CUDA 13.1 from nvcc)
./scripts/build_b200.sh

# Cross-compile from another machine
CUDARC_CUDA_VERSION=13010 ./scripts/build_b200.sh --cross
```

Or with install script: `bash scripts/install.sh --arch-list 100`

**One-line install + run mega test on B200:**

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install_and_run_b200.sh | bash
```

Clones feRcuda, builds for arch 100, runs `mega_stress_b200`. Requires CUDA 13.1, cmake, rustc, and `pip install torch`.

**Note:** PyTorch (ferrite-torch) requires libtorch built for CUDA 13.1. If pip `torch` doesn't yet support 13.1, set `LIBTORCH` to a compatible libtorch build.

If needed at runtime:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
```

Interception (LD_PRELOAD `libptx_hook.so`):
- `PTX_HOOK_VERBOSE=1` — print allocation/free info
- `PTX_HOOK_DEVICE=0` — CUDA device ID (default: 0)
- `PTX_HOOK_DISABLE=1` — disable hook
- `PTX_HOOK_MODE=tlsf|cuda|hybrid` — allocator mode (default: `tlsf`)
- `PTX_HOOK_HYBRID_FALLBACK=1` — allow TLSF alloc to fall back to CUDA (hybrid mode only)

### Cache Trimming

To reclaim rebuildable disk usage:

```bash
bash scripts/trim_cache.sh --dry-run
bash scripts/trim_cache.sh
```

Optional aggressive cleanup:

```bash
bash scripts/trim_cache.sh --global-cargo --feros-state
```

## Rust Layer

- `rust/fercuda-ffi` — C API bridge
- `rust/deps/ptx-os` — PTX-OS FFI (TLSF, streams, runtime)
- `rust/deps/cudarc-ptx` — CUDA driver wrapper with PTX-OS integration
- `rust/deps/candle-ptx-os` — Candle ML backend on PTX-OS
- `rust/deps/aten-ptx` — PyTorch/ATen TLSF allocator (patches libtorch CUDACachingAllocator)
- `rust/deps/ferrite-torch` — PyTorch (tch-rs) + TLSF examples
- `rust/deps/fercuda-ml-lite`, `fercuda-math`, `fercuda-vision-lite`

### PyTorch + TLSF (aten-ptx / ferrite-torch)

aten-ptx replaces PyTorch's CUDA allocator with PTX-OS TLSF. Auto-detects pip-installed torch and fixes CUDA version mismatch (preloads nvidia/nvjitlink before libtorch).

```bash
# Build native layer first
cmake -S . -B build && cmake --build build -j$(nproc)

# Run (auto-detects torch from pip, sets LD_LIBRARY_PATH)
./rust/deps/ferrite-torch/run_torch_example.sh torch_basic
./rust/deps/ferrite-torch/run_torch_example.sh torch_training
./rust/deps/ferrite-torch/run_torch_example.sh candle_torch_cohab   # Candle + PyTorch in one process
```

Or manually: `LIBTORCH=$(python -c "import torch; print(torch.__path__[0])") LD_LIBRARY_PATH=build:$LIBTORCH/lib cargo run -p ferrite-torch --example torch_basic --release`

Enable in fercuda-ffi: `cargo build -F torch`

## Compatibility Test Harness

Intercept compatibility smoke test (verifies `libptx_hook.so` ABI):

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)" --target test_intercept_compat
ctest --test-dir build --output-on-failure -R test_intercept_compat
```

Native layer tests (TLSF, tensor ops, production stress, fragmentation, concurrent):

```bash
ctest --test-dir build --output-on-failure -R "test_ptx_|test_intercept"
```

- **test_ptx_fragmentation**: alternating-free pattern, ratio bounds, O(1) alloc under fragmentation
- **test_ptx_concurrent_stress**: 24 threads × 5k ops, alloc+compute, burst contention — OS-level stability

## TLSF vs cudaMalloc Benchmark

Compare alloc/free latency: TLSF (libptx_hook) vs native cudaMalloc:

```bash
cmake -S . -B build && cmake --build build -j"$(nproc)" --target bench_alloc_vs_cudamalloc
bash scripts/bench_alloc_vs_cudamalloc.sh
```

Runs the same workload twice (native, then `LD_PRELOAD=libptx_hook.so`) and reports ns/op and speedup. Typical result: TLSF ~100–200x faster than cudaMalloc for alloc+free.

Manual run:
```bash
./build/bench_alloc_vs_cudamalloc                    # native
LD_PRELOAD=./build/libptx_hook.so ./build/bench_alloc_vs_cudamalloc  # TLSF
```

## Repository Layout

- `include/` — public headers (fercuda, ptx)
- `native/core/` — PTX-OS layer (TLSF, hot runtime, tensor kernels, JIT, hooks)
- `src/` — feRcuda runtime (session, ElasticPool, regime, MCP adapter)
- `tests/` — runtime, C API, intercept, native tests
- `rust/` — Rust adapters (fercuda-ffi, ptx-os, cudarc-ptx, candle-ptx-os)
- `scripts/install.sh` — managed bootstrap installer

## Agent Control Plane Design

- `JIT_CONTRACTS.md` defines the in-runtime JIT/script contracts.
- `AGENT_MCP_CONTRACTS.md` defines the MCP-facing agent control plane (`ferrite-mcp` <-> `feRcuda`) for headless feR-os operation.
