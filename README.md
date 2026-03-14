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
- `rust/deps/fercuda-ml-lite`, `fercuda-math`, `fercuda-vision-lite`

## Compatibility Test Harness

Intercept compatibility smoke test (verifies `libptx_hook.so` ABI):

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)" --target test_intercept_compat
ctest --test-dir build --output-on-failure -R test_intercept_compat
```

Native layer tests (TLSF, tensor ops, production stress):

```bash
ctest --test-dir build --output-on-failure -R "test_ptx_|test_intercept"
```

## External Runtime Benchmark

To compare CUDA alloc/free in native vs TLSF intercept mode:

```bash
# Build
cmake -S . -B build && cmake --build build -j"$(nproc)"

# Run with TLSF intercept
LD_PRELOAD=./build/libptx_hook.so your_cuda_app
```

The intercept library `libptx_hook.so` redirects `cudaMalloc`/`cudaFree` through the PTX-OS TLSF allocator. See `native/core/hooks/cuda_intercept.c` for implementation.

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
