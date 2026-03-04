# feR-os (feRcuda)

`feR-os` is a deterministic CUDA runtime substrate for GPU-managed execution.
It gives you a stable control plane to run memory-managed GPU workloads and bridge existing Rust/C/CUDA code onto your regime.

## What It Is

`feR-os` combines:
- deterministic numeric behavior (no implicit fast-math/fusion drift)
- reclaimable pool allocation (slab + TLSF path)
- persistent scheduler execution model
- C API surface for language/runtime adapters
- optional CUDA allocation interception for regime handoff

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
- managed build dir: `~/.local/share/fer-os/build`
- install prefix: `~/.local`
- tracked branch: `main`
- Rust bootstrap: enabled (auto-fetches crate dependencies)

Re-running the installer updates to the latest `origin/main`, rebuilds, and reinstalls.

## Installer Behavior (Managed)

`scripts/install.sh` is stateful and self-managed:
1. keeps a local managed git checkout
2. fetches latest remote commit on rerun
3. hard-resets managed checkout to tracked branch head
4. configures/builds/installs from managed source
5. writes install metadata to `~/.local/share/fer-os/install-meta.txt`
6. bootstraps Rust deps via `cargo fetch` across runtime crates

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
```

## Build / Runtime Notes

Default CUDA architectures are set in `CMakeLists.txt` and can be overridden with `--arch-list`.

If needed at runtime:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
```

Interception compatibility env knobs:
- `FERCUDA_GPU_HOT_SO=/absolute/path/to/libptx_os_shared.so`
- `FERCUDA_GPU_HOT_SO_PATHS=/path/a.so:/path/b.so:/path/c.so`
- `FERCUDA_INTERCEPT_ENABLE=1` (or `0` to force pure CUDA fallback)
- `FERCUDA_INTERCEPT_MODE=permissive|strict` (default: `permissive`)
- `FERCUDA_INTERCEPT_REGIME=tlsf|regime2|segv2` (default: `tlsf`)
- `FERCUDA_INTERCEPT_ASYNC_REGIME=1` to route async alloc APIs through active regime path
- `FERCUDA_INTERCEPT_ASYNC_TLSF=1` legacy alias for async regime routing
- `FERCUDA_INTERCEPT_DLCLOSE=1` to close shared libs on shutdown (default keeps handles open for safer teardown ordering)

If async CUDA APIs are unavailable on a target runtime, the intercept layer now degrades to sync CUDA alloc/free fallback instead of hard failure.

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

- `rust/fercuda-ffi` (C API bridge)
- `rust/deps/fercuda-ml-lite`
- `rust/deps/fercuda-math`
- `rust/deps/fercuda-vision-lite`

## Compatibility Test Harness

Intercept compatibility smoke test target:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)" --target test_intercept_compat
ctest --test-dir build --output-on-failure -R test_intercept_compat
```

This test exercises:
- permissive mode fallback paths
- managed/host/pitch API telemetry accounting
- strict mode rejection path in a subprocess

## External Runtime Benchmark

Compare external CUDA alloc/free workload in:
- native mode
- `feR-os` intercept/runtime mode (`LD_PRELOAD`)

```bash
bash scripts/benchmark_external_runtime.sh --iters 20000 --bytes 65536
bash scripts/benchmark_external_runtime.sh --iters 20000 --bytes 65536 --regime sizeclass
```

Optional:

```bash
# Async alloc API path
bash scripts/benchmark_external_runtime.sh --iters 20000 --bytes 65536 --async --async-tlsf --regime sizeclass

# Explicit GPU hot runtime shared library
bash scripts/benchmark_external_runtime.sh --gpu-hot-so /path/to/libptx_os_shared.so
```

## Repository Layout

- `include/` public headers
- `src/` CUDA/C++ runtime
- `tests/` runtime and integration tests
- `examples/` usage demos
- `rust/` Rust adapters/deps
- `scripts/install.sh` managed installer
