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

Re-running the installer updates to the latest `origin/main`, rebuilds, and reinstalls.

## Installer Behavior (Managed)

`scripts/install.sh` is stateful and self-managed:
1. keeps a local managed git checkout
2. fetches latest remote commit on rerun
3. hard-resets managed checkout to tracked branch head
4. configures/builds/installs from managed source
5. writes install metadata to `~/.local/share/fer-os/install-meta.txt`

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
```

## Build / Runtime Notes

Default CUDA architectures are set in `CMakeLists.txt` and can be overridden with `--arch-list`.

If needed at runtime:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
```

## Rust Layer

- `rust/fercuda-ffi` (C API bridge)
- `rust/deps/fercuda-ml-lite`
- `rust/deps/fercuda-math`
- `rust/deps/fercuda-vision-lite`

## Repository Layout

- `include/` public headers
- `src/` CUDA/C++ runtime
- `tests/` runtime and integration tests
- `examples/` usage demos
- `rust/` Rust adapters/deps
- `scripts/install.sh` managed installer
