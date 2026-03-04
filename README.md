# feR-os (feRcuda)

Deterministic CUDA runtime + memory/scheduler substrate for building GPU-orchestrated systems.

`feR-os` is the runtime/OS-facing layer built from the `feRcuda` codebase:
- deterministic numeric wrappers and typed tensor views
- slab + TLSF-style reclaimable memory regime integration
- persistent GPU scheduler and op dispatch plane
- C API bridge (`libfercuda_capi.so`) for Rust/Python/C integration
- optional CUDA allocation interception (`libfercuda_intercept.so`)
- Rust deps layer (`rust/deps/*`) for ML/math/vision adapters

## One-line Install

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/feRcuda/main/scripts/install.sh | bash
```

This installs to `~/.local` by default.

## Quick Build (Manual)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/.local"
```

## Installer Options

```bash
bash scripts/install.sh --help
```

Common examples:

```bash
# Install to default ~/.local
bash scripts/install.sh

# Install to custom prefix
bash scripts/install.sh --prefix /opt/feros

# Build for specific GPU targets
bash scripts/install.sh --arch-list "75;86;89"

# Build + run C++/CUDA tests
bash scripts/install.sh --with-tests
```

## Build Targets and Architecture Coverage

Default CUDA arch list is configured in `CMakeLists.txt`:
- `75;80;86;87;89;90`

You can override at configure/install time with:
- `-DCMAKE_CUDA_ARCHITECTURES="..."` or installer `--arch-list`

For Jetson-specific builds, set only the target architecture(s) used by the board.

## Rust Layer

Rust crates are under `rust/`:
- `rust/fercuda-ffi` (FFI bridge)
- `rust/deps/fercuda-ml-lite`
- `rust/deps/fercuda-math`
- `rust/deps/fercuda-vision-lite`

Run checks:

```bash
cd rust/fercuda-ffi
cargo test
```

## Runtime Libraries

Installed libraries:
- `libfercuda.a`
- `libfercuda_capi.so`
- `libfercuda_intercept.so` (Linux)

Headers install under `<prefix>/include`.

## Environment

At runtime, ensure dynamic linker can find installed libs:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
```

## Repository Layout

- `include/` public headers
- `src/` CUDA/C++ runtime implementation
- `tests/` unit/integration tests
- `examples/` runtime demos
- `rust/` Rust integration crates
- `scripts/install.sh` one-line install entry

## Status

Core runtime, allocator reclaim path, C API bridge, and Rust adapter crates are active development targets for `feR-os`.
