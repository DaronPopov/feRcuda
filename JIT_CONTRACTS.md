# feRcuda JIT Contracts (Generic Rust Scripting)

This document defines the first contract surface for dynamic JIT programming in feRcuda.
It is explicitly *not* a DSL design. Scripts are generic Rust programs that call host APIs.

## Goals

- Allow runtime-loaded Rust scripts to orchestrate GPU work.
- Allow scripts to compile/load external CUDA kernels at runtime (JIT).
- Keep all memory ownership and policy inside feRcuda regimes.
- Reuse existing session/job control plane instead of bypassing it.

## Non-Goals (MVP)

- No Rust kernel language or macro DSL.
- No direct script access to raw device pointers by default.
- No in-process arbitrary dynamic linking policy beyond explicit host controls.

## Layered Surface

1. Script ABI (`fer_script_*`):
   - Runtime loads a Rust-produced shared object.
   - Runtime calls one stable entrypoint.
2. JIT Kernel ABI (`fer_jit_*`):
   - Script compiles CUDA source (NVRTC backend) into a program/module.
   - Script resolves a named kernel handle.
   - Script launches via feRcuda session using buffer IDs + launch config.
3. Runtime/Regime bridge:
   - Buffer IDs resolve to active regime allocations.
   - Launch path validates read/write intents and dtype/size contracts.

## Script Model

Script artifact:
- Recommended crate type: `cdylib`.
- Required symbol:
  - `fer_script_main(const fer_script_context_t*, const char* args_json, char* err_buf, size_t err_buf_len) -> fer_status_t`

Runtime responsibilities:
- Prepare context with host callbacks and an attached `fer_session_t`.
- Invoke script entrypoint.
- Collect status/errors and optional script log output.

## JIT Program Model

`fer_jit_program_t`:
- Opaque compiled module handle (PTX/CUBIN backend-specific).
- Created from source + compile options.
- Cache-keyed by:
  - source bytes
  - compile opts
  - target device arch
  - toolkit/runtime build ID

`fer_jit_kernel_t`:
- Opaque kernel symbol handle from a program.
- Bound to a validated argument schema (`fer_jit_arg_desc_t[]`).

## Memory-Regime Contract

- Script/kernel launches must pass feRcuda buffer IDs, not arbitrary pointers.
- Runtime resolves IDs through session/regime manager at launch time.
- Runtime enforces:
  - existence and liveness of all buffers
  - expected dtype/shape/bytes
  - declared access mode (read/write/read_write)
  - optional regime constraints
- Any mismatch returns `FER_STATUS_INVALID_ARGUMENT` or `FER_STATUS_NOT_FOUND`.

## Scheduling Contract

- JIT launches are submitted through session-owned execution path.
- Launch is asynchronous and returns a job ID.
- Existing `fer_job_status/fer_job_wait` semantics remain valid.

## Safety Modes

- `FER_JIT_MODE_PERMISSIVE`:
  - compile warnings allowed
  - unresolved optional metadata tolerated
- `FER_JIT_MODE_STRICT`:
  - requires explicit arg schema and access declarations
  - rejects ambiguous scalar/ptr coercions

## Caching Contract

- In-memory cache always enabled for process lifetime.
- Optional disk cache path configurable in `fer_jit_options_t`.
- Cache invalidation is automatic on key mismatch.
- Explicit eviction API included for operational control.

## Observability Contract

- Compilation diagnostics returned in `fer_jit_compile_result_t`.
- Launch telemetry includes:
  - compile cache hit/miss
  - launch count
  - compile and launch latency
  - backend used

## MVP Sequence

1. Add `include/fercuda/jit/types.h` and `include/fercuda/jit/api.h`.
2. Add no-op/stub implementation in `src/jit` returning "not implemented".
3. Wire C API entry points in runtime C API bridge.
4. Add integration test:
   - compile tiny vector add kernel
   - launch with regime buffers
   - validate output and job completion
5. Add script loader contract test with minimal Rust `cdylib`.
