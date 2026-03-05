# feR-os Agent MCP Contracts (Draft)

This document defines the first stable control-plane design for LLM agents (Codex/GPT) to operate feR-os through MCP.

Scope:
- `ferrite-mcp` (separate repo): tool transport, auth, policy, audit.
- `feRcuda` (this repo): deterministic execution/runtime contracts.
- Adapter layer (new): strict schema bridge from MCP JSON to `fer_*` C API.

This is intentionally narrow: typed, composable tools with explicit safety boundaries.

## Goals

- Make GPU workflows agent-operable with predictable behavior.
- Keep memory and execution ownership in feR-os regimes.
- Enable external kernel workflows (Candle/other libs/custom kernels) via pointer-safe interop.
- Keep contracts machine-checkable and versioned.

## Non-Goals (MVP)

- No natural-language parsing in runtime.
- No raw unrestricted pointer execution surface.
- No unconstrained shell-like execution through MCP.

## Layer Model

1. MCP Tool Layer (`ferrite-mcp`)
- Validates JSON schema.
- Applies policy and quotas.
- Emits structured audit events.

2. Agent Adapter Layer (`agent_mcp_adapter` in feR-os packaging)
- Converts MCP request DTOs -> `fer_*` calls.
- Normalizes errors to a stable `error` object.
- Handles id mapping (`session_id`, `tensor_id`, `job_id`).

3. Runtime Execution Layer (`feRcuda`)
- Session, memory regimes, JIT, intent lowering, launches, jobs.

## Versioning

- `agent_api_version`: `"v1alpha1"` for all requests in MVP.
- Runtime version probe tool returns:
  - `agent_api_version`
  - `fercuda_abi_version`
  - `jit_intent_abi_version`
  - supported feature flags

## Tool Surface (MVP)

All tools return:
- `ok: bool`
- `request_id: string`
- either `result` or `error`

Error shape:
```json
{
  "code": "INVALID_ARGUMENT|NOT_FOUND|INTERNAL|POLICY_DENIED|RESOURCE_EXHAUSTED|TIMEOUT",
  "message": "human readable",
  "details": {}
}
```

### 1) `fer.session.create`
Creates a runtime session.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "device": 0,
  "pool": { "mutable_bytes": 67108864, "immutable_bytes": 67108864, "memory_regime": "custom_pool" }
}
```

Result:
```json
{
  "session_id": "ses_...",
  "default_memory_regime": "custom_pool"
}
```

### 2) `fer.session.destroy`
Destroys session and all owned handles.

### 3) `fer.tensor.create`
Typed tensor allocation in a regime.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "dtype": "f32",
  "shape": [1024],
  "memory_regime": "custom_pool",
  "immutable": false,
  "tag": 0
}
```

Result:
```json
{ "tensor_id": "ten_..." }
```

### 4) `fer.tensor.import_external`
Imports an external device pointer as a managed tensor handle.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "dtype": "f32",
  "shape": [1024],
  "device_ptr_u64": "0x...",
  "memory_regime": "custom_pool",
  "ownership": "external_no_free|external_with_deleter"
}
```

### 5) `fer.tensor.copy`
Directional transfer helper.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "tensor_id": "ten_...",
  "direction": "host_to_device|device_to_host",
  "dtype": "f32",
  "count": 1024,
  "host_blob_ref": "blob_..."
}
```

### 5a) `fer.blob.put`
Stores host payload in adapter-managed blob store.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "host_blob_ref": "blob_in",
  "blob_hex": "001122..."
}
```

### 5b) `fer.blob.get`
Fetches blob payload from adapter-managed store.

### 6) `fer.jit.intent.run`
High-level semantic launch (maps to `fer_jit_run_intent`).

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "intent": {
    "op": "affine_f32",
    "n": 1024,
    "alpha": 2.0,
    "beta": 1.0,
    "fusion_mask": ["relu"],
    "caps_mask": [],
    "memory_regime": "custom_pool"
  },
  "bindings": { "input": "ten_in", "output": "ten_out" }
}
```

Result:
```json
{ "job_id": "job_..." }
```

### 7) `fer.jit.kernel.compile`
Compiles user-provided CUDA kernel source.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "source": "...cuda code...",
  "language": "cuda",
  "options": { "optimization_level": 2, "strict": true }
}
```

Result:
```json
{
  "program_id": "prg_...",
  "cache": { "hit": false },
  "diagnostics": { "warnings": [], "errors": [] }
}
```

### 8) `fer.jit.kernel.launch`
Launches a kernel handle over managed tensor arguments.

Request:
```json
{
  "agent_api_version": "v1alpha1",
  "session_id": "ses_...",
  "program_id": "prg_...",
  "kernel_name": "axpy",
  "grid": [256, 1, 1],
  "block": [256, 1, 1],
  "shared_mem_bytes": 0,
  "args": [
    { "kind": "tensor", "tensor_id": "ten_x", "access": "read" },
    { "kind": "tensor", "tensor_id": "ten_y", "access": "read_write" },
    { "kind": "f32", "value": 2.0 },
    { "kind": "u32", "value": 1024 }
  ]
}
```

Result:
```json
{ "job_id": "job_..." }
```

### 9) `fer.job.wait`
Waits for completion and returns execution summary.

### 10) `fer.runtime.inspect`
Reports capabilities and live stats.

Result:
```json
{
  "agent_api_version": "v1alpha1",
  "supports": {
    "jit_compile": true,
    "jit_intent": true,
    "external_ptr_import": true,
    "session_stream_handoff": true
  }
}
```

## Policy Model (must be adapter-enforced before runtime call)

Policy object attaches to `session_id`:
- `allow_compile: bool`
- `allow_launch: bool`
- `allow_external_ptr: bool`
- `max_tensor_bytes`
- `max_total_session_bytes`
- `max_jobs_in_flight`
- `max_kernel_wall_ms`
- `allow_dtypes: []`
- `allow_regimes: []`

Any policy violation maps to `POLICY_DENIED` or `RESOURCE_EXHAUSTED`.

## Deterministic Execution Rules

- MCP tools only accept typed JSON; reject ambiguous forms.
- Tensor args to kernels must be by `tensor_id`, not arbitrary pointers.
- External pointers are allowed only through `fer.tensor.import_external`.
- All launches are async and must return `job_id`.
- No hidden implicit sync except `fer.job.wait`.

## Adapter Mapping to Existing C API

- `fer.session.create` -> `fer_session_create`
- `fer.session.destroy` -> `fer_session_destroy`
- `fer.tensor.create` -> `fer_tensor_create`
- `fer.tensor.import_external` -> `fer_tensor_attach_external` or `fer_import_external_buffer_*`
- `fer.tensor.copy` -> `fer_tensor_upload_*` / `fer_tensor_download_*`
- `fer.jit.intent.run` -> `fer_jit_run_intent` / `fer_tensor_run_affine_f32`
- `fer.jit.kernel.compile` -> `fer_jit_compile`
- `fer.jit.kernel.launch` -> `fer_jit_get_kernel` + `fer_jit_launch`
- `fer.job.wait` -> `fer_job_wait`

## Observability / Audit Fields

Every tool call should log:
- `timestamp_ms`
- `request_id`
- `session_id`
- `tool_name`
- `latency_ms`
- `status_code`
- `policy_decision`
- optional: `program_id`, `kernel_name`, `job_id`, `memory_regime`

## Proposed Repo Split

- `ferrite-mcp` repo:
  - MCP JSON schemas
  - policy enforcement
  - transport/auth/audit sinks

- `feRcuda` repo:
  - runtime APIs + typed tensor helpers + JIT/intent execution
  - adapter contract header and conformance tests

## Immediate Next Step (Implementation)

Implement `v1alpha1` adapter with only:
1. `fer.session.create`
2. `fer.tensor.create`
3. `fer.tensor.copy`
4. `fer.jit.intent.run` (affine_f32 only)
5. `fer.job.wait`
6. `fer.runtime.inspect`

Then add compile/launch tools once policy and arg-schema validation are hardened.
