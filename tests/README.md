# feRcuda Test Layout

Domain-oriented test folders:

- `core/`: numeric types and determinism
- `memory/`: pool allocator, reclaim, fragmentation, memory regimes
- `scheduler/`: persistent scheduler behavior and interop
- `runtime/`: runtime session and registry behavior
- `c_api/`: C API bridge and allocator integration tests
- `integration/`: external runtime integration tests (PTX-OS / feR-os)
- `h2d/`: host-to-device control/data path tests
- `experimental/`: experimental CA algorithm tests

Guidelines:

- Keep test target names stable (e.g. `test_pool_reclaim`) to avoid script churn.
- Place new tests in the matching domain folder.
- Prefer one behavior focus per test file.
