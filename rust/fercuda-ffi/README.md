# fercuda-ffi

Minimal safe Rust wrapper over `libfercuda_capi.so`.

## Link setup

By default `build.rs` links from:
- `$HOME/.local/lib`

Override with:
- `FERCUDA_LIB_DIR=/path/to/lib`

At runtime, ensure the loader can find `libfercuda_capi.so`, for example:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH}
```

## Example

```rust
use fercuda_ffi::{BufferDesc, LayerNormRequest, MatmulRequest, Session} ; fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sess = Session::new(0, None)? ; let a = sess.alloc_buffer(BufferDesc::f32_2d(2, 2))? ; let b = sess.alloc_buffer(BufferDesc::f32_2d(2, 2))? ; let out = sess.alloc_buffer(BufferDesc::f32_2d(2, 2))? ; sess.upload_f32(a, &[1.0, 2.0, 3.0, 4.0])? ; sess.upload_f32(b, &[5.0, 6.0, 7.0, 8.0])? ; let job = sess.submit_matmul(MatmulRequest { a, b, out })? ; sess.job_wait(job)? ; let mut y = [0.0f32 ; 4] ; sess.download_f32(out, &mut y)? ; let x = sess.alloc_buffer(BufferDesc::f32_1d(4))? ; let ln_out = sess.alloc_buffer(BufferDesc::f32_1d(4))? ; sess.upload_f32(x, &[1.0, 2.0, 3.0, 4.0])? ; let ln_job = sess.submit_layer_norm(LayerNormRequest { x, out: ln_out, eps: 1e-6 })? ; sess.job_wait(ln_job)? ; Ok(())
}
```

## Adapter Backends

- Shared interface:
  - `AdapterExecutionBackend` (see `src/adapter_backend.rs`)
  - backend slot-in checklist:
    - `BACKEND_SLOT_IN_PATTERN`

- `candle` feature:
  - `CandleSessionAdapter` for `Tensor -> feR-os -> Tensor` execution
  - ops: `matmul`, `layer_norm`
  - example: `cargo run --example candle_regime_bridge --features candle`

- `cudarc` feature:
  - `CudarcSessionAdapter` for `CudaSlice<f32> -> feR-os -> CudaSlice<f32>` execution
  - ops: `matmul`, `layer_norm`
  - example: `cargo run --example cudarc_regime_bridge --features cudarc`

## Adding A New Backend

Implement the same pattern used by `candle_adapter` and `cudarc_adapter`:

1. Add `src/<backend>_adapter.rs`
2. Define backend error + tensor handle type
3. Add shape validators for `matmul`/`layer_norm`
4. Implement `AdapterExecutionBackend` for `<Backend>SessionAdapter<'a>`
5. Add feature-gated module/exports in `src/lib.rs`
6. Add `examples/<backend>_regime_bridge.rs`
7. Add feature-gated unit tests (validation + trait impl)
8. Document feature and usage in this README
