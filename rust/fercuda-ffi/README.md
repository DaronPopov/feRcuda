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
