//! Reduction operations - Zero-copy GPU execution with parallel streams
//!
//! All operations execute directly on GPU via native CUDA kernels.
//! NO CPU fallback - data never leaves the GPU.
//! Operations are dispatched across multiple streams for parallel execution.

use crate::error::{PtxCandleError, Result};
use crate::ffi;
use crate::storage::PtxStorage;
use candle_core::op::ReduceOp;
use candle_core::{DType, Layout, Shape};
use std::ffi::c_void;

/// Compute reduction dimensions for [outer, reduce, inner] layout
fn compute_reduce_dims(shape: &Shape, reduce_dims: &[usize]) -> (usize, usize, usize) {
    let dims = shape.dims();
    let ndim = dims.len();

    if reduce_dims.is_empty() || ndim == 0 {
        return (1, shape.elem_count(), 1);
    }

    // For simplicity, handle single dimension reduction
    // Multi-dim reduction is decomposed into sequential single-dim reductions
    if reduce_dims.len() == 1 {
        let reduce_dim = reduce_dims[0];
        let outer: usize = dims[..reduce_dim].iter().product();
        let reduce = dims[reduce_dim];
        let inner: usize = dims[reduce_dim + 1..].iter().product();
        (outer.max(1), reduce.max(1), inner.max(1))
    } else {
        // Multi-dim: compute total reduction size
        let mut outer = 1usize;
        let mut reduce = 1usize;
        let mut inner = 1usize;
        let mut in_reduce = false;
        let mut past_reduce = false;

        for (i, &d) in dims.iter().enumerate() {
            if reduce_dims.contains(&i) {
                reduce *= d;
                in_reduce = true;
            } else if in_reduce {
                inner *= d;
                past_reduce = true;
            } else if !past_reduce {
                outer *= d;
            } else {
                inner *= d;
            }
        }

        (outer.max(1), reduce.max(1), inner.max(1))
    }
}

/// Apply reduction operation along specified dimensions (zero-copy GPU execution)
pub fn reduce(
    storage: &PtxStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> Result<PtxStorage> {
    let device = storage.device.clone();
    let dtype = storage.dtype();
    let shape = layout.shape();

    // Compute reduction layout
    let (outer, reduce_size, inner) = compute_reduce_dims(shape, reduce_dims);
    let out_len = outer * inner;

    // Allocate output
    let out_size = out_len * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    let inp_ptr = storage.as_ptr() as *mut c_void;
    // Use next available stream for parallel execution
    let stream = device.next_stream();

    unsafe {
        match dtype {
            DType::F32 => {
                dispatch_reduce_f32(
                    inp_ptr as *mut f32,
                    out_ptr as *mut f32,
                    outer,
                    reduce_size,
                    inner,
                    op,
                    stream,
                )?;
            }
            DType::F64 => {
                dispatch_reduce_f64(
                    inp_ptr as *mut f64,
                    out_ptr as *mut f64,
                    outer,
                    reduce_size,
                    inner,
                    op,
                    stream,
                )?;
            }
            _ => {
                device.free_raw(out_ptr);
                return Err(PtxCandleError::UnsupportedDtype(dtype));
            }
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, out_len, dtype, device))
}

/// Dispatch F32 reduction operations
unsafe fn dispatch_reduce_f32(
    inp: *mut f32,
    out: *mut f32,
    outer: usize,
    reduce: usize,
    inner: usize,
    op: ReduceOp,
    stream: ffi::CudaStream,
) -> Result<()> {
    match op {
        ReduceOp::Sum => ffi::ptx_tensor_reduce_sum_f32(inp, out, outer, reduce, inner, stream),
        ReduceOp::Max => ffi::ptx_tensor_reduce_max_f32(inp, out, outer, reduce, inner, stream),
        ReduceOp::Min => ffi::ptx_tensor_reduce_min_f32(inp, out, outer, reduce, inner, stream),
        ReduceOp::ArgMax | ReduceOp::ArgMin => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "reduce op {:?} not yet implemented",
                op
            )))
        }
    }
    Ok(())
}

/// Dispatch F64 reduction operations
unsafe fn dispatch_reduce_f64(
    inp: *mut f64,
    out: *mut f64,
    outer: usize,
    reduce: usize,
    inner: usize,
    op: ReduceOp,
    stream: ffi::CudaStream,
) -> Result<()> {
    match op {
        ReduceOp::Sum => ffi::ptx_tensor_reduce_sum_f64(inp, out, outer, reduce, inner, stream),
        ReduceOp::Max => ffi::ptx_tensor_reduce_max_f64(inp, out, outer, reduce, inner, stream),
        ReduceOp::Min => ffi::ptx_tensor_reduce_min_f64(inp, out, outer, reduce, inner, stream),
        ReduceOp::ArgMax | ReduceOp::ArgMin => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "reduce op {:?} not yet implemented for F64",
                op
            )))
        }
    }
    Ok(())
}

/// Softmax operation along last dimension (zero-copy GPU execution)
pub fn softmax(storage: &PtxStorage, layout: &Layout) -> Result<PtxStorage> {
    let device = storage.device.clone();
    let dtype = storage.dtype();
    let shape = layout.shape();
    let dims = shape.dims();

    if dims.is_empty() {
        return Err(PtxCandleError::UnsupportedOp(
            "softmax on scalar".to_string(),
        ));
    }

    let last_dim = dims[dims.len() - 1];
    let batch_size = shape.elem_count() / last_dim;
    let n = shape.elem_count();

    // Allocate output (same size as input)
    let out_size = n * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    let inp_ptr = storage.as_ptr() as *mut c_void;
    // Use next available stream for parallel execution
    let stream = device.next_stream();

    unsafe {
        match dtype {
            DType::F32 => {
                ffi::ptx_tensor_softmax_f32(
                    inp_ptr as *mut f32,
                    out_ptr as *mut f32,
                    batch_size,
                    last_dim,
                    stream,
                );
            }
            DType::F64 => {
                ffi::ptx_tensor_softmax_f64(
                    inp_ptr as *mut f64,
                    out_ptr as *mut f64,
                    batch_size,
                    last_dim,
                    stream,
                );
            }
            DType::F16 => {
                ffi::ptx_tensor_softmax_f16(inp_ptr, out_ptr, batch_size, last_dim, stream);
            }
            _ => {
                device.free_raw(out_ptr);
                return Err(PtxCandleError::UnsupportedDtype(dtype));
            }
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, n, dtype, device))
}

/// Log softmax operation along last dimension (zero-copy GPU execution)
pub fn log_softmax(storage: &PtxStorage, layout: &Layout) -> Result<PtxStorage> {
    let device = storage.device.clone();
    let dtype = storage.dtype();
    let shape = layout.shape();
    let dims = shape.dims();

    if dims.is_empty() {
        return Err(PtxCandleError::UnsupportedOp(
            "log_softmax on scalar".to_string(),
        ));
    }

    let last_dim = dims[dims.len() - 1];
    let batch_size = shape.elem_count() / last_dim;
    let n = shape.elem_count();

    let out_size = n * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    let inp_ptr = storage.as_ptr() as *mut c_void;
    // Use next available stream for parallel execution
    let stream = device.next_stream();

    unsafe {
        match dtype {
            DType::F32 => {
                ffi::ptx_tensor_log_softmax_f32(
                    inp_ptr as *mut f32,
                    out_ptr as *mut f32,
                    batch_size,
                    last_dim,
                    stream,
                );
            }
            _ => {
                device.free_raw(out_ptr);
                return Err(PtxCandleError::UnsupportedOp(format!(
                    "log_softmax not implemented for {:?}",
                    dtype
                )));
            }
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, n, dtype, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_reduce_dims() {
        // Shape [2, 3, 4], reduce dim 1 -> outer=2, reduce=3, inner=4
        let shape = Shape::from_dims(&[2, 3, 4]);
        let (outer, reduce, inner) = compute_reduce_dims(&shape, &[1]);
        assert_eq!(outer, 2);
        assert_eq!(reduce, 3);
        assert_eq!(inner, 4);

        // Shape [10], reduce dim 0 -> outer=1, reduce=10, inner=1
        let shape = Shape::from_dims(&[10]);
        let (outer, reduce, inner) = compute_reduce_dims(&shape, &[0]);
        assert_eq!(outer, 1);
        assert_eq!(reduce, 10);
        assert_eq!(inner, 1);
    }
}
