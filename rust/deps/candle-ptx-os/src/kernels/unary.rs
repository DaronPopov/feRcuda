//! Unary operations - Zero-copy GPU execution with parallel streams
//!
//! All operations execute directly on GPU via native CUDA kernels.
//! NO CPU fallback - data never leaves the GPU.
//! Operations are dispatched across multiple streams for parallel execution.

use crate::error::{PtxCandleError, Result};
use crate::ffi;
use crate::storage::PtxStorage;
use candle_core::op::UnaryOpT;
use candle_core::{DType, Layout};
use std::ffi::c_void;

/// Apply unary operation to storage (zero-copy GPU execution)
pub fn unary<B: UnaryOpT>(storage: &PtxStorage, layout: &Layout) -> Result<PtxStorage> {
    let device = storage.device.clone();
    let dtype = storage.dtype();
    let n = layout.shape().elem_count();

    // Allocate output
    let out_size = n * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    let inp_ptr = storage.as_ptr() as *mut c_void;
    // Use next available stream for parallel execution
    let stream = device.next_stream();

    unsafe {
        match dtype {
            DType::F32 => {
                dispatch_unary_f32::<B>(inp_ptr as *mut f32, out_ptr as *mut f32, n, stream)?;
            }
            DType::F64 => {
                dispatch_unary_f64::<B>(inp_ptr as *mut f64, out_ptr as *mut f64, n, stream)?;
            }
            DType::F16 => {
                dispatch_unary_f16::<B>(inp_ptr, out_ptr, n, stream)?;
            }
            _ => {
                device.free_raw(out_ptr);
                return Err(PtxCandleError::UnsupportedDtype(dtype));
            }
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, n, dtype, device))
}

/// Dispatch F32 unary operations to native CUDA kernels
unsafe fn dispatch_unary_f32<B: UnaryOpT>(
    inp: *mut f32,
    out: *mut f32,
    n: usize,
    stream: ffi::CudaStream,
) -> Result<()> {
    match B::NAME {
        "neg" => ffi::ptx_tensor_neg_f32(inp, out, n, stream),
        "abs" => ffi::ptx_tensor_abs_f32(inp, out, n, stream),
        "exp" => ffi::ptx_tensor_exp_f32(inp, out, n, stream),
        "log" => ffi::ptx_tensor_log_f32(inp, out, n, stream),
        "sqrt" => ffi::ptx_tensor_sqrt_f32(inp, out, n, stream),
        "sin" => ffi::ptx_tensor_sin_f32(inp, out, n, stream),
        "cos" => ffi::ptx_tensor_cos_f32(inp, out, n, stream),
        "tanh" => ffi::ptx_tensor_tanh_f32(inp, out, n, stream),
        "ceil" => ffi::ptx_tensor_ceil_f32(inp, out, n, stream),
        "floor" => ffi::ptx_tensor_floor_f32(inp, out, n, stream),
        "round" => ffi::ptx_tensor_round_f32(inp, out, n, stream),
        "sqr" => ffi::ptx_tensor_sqr_f32(inp, out, n, stream),
        "recip" => ffi::ptx_tensor_recip_f32(inp, out, n, stream),
        "relu" => ffi::ptx_tensor_relu_f32(inp, out, n, stream),
        "gelu" => ffi::ptx_tensor_gelu_f32(inp, out, n, stream),
        "gelu_erf" => ffi::ptx_tensor_gelu_f32(inp, out, n, stream), // Same impl
        "sigmoid" => ffi::ptx_tensor_sigmoid_f32(inp, out, n, stream),
        "silu" => ffi::ptx_tensor_silu_f32(inp, out, n, stream),
        "selu" => ffi::ptx_tensor_selu_f32(inp, out, n, stream),
        "softplus" => ffi::ptx_tensor_softplus_f32(inp, out, n, stream),
        "mish" => ffi::ptx_tensor_mish_f32(inp, out, n, stream),
        op => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "unary op '{}' not implemented for F32",
                op
            )))
        }
    }
    Ok(())
}

/// Dispatch F64 unary operations to native CUDA kernels
unsafe fn dispatch_unary_f64<B: UnaryOpT>(
    inp: *mut f64,
    out: *mut f64,
    n: usize,
    stream: ffi::CudaStream,
) -> Result<()> {
    match B::NAME {
        "neg" => ffi::ptx_tensor_neg_f64(inp, out, n, stream),
        "exp" => ffi::ptx_tensor_exp_f64(inp, out, n, stream),
        "log" => ffi::ptx_tensor_log_f64(inp, out, n, stream),
        "sqrt" => ffi::ptx_tensor_sqrt_f64(inp, out, n, stream),
        "tanh" => ffi::ptx_tensor_tanh_f64(inp, out, n, stream),
        "relu" => ffi::ptx_tensor_relu_f64(inp, out, n, stream),
        "gelu" | "gelu_erf" => ffi::ptx_tensor_gelu_f64(inp, out, n, stream),
        "sigmoid" => ffi::ptx_tensor_sigmoid_f64(inp, out, n, stream),
        op => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "unary op '{}' not implemented for F64",
                op
            )))
        }
    }
    Ok(())
}

/// Dispatch F16 unary operations to native CUDA kernels
unsafe fn dispatch_unary_f16<B: UnaryOpT>(
    inp: *mut c_void,
    out: *mut c_void,
    n: usize,
    stream: ffi::CudaStream,
) -> Result<()> {
    match B::NAME {
        "relu" => ffi::ptx_tensor_relu_f16(inp, out, n, stream),
        "gelu" | "gelu_erf" => ffi::ptx_tensor_gelu_f16(inp, out, n, stream),
        "sigmoid" => ffi::ptx_tensor_sigmoid_f16(inp, out, n, stream),
        op => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "unary op '{}' not implemented for F16",
                op
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {}
