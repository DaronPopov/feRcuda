//! Binary operations - Zero-copy GPU execution with parallel streams
//!
//! All operations execute directly on GPU via native CUDA kernels.
//! NO CPU fallback - data never leaves the GPU.
//! Operations are dispatched across multiple streams for parallel execution.

use crate::error::{PtxCandleError, Result};
use crate::ffi;
use crate::storage::PtxStorage;
use candle_core::op::BinaryOpT;
use candle_core::{DType, Layout};
use std::ffi::c_void;

/// Apply binary operation to two storages (zero-copy GPU execution)
pub fn binary<B: BinaryOpT>(
    lhs: &PtxStorage,
    rhs: &PtxStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<PtxStorage> {
    if lhs.dtype() != rhs.dtype() {
        return Err(PtxCandleError::UnsupportedOp(format!(
            "binary {} requires same dtype, got {:?} and {:?}",
            B::NAME,
            lhs.dtype(),
            rhs.dtype()
        )));
    }

    let device = lhs.device.clone();
    let dtype = lhs.dtype();
    let lhs_len = lhs_layout.shape().elem_count();
    let rhs_len = rhs_layout.shape().elem_count();

    // Determine output size and broadcast mode
    let (out_len, broadcast_mode) = if lhs_len == rhs_len {
        (lhs_len, BroadcastMode::Same)
    } else if rhs_len == 1 {
        (lhs_len, BroadcastMode::ScalarRight)
    } else if lhs_len == 1 {
        (rhs_len, BroadcastMode::ScalarLeft)
    } else {
        return Err(PtxCandleError::UnsupportedOp(format!(
            "binary {} broadcasting not yet implemented for shapes {:?} vs {:?}",
            B::NAME,
            lhs_layout.shape(),
            rhs_layout.shape()
        )));
    };

    // Allocate output
    let out_size = out_len * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    // Get raw pointers
    let lhs_ptr = lhs.as_ptr() as *mut c_void;
    let rhs_ptr = rhs.as_ptr() as *mut c_void;

    // Use next available stream for parallel execution
    let stream = device.next_stream();

    unsafe {
        match dtype {
            DType::F32 => {
                dispatch_binary_f32::<B>(
                    lhs_ptr as *mut f32,
                    rhs_ptr as *mut f32,
                    out_ptr as *mut f32,
                    out_len,
                    broadcast_mode,
                    stream,
                )?;
            }
            DType::F64 => {
                dispatch_binary_f64::<B>(
                    lhs_ptr as *mut f64,
                    rhs_ptr as *mut f64,
                    out_ptr as *mut f64,
                    out_len,
                    broadcast_mode,
                    stream,
                )?;
            }
            DType::F16 => {
                dispatch_binary_f16::<B>(
                    lhs_ptr,
                    rhs_ptr,
                    out_ptr,
                    out_len,
                    broadcast_mode,
                    stream,
                )?;
            }
            _ => {
                // Free allocated memory before returning error
                device.free_raw(out_ptr);
                return Err(PtxCandleError::UnsupportedDtype(dtype));
            }
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, out_len, dtype, device))
}

#[derive(Debug, Clone, Copy)]
enum BroadcastMode {
    Same,
    ScalarRight,
    ScalarLeft,
}

/// Dispatch F32 binary operations to native CUDA kernels
unsafe fn dispatch_binary_f32<B: BinaryOpT>(
    lhs: *mut f32,
    rhs: *mut f32,
    out: *mut f32,
    n: usize,
    mode: BroadcastMode,
    stream: ffi::CudaStream,
) -> Result<()> {
    match B::NAME {
        "add" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_add_f32(lhs, rhs, out, n, stream),
            BroadcastMode::ScalarRight => {
                // Read scalar value and use scalar broadcast kernel
                let scalar = std::ptr::read(rhs);
                ffi::ptx_tensor_add_scalar_f32(lhs, scalar, out, n, stream);
            }
            BroadcastMode::ScalarLeft => {
                let scalar = std::ptr::read(lhs);
                ffi::ptx_tensor_add_scalar_f32(rhs, scalar, out, n, stream);
            }
        },
        "sub" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_sub_f32(lhs, rhs, out, n, stream),
            BroadcastMode::ScalarRight => {
                let scalar = std::ptr::read(rhs);
                ffi::ptx_tensor_sub_scalar_f32(lhs, scalar, out, n, stream);
            }
            BroadcastMode::ScalarLeft => {
                // a - b where a is scalar: negate b, then add scalar
                ffi::ptx_tensor_neg_f32(rhs, out, n, stream);
                let scalar = std::ptr::read(lhs);
                ffi::ptx_tensor_add_scalar_f32(out, scalar, out, n, stream);
            }
        },
        "mul" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_mul_f32(lhs, rhs, out, n, stream),
            BroadcastMode::ScalarRight => {
                let scalar = std::ptr::read(rhs);
                ffi::ptx_tensor_mul_scalar_f32(lhs, scalar, out, n, stream);
            }
            BroadcastMode::ScalarLeft => {
                let scalar = std::ptr::read(lhs);
                ffi::ptx_tensor_mul_scalar_f32(rhs, scalar, out, n, stream);
            }
        },
        "div" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_div_f32(lhs, rhs, out, n, stream),
            BroadcastMode::ScalarRight => {
                let scalar = std::ptr::read(rhs);
                ffi::ptx_tensor_div_scalar_f32(lhs, scalar, out, n, stream);
            }
            BroadcastMode::ScalarLeft => {
                // a / b where a is scalar: compute reciprocal of b, then mul by scalar
                ffi::ptx_tensor_recip_f32(rhs, out, n, stream);
                let scalar = std::ptr::read(lhs);
                ffi::ptx_tensor_mul_scalar_f32(out, scalar, out, n, stream);
            }
        },
        "maximum" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_max_f32(lhs, rhs, out, n, stream),
            _ => {
                return Err(PtxCandleError::UnsupportedOp(
                    "maximum with broadcast not yet implemented".to_string(),
                ))
            }
        },
        "minimum" => match mode {
            BroadcastMode::Same => ffi::ptx_tensor_min_f32(lhs, rhs, out, n, stream),
            _ => {
                return Err(PtxCandleError::UnsupportedOp(
                    "minimum with broadcast not yet implemented".to_string(),
                ))
            }
        },
        op => {
            return Err(PtxCandleError::UnsupportedOp(format!(
                "binary op '{}' not implemented for F32",
                op
            )))
        }
    }
    Ok(())
}

/// Dispatch F64 binary operations to native CUDA kernels
unsafe fn dispatch_binary_f64<B: BinaryOpT>(
    lhs: *mut f64,
    rhs: *mut f64,
    out: *mut f64,
    n: usize,
    mode: BroadcastMode,
    stream: ffi::CudaStream,
) -> Result<()> {
    match mode {
        BroadcastMode::Same => match B::NAME {
            "add" => ffi::ptx_tensor_add_f64(lhs, rhs, out, n, stream),
            "sub" => ffi::ptx_tensor_sub_f64(lhs, rhs, out, n, stream),
            "mul" => ffi::ptx_tensor_mul_f64(lhs, rhs, out, n, stream),
            "div" => ffi::ptx_tensor_div_f64(lhs, rhs, out, n, stream),
            op => {
                return Err(PtxCandleError::UnsupportedOp(format!(
                    "binary op '{}' not implemented for F64",
                    op
                )))
            }
        },
        _ => {
            return Err(PtxCandleError::UnsupportedOp(
                "F64 broadcast operations not yet implemented".to_string(),
            ))
        }
    }
    Ok(())
}

/// Dispatch F16 binary operations to native CUDA kernels
unsafe fn dispatch_binary_f16<B: BinaryOpT>(
    lhs: *mut c_void,
    rhs: *mut c_void,
    out: *mut c_void,
    n: usize,
    mode: BroadcastMode,
    stream: ffi::CudaStream,
) -> Result<()> {
    match mode {
        BroadcastMode::Same => match B::NAME {
            "add" => ffi::ptx_tensor_add_f16(lhs, rhs, out, n, stream),
            "sub" => ffi::ptx_tensor_sub_f16(lhs, rhs, out, n, stream),
            "mul" => ffi::ptx_tensor_mul_f16(lhs, rhs, out, n, stream),
            "div" => ffi::ptx_tensor_div_f16(lhs, rhs, out, n, stream),
            op => {
                return Err(PtxCandleError::UnsupportedOp(format!(
                    "binary op '{}' not implemented for F16",
                    op
                )))
            }
        },
        _ => {
            return Err(PtxCandleError::UnsupportedOp(
                "F16 broadcast operations not yet implemented".to_string(),
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {}
