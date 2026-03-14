//! Matrix multiplication via cuBLAS

use crate::error::{PtxCandleError, Result};
use crate::storage::{PtxStorage, PtxStorageSlice};
use candle_core::{DType, Layout};
use std::ffi::c_void;

// cuBLAS FFI bindings
type CublasHandle = *mut c_void;
#[allow(dead_code)]
type CudaStream = *mut c_void;

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasOperation {
    N = 0, // No transpose
    T = 1, // Transpose
    C = 2, // Conjugate transpose
}

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut CublasHandle) -> i32;
    #[allow(dead_code)]
    fn cublasDestroy_v2(handle: CublasHandle) -> i32;
    #[allow(dead_code)]
    fn cublasSetStream_v2(handle: CublasHandle, stream: CudaStream) -> i32;

    // Single precision GEMM
    fn cublasSgemm_v2(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;

    // Double precision GEMM
    fn cublasDgemm_v2(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: *const f64,
        c: *mut f64,
        ldc: i32,
    ) -> i32;

    // Half precision GEMM (with float compute)
    fn cublasGemmEx(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute_type: i32,
        algo: i32,
    ) -> i32;

    // Batched GEMM
    fn cublasSgemmStridedBatched(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;

    fn cublasDgemmStridedBatched(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f64,
        a: *const f64,
        lda: i32,
        stride_a: i64,
        b: *const f64,
        ldb: i32,
        stride_b: i64,
        beta: *const f64,
        c: *mut f64,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;
}

// cuBLAS data types
const CUDA_R_16F: i32 = 2; // half
#[allow(dead_code)]
const CUDA_R_32F: i32 = 0; // float
#[allow(dead_code)]
const CUDA_R_64F: i32 = 1; // double
const CUDA_R_16BF: i32 = 14; // bfloat16

// cuBLAS compute types
#[allow(dead_code)]
const CUBLAS_COMPUTE_16F: i32 = 64;
const CUBLAS_COMPUTE_32F: i32 = 68;
#[allow(dead_code)]
const CUBLAS_COMPUTE_64F: i32 = 70;

// Default algorithm
const CUBLAS_GEMM_DEFAULT: i32 = -1;

// Thread-local cuBLAS handle
thread_local! {
    static CUBLAS_HANDLE: std::cell::RefCell<Option<CublasHandle>> = const { std::cell::RefCell::new(None) };
}

fn get_cublas_handle() -> Result<CublasHandle> {
    CUBLAS_HANDLE.with(|cell| {
        let mut handle_opt = cell.borrow_mut();
        if handle_opt.is_none() {
            let mut handle: CublasHandle = std::ptr::null_mut();
            let status = unsafe { cublasCreate_v2(&mut handle) };
            if status != 0 {
                return Err(PtxCandleError::CuBlas {
                    message: format!("cublasCreate failed with status {}", status),
                });
            }
            *handle_opt = Some(handle);
        }
        Ok(handle_opt.unwrap())
    })
}

fn check_cublas(status: i32, op: &str) -> Result<()> {
    if status != 0 {
        Err(PtxCandleError::CuBlas {
            message: format!("{} failed with status {}", op, status),
        })
    } else {
        Ok(())
    }
}

/// Matrix multiplication: C = A @ B
///
/// For batched matmul: (b, m, n, k) means batch_size=b, A is (m, k), B is (k, n), C is (m, n)
pub fn matmul(
    lhs: &PtxStorage,
    rhs: &PtxStorage,
    (batch, m, n, k): (usize, usize, usize, usize),
    _lhs_layout: &Layout,
    _rhs_layout: &Layout,
) -> Result<PtxStorage> {
    if lhs.dtype() != rhs.dtype() {
        return Err(PtxCandleError::UnsupportedOp(format!(
            "matmul requires same dtype, got {:?} and {:?}",
            lhs.dtype(),
            rhs.dtype()
        )));
    }

    let dtype = lhs.dtype();
    let device = lhs.device.clone();

    // Allocate output
    let out_len = batch * m * n;
    let out_size = out_len * dtype.size_in_bytes();
    let out_ptr = device.alloc_raw(out_size)?;

    let handle = get_cublas_handle()?;

    // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    // which gives us C in row-major layout
    // Alternatively, we can use transposed inputs

    match dtype {
        DType::F32 => {
            let a_ptr = match &lhs.slice {
                PtxStorageSlice::F32(s) => s.as_ptr(),
                _ => unreachable!(),
            };
            let b_ptr = match &rhs.slice {
                PtxStorageSlice::F32(s) => s.as_ptr(),
                _ => unreachable!(),
            };
            let c_ptr = out_ptr as *mut f32;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            if batch == 1 {
                // Regular GEMM
                // cuBLAS is column-major, so we compute C = B^T @ A^T which is (A @ B)^T
                // But we want row-major output, so we do: C = B @ A with swapped dimensions
                let status = unsafe {
                    cublasSgemm_v2(
                        handle,
                        CublasOperation::N as i32, // B not transposed
                        CublasOperation::N as i32, // A not transposed
                        n as i32,                  // rows of B (cols of result)
                        m as i32,                  // cols of A (rows of result)
                        k as i32,                  // shared dimension
                        &alpha,
                        b_ptr,                     // B
                        n as i32,                  // ldb
                        a_ptr,                     // A
                        k as i32,                  // lda
                        &beta,
                        c_ptr,
                        n as i32,                  // ldc
                    )
                };
                check_cublas(status, "cublasSgemm")?;
            } else {
                // Batched GEMM
                let stride_a = (m * k) as i64;
                let stride_b = (k * n) as i64;
                let stride_c = (m * n) as i64;

                let status = unsafe {
                    cublasSgemmStridedBatched(
                        handle,
                        CublasOperation::N as i32,
                        CublasOperation::N as i32,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        b_ptr,
                        n as i32,
                        stride_b,
                        a_ptr,
                        k as i32,
                        stride_a,
                        &beta,
                        c_ptr,
                        n as i32,
                        stride_c,
                        batch as i32,
                    )
                };
                check_cublas(status, "cublasSgemmStridedBatched")?;
            }
        }
        DType::F64 => {
            let a_ptr = match &lhs.slice {
                PtxStorageSlice::F64(s) => s.as_ptr(),
                _ => unreachable!(),
            };
            let b_ptr = match &rhs.slice {
                PtxStorageSlice::F64(s) => s.as_ptr(),
                _ => unreachable!(),
            };
            let c_ptr = out_ptr as *mut f64;

            let alpha: f64 = 1.0;
            let beta: f64 = 0.0;

            if batch == 1 {
                let status = unsafe {
                    cublasDgemm_v2(
                        handle,
                        CublasOperation::N as i32,
                        CublasOperation::N as i32,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        b_ptr,
                        n as i32,
                        a_ptr,
                        k as i32,
                        &beta,
                        c_ptr,
                        n as i32,
                    )
                };
                check_cublas(status, "cublasDgemm")?;
            } else {
                let stride_a = (m * k) as i64;
                let stride_b = (k * n) as i64;
                let stride_c = (m * n) as i64;

                let status = unsafe {
                    cublasDgemmStridedBatched(
                        handle,
                        CublasOperation::N as i32,
                        CublasOperation::N as i32,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        b_ptr,
                        n as i32,
                        stride_b,
                        a_ptr,
                        k as i32,
                        stride_a,
                        &beta,
                        c_ptr,
                        n as i32,
                        stride_c,
                        batch as i32,
                    )
                };
                check_cublas(status, "cublasDgemmStridedBatched")?;
            }
        }
        DType::F16 => {
            let a_ptr = match &lhs.slice {
                PtxStorageSlice::F16(s) => s.as_ptr() as *const c_void,
                _ => unreachable!(),
            };
            let b_ptr = match &rhs.slice {
                PtxStorageSlice::F16(s) => s.as_ptr() as *const c_void,
                _ => unreachable!(),
            };
            let c_ptr = out_ptr;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let status = unsafe {
                cublasGemmEx(
                    handle,
                    CublasOperation::N as i32,
                    CublasOperation::N as i32,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha as *const f32 as *const c_void,
                    b_ptr,
                    CUDA_R_16F,
                    n as i32,
                    a_ptr,
                    CUDA_R_16F,
                    k as i32,
                    &beta as *const f32 as *const c_void,
                    c_ptr,
                    CUDA_R_16F,
                    n as i32,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            };
            check_cublas(status, "cublasGemmEx F16")?;
        }
        DType::BF16 => {
            let a_ptr = match &lhs.slice {
                PtxStorageSlice::BF16(s) => s.as_ptr() as *const c_void,
                _ => unreachable!(),
            };
            let b_ptr = match &rhs.slice {
                PtxStorageSlice::BF16(s) => s.as_ptr() as *const c_void,
                _ => unreachable!(),
            };
            let c_ptr = out_ptr;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let status = unsafe {
                cublasGemmEx(
                    handle,
                    CublasOperation::N as i32,
                    CublasOperation::N as i32,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha as *const f32 as *const c_void,
                    b_ptr,
                    CUDA_R_16BF,
                    n as i32,
                    a_ptr,
                    CUDA_R_16BF,
                    k as i32,
                    &beta as *const f32 as *const c_void,
                    c_ptr,
                    CUDA_R_16BF,
                    n as i32,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            };
            check_cublas(status, "cublasGemmEx BF16")?;
        }
        _ => {
            return Err(PtxCandleError::UnsupportedDtype(dtype));
        }
    }

    Ok(PtxStorage::from_raw_ptr(out_ptr, out_len, dtype, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cublas_constants() {
        assert_eq!(CublasOperation::N as i32, 0);
        assert_eq!(CublasOperation::T as i32, 1);
    }
}
