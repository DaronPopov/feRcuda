//! CUDA utility functions for streams and memory transfers

use crate::error::{PtxCandleError, Result};
use ptx_os::RegimeRuntimeCore;
use std::ffi::c_void;

/// CUDA memory copy direction
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

// Raw CUDA FFI bindings for memory operations
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    #[allow(dead_code)]
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
}

/// Convert CUDA error code to Result
fn check_cuda_error(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                format!("CUDA error code: {}", code)
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(PtxCandleError::Cuda { message: msg })
    }
}

/// Copy data from host to device memory
///
/// # Safety
/// - `dst` must be a valid device pointer with at least `size` bytes allocated
/// - `src` must be a valid host pointer with at least `size` readable bytes
pub unsafe fn memcpy_htod(dst: *mut c_void, src: *const c_void, size: usize) -> Result<()> {
    let code = cudaMemcpy(dst, src, size, CudaMemcpyKind::HostToDevice as i32);
    check_cuda_error(code)
}

/// Copy data from device to host memory
///
/// # Safety
/// - `dst` must be a valid host pointer with at least `size` bytes writable
/// - `src` must be a valid device pointer with at least `size` bytes allocated
pub unsafe fn memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: usize) -> Result<()> {
    let code = cudaMemcpy(dst, src, size, CudaMemcpyKind::DeviceToHost as i32);
    check_cuda_error(code)
}

/// Copy data from device to device memory
///
/// # Safety
/// - Both pointers must be valid device pointers with at least `size` bytes
pub unsafe fn memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize) -> Result<()> {
    let code = cudaMemcpy(dst, src, size, CudaMemcpyKind::DeviceToDevice as i32);
    check_cuda_error(code)
}

/// Async copy from host to device
///
/// # Safety
/// - Same requirements as `memcpy_htod`
/// - `stream` must be a valid CUDA stream or null for default stream
pub unsafe fn memcpy_htod_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let code = cudaMemcpyAsync(dst, src, size, CudaMemcpyKind::HostToDevice as i32, stream);
    check_cuda_error(code)
}

/// Async copy from device to host
///
/// # Safety
/// - Same requirements as `memcpy_dtoh`
/// - `stream` must be a valid CUDA stream or null for default stream
pub unsafe fn memcpy_dtoh_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let code = cudaMemcpyAsync(dst, src, size, CudaMemcpyKind::DeviceToHost as i32, stream);
    check_cuda_error(code)
}

/// Set device memory to a value (byte-wise)
///
/// # Safety
/// - `ptr` must be a valid device pointer with at least `size` bytes
pub unsafe fn memset(ptr: *mut c_void, value: i32, size: usize) -> Result<()> {
    let code = cudaMemset(ptr, value, size);
    check_cuda_error(code)
}

/// Async memset
///
/// # Safety
/// - Same as `memset`
/// - `stream` must be a valid CUDA stream or null
pub unsafe fn memset_async(
    ptr: *mut c_void,
    value: i32,
    size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let code = cudaMemsetAsync(ptr, value, size, stream);
    check_cuda_error(code)
}

/// Synchronize all CUDA operations on the current device
pub fn device_synchronize() -> Result<()> {
    let code = unsafe { cudaDeviceSynchronize() };
    check_cuda_error(code)
}

/// Synchronize a specific stream
///
/// # Safety
/// - `stream` must be a valid CUDA stream or null
pub unsafe fn stream_synchronize(stream: *mut c_void) -> Result<()> {
    let code = cudaStreamSynchronize(stream);
    check_cuda_error(code)
}

/// Get the default CUDA stream from PTX-OS runtime
///
/// # Safety
/// - Runtime must be initialized
pub unsafe fn get_stream(runtime: &RegimeRuntimeCore) -> *mut c_void {
    ptx_os::ffi::gpu_hot_get_stream(runtime.as_ptr(), 0)
}

/// Get a priority stream from PTX-OS runtime
///
/// # Safety
/// - Runtime must be initialized
pub unsafe fn get_priority_stream(runtime: &RegimeRuntimeCore, priority: i32) -> *mut c_void {
    ptx_os::ffi::gpu_hot_get_priority_stream(runtime.as_ptr(), priority)
}

/// Copy typed slice from host to device
pub fn copy_slice_to_device<T: Copy>(src: &[T], dst: *mut T) -> Result<()> {
    let size = std::mem::size_of_val(src);
    unsafe { memcpy_htod(dst as *mut c_void, src.as_ptr() as *const c_void, size) }
}

/// Copy typed slice from device to host
pub fn copy_slice_from_device<T: Copy>(src: *const T, dst: &mut [T]) -> Result<()> {
    let size = std::mem::size_of_val(dst);
    unsafe { memcpy_dtoh(dst.as_mut_ptr() as *mut c_void, src as *const c_void, size) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memcpy_kind_values() {
        assert_eq!(CudaMemcpyKind::HostToHost as i32, 0);
        assert_eq!(CudaMemcpyKind::HostToDevice as i32, 1);
        assert_eq!(CudaMemcpyKind::DeviceToHost as i32, 2);
        assert_eq!(CudaMemcpyKind::DeviceToDevice as i32, 3);
    }
}
