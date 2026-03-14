//! Safe GPU memory allocation types

use crate::error::{PtxError, Result};
use crate::runtime::RegimeRuntimeCore;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

// CUDA memcpy direction enum
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
enum CudaMemcpyKind {
    _HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    _DeviceToDevice = 3,
}

// CUDA runtime FFI for memory operations
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
}

/// Check CUDA error code and convert to Result
fn check_cuda_error(code: i32, op: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(PtxError::CudaError {
            message: format!("{} failed with error code {}", op, code),
        })
    }
}

/// A smart pointer to GPU memory, similar to Box<T> but for device memory.
///
/// Automatically frees GPU memory when dropped.
///
/// # Example
/// ```rust,ignore
/// let runtime = RegimeRuntimeCore::new(0)?;
/// let buffer: DeviceBox<[f32; 1024]> = DeviceBox::new(&runtime)?;
/// ```
pub struct DeviceBox<T: ?Sized> {
    ptr: NonNull<T>,
    runtime: RegimeRuntimeCore,
    _marker: PhantomData<T>,
}

impl<T> DeviceBox<T> {
    /// Allocate uninitialized GPU memory for type T.
    pub fn new_uninit(runtime: &RegimeRuntimeCore) -> Result<DeviceBox<std::mem::MaybeUninit<T>>> {
        let size = std::mem::size_of::<T>();
        let ptr = runtime.alloc_raw(size)?;

        Ok(DeviceBox {
            ptr: NonNull::new(ptr as *mut std::mem::MaybeUninit<T>).unwrap(),
            runtime: runtime.clone(),
            _marker: PhantomData,
        })
    }

    /// Allocate zeroed GPU memory for type T.
    pub fn new_zeroed(runtime: &RegimeRuntimeCore) -> Result<Self>
    where
        T: bytemuck::Zeroable,
    {
        let size = std::mem::size_of::<T>();
        let ptr = runtime.alloc_raw(size)?;

        // Zero the memory using cudaMemset
        let code = unsafe { cudaMemset(ptr, 0, size) };
        check_cuda_error(code, "cudaMemset")?;

        Ok(DeviceBox {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            runtime: runtime.clone(),
            _marker: PhantomData,
        })
    }

    /// Get raw pointer to device memory.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer to device memory.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T> DeviceBox<[T]> {
    /// Allocate a device slice of the given length.
    pub fn new_slice(runtime: &RegimeRuntimeCore, len: usize) -> Result<Self> {
        let size = std::mem::size_of::<T>() * len;
        let ptr = runtime.alloc_raw(size)?;

        // Create a fat pointer for the slice
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr as *mut T, len);

        Ok(DeviceBox {
            ptr: NonNull::new(slice_ptr).unwrap(),
            runtime: runtime.clone(),
            _marker: PhantomData,
        })
    }

    /// Get the length of the slice.
    pub fn len(&self) -> usize {
        // Safety: ptr points to valid slice metadata
        unsafe { (&*self.ptr.as_ptr()).len() }
    }

    /// Check if slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ?Sized> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        unsafe {
            self.runtime.free_raw(self.ptr.as_ptr() as *mut _);
        }
    }
}

// Safety: DeviceBox owns the memory and can be sent across threads
unsafe impl<T: ?Sized + Send> Send for DeviceBox<T> {}
unsafe impl<T: ?Sized + Sync> Sync for DeviceBox<T> {}

impl<T: ?Sized> std::fmt::Debug for DeviceBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBox")
            .field("ptr", &self.ptr)
            .field("device", &self.runtime.device_id())
            .finish()
    }
}

/// A borrowed view of GPU memory, similar to &[T].
pub struct DeviceSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> DeviceSlice<'a, T> {
    /// Create a device slice from a raw pointer and length.
    ///
    /// # Safety
    /// The pointer must be valid device memory for `len` elements.
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// A mutable borrowed view of GPU memory, similar to &mut [T].
pub struct DeviceSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> DeviceSliceMut<'a, T> {
    /// Create a mutable device slice from a raw pointer and length.
    ///
    /// # Safety
    /// The pointer must be valid device memory for `len` elements.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Convert to immutable slice.
    pub fn as_slice(&self) -> DeviceSlice<'_, T> {
        DeviceSlice {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

/// Extension trait for copying data to/from device memory.
pub trait DeviceMemcpy<T> {
    /// Copy data from host to device.
    fn copy_from_host(&mut self, src: &[T]) -> Result<()>;

    /// Copy data from device to host.
    fn copy_to_host(&self, dst: &mut [T]) -> Result<()>;
}

impl<T: Copy> DeviceMemcpy<T> for DeviceBox<[T]> {
    fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(PtxError::InvalidArgument {
                message: format!(
                    "Size mismatch: src={}, dst={}",
                    src.len(),
                    self.len()
                ),
            });
        }

        let size = std::mem::size_of_val(src);
        let code = unsafe {
            cudaMemcpy(
                self.ptr.as_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                size,
                CudaMemcpyKind::HostToDevice as i32,
            )
        };
        check_cuda_error(code, "cudaMemcpy HostToDevice")
    }

    fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len() {
            return Err(PtxError::InvalidArgument {
                message: format!(
                    "Size mismatch: src={}, dst={}",
                    self.len(),
                    dst.len()
                ),
            });
        }

        let size = std::mem::size_of_val(dst);
        let code = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr() as *const c_void,
                size,
                CudaMemcpyKind::DeviceToHost as i32,
            )
        };
        check_cuda_error(code, "cudaMemcpy DeviceToHost")
    }
}

/// Pinned host memory for efficient DMA transfers.
pub struct PinnedMemory<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> PinnedMemory<T> {
    /// Allocate pinned host memory.
    pub fn new(len: usize) -> Result<Self> {
        let size = std::mem::size_of::<T>() * len;
        let mut ptr: *mut c_void = std::ptr::null_mut();

        let code = unsafe { cudaMallocHost(&mut ptr, size) };
        check_cuda_error(code, "cudaMallocHost")?;

        Ok(Self {
            ptr: NonNull::new(ptr as *mut T).ok_or(PtxError::AllocationFailed { size })?,
            len,
            _marker: PhantomData,
        })
    }

    /// Get the length of the pinned memory region.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the pinned memory region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            cudaFreeHost(self.ptr.as_ptr() as *mut c_void);
        }
    }
}

impl<T> Deref for PinnedMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for PinnedMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
