//! GPU Tensor type for PTX-OS

use crate::error::{PtxError, Result};
use crate::runtime::RegimeRuntimeCore;
use std::ptr::NonNull;

/// Data type for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int8 = 3,
}

impl DType {
    /// Size of one element in bytes
    pub fn size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::Int32 => 4,
            DType::Int8 => 1,
        }
    }
}

/// A multi-dimensional tensor stored in GPU memory.
#[derive(Debug)]
pub struct Tensor {
    ptr: NonNull<std::ffi::c_void>,
    shape: Vec<usize>,
    dtype: DType,
    runtime: RegimeRuntimeCore,
}

impl Tensor {
    /// Create a new tensor with the given shape and dtype.
    pub fn new(runtime: &RegimeRuntimeCore, shape: &[usize], dtype: DType) -> Result<Self> {
        let num_elements: usize = shape.iter().product();
        let size_bytes = num_elements * dtype.size();

        let ptr = runtime.alloc_raw(size_bytes)?;

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            shape: shape.to_vec(),
            dtype,
            runtime: runtime.clone(),
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(runtime: &RegimeRuntimeCore, shape: &[usize], dtype: DType) -> Result<Self> {
        let tensor = Self::new(runtime, shape, dtype)?;
        // Memory is already zeroed by the allocator
        Ok(tensor)
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// Get raw pointer to data.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer to data.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.ptr.as_ptr()
    }

    /// Reshape tensor (must have same number of elements).
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(PtxError::InvalidArgument {
                message: format!(
                    "Cannot reshape tensor of {} elements to shape with {} elements",
                    self.numel(),
                    new_numel
                ),
            });
        }
        self.shape = new_shape.to_vec();
        Ok(())
    }

    /// Clone tensor (allocate new memory and copy).
    pub fn clone_tensor(&self) -> Result<Self> {
        let new_tensor = Self::new(&self.runtime, &self.shape, self.dtype)?;
        // TODO: cudaMemcpy from self to new_tensor
        Ok(new_tensor)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            self.runtime.free_raw(self.ptr.as_ptr());
        }
    }
}

// Safety: Tensor owns the GPU memory
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

/// Tensor operations trait
pub trait TensorOps {
    /// Fill tensor with a value.
    fn fill(&mut self, value: f32) -> Result<()>;

    /// Add two tensors element-wise.
    fn add(&self, other: &Self) -> Result<Tensor>;

    /// Multiply two tensors element-wise.
    fn mul(&self, other: &Self) -> Result<Tensor>;

    /// Matrix multiplication.
    fn matmul(&self, other: &Self) -> Result<Tensor>;
}

// Note: TensorOps would require CUDA kernels to implement
// These would be added as the project develops
