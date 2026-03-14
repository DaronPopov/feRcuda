//! PtxTensor - Seamless Candle Tensor integration with PTX-OS
//!
//! This module provides a drop-in tensor API that uses PTX-OS's O(1) TLSF
//! allocator under the hood while maintaining full compatibility with
//! Candle's Tensor operations.
//!
//! # Example
//!
//! ```ignore
//! use candle_ptx_os::{PtxDevice, PtxTensor};
//!
//! let device = PtxDevice::new(0)?;
//!
//! // Create tensors using familiar Candle-like API
//! let a = PtxTensor::randn(&device, (64, 128), 0.0, 1.0)?;
//! let b = PtxTensor::randn(&device, (128, 256), 0.0, 1.0)?;
//!
//! // Operations work seamlessly
//! let c = a.matmul(&b)?;
//! ```

use crate::device::PtxDevice;
use crate::error::{PtxCandleError, Result};
use crate::storage::PtxStorage;
use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::{DType, Device, Shape, Tensor, WithDType};
use std::ops::{Add, Div, Mul, Sub};

/// A tensor backed by PTX-OS's O(1) TLSF allocator.
///
/// `PtxTensor` wraps Candle's `Tensor` type but ensures all GPU memory
/// allocations go through PTX-OS's fast TLSF allocator instead of cudaMalloc.
///
/// All standard tensor operations (matmul, add, etc.) are supported through
/// automatic conversion to Candle tensors for computation, then back to
/// PtxTensor for storage.
#[derive(Clone, Debug)]
pub struct PtxTensor {
    storage: PtxStorage,
    shape: Shape,
    device: PtxDevice,
}

impl PtxTensor {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor filled with zeros
    pub fn zeros<S: Into<Shape>>(device: &PtxDevice, shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = device.zeros_impl(&shape, dtype)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor filled with ones
    pub fn ones<S: Into<Shape>>(device: &PtxDevice, shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = device.ones_impl(&shape, dtype)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor filled with a constant value
    pub fn full<S: Into<Shape>, T: WithDType>(
        device: &PtxDevice,
        shape: S,
        value: T,
    ) -> Result<Self> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let data = vec![value; elem_count];
        Self::from_slice(device, &data, shape)
    }

    /// Create a tensor with random uniform values in [min, max)
    pub fn rand<S: Into<Shape>>(
        device: &PtxDevice,
        shape: S,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.rand_uniform(&shape, dtype, min, max)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor with random normal (Gaussian) values
    pub fn randn<S: Into<Shape>>(
        device: &PtxDevice,
        shape: S,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.rand_normal(&shape, DType::F32, mean, std)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor with random normal values of specific dtype
    pub fn randn_dtype<S: Into<Shape>>(
        device: &PtxDevice,
        shape: S,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.rand_normal(&shape, dtype, mean, std)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor from a slice of data
    pub fn from_slice<T: WithDType, S: Into<Shape>>(
        device: &PtxDevice,
        data: &[T],
        shape: S,
    ) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(PtxCandleError::ShapeMismatch {
                expected: shape.elem_count(),
                got: data.len(),
            });
        }
        let storage = device.storage_from_slice(data)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create a tensor from a Vec, consuming the data
    pub fn from_vec<T: WithDType, S: Into<Shape>>(
        device: &PtxDevice,
        data: Vec<T>,
        shape: S,
    ) -> Result<Self> {
        Self::from_slice(device, &data, shape)
    }

    /// Create an uninitialized tensor (for advanced use)
    ///
    /// # Safety
    /// Contents are undefined until written to.
    pub unsafe fn uninit<S: Into<Shape>>(
        device: &PtxDevice,
        shape: S,
        dtype: DType,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.alloc_uninit(&shape, dtype)?;
        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create an identity matrix
    pub fn eye(device: &PtxDevice, n: usize, dtype: DType) -> Result<Self> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        match dtype {
            DType::F32 => Self::from_slice(device, &data, (n, n)),
            DType::F64 => {
                let data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                Self::from_slice(device, &data, (n, n))
            }
            _ => Err(PtxCandleError::UnsupportedDtype(dtype)),
        }
    }

    /// Create a tensor with values from start to end (exclusive)
    pub fn arange(device: &PtxDevice, start: f32, end: f32, dtype: DType) -> Result<Self> {
        let len = (end - start) as usize;
        let data: Vec<f32> = (0..len).map(|i| start + i as f32).collect();

        match dtype {
            DType::F32 => Self::from_slice(device, &data, len),
            DType::F64 => {
                let data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                Self::from_slice(device, &data, len)
            }
            DType::I64 => {
                let data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Self::from_slice(device, &data, len)
            }
            _ => Err(PtxCandleError::UnsupportedDtype(dtype)),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the tensor's shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the tensor's dimensions
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the tensor's rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.dims().len()
    }

    /// Get the total number of elements
    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get the device
    pub fn device(&self) -> &PtxDevice {
        &self.device
    }

    /// Get the underlying storage
    pub fn storage(&self) -> &PtxStorage {
        &self.storage
    }

    // ========================================================================
    // Candle Interop
    // ========================================================================

    /// Convert to a Candle Tensor on CPU for operations
    ///
    /// This copies data to CPU, performs the operation there, then
    /// optionally copies back. For GPU-native operations, use the
    /// direct methods on PtxTensor instead.
    pub fn to_candle_cpu(&self) -> Result<Tensor> {
        let cpu_storage = self.storage.to_cpu_storage()?;

        // Create tensor from the CpuStorage data
        let tensor = match cpu_storage {
            candle_core::CpuStorage::F32(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::F64(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::F16(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::BF16(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::U8(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::U32(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
            candle_core::CpuStorage::I64(data) => {
                Tensor::from_vec(data, self.shape.clone(), &Device::Cpu)?
            }
        };
        Ok(tensor)
    }

    /// Create from a Candle Tensor (copies data if not on CPU)
    pub fn from_candle(device: &PtxDevice, tensor: &Tensor) -> Result<Self> {
        // Ensure tensor is on CPU
        let cpu_tensor = if tensor.device().is_cpu() {
            tensor.clone()
        } else {
            tensor.to_device(&Device::Cpu)?
        };

        let shape = cpu_tensor.shape().clone();
        let dtype = cpu_tensor.dtype();

        // Extract data and create PtxStorage
        let storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::F64 => {
                let data: Vec<f64> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::F16 => {
                let data: Vec<half::f16> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::U8 => {
                let data: Vec<u8> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::U32 => {
                let data: Vec<u32> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
            DType::I64 => {
                let data: Vec<i64> = cpu_tensor.flatten_all()?.to_vec1()?;
                device.storage_from_slice(&data)?
            }
        };

        Ok(Self {
            storage,
            shape,
            device: device.clone(),
        })
    }

    /// Create from a Candle CPU tensor (alias for backwards compatibility)
    pub fn from_candle_cpu(device: &PtxDevice, tensor: &Tensor) -> Result<Self> {
        Self::from_candle(device, tensor)
    }

    /// Copy tensor data to CPU as a Vec
    pub fn to_vec<T: WithDType + Clone>(&self) -> Result<Vec<T>> {
        self.storage.to_vec()
    }

    /// Copy tensor data to CPU as a 1D Vec (flattened)
    pub fn to_vec1<T: WithDType + Clone>(&self) -> Result<Vec<T>> {
        self.to_vec()
    }

    /// Copy tensor data to CPU as a 2D Vec
    pub fn to_vec2<T: WithDType + Clone>(&self) -> Result<Vec<Vec<T>>> {
        let dims = self.dims();
        if dims.len() != 2 {
            return Err(PtxCandleError::InvalidArgument(
                format!("Expected 2D tensor, got {}D", dims.len())
            ));
        }
        let flat: Vec<T> = self.to_vec()?;
        let rows = dims[0];
        let cols = dims[1];
        let result: Vec<Vec<T>> = (0..rows)
            .map(|i| flat[i * cols..(i + 1) * cols].to_vec())
            .collect();
        Ok(result)
    }

    // ========================================================================
    // Shape Operations
    // ========================================================================

    /// Reshape the tensor
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let new_shape = shape.into();
        if new_shape.elem_count() != self.elem_count() {
            return Err(PtxCandleError::ShapeMismatch {
                expected: self.elem_count(),
                got: new_shape.elem_count(),
            });
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            device: self.device.clone(),
        })
    }

    /// Transpose the tensor (swap last two dimensions)
    pub fn t(&self) -> Result<Self> {
        let dims = self.dims();
        if dims.len() < 2 {
            return Err(PtxCandleError::InvalidArgument(
                "Cannot transpose tensor with less than 2 dimensions".to_string()
            ));
        }

        let mut new_dims = dims.to_vec();
        let len = new_dims.len();
        new_dims.swap(len - 2, len - 1);

        // For actual transpose, we need to perform the operation via CPU
        let cpu_tensor = self.to_candle_cpu()?;
        let transposed = cpu_tensor.t()?;
        Self::from_candle_cpu(&self.device, &transposed)
    }

    /// Flatten the tensor to 1D
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(self.elem_count())
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() || dims[dim] != 1 {
            return Err(PtxCandleError::InvalidArgument(
                format!("Cannot squeeze dimension {} with size {}", dim, dims.get(dim).unwrap_or(&0))
            ));
        }

        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);
        self.reshape(new_dims)
    }

    /// Unsqueeze: add a dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim > dims.len() {
            return Err(PtxCandleError::InvalidArgument(
                format!("Cannot unsqueeze at dimension {}, tensor has {} dims", dim, dims.len())
            ));
        }

        let mut new_dims = dims.to_vec();
        new_dims.insert(dim, 1);
        self.reshape(new_dims)
    }

    // ========================================================================
    // Math Operations (via CPU fallback for now)
    // ========================================================================

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let b = other.to_candle_cpu()?;
        let c = a.matmul(&b)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let b = other.to_candle_cpu()?;
        let c = (&a + &b)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let b = other.to_candle_cpu()?;
        let c = (&a - &b)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let b = other.to_candle_cpu()?;
        let c = (&a * &b)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Element-wise division
    pub fn div(&self, other: &Self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let b = other.to_candle_cpu()?;
        let c = (&a / &b)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Add a scalar
    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = (a + scalar)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Multiply by a scalar
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = (a * scalar)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Square root
    pub fn sqrt(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.sqrt()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Exponential
    pub fn exp(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.exp()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Natural logarithm
    pub fn log(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.log()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Power
    pub fn powf(&self, exp: f64) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.powf(exp)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Sum all elements
    pub fn sum_all(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.sum_all()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Sum along a dimension
    pub fn sum(&self, dim: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.sum(dim)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Mean of all elements
    pub fn mean_all(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.mean_all()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Mean along a dimension
    pub fn mean(&self, dim: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.mean(dim)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Max of all elements
    pub fn max_all(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.max(0)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Min of all elements
    pub fn min_all(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.min(0)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    /// ReLU activation
    pub fn relu(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.relu()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// GELU activation
    pub fn gelu(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.gelu()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let neg_a = a.neg()?;
        let exp_neg = neg_a.exp()?;
        let one_plus = (exp_neg + 1.0)?;
        let c = one_plus.recip()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Softmax along a dimension: exp(x) / sum(exp(x), dim)
    pub fn softmax(&self, dim: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        // For numerical stability, subtract max before exp
        let max_val = a.max_keepdim(dim)?;
        let shifted = a.broadcast_sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum_keepdim(dim)?;
        let c = exp_vals.broadcast_div(&sum_exp)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.tanh()?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// SiLU (Swish) activation: x * sigmoid(x)
    pub fn silu(&self) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let neg_a = a.neg()?;
        let exp_neg = neg_a.exp()?;
        let one_plus = (exp_neg + 1.0)?;
        let sigmoid = one_plus.recip()?;
        let c = (&a * &sigmoid)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    // ========================================================================
    // Slicing and Indexing
    // ========================================================================

    /// Narrow the tensor along a dimension
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.narrow(dim, start, len)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Index select along a dimension
    pub fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let idx = indexes.to_candle_cpu()?;
        let c = a.index_select(&idx, dim)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    /// Gather along a dimension
    pub fn gather(&self, indexes: &Self, dim: usize) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let idx = indexes.to_candle_cpu()?;
        let c = a.gather(&idx, dim)?;
        Self::from_candle_cpu(&self.device, &c)
    }

    // ========================================================================
    // Contiguous & Type Conversion
    // ========================================================================

    /// Make tensor contiguous in memory
    pub fn contiguous(&self) -> Result<Self> {
        // PtxTensor is always contiguous since we store raw storage
        Ok(self.clone())
    }

    /// Cast to a different dtype
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        let a = self.to_candle_cpu()?;
        let c = a.to_dtype(dtype)?;
        Self::from_candle_cpu(&self.device, &c)
    }
}

// ========================================================================
// Operator Overloads
// ========================================================================

impl Add for &PtxTensor {
    type Output = Result<PtxTensor>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl Sub for &PtxTensor {
    type Output = Result<PtxTensor>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

impl Mul for &PtxTensor {
    type Output = Result<PtxTensor>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

impl Div for &PtxTensor {
    type Output = Result<PtxTensor>;

    fn div(self, other: Self) -> Self::Output {
        self.div(other)
    }
}

// ========================================================================
// Extension trait for PtxDevice
// ========================================================================

/// Extension trait adding tensor creation methods to PtxDevice
#[allow(clippy::wrong_self_convention)]
pub trait PtxDeviceExt {
    fn zeros<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Result<PtxTensor>;
    fn ones<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Result<PtxTensor>;
    fn randn<S: Into<Shape>>(&self, shape: S, mean: f64, std: f64) -> Result<PtxTensor>;
    fn rand<S: Into<Shape>>(&self, shape: S, dtype: DType, min: f64, max: f64) -> Result<PtxTensor>;
    fn from_slice<T: WithDType, S: Into<Shape>>(&self, data: &[T], shape: S) -> Result<PtxTensor>;
    fn from_vec<T: WithDType, S: Into<Shape>>(&self, data: Vec<T>, shape: S) -> Result<PtxTensor>;
}

impl PtxDeviceExt for PtxDevice {
    fn zeros<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Result<PtxTensor> {
        PtxTensor::zeros(self, shape, dtype)
    }

    fn ones<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Result<PtxTensor> {
        PtxTensor::ones(self, shape, dtype)
    }

    fn randn<S: Into<Shape>>(&self, shape: S, mean: f64, std: f64) -> Result<PtxTensor> {
        PtxTensor::randn(self, shape, mean, std)
    }

    fn rand<S: Into<Shape>>(&self, shape: S, dtype: DType, min: f64, max: f64) -> Result<PtxTensor> {
        PtxTensor::rand(self, shape, dtype, min, max)
    }

    fn from_slice<T: WithDType, S: Into<Shape>>(&self, data: &[T], shape: S) -> Result<PtxTensor> {
        PtxTensor::from_slice(self, data, shape)
    }

    fn from_vec<T: WithDType, S: Into<Shape>>(&self, data: Vec<T>, shape: S) -> Result<PtxTensor> {
        PtxTensor::from_vec(self, data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<()> {
        let device = PtxDevice::with_pool_size(0, 128 * 1024 * 1024)?;

        let zeros = PtxTensor::zeros(&device, (2, 3), DType::F32)?;
        assert_eq!(zeros.dims(), &[2, 3]);
        assert_eq!(zeros.elem_count(), 6);

        let ones = PtxTensor::ones(&device, (4, 4), DType::F32)?;
        assert_eq!(ones.dims(), &[4, 4]);

        let randn = PtxTensor::randn(&device, (10, 10), 0.0, 1.0)?;
        assert_eq!(randn.elem_count(), 100);

        Ok(())
    }

    #[test]
    fn test_device_ext() -> Result<()> {
        let device = PtxDevice::with_pool_size(0, 128 * 1024 * 1024)?;

        let t = device.zeros((3, 3), DType::F32)?;
        assert_eq!(t.dims(), &[3, 3]);

        let t = device.randn((5, 5), 0.0, 1.0)?;
        assert_eq!(t.elem_count(), 25);

        Ok(())
    }
}
