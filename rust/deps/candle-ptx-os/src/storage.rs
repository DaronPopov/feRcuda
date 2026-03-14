//! PtxStorage - Candle BackendStorage implementation using PTX-OS

use crate::cuda_utils;
use crate::device::PtxDevice;
use crate::error::{PtxCandleError, Result};
use crate::kernels;
use candle_core::backend::BackendStorage;
use candle_core::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use candle_core::{CpuStorage, DType, Layout};
use std::ffi::c_void;

/// Typed GPU memory slice with RAII cleanup
pub struct PtxSlice<T> {
    ptr: *mut T,
    len: usize,
    device: PtxDevice,
}

impl<T> PtxSlice<T> {
    /// Create a new slice from raw parts
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer allocated by `device`
    /// - `len` must be the correct number of elements
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, device: PtxDevice) -> Self {
        Self { ptr, len, device }
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get element count
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Get the device
    pub fn device(&self) -> &PtxDevice {
        &self.device
    }

    /// Copy to host Vec
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy + Default,
    {
        let mut result = vec![T::default(); self.len];
        cuda_utils::copy_slice_from_device(self.ptr, &mut result)?;
        Ok(result)
    }
}

impl<T> Drop for PtxSlice<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                self.device.free_raw(self.ptr as *mut c_void);
            }
        }
    }
}

// Safety: PtxSlice owns its memory and can be sent across threads
unsafe impl<T: Send> Send for PtxSlice<T> {}
unsafe impl<T: Sync> Sync for PtxSlice<T> {}

impl<T> std::fmt::Debug for PtxSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtxSlice")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("device_id", &self.device.device_id)
            .finish()
    }
}

/// Storage slice enum for different dtypes
pub enum PtxStorageSlice {
    F32(PtxSlice<f32>),
    F64(PtxSlice<f64>),
    F16(PtxSlice<half::f16>),
    BF16(PtxSlice<half::bf16>),
    U8(PtxSlice<u8>),
    U32(PtxSlice<u32>),
    I64(PtxSlice<i64>),
}

impl PtxStorageSlice {
    /// Get dtype
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
        }
    }

    /// Get element count
    pub fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::F64(s) => s.len(),
            Self::F16(s) => s.len(),
            Self::BF16(s) => s.len(),
            Self::U8(s) => s.len(),
            Self::U32(s) => s.len(),
            Self::I64(s) => s.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get raw pointer as void*
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            Self::F32(s) => s.as_ptr() as *const c_void,
            Self::F64(s) => s.as_ptr() as *const c_void,
            Self::F16(s) => s.as_ptr() as *const c_void,
            Self::BF16(s) => s.as_ptr() as *const c_void,
            Self::U8(s) => s.as_ptr() as *const c_void,
            Self::U32(s) => s.as_ptr() as *const c_void,
            Self::I64(s) => s.as_ptr() as *const c_void,
        }
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32(s) => s.size_bytes(),
            Self::F64(s) => s.size_bytes(),
            Self::F16(s) => s.size_bytes(),
            Self::BF16(s) => s.size_bytes(),
            Self::U8(s) => s.size_bytes(),
            Self::U32(s) => s.size_bytes(),
            Self::I64(s) => s.size_bytes(),
        }
    }
}

impl std::fmt::Debug for PtxStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32(s) => f.debug_tuple("F32").field(s).finish(),
            Self::F64(s) => f.debug_tuple("F64").field(s).finish(),
            Self::F16(s) => f.debug_tuple("F16").field(s).finish(),
            Self::BF16(s) => f.debug_tuple("BF16").field(s).finish(),
            Self::U8(s) => f.debug_tuple("U8").field(s).finish(),
            Self::U32(s) => f.debug_tuple("U32").field(s).finish(),
            Self::I64(s) => f.debug_tuple("I64").field(s).finish(),
        }
    }
}

/// PTX-OS backed GPU storage for Candle tensors
pub struct PtxStorage {
    pub(crate) slice: PtxStorageSlice,
    pub(crate) device: PtxDevice,
}

impl PtxStorage {
    /// Create storage from raw pointer
    pub(crate) fn from_raw_ptr(
        ptr: *mut c_void,
        len: usize,
        dtype: DType,
        device: PtxDevice,
    ) -> Self {
        let slice = match dtype {
            DType::F32 => PtxStorageSlice::F32(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut f32, len, device.clone())
            }),
            DType::F64 => PtxStorageSlice::F64(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut f64, len, device.clone())
            }),
            DType::F16 => PtxStorageSlice::F16(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut half::f16, len, device.clone())
            }),
            DType::BF16 => PtxStorageSlice::BF16(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut half::bf16, len, device.clone())
            }),
            DType::U8 => PtxStorageSlice::U8(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut u8, len, device.clone())
            }),
            DType::U32 => PtxStorageSlice::U32(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut u32, len, device.clone())
            }),
            DType::I64 => PtxStorageSlice::I64(unsafe {
                PtxSlice::from_raw_parts(ptr as *mut i64, len, device.clone())
            }),
        };

        Self { slice, device }
    }

    /// Get the underlying slice
    pub fn slice(&self) -> &PtxStorageSlice {
        &self.slice
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const c_void {
        self.slice.as_ptr()
    }

    /// Get element count
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Copy F32 tensor data to CPU as a Vec<f32>
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        match &self.slice {
            PtxStorageSlice::F32(s) => s.to_vec(),
            _ => Err(PtxCandleError::InvalidArgument(format!(
                "Expected F32 storage, got {:?}",
                self.dtype()
            ))),
        }
    }

    /// Copy F64 tensor data to CPU as a Vec<f64>
    pub fn to_vec_f64(&self) -> Result<Vec<f64>> {
        match &self.slice {
            PtxStorageSlice::F64(s) => s.to_vec(),
            _ => Err(PtxCandleError::InvalidArgument(format!(
                "Expected F64 storage, got {:?}",
                self.dtype()
            ))),
        }
    }

    /// Copy tensor data to CPU, converting to the requested type
    ///
    /// This is a generic method that dispatches to the appropriate type.
    pub fn to_vec<T: candle_core::WithDType + Clone>(&self) -> Result<Vec<T>> {
        // Convert through CpuStorage which handles all types
        let cpu_storage = self.to_cpu_storage().map_err(PtxCandleError::from)?;

        match cpu_storage {
            CpuStorage::F32(v) => {
                if T::DTYPE == DType::F32 {
                    // Safety: Same type
                    Ok(unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is F32, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::F64(v) => {
                if T::DTYPE == DType::F64 {
                    Ok(unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is F64, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::U8(v) => {
                if T::DTYPE == DType::U8 {
                    Ok(unsafe { std::mem::transmute::<Vec<u8>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is U8, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::U32(v) => {
                if T::DTYPE == DType::U32 {
                    Ok(unsafe { std::mem::transmute::<Vec<u32>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is U32, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::I64(v) => {
                if T::DTYPE == DType::I64 {
                    Ok(unsafe { std::mem::transmute::<Vec<i64>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is I64, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::F16(v) => {
                if T::DTYPE == DType::F16 {
                    Ok(unsafe { std::mem::transmute::<Vec<half::f16>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is F16, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
            CpuStorage::BF16(v) => {
                if T::DTYPE == DType::BF16 {
                    Ok(unsafe { std::mem::transmute::<Vec<half::bf16>, Vec<T>>(v) })
                } else {
                    Err(PtxCandleError::InvalidArgument(format!(
                        "Type mismatch: storage is BF16, requested {:?}",
                        T::DTYPE
                    )))
                }
            }
        }
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.slice.dtype()
    }
}

impl std::fmt::Debug for PtxStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtxStorage")
            .field("slice", &self.slice)
            .field("device_id", &self.device.device_id)
            .finish()
    }
}

impl Clone for PtxStorage {
    fn clone(&self) -> Self {
        // Deep clone - allocate new memory and copy
        let size = self.slice.size_bytes();
        let len = self.slice.len();
        let dtype = self.slice.dtype();

        let new_ptr = self.device.alloc_raw(size).expect("Clone allocation failed");
        unsafe {
            cuda_utils::memcpy_dtod(new_ptr, self.slice.as_ptr(), size)
                .expect("Clone memcpy failed");
        }

        Self::from_raw_ptr(new_ptr, len, dtype, self.device.clone())
    }
}

impl BackendStorage for PtxStorage {
    type Device = PtxDevice;

    fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn try_clone(&self, _layout: &Layout) -> candle_core::Result<Self> {
        Ok(self.clone())
    }

    fn to_cpu_storage(&self) -> candle_core::Result<CpuStorage> {
        match &self.slice {
            PtxStorageSlice::F32(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::F32(data))
            }
            PtxStorageSlice::F64(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::F64(data))
            }
            PtxStorageSlice::F16(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::F16(data))
            }
            PtxStorageSlice::BF16(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::BF16(data))
            }
            PtxStorageSlice::U8(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::U8(data))
            }
            PtxStorageSlice::U32(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::U32(data))
            }
            PtxStorageSlice::I64(s) => {
                let data = s.to_vec().map_err(candle_core::Error::wrap)?;
                Ok(CpuStorage::I64(data))
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> candle_core::Result<Self> {
        kernels::affine(self, layout, mul, add).map_err(candle_core::Error::wrap)
    }

    fn powf(&self, layout: &Layout, exp: f64) -> candle_core::Result<Self> {
        kernels::powf(self, layout, exp).map_err(candle_core::Error::wrap)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> candle_core::Result<Self> {
        kernels::elu(self, layout, alpha).map_err(candle_core::Error::wrap)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> candle_core::Result<Self> {
        kernels::unary::<B>(self, layout).map_err(candle_core::Error::wrap)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> candle_core::Result<Self> {
        kernels::binary::<B>(self, rhs, lhs_layout, rhs_layout).map_err(candle_core::Error::wrap)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> candle_core::Result<Self> {
        kernels::to_dtype(self, layout, dtype).map_err(candle_core::Error::wrap)
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> candle_core::Result<Self> {
        kernels::reduce(self, op, layout, reduce_dims).map_err(candle_core::Error::wrap)
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> candle_core::Result<Self> {
        kernels::cmp(self, op, rhs, lhs_layout, rhs_layout).map_err(candle_core::Error::wrap)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_layout: &Layout,
        f: &Self,
        f_layout: &Layout,
    ) -> candle_core::Result<Self> {
        kernels::where_cond(self, layout, t, t_layout, f, f_layout).map_err(candle_core::Error::wrap)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> candle_core::Result<Self> {
        kernels::matmul(self, rhs, (b, m, n, k), lhs_layout, rhs_layout)
            .map_err(candle_core::Error::wrap)
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> candle_core::Result<()> {
        kernels::copy_strided(self, dst, dst_offset, src_layout).map_err(candle_core::Error::wrap)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> candle_core::Result<()> {
        kernels::copy2d(
            self,
            dst,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
        .map_err(candle_core::Error::wrap)
    }

    fn index_select(
        &self,
        ids: &Self,
        layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> candle_core::Result<Self> {
        kernels::index_select(self, ids, layout, ids_layout, dim).map_err(candle_core::Error::wrap)
    }

    fn gather(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        dim: usize,
    ) -> candle_core::Result<Self> {
        kernels::gather(self, layout, ids, ids_layout, dim).map_err(candle_core::Error::wrap)
    }

    fn scatter_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> candle_core::Result<Self> {
        kernels::scatter_add(self, layout, ids, ids_layout, src, src_layout, dim)
            .map_err(candle_core::Error::wrap)
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> candle_core::Result<Self> {
        kernels::index_add(self, layout, ids, ids_layout, src, src_layout, dim)
            .map_err(candle_core::Error::wrap)
    }

    fn conv1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &candle_core::conv::ParamsConv1D,
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "conv1d".to_string(),
        )))
    }

    fn conv2d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &candle_core::conv::ParamsConv2D,
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "conv2d".to_string(),
        )))
    }

    fn conv_transpose1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &candle_core::conv::ParamsConvTranspose1D,
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "conv_transpose1d".to_string(),
        )))
    }

    fn conv_transpose2d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_layout: &Layout,
        _params: &candle_core::conv::ParamsConvTranspose2D,
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "conv_transpose2d".to_string(),
        )))
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "avg_pool2d".to_string(),
        )))
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "max_pool2d".to_string(),
        )))
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "upsample_nearest1d".to_string(),
        )))
    }

    fn upsample_nearest2d(
        &self,
        _layout: &Layout,
        _h: usize,
        _w: usize,
    ) -> candle_core::Result<Self> {
        Err(candle_core::Error::wrap(PtxCandleError::UnsupportedOp(
            "upsample_nearest2d".to_string(),
        )))
    }
}
