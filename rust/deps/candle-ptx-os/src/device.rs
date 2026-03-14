//! PtxDevice - Candle BackendDevice implementation using PTX-OS
//!
//! Features:
//! - O(1) TLSF memory allocation (~130ns alloc, ~55ns free)
//! - CUDA Graph capture/replay (~0.5μs vs ~5μs kernel launch)
//! - Multi-stream parallel execution (16 streams)
//! - Priority-aware stream scheduling

use crate::cuda_utils;
use crate::error::{PtxCandleError, Result};
use crate::ffi::CudaStream;
use crate::graph_fusion::{FusedExecutor, FusedExecutorStats};
use crate::storage::PtxStorage;
use candle_core::backend::BackendDevice;
use candle_core::{CpuStorage, DType, DeviceLocation, Shape};
use parking_lot::Mutex;
use ptx_os::{RegimeConfig, RegimeRuntimeCore};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Number of CUDA streams for parallel execution
const NUM_STREAMS: usize = 16;
/// Graph cache capacity
const GRAPH_CACHE_SIZE: usize = 64;

/// PTX-OS backed CUDA device for Candle ML framework
///
/// Uses PTX-OS's O(1) TLSF allocator for fast GPU memory management.
/// Supports CUDA Graph fusion and multi-stream parallel execution.
#[derive(Clone)]
pub struct PtxDevice {
    pub(crate) runtime: RegimeRuntimeCore,
    pub(crate) device_id: usize,
    pub(crate) rng: Arc<Mutex<StdRng>>,
    pub(crate) executor: Arc<FusedExecutor>,
    pub(crate) _stream_counter: Arc<AtomicUsize>,
}

impl PtxDevice {
    /// Create a new PtxDevice on the specified GPU
    ///
    /// Initializes:
    /// - O(1) TLSF memory allocator
    /// - 16-stream parallel executor
    /// - 64-entry graph cache
    pub fn new(device_id: usize) -> Result<Self> {
        let runtime = RegimeRuntimeCore::new(device_id as i32)?;
        let rng = Arc::new(Mutex::new(StdRng::from_entropy()));
        let executor = Arc::new(unsafe {
            FusedExecutor::new(
                runtime.as_ptr(),
                NUM_STREAMS,
                GRAPH_CACHE_SIZE,
            )
        });

        Ok(Self {
            runtime,
            device_id,
            rng,
            executor,
            _stream_counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create with custom PTX-OS configuration
    #[deprecated(note = "Use with_regime(device_id, RegimeConfig) as the canonical OS syntax.")]
    pub fn with_config(
        device_id: usize,
        config: ptx_os::runtime::RuntimeConfig,
    ) -> Result<Self> {
        Self::with_regime(device_id, RegimeConfig::from_runtime_config(config))
    }

    /// Create with explicit OS-level regime policy.
    pub fn with_regime(device_id: usize, regime: RegimeConfig) -> Result<Self> {
        let runtime = RegimeRuntimeCore::with_regime(device_id as i32, regime)?;
        let rng = Arc::new(Mutex::new(StdRng::from_entropy()));
        let executor = Arc::new(unsafe {
            FusedExecutor::new(
                runtime.as_ptr(),
                NUM_STREAMS,
                GRAPH_CACHE_SIZE,
            )
        });

        Ok(Self {
            runtime,
            device_id,
            rng,
            executor,
            _stream_counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create with a specific pool size (in bytes)
    /// Use this to limit TLSF overhead, leaving more VRAM for compute
    pub fn with_pool_size(device_id: usize, size_bytes: usize) -> Result<Self> {
        Self::with_regime(
            device_id,
            RegimeConfig::new().pool_fixed_bytes(size_bytes),
        )
    }

    /// Create with maximum available VRAM
    pub fn max_vram(device_id: usize) -> Result<Self> {
        Self::with_regime(device_id, RegimeConfig::max_vram())
    }

    /// Get the PTX-OS runtime handle
    pub fn runtime(&self) -> &RegimeRuntimeCore {
        &self.runtime
    }

    /// Get the device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get TLSF pool statistics
    pub fn pool_stats(&self) -> ptx_os::runtime::PoolStats {
        self.runtime.pool_stats()
    }

    /// Get global GPU runtime statistics
    pub fn stats(&self) -> ptx_os::runtime::RuntimeStats {
        self.runtime.stats()
    }

    /// Allocate raw GPU memory using O(1) TLSF allocator
    ///
    /// # Safety
    /// Returns uninitialized memory - caller must initialize before use
    pub fn alloc_raw(&self, size: usize) -> Result<*mut std::ffi::c_void> {
        self.runtime
            .alloc_raw(size)
            .map_err(PtxCandleError::from)
    }

    /// Free GPU memory
    ///
    /// # Safety
    /// Pointer must have been allocated by this device's runtime
    pub unsafe fn free_raw(&self, ptr: *mut std::ffi::c_void) {
        self.runtime.free_raw(ptr);
    }

    /// Synchronize all operations on this device
    pub fn synchronize(&self) -> Result<()> {
        cuda_utils::device_synchronize()
    }

    /// Allocate and zero GPU memory
    pub fn alloc_zeros(&self, size: usize) -> Result<*mut std::ffi::c_void> {
        let ptr = self.alloc_raw(size)?;
        unsafe {
            cuda_utils::memset(ptr, 0, size)?;
        }
        Ok(ptr)
    }

    /// Calculate element count from shape
    pub fn elem_count(shape: &Shape) -> usize {
        shape.elem_count()
    }

    /// Calculate byte size for shape and dtype
    pub fn byte_size(shape: &Shape, dtype: DType) -> usize {
        shape.elem_count() * dtype.size_in_bytes()
    }

    // ========================================================================
    // Stream & Graph Fusion Methods
    // ========================================================================

    /// Get the next CUDA stream for parallel execution (round-robin)
    pub fn next_stream(&self) -> CudaStream {
        self.executor.stream_pool.next_stream()
    }

    /// Get a high-priority stream for critical operations
    pub fn priority_stream(&self) -> CudaStream {
        self.executor.stream_pool.priority_stream()
    }

    /// Get a specific stream by index
    pub fn stream(&self, idx: usize) -> CudaStream {
        self.executor.stream_pool.stream(idx)
    }

    /// Get the fused executor for advanced graph operations
    pub fn executor(&self) -> &FusedExecutor {
        &self.executor
    }

    /// Get graph execution statistics
    pub fn executor_stats(&self) -> FusedExecutorStats {
        self.executor.stats()
    }

    /// Synchronize all parallel streams
    pub fn sync_streams(&self) {
        self.executor.sync_all();
    }

    /// Number of available parallel streams
    pub fn num_streams(&self) -> usize {
        self.executor.stream_pool.num_streams()
    }

    /// Clear the graph cache (useful for memory pressure)
    pub fn clear_graph_cache(&self) {
        self.executor.graph_cache.clear();
    }
}

impl std::fmt::Debug for PtxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtxDevice")
            .field("device_id", &self.device_id)
            .field("pool_stats", &self.runtime.pool_stats())
            .finish()
    }
}

impl BackendDevice for PtxDevice {
    type Storage = PtxStorage;

    fn new(device_id: usize) -> candle_core::Result<Self> {
        PtxDevice::new(device_id).map_err(candle_core::Error::wrap)
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Cuda {
            gpu_id: self.device_id,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.device_id == other.device_id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> candle_core::Result<Self::Storage> {
        let size = Self::byte_size(shape, dtype);
        let ptr = self.alloc_zeros(size).map_err(candle_core::Error::wrap)?;

        Ok(PtxStorage::from_raw_ptr(ptr, shape.elem_count(), dtype, self.clone()))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> candle_core::Result<Self::Storage> {
        // Create ones on CPU and copy to device
        let elem_count = shape.elem_count();
        let cpu_storage = match dtype {
            DType::F32 => CpuStorage::F32(vec![1.0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![1.0f64; elem_count]),
            DType::F16 => CpuStorage::F16(vec![half::f16::from_f32(1.0); elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::from_f32(1.0); elem_count]),
            DType::U8 => CpuStorage::U8(vec![1u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![1u32; elem_count]),
            DType::I64 => CpuStorage::I64(vec![1i64; elem_count]),
        };
        self.storage_from_cpu_storage_owned(cpu_storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> candle_core::Result<Self::Storage> {
        let size = Self::byte_size(shape, dtype);
        let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;

        Ok(PtxStorage::from_raw_ptr(ptr, shape.elem_count(), dtype, self.clone()))
    }

    fn storage_from_slice<T: candle_core::WithDType>(&self, data: &[T]) -> candle_core::Result<Self::Storage> {
        let size = std::mem::size_of_val(data);
        let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;

        unsafe {
            cuda_utils::memcpy_htod(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                size,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        Ok(PtxStorage::from_raw_ptr(ptr, data.len(), T::DTYPE, self.clone()))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> candle_core::Result<Self::Storage> {
        let (ptr, len, dtype) = match storage {
            CpuStorage::F32(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::F32)
            }
            CpuStorage::F64(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::F64)
            }
            CpuStorage::F16(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::F16)
            }
            CpuStorage::BF16(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::BF16)
            }
            CpuStorage::U8(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::U8)
            }
            CpuStorage::U32(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::U32)
            }
            CpuStorage::I64(data) => {
                let size = std::mem::size_of_val(data.as_slice());
                let ptr = self.alloc_raw(size).map_err(candle_core::Error::wrap)?;
                unsafe {
                    cuda_utils::memcpy_htod(ptr, data.as_ptr() as *const _, size)
                        .map_err(candle_core::Error::wrap)?;
                }
                (ptr, data.len(), DType::I64)
            }
        };

        Ok(PtxStorage::from_raw_ptr(ptr, len, dtype, self.clone()))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> candle_core::Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> candle_core::Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut rng = self.rng.lock();

        let cpu_storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| rng.gen_range(min as f32..max as f32))
                    .collect();
                CpuStorage::F32(data)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| rng.gen_range(min..max))
                    .collect();
                CpuStorage::F64(data)
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..elem_count)
                    .map(|_| half::f16::from_f32(rng.gen_range(min as f32..max as f32)))
                    .collect();
                CpuStorage::F16(data)
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..elem_count)
                    .map(|_| half::bf16::from_f32(rng.gen_range(min as f32..max as f32)))
                    .collect();
                CpuStorage::BF16(data)
            }
            _ => {
                return Err(candle_core::Error::wrap(PtxCandleError::UnsupportedDtype(
                    dtype,
                )))
            }
        };
        drop(rng);

        self.storage_from_cpu_storage_owned(cpu_storage)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> candle_core::Result<Self::Storage> {
        use rand_distr::{Distribution, Normal};

        let elem_count = shape.elem_count();
        let mut rng = self.rng.lock();
        let normal = Normal::new(mean, std).map_err(|e| {
            candle_core::Error::wrap(PtxCandleError::InvalidArgument(format!(
                "Invalid normal distribution params: {}",
                e
            )))
        })?;

        let cpu_storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| normal.sample(&mut *rng) as f32)
                    .collect();
                CpuStorage::F32(data)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| normal.sample(&mut *rng))
                    .collect();
                CpuStorage::F64(data)
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..elem_count)
                    .map(|_| half::f16::from_f32(normal.sample(&mut *rng) as f32))
                    .collect();
                CpuStorage::F16(data)
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..elem_count)
                    .map(|_| half::bf16::from_f32(normal.sample(&mut *rng) as f32))
                    .collect();
                CpuStorage::BF16(data)
            }
            _ => {
                return Err(candle_core::Error::wrap(PtxCandleError::UnsupportedDtype(
                    dtype,
                )))
            }
        };
        drop(rng);

        self.storage_from_cpu_storage_owned(cpu_storage)
    }

    fn set_seed(&self, seed: u64) -> candle_core::Result<()> {
        let mut rng = self.rng.lock();
        *rng = StdRng::seed_from_u64(seed);
        Ok(())
    }

    fn synchronize(&self) -> candle_core::Result<()> {
        self.runtime.sync_all();
        cuda_utils::device_synchronize().map_err(candle_core::Error::wrap)
    }
}
