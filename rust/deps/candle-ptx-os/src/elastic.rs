//! Elastic Pool - Two-tier GPU memory management
//!
//! - Tier 1 (TLSF): Mutable allocations for KV cache, activations
//! - Tier 2 (Immutable): Append-only storage for model weights
//!
//! The immutable tier dynamically polls available VRAM and expands as needed.

use crate::error::{PtxCandleError, Result};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use parking_lot::Mutex;

/// Configuration for elastic pool
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ElasticPoolConfig {
    /// Initial TLSF pool size (default: 512MB)
    pub tlsf_initial_size: usize,
    /// Maximum TLSF pool size (default: 2GB)
    pub tlsf_max_size: usize,
    /// Size of immutable region chunks (default: 256MB)
    pub immutable_chunk_size: usize,
    /// Reserved space in immutable region (default: 256MB)
    pub immutable_reserve: usize,
    /// Reserved for CUDA runtime (default: 256MB)
    pub cuda_reserve: usize,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for ElasticPoolConfig {
    fn default() -> Self {
        Self {
            tlsf_initial_size: 512 * 1024 * 1024,      // 512MB
            tlsf_max_size: 2 * 1024 * 1024 * 1024,     // 2GB
            immutable_chunk_size: 256 * 1024 * 1024,   // 256MB
            immutable_reserve: 256 * 1024 * 1024,      // 256MB
            cuda_reserve: 256 * 1024 * 1024,           // 256MB
            verbose: false,
        }
    }
}

impl ElasticPoolConfig {
    /// Create config optimized for model loading
    pub fn for_model_loading() -> Self {
        Self {
            tlsf_initial_size: 256 * 1024 * 1024,      // 256MB for KV cache
            tlsf_max_size: 1024 * 1024 * 1024,         // 1GB max
            immutable_chunk_size: 512 * 1024 * 1024,   // 512MB chunks
            immutable_reserve: 128 * 1024 * 1024,      // 128MB reserve
            cuda_reserve: 256 * 1024 * 1024,           // 256MB for CUDA
            verbose: true,
        }
    }

    /// Create config with verbose logging
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Statistics for the elastic pool
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct ElasticPoolStats {
    // TLSF (mutable) region
    pub tlsf_total: usize,
    pub tlsf_used: usize,
    pub tlsf_free: usize,

    // Immutable region
    pub immutable_total: usize,
    pub immutable_used: usize,
    pub immutable_committed: usize,
    pub immutable_blocks: u32,

    // System
    pub vram_total: usize,
    pub vram_available: usize,
    pub vram_reserved: usize,

    // Performance
    pub compression_ratio: f32,
    pub total_loads: u64,
    pub avg_load_time_us: f32,
}

// FFI declarations
#[repr(C)]
pub struct CElasticPool {
    _private: [u8; 0],
}

extern "C" {
    fn elastic_pool_create(device_id: i32, config: *const ElasticPoolConfig) -> *mut CElasticPool;
    fn elastic_pool_destroy(pool: *mut CElasticPool);

    fn elastic_pool_alloc_mutable(pool: *mut CElasticPool, size: usize) -> *mut c_void;
    fn elastic_pool_free_mutable(pool: *mut CElasticPool, ptr: *mut c_void);

    fn elastic_pool_alloc_immutable(
        pool: *mut CElasticPool,
        size: usize,
        layer_id: u32,
        tensor_id: u32,
        is_quantized: bool,
    ) -> *mut c_void;
    fn elastic_pool_load_weight(
        pool: *mut CElasticPool,
        host_data: *const c_void,
        size: usize,
        layer_id: u32,
        tensor_id: u32,
    ) -> *mut c_void;
    fn elastic_pool_get_immutable(
        pool: *mut CElasticPool,
        layer_id: u32,
        tensor_id: u32,
    ) -> *mut c_void;

    fn elastic_pool_poll_expand(pool: *mut CElasticPool) -> usize;
    fn elastic_pool_defragment(pool: *mut CElasticPool);
    fn elastic_pool_get_stats(pool: *mut CElasticPool, stats: *mut ElasticPoolStats);
    fn elastic_pool_print_map(pool: *mut CElasticPool);

    fn elastic_pool_can_grow(pool: *mut CElasticPool, additional: usize) -> bool;
    fn elastic_pool_owns_mutable(pool: *mut CElasticPool, ptr: *mut c_void) -> bool;
}

/// Inner state for ElasticPool
struct ElasticPoolInner {
    ptr: NonNull<CElasticPool>,
    device_id: i32,
}

unsafe impl Send for ElasticPoolInner {}
unsafe impl Sync for ElasticPoolInner {}

impl Drop for ElasticPoolInner {
    fn drop(&mut self) {
        unsafe {
            elastic_pool_destroy(self.ptr.as_ptr());
        }
    }
}

/// Elastic GPU Memory Pool
///
/// Two-tier memory management:
/// - TLSF tier for mutable allocations (KV cache, activations)
/// - Immutable tier for model weights (grows dynamically)
#[derive(Clone)]
pub struct ElasticPool {
    inner: Arc<Mutex<ElasticPoolInner>>,
}

impl ElasticPool {
    /// Create a new elastic pool with default configuration
    pub fn new(device_id: i32) -> Result<Self> {
        Self::with_config(device_id, ElasticPoolConfig::default())
    }

    /// Create an elastic pool optimized for model loading
    pub fn for_model_loading(device_id: i32) -> Result<Self> {
        Self::with_config(device_id, ElasticPoolConfig::for_model_loading())
    }

    /// Create with custom configuration
    pub fn with_config(device_id: i32, config: ElasticPoolConfig) -> Result<Self> {
        let ptr = unsafe { elastic_pool_create(device_id, &config) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            PtxCandleError::AllocationFailed { size: 0 }
        })?;

        Ok(Self {
            inner: Arc::new(Mutex::new(ElasticPoolInner { ptr, device_id })),
        })
    }

    /// Allocate mutable memory (via TLSF)
    ///
    /// Use for KV cache, activations, intermediate tensors.
    pub fn alloc_mutable(&self, size: usize) -> Result<*mut c_void> {
        let guard = self.inner.lock();
        let ptr = unsafe { elastic_pool_alloc_mutable(guard.ptr.as_ptr(), size) };
        if ptr.is_null() {
            Err(PtxCandleError::AllocationFailed { size })
        } else {
            Ok(ptr)
        }
    }

    /// Free mutable memory
    ///
    /// # Safety
    /// `ptr` must have been allocated by this pool's `alloc_mutable`.
    pub unsafe fn free_mutable(&self, ptr: *mut c_void) {
        let guard = self.inner.lock();
        elastic_pool_free_mutable(guard.ptr.as_ptr(), ptr);
    }

    /// Allocate immutable memory for a weight tensor
    ///
    /// The immutable region grows dynamically as needed.
    pub fn alloc_immutable(
        &self,
        size: usize,
        layer_id: u32,
        tensor_id: u32,
        is_quantized: bool,
    ) -> Result<*mut c_void> {
        let guard = self.inner.lock();
        let ptr = unsafe {
            elastic_pool_alloc_immutable(guard.ptr.as_ptr(), size, layer_id, tensor_id, is_quantized)
        };
        if ptr.is_null() {
            Err(PtxCandleError::AllocationFailed { size })
        } else {
            Ok(ptr)
        }
    }

    /// Load quantized weight data directly into immutable region
    ///
    /// This copies from host to GPU and returns a device pointer.
    pub fn load_weight(
        &self,
        data: &[u8],
        layer_id: u32,
        tensor_id: u32,
    ) -> Result<*mut c_void> {
        let guard = self.inner.lock();
        let ptr = unsafe {
            elastic_pool_load_weight(
                guard.ptr.as_ptr(),
                data.as_ptr() as *const c_void,
                data.len(),
                layer_id,
                tensor_id,
            )
        };
        if ptr.is_null() {
            Err(PtxCandleError::AllocationFailed { size: data.len() })
        } else {
            Ok(ptr)
        }
    }

    /// Get pointer to a previously loaded weight
    pub fn get_weight(&self, layer_id: u32, tensor_id: u32) -> Option<*mut c_void> {
        let guard = self.inner.lock();
        let ptr = unsafe { elastic_pool_get_immutable(guard.ptr.as_ptr(), layer_id, tensor_id) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    /// Poll available VRAM and expand immutable region if possible
    ///
    /// Returns bytes allocated (0 if no expansion possible).
    pub fn poll_and_expand(&self) -> usize {
        let guard = self.inner.lock();
        unsafe { elastic_pool_poll_expand(guard.ptr.as_ptr()) }
    }

    /// Defragment the mutable (TLSF) region
    pub fn defragment(&self) {
        let guard = self.inner.lock();
        unsafe { elastic_pool_defragment(guard.ptr.as_ptr()) }
    }

    /// Get current statistics
    pub fn stats(&self) -> ElasticPoolStats {
        let guard = self.inner.lock();
        let mut stats = ElasticPoolStats::default();
        unsafe {
            elastic_pool_get_stats(guard.ptr.as_ptr(), &mut stats);
        }
        stats
    }

    /// Print memory map to stdout
    pub fn print_memory_map(&self) {
        let guard = self.inner.lock();
        unsafe { elastic_pool_print_map(guard.ptr.as_ptr()) }
    }

    /// Check if we can grow the immutable region by the given amount
    pub fn can_grow(&self, additional: usize) -> bool {
        let guard = self.inner.lock();
        unsafe { elastic_pool_can_grow(guard.ptr.as_ptr(), additional) }
    }

    /// Check if a pointer belongs to the mutable (TLSF) region
    ///
    /// # Safety
    /// `ptr` must be a valid pointer (or null).
    pub unsafe fn owns_mutable(&self, ptr: *mut c_void) -> bool {
        let guard = self.inner.lock();
        elastic_pool_owns_mutable(guard.ptr.as_ptr(), ptr)
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.inner.lock().device_id
    }
}

impl std::fmt::Debug for ElasticPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ElasticPool")
            .field("device_id", &self.device_id())
            .field("tlsf_used_mb", &(stats.tlsf_used as f64 / 1024.0 / 1024.0))
            .field("immutable_used_mb", &(stats.immutable_used as f64 / 1024.0 / 1024.0))
            .field("immutable_blocks", &stats.immutable_blocks)
            .finish()
    }
}
