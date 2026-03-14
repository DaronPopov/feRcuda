//! Stream Provider - Abstraction for CUDA stream sources
//!
//! This module provides a trait for abstracting where CUDA streams come from,
//! allowing cudarc to use either native CUDA streams or external stream pools
//! like PTX-OS KernelAccelerator.

use crate::driver::{result::DriverError, sys};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::CudaContext;

/// Abstraction over CUDA stream sources.
///
/// Implementations can provide streams from different sources:
/// - Single native CUDA stream (default behavior)
/// - External stream pools (PTX-OS, custom pools)
/// - Round-robin stream distribution
pub trait StreamProvider: Send + Sync {
    /// Get the next stream for kernel execution.
    ///
    /// For single-stream providers, this always returns the same stream.
    /// For pool-based providers, this may round-robin through available streams.
    fn next_stream(&self) -> sys::CUstream;

    /// Get a high-priority stream for latency-critical operations.
    ///
    /// Default implementation returns the same as `next_stream()`.
    fn priority_stream(&self) -> sys::CUstream {
        self.next_stream()
    }

    /// Get a specific stream by index.
    ///
    /// Default implementation ignores the index and returns `next_stream()`.
    fn stream(&self, _idx: usize) -> sys::CUstream {
        self.next_stream()
    }

    /// Synchronize all streams managed by this provider.
    fn sync_all(&self) -> Result<(), DriverError>;

    /// Number of streams available from this provider.
    fn num_streams(&self) -> usize;

    /// Whether this provider owns the streams (and should destroy them on drop).
    ///
    /// External providers (like PTX-OS) own their streams externally,
    /// so we shouldn't destroy them when the CudaStream wrapper drops.
    fn owns_streams(&self) -> bool {
        true
    }
}

/// Single stream provider - wraps a single CUDA stream (default behavior).
pub struct SingleStreamProvider {
    stream: sys::CUstream,
    ctx: Arc<CudaContext>,
}

// Safety: CUDA streams are thread-safe by design
unsafe impl Send for SingleStreamProvider {}
unsafe impl Sync for SingleStreamProvider {}

impl SingleStreamProvider {
    /// Create a new single stream provider.
    pub fn new(stream: sys::CUstream, ctx: Arc<CudaContext>) -> Self {
        Self { stream, ctx }
    }
}

impl StreamProvider for SingleStreamProvider {
    fn next_stream(&self) -> sys::CUstream {
        self.stream
    }

    fn sync_all(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { crate::driver::result::stream::synchronize(self.stream) }
    }

    fn num_streams(&self) -> usize {
        1
    }
}

/// External stream pool provider - wraps streams from an external source.
///
/// This is used for PTX-OS and other external stream pools. The streams
/// are not owned by this provider and won't be destroyed when dropped.
pub struct ExternalStreamPool {
    streams: Vec<sys::CUstream>,
    priority_stream: Option<sys::CUstream>,
    current_idx: AtomicUsize,
    sync_fn: Option<Box<dyn Fn() + Send + Sync>>,
}

// Safety: CUDA streams are thread-safe by design
unsafe impl Send for ExternalStreamPool {}
unsafe impl Sync for ExternalStreamPool {}

impl ExternalStreamPool {
    /// Create a new external stream pool.
    ///
    /// # Arguments
    /// * `streams` - Vector of external stream handles
    /// * `priority_stream` - Optional high-priority stream
    /// * `sync_fn` - Optional function to synchronize all streams
    pub fn new(
        streams: Vec<sys::CUstream>,
        priority_stream: Option<sys::CUstream>,
        sync_fn: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self {
            streams,
            priority_stream,
            current_idx: AtomicUsize::new(0),
            sync_fn,
        }
    }

    /// Create from raw pointers (for FFI).
    ///
    /// # Safety
    /// The caller must ensure the stream pointers are valid CUDA streams
    /// that will outlive this pool.
    pub unsafe fn from_raw_streams(
        stream_ptrs: &[*mut std::ffi::c_void],
        priority_stream: Option<*mut std::ffi::c_void>,
        sync_fn: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Self {
        let streams = stream_ptrs
            .iter()
            .map(|&ptr| ptr as sys::CUstream)
            .collect();
        Self {
            streams,
            priority_stream: priority_stream.map(|p| p as sys::CUstream),
            current_idx: AtomicUsize::new(0),
            sync_fn,
        }
    }
}

impl StreamProvider for ExternalStreamPool {
    fn next_stream(&self) -> sys::CUstream {
        if self.streams.is_empty() {
            return std::ptr::null_mut();
        }
        let idx = self.current_idx.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx]
    }

    fn priority_stream(&self) -> sys::CUstream {
        self.priority_stream.unwrap_or_else(|| self.next_stream())
    }

    fn stream(&self, idx: usize) -> sys::CUstream {
        if self.streams.is_empty() {
            return std::ptr::null_mut();
        }
        self.streams[idx % self.streams.len()]
    }

    fn sync_all(&self) -> Result<(), DriverError> {
        if let Some(ref sync_fn) = self.sync_fn {
            sync_fn();
            Ok(())
        } else {
            // Fallback: synchronize each stream individually
            for &stream in &self.streams {
                unsafe { crate::driver::result::stream::synchronize(stream)? };
            }
            Ok(())
        }
    }

    fn num_streams(&self) -> usize {
        self.streams.len()
    }

    fn owns_streams(&self) -> bool {
        false // External streams are not owned by us
    }
}

/// PTX-OS stream pool provider.
///
/// This integrates with PTX-OS KernelAccelerator to use pre-warmed streams
/// for faster kernel dispatch.
#[cfg(feature = "ptx-os")]
pub struct PtxStreamPool {
    runtime_ptr: *mut std::ffi::c_void,
    num_streams: usize,
    current_idx: AtomicUsize,
}

#[cfg(feature = "ptx-os")]
extern "C" {
    fn ptx_hook_init(runtime: *mut std::ffi::c_void, verbose: bool);
}

#[cfg(feature = "ptx-os")]
unsafe impl Send for PtxStreamPool {}
#[cfg(feature = "ptx-os")]
unsafe impl Sync for PtxStreamPool {}

#[cfg(feature = "ptx-os")]
impl PtxStreamPool {
    /// Create a new PTX-OS stream pool.
    ///
    /// # Arguments
    /// * `runtime_ptr` - Pointer to GPUHotRuntime from PTX-OS
    /// * `num_streams` - Number of streams in the pool (typically 16)
    ///
    /// # Safety
    /// The runtime_ptr must be a valid GPUHotRuntime pointer that will
    /// outlive this pool.
    pub unsafe fn new(runtime_ptr: *mut std::ffi::c_void, num_streams: usize) -> Self {
        if !runtime_ptr.is_null() {
            ptx_hook_init(runtime_ptr, false);
        }
        Self {
            runtime_ptr,
            num_streams,
            current_idx: AtomicUsize::new(0),
        }
    }
}

#[cfg(feature = "ptx-os")]
impl StreamProvider for PtxStreamPool {
    fn next_stream(&self) -> sys::CUstream {
        if self.runtime_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let idx = self.current_idx.fetch_add(1, Ordering::Relaxed) % self.num_streams;
        // Call PTX-OS to get the stream
        // This will be linked at compile time when ptx-os feature is enabled
        extern "C" {
            fn gpu_hot_get_stream(runtime: *mut std::ffi::c_void, stream_id: i32) -> *mut std::ffi::c_void;
        }
        unsafe { gpu_hot_get_stream(self.runtime_ptr, idx as i32) as sys::CUstream }
    }

    fn priority_stream(&self) -> sys::CUstream {
        if self.runtime_ptr.is_null() {
            return std::ptr::null_mut();
        }
        extern "C" {
            fn gpu_hot_get_priority_stream(runtime: *mut std::ffi::c_void, priority: i32) -> *mut std::ffi::c_void;
        }
        unsafe { gpu_hot_get_priority_stream(self.runtime_ptr, 0) as sys::CUstream }
    }

    fn stream(&self, idx: usize) -> sys::CUstream {
        if self.runtime_ptr.is_null() {
            return std::ptr::null_mut();
        }
        extern "C" {
            fn gpu_hot_get_stream(runtime: *mut std::ffi::c_void, stream_id: i32) -> *mut std::ffi::c_void;
        }
        unsafe { gpu_hot_get_stream(self.runtime_ptr, (idx % self.num_streams) as i32) as sys::CUstream }
    }

    fn sync_all(&self) -> Result<(), DriverError> {
        if self.runtime_ptr.is_null() {
            return Ok(());
        }
        extern "C" {
            fn gpu_hot_sync_all(runtime: *mut std::ffi::c_void);
        }
        unsafe { gpu_hot_sync_all(self.runtime_ptr) };
        Ok(())
    }

    fn num_streams(&self) -> usize {
        self.num_streams
    }

    fn owns_streams(&self) -> bool {
        false // PTX-OS owns the streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_pool_round_robin() {
        // Create fake stream pointers for testing
        let streams: Vec<sys::CUstream> = (1..=4).map(|i| i as sys::CUstream).collect();
        let pool = ExternalStreamPool::new(streams.clone(), None, None);

        // Verify round-robin behavior
        assert_eq!(pool.next_stream(), streams[0]);
        assert_eq!(pool.next_stream(), streams[1]);
        assert_eq!(pool.next_stream(), streams[2]);
        assert_eq!(pool.next_stream(), streams[3]);
        assert_eq!(pool.next_stream(), streams[0]); // Wraps around
    }

    #[test]
    fn test_external_pool_specific_stream() {
        let streams: Vec<sys::CUstream> = (1..=4).map(|i| i as sys::CUstream).collect();
        let pool = ExternalStreamPool::new(streams.clone(), None, None);

        assert_eq!(pool.stream(0), streams[0]);
        assert_eq!(pool.stream(2), streams[2]);
        assert_eq!(pool.stream(5), streams[1]); // Wraps: 5 % 4 = 1
    }

    #[test]
    fn test_external_pool_does_not_own_streams() {
        let streams: Vec<sys::CUstream> = (1..=4).map(|i| i as sys::CUstream).collect();
        let pool = ExternalStreamPool::new(streams, None, None);
        assert!(!pool.owns_streams());
    }
}
