//! PTX-OS Integration - High-level API for using PTX-OS streams with cudarc
//!
//! This module provides convenient wrappers for integrating PTX-OS KernelAccelerator
//! with cudarc's CUDA abstractions.
//!
//! # Example
//! ```ignore
//! use cudarc_ptx::driver::{CudaContext, PtxStreamManager};
//!
//! // Initialize PTX-OS runtime (from your PTX-OS code)
//! let runtime_ptr = /* ... */;
//!
//! // Create context and stream manager
//! let ctx = CudaContext::new(0)?;
//! let streams = PtxStreamManager::new(&ctx, runtime_ptr, 16)?;
//!
//! // Get streams for kernel launches - round-robin across 16 pre-warmed streams
//! let stream = streams.next_stream();
//! stream.launch_builder(&kernel).arg(&data).launch(cfg)?;
//!
//! // Sync all PTX-OS streams
//! streams.sync_all()?;
//! ```

#![cfg(feature = "ptx-os")]

use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
use crate::driver::{result::DriverError, sys, safe::{CudaContext, CudaStream}};

/// High-level PTX-OS stream manager.
///
/// Wraps PTX-OS KernelAccelerator streams for use with cudarc.
/// Provides round-robin stream distribution for parallel kernel execution.
pub struct PtxStreamManager {
    ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
    priority_stream: Option<Arc<CudaStream>>,
    runtime_ptr: *mut std::ffi::c_void,
    current_idx: AtomicUsize,
}

// Safety: The runtime_ptr is thread-safe (PTX-OS guarantees this)
unsafe impl Send for PtxStreamManager {}
unsafe impl Sync for PtxStreamManager {}

// FFI declarations for PTX-OS
extern "C" {
    fn gpu_hot_get_stream(runtime: *mut std::ffi::c_void, stream_id: i32) -> *mut std::ffi::c_void;
    fn gpu_hot_get_priority_stream(runtime: *mut std::ffi::c_void, priority: i32) -> *mut std::ffi::c_void;
    fn gpu_hot_sync_all(runtime: *mut std::ffi::c_void);
    fn ptx_hook_init(runtime: *mut std::ffi::c_void, verbose: bool);
}

impl PtxStreamManager {
    /// Create a new PTX-OS stream manager.
    ///
    /// # Arguments
    /// * `ctx` - The CUDA context (must be on the same device as PTX-OS runtime)
    /// * `runtime_ptr` - Pointer to GPUHotRuntime from PTX-OS
    /// * `num_streams` - Number of streams to use (typically 16)
    ///
    /// # Safety
    /// The runtime_ptr must be a valid GPUHotRuntime pointer that will
    /// outlive this manager.
    pub unsafe fn new(
        ctx: &Arc<CudaContext>,
        runtime_ptr: *mut std::ffi::c_void,
        num_streams: usize,
    ) -> Result<Self, DriverError> {
        if !runtime_ptr.is_null() {
            ptx_hook_init(runtime_ptr, false);
        }
        let mut streams = Vec::with_capacity(num_streams);

        for i in 0..num_streams {
            let cu_stream = gpu_hot_get_stream(runtime_ptr, i as i32) as sys::CUstream;
            let stream = ctx.wrap_external_stream(cu_stream);
            streams.push(stream);
        }

        // Get the priority stream
        let priority_cu_stream = gpu_hot_get_priority_stream(runtime_ptr, 0) as sys::CUstream;
        let priority_stream = if !priority_cu_stream.is_null() {
            Some(ctx.wrap_external_stream(priority_cu_stream))
        } else {
            None
        };

        Ok(Self {
            ctx: ctx.clone(),
            streams,
            priority_stream,
            runtime_ptr,
            current_idx: AtomicUsize::new(0),
        })
    }

    /// Get the next stream (round-robin).
    ///
    /// This distributes kernel launches across all available streams
    /// for maximum parallelism.
    pub fn next_stream(&self) -> Arc<CudaStream> {
        let idx = self.current_idx.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx].clone()
    }

    /// Get a high-priority stream for latency-critical operations.
    ///
    /// Use this for attention score computation, final logits, etc.
    pub fn priority_stream(&self) -> Arc<CudaStream> {
        self.priority_stream.clone().unwrap_or_else(|| self.next_stream())
    }

    /// Get a specific stream by index.
    pub fn stream(&self, idx: usize) -> Arc<CudaStream> {
        self.streams[idx % self.streams.len()].clone()
    }

    /// Get all streams (useful for parallel operations).
    pub fn all_streams(&self) -> &[Arc<CudaStream>] {
        &self.streams
    }

    /// Synchronize all PTX-OS streams.
    pub fn sync_all(&self) -> Result<(), DriverError> {
        unsafe { gpu_hot_sync_all(self.runtime_ptr) };
        Ok(())
    }

    /// Number of streams available.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// The underlying context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Reset the round-robin counter.
    pub fn reset_counter(&self) {
        self.current_idx.store(0, Ordering::Relaxed);
    }
}

/// Extension trait for CudaContext to easily create PTX-OS stream managers.
pub trait PtxContextExt {
    /// Create a PTX-OS stream manager for this context.
    ///
    /// # Safety
    /// The runtime_ptr must be a valid GPUHotRuntime pointer.
    unsafe fn ptx_stream_manager(
        self: &Arc<Self>,
        runtime_ptr: *mut std::ffi::c_void,
        num_streams: usize,
    ) -> Result<PtxStreamManager, DriverError>;
}

impl PtxContextExt for CudaContext {
    unsafe fn ptx_stream_manager(
        self: &Arc<Self>,
        runtime_ptr: *mut std::ffi::c_void,
        num_streams: usize,
    ) -> Result<PtxStreamManager, DriverError> {
        PtxStreamManager::new(self, runtime_ptr, num_streams)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a working PTX-OS runtime
    // They are disabled by default

    #[test]
    #[ignore = "requires PTX-OS runtime"]
    fn test_stream_manager_round_robin() {
        // This would test round-robin behavior with a real PTX-OS runtime
    }
}
