//! Bridge between KernelAccelerator and cudarc-ptx
//!
//! This module provides integration between PTX-OS streams and cudarc-ptx,
//! enabling candle CUDA operations to use pre-warmed PTX-OS streams.

use crate::accelerator::KernelAccelerator;
use crate::error::{PtxCandleError, Result};
use cudarc_ptx::driver::{CudaContext, CudaStream, PtxStreamManager, PtxContextExt};
use std::sync::Arc;

/// Wrapper that combines cudarc-ptx CudaContext with PTX-OS stream management.
///
/// This allows using cudarc-ptx's CUDA abstractions with PTX-OS pre-warmed streams.
pub struct PtxCudaBridge {
    ctx: Arc<CudaContext>,
    stream_manager: PtxStreamManager,
    accelerator: &'static KernelAccelerator,
}

impl PtxCudaBridge {
    /// Create a new bridge from an existing KernelAccelerator.
    ///
    /// This creates a cudarc-ptx CudaContext and wraps the PTX-OS streams
    /// so they can be used with cudarc-ptx APIs.
    pub fn new(accelerator: &'static KernelAccelerator) -> Result<Self> {
        let device_id = accelerator.device_id();

        // Create cudarc-ptx context
        let ctx = CudaContext::new(device_id as usize)
            .map_err(|e| PtxCandleError::Cuda {
                message: format!("Failed to create CudaContext: {:?}", e),
            })?;

        // Create stream manager wrapping PTX-OS streams
        let runtime_ptr = accelerator.runtime_ptr() as *mut std::ffi::c_void;
        let stream_manager = unsafe {
            ctx.ptx_stream_manager(runtime_ptr, 16)
                .map_err(|e| PtxCandleError::Cuda {
                    message: format!("Failed to create PtxStreamManager: {:?}", e),
                })?
        };

        Ok(Self {
            ctx,
            stream_manager,
            accelerator,
        })
    }

    /// Create from global accelerator instance.
    pub fn from_global(device_id: i32) -> Result<Self> {
        let accelerator = KernelAccelerator::global(device_id)?;
        Self::new(accelerator)
    }

    /// Get the cudarc-ptx CudaContext.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get the next stream (round-robin across 16 pre-warmed streams).
    pub fn next_stream(&self) -> Arc<CudaStream> {
        self.stream_manager.next_stream()
    }

    /// Get a high-priority stream.
    pub fn priority_stream(&self) -> Arc<CudaStream> {
        self.stream_manager.priority_stream()
    }

    /// Get a specific stream by index.
    pub fn stream(&self, idx: usize) -> Arc<CudaStream> {
        self.stream_manager.stream(idx)
    }

    /// Get all streams.
    pub fn all_streams(&self) -> &[Arc<CudaStream>] {
        self.stream_manager.all_streams()
    }

    /// Synchronize all streams.
    pub fn sync_all(&self) -> Result<()> {
        self.stream_manager.sync_all()
            .map_err(|e| PtxCandleError::Cuda {
                message: format!("Failed to sync streams: {:?}", e),
            })
    }

    /// Get the underlying KernelAccelerator.
    pub fn accelerator(&self) -> &'static KernelAccelerator {
        self.accelerator
    }

    /// Get accelerator statistics.
    pub fn stats(&self) -> crate::accelerator::AcceleratorStats {
        self.accelerator.stats()
    }

    /// Number of streams available.
    pub fn num_streams(&self) -> usize {
        self.stream_manager.num_streams()
    }
}

/// Extension trait for using PtxCudaBridge with kernel launches.
pub trait PtxLaunchExt {
    /// Execute a kernel on the next available PTX-OS stream.
    fn launch_on_ptx<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Arc<CudaStream>) -> std::result::Result<R, cudarc_ptx::driver::DriverError>;

    /// Execute a kernel on the priority stream.
    fn launch_priority<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Arc<CudaStream>) -> std::result::Result<R, cudarc_ptx::driver::DriverError>;
}

impl PtxLaunchExt for PtxCudaBridge {
    fn launch_on_ptx<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Arc<CudaStream>) -> std::result::Result<R, cudarc_ptx::driver::DriverError>,
    {
        let stream = self.next_stream();
        f(&stream).map_err(|e| e.into())
    }

    fn launch_priority<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Arc<CudaStream>) -> std::result::Result<R, cudarc_ptx::driver::DriverError>,
    {
        let stream = self.priority_stream();
        f(&stream).map_err(|e| e.into())
    }
}

/// Global bridge instance for convenience.
static BRIDGE: std::sync::OnceLock<PtxCudaBridge> = std::sync::OnceLock::new();

/// Get or create the global PtxCudaBridge instance.
pub fn global_bridge(device_id: i32) -> Result<&'static PtxCudaBridge> {
    if let Some(bridge) = BRIDGE.get() {
        return Ok(bridge);
    }

    let bridge = PtxCudaBridge::from_global(device_id)?;
    let _ = BRIDGE.set(bridge);
    Ok(BRIDGE.get().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU"]
    fn test_bridge_creation() {
        let bridge = PtxCudaBridge::from_global(0).unwrap();
        assert_eq!(bridge.num_streams(), 16);
    }
}
