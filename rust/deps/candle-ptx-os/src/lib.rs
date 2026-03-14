//! Candle ML backend using PTX-OS's O(1) TLSF GPU allocator
//!
//! This crate provides a seamless tensor API that uses PTX-OS for GPU memory
//! management. The key advantage is O(1) allocation and deallocation times
//! (~130ns alloc, ~55ns free) compared to cudaMalloc's typical ~2µs latency.
//!
//! # Example
//!
//! ```ignore
//! use candle_ptx_os::{PtxDevice, PtxTensor, PtxDeviceExt};
//! use candle_core::DType;
//!
//! // Create a PTX-OS backed device with 256MB pool
//! let device = PtxDevice::with_pool_size(0, 256 * 1024 * 1024)?;
//!
//! // Create tensors using PtxTensor - all allocations use O(1) TLSF
//! let a = PtxTensor::randn(&device, (64, 128), 0.0, 1.0)?;
//! let b = PtxTensor::randn(&device, (128, 256), 0.0, 1.0)?;
//!
//! // Matrix multiplication
//! let c = a.matmul(&b)?;
//! assert_eq!(c.dims(), &[64, 256]);
//!
//! // Or use the extension trait on PtxDevice
//! let zeros = device.zeros((32, 32), DType::F32)?;
//! let random = device.randn((100,), 0.0, 1.0)?;
//!
//! // Full activation function support
//! let activated = random.relu()?.gelu()?.softmax(0)?;
//!
//! // Convert to/from Candle tensors
//! let candle_tensor = c.to_candle_cpu()?;
//! let back_to_ptx = PtxTensor::from_candle(&device, &candle_tensor)?;
//! ```
//!
//! # Architecture
//!
//! - **PtxDevice**: Implements `BackendDevice`, manages RegimeRuntimeCore and CUDA streams
//! - **PtxTensor**: High-level tensor API with all standard operations
//! - **PtxStorage**: Implements `BackendStorage`, wraps typed GPU memory slices
//! - **PtxSlice<T>**: RAII wrapper for GPU memory with automatic cleanup
//!
//! # Performance
//!
//! The main performance benefit comes from PTX-OS's TLSF allocator:
//! - Allocation: O(1) time, ~130ns average
//! - Deallocation: O(1) time, ~55ns average
//! - No CUDA driver overhead per allocation
//! - Zero fragmentation by design
//!
//! This is especially beneficial for:
//! - Dynamic tensor shapes (NLP, transformers)
//! - Many small allocations (attention heads)
//! - Real-time inference workloads
//! - Long-running services that need stable memory

pub mod accelerator;
pub mod cuda_utils;
pub mod cudarc_bridge;
pub mod device;
pub mod elastic;
pub mod error;
pub mod ffi;
pub mod graph_fusion;
pub mod kernels;
pub mod quantized;
pub mod storage;
pub mod tensor;

pub use accelerator::{KernelAccelerator, AcceleratorStats};
pub use cudarc_bridge::{PtxCudaBridge, PtxLaunchExt, global_bridge};
pub use device::PtxDevice;
pub use elastic::{ElasticPool, ElasticPoolConfig, ElasticPoolStats};
pub use error::{PtxCandleError, Result};
pub use graph_fusion::{
    BatchBuilder, FusedExecutor, FusedExecutorStats, GraphCache, GraphCapture,
    OpSignature, ParallelOps,
};
pub use quantized::{PtxQuantizedTensor, PtxQuantizedExt};
pub use storage::{PtxSlice, PtxStorage, PtxStorageSlice};
pub use tensor::{PtxTensor, PtxDeviceExt};

// Re-export ptx-os core functionality
pub use ptx_os::{self as os, RegimeRuntimeCore, PtxError};
pub use ptx_os::runtime::{PoolStats, RuntimeConfig, Priority};

pub mod prelude {
    pub use crate::accelerator::{KernelAccelerator, AcceleratorStats};
    pub use crate::cudarc_bridge::{PtxCudaBridge, PtxLaunchExt, global_bridge};
    pub use crate::device::PtxDevice;
    pub use crate::elastic::{ElasticPool, ElasticPoolConfig, ElasticPoolStats};
    pub use crate::error::{PtxCandleError, Result};
    pub use crate::graph_fusion::{BatchBuilder, FusedExecutor, ParallelOps, OpSignature};
    pub use crate::quantized::{PtxQuantizedTensor, PtxQuantizedExt};
    pub use crate::storage::{PtxSlice, PtxStorage};
    pub use crate::tensor::{PtxTensor, PtxDeviceExt};
    pub use ptx_os::{RegimeRuntimeCore, PtxError};
    pub use ptx_os::runtime::{PoolStats, RuntimeConfig};
    // Re-export cudarc-ptx for direct access
    pub use cudarc_ptx;
}
