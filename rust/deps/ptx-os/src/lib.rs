//! PTX-OS: Persistent GPU Operating System
//!
//! Safe Rust bindings for the PTX-OS GPU runtime.
//!
//! # Example
//!
//! ```rust,ignore
//! use ptx_os::{RegimeRuntimeCore, DeviceBox};
//!
//! let runtime = RegimeRuntimeCore::new(0)?;
//! let mut buffer: DeviceBox<[f32; 1024]> = runtime.alloc()?;
//! ```

pub mod ffi;
pub mod runtime;
pub mod memory;
pub mod tensor;
pub mod vfs;
pub mod vmm;
pub mod error;

pub use runtime::{
    AllocMode, Degraded, HotRuntime, PoolStats, Priority, Ready, RegimeConfig, RegimeLocked,
    RegimeRuntime, RegimeRuntimeCore, RuntimeConfig, Shutdown, Uninitialized,
};
pub use memory::{DeviceBox, DeviceSlice};
pub use tensor::Tensor;
pub use vfs::VirtualFs;
pub use vmm::VirtualMemory;
pub use error::{PtxError, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::runtime::{
        AllocMode, HotRuntime, PoolStats, RegimeConfig, RegimeRuntimeCore, RuntimeConfig,
    };
    pub use crate::runtime::{
        Degraded, Ready, RegimeLocked, RegimeRuntime, Shutdown, Uninitialized,
    };
    pub use crate::memory::{DeviceBox, DeviceSlice};
    pub use crate::tensor::Tensor;
    pub use crate::error::{PtxError, Result};
}
