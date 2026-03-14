//! Error types for PTX-OS

use thiserror::Error;

/// PTX-OS Error type
#[derive(Error, Debug)]
pub enum PtxError {
    #[error("Failed to initialize GPU runtime on device {device_id}")]
    InitializationFailed { device_id: i32 },

    #[error("Memory allocation failed: requested {size} bytes")]
    AllocationFailed { size: usize },

    #[error("Out of GPU memory")]
    OutOfMemory,

    #[error("Invalid device pointer")]
    InvalidPointer,

    #[error("CUDA error: {message}")]
    CudaError { message: String },

    #[error("Graph capture failed: {message}")]
    GraphCaptureFailed { message: String },

    #[error("VFS error: {message}")]
    VfsError { message: String },

    #[error("VMM error: {message}")]
    VmmError { message: String },

    #[error("IPC/Shared memory error: {message}")]
    IpcError { message: String },

    #[error("Watchdog timeout")]
    WatchdogTimeout,

    #[error(
        "Regime allocation gate violation: allocations {alloc_before}->{alloc_after}, frees {free_before}->{free_after}"
    )]
    RegimeAllocationViolation {
        alloc_before: u64,
        alloc_after: u64,
        free_before: u64,
        free_after: u64,
    },

    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    #[error("Resource not found: {name}")]
    NotFound { name: String },

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Resource busy")]
    Busy,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for PTX-OS operations
pub type Result<T> = std::result::Result<T, PtxError>;
