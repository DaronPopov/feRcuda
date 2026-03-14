//! Error types bridging PTX-OS and Candle errors

use thiserror::Error;

/// Error type for candle-ptx-os operations
#[derive(Error, Debug)]
pub enum PtxCandleError {
    #[error("PTX-OS error: {0}")]
    Ptx(#[from] ptx_os::PtxError),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("CUDA error: {message}")]
    Cuda { message: String },

    #[error("cuBLAS error: {message}")]
    CuBlas { message: String },

    #[error("Memory allocation failed: requested {size} bytes")]
    AllocationFailed { size: usize },

    #[error("Invalid device pointer")]
    InvalidPointer,

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatchVec {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Element count mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        expected: usize,
        got: usize,
    },

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(candle_core::DType),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Device mismatch: operation requires tensors on same device")]
    DeviceMismatch,

    #[error("Synchronization error")]
    SyncError,
}

impl From<PtxCandleError> for candle_core::Error {
    fn from(e: PtxCandleError) -> Self {
        candle_core::Error::wrap(e)
    }
}

impl From<cudarc_ptx::driver::DriverError> for PtxCandleError {
    fn from(e: cudarc_ptx::driver::DriverError) -> Self {
        PtxCandleError::Cuda {
            message: format!("{:?}", e),
        }
    }
}

impl From<cudarc_ptx::cublas::result::CublasError> for PtxCandleError {
    fn from(e: cudarc_ptx::cublas::result::CublasError) -> Self {
        PtxCandleError::CuBlas {
            message: format!("{:?}", e),
        }
    }
}

/// Result type alias for candle-ptx-os operations
pub type Result<T> = std::result::Result<T, PtxCandleError>;
