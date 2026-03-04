/// Shared execution contract for backend adapters that route ops through feR-os.
///
/// Backend implementers should:
/// 1. define a backend-native tensor handle type (`TensorHandle`)
/// 2. implement host upload/download conversion
/// 3. implement `matmul` and `layer_norm` by dispatching into the session plane
///
/// This trait keeps integration shape stable across Candle, cudarc, and future backends.
pub trait AdapterExecutionBackend {
    type TensorHandle;
    type BackendError: std::error::Error + Send + Sync + 'static;

    /// Create a backend-native tensor handle from host f32 data.
    fn upload_host_f32(
        &self,
        host: &[f32],
        dims: &[usize],
    ) -> Result<Self::TensorHandle, Self::BackendError>;

    /// Copy a backend-native tensor handle back to host f32 data.
    fn download_host_f32(
        &self,
        tensor: &Self::TensorHandle,
    ) -> Result<Vec<f32>, Self::BackendError>;

    /// Execute matrix multiplication through feR-os runtime path.
    fn matmul(
        &self,
        a: &Self::TensorHandle,
        b: &Self::TensorHandle,
    ) -> Result<Self::TensorHandle, Self::BackendError>;

    /// Execute layer norm through feR-os runtime path.
    fn layer_norm(
        &self,
        x: &Self::TensorHandle,
        eps: f32,
    ) -> Result<Self::TensorHandle, Self::BackendError>;
}

/// Backend author checklist for consistent slot-in behavior.
pub const BACKEND_SLOT_IN_PATTERN: &[&str] = &[
    "Create src/<backend>_adapter.rs",
    "Define backend error + tensor handle type",
    "Implement shape validation helpers",
    "Implement AdapterExecutionBackend for <Backend>SessionAdapter<'a>",
    "Export module + types in src/lib.rs behind feature gate",
    "Add an examples/<backend>_regime_bridge.rs runnable path",
    "Add feature-gated unit tests for validation + trait impl",
    "Document feature flag and usage in rust/fercuda-ffi/README.md",
];
