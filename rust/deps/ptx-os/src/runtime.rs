//! Safe wrapper for feR-os runtime

use crate::error::{PtxError, Result};
use crate::ffi;
use parking_lot::Mutex;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

/// feR-os runtime statistics
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    pub vram_allocated: usize,
    pub vram_used: usize,
    pub vram_free: usize,
    pub gpu_utilization: f32,
    pub active_streams: i32,
    pub registered_kernels: i32,
    pub shm_count: i32,
    pub total_ops: u64,
    pub avg_latency_us: f32,
    pub watchdog_tripped: bool,
}

impl From<ffi::GPUHotStats> for RuntimeStats {
    fn from(stats: ffi::GPUHotStats) -> Self {
        Self {
            vram_allocated: stats.vram_allocated,
            vram_used: stats.vram_used,
            vram_free: stats.vram_free,
            gpu_utilization: stats.gpu_utilization,
            active_streams: stats.active_streams,
            registered_kernels: stats.registered_kernels,
            shm_count: stats.shm_count,
            total_ops: stats.total_ops,
            avg_latency_us: stats.avg_latency_us,
            watchdog_tripped: stats.watchdog_tripped,
        }
    }
}

/// TLSF memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_size: usize,
    pub allocated: usize,
    pub free: usize,
    pub peak_allocated: usize,
    pub utilization_percent: f32,
    pub fragmentation_ratio: f32,
    pub is_healthy: bool,
    pub needs_defrag: bool,
    pub total_allocations: u64,
    pub total_frees: u64,
}

impl From<ffi::TLSFPoolStats> for PoolStats {
    fn from(stats: ffi::TLSFPoolStats) -> Self {
        Self {
            total_size: stats.total_pool_size,
            allocated: stats.allocated_bytes,
            free: stats.free_bytes,
            peak_allocated: stats.peak_allocated,
            utilization_percent: stats.utilization_percent,
            fragmentation_ratio: stats.fragmentation_ratio,
            is_healthy: stats.is_healthy,
            needs_defrag: stats.needs_defrag,
            total_allocations: stats.total_allocations,
            total_frees: stats.total_frees,
        }
    }
}

/// System state snapshot
#[derive(Debug, Clone, Default)]
pub struct SystemSnapshot {
    pub total_ops: u64,
    pub active_processes: i32,
    pub active_tasks: i32,
    pub kernel_running: bool,
    pub shutdown_requested: bool,
    pub watchdog_alert: bool,
    pub signal_mask: u64,
    pub queue_depth: u32,
}

impl From<ffi::GPUHotSystemSnapshot> for SystemSnapshot {
    fn from(snap: ffi::GPUHotSystemSnapshot) -> Self {
        Self {
            total_ops: snap.total_ops,
            active_processes: snap.active_processes,
            active_tasks: snap.active_tasks,
            kernel_running: snap.kernel_running,
            shutdown_requested: snap.shutdown_requested,
            watchdog_alert: snap.watchdog_alert,
            signal_mask: snap.signal_mask,
            queue_depth: snap.queue_head.wrapping_sub(snap.queue_tail),
        }
    }
}

/// Priority levels for stream scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Realtime = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

/// Inner runtime state (shared between clones)
struct RuntimeInner {
    ptr: NonNull<ffi::GPUHotRuntime>,
    device_id: i32,
}

// Safety: The runtime uses internal synchronization via CUDA
unsafe impl Send for RuntimeInner {}
unsafe impl Sync for RuntimeInner {}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        unsafe {
            ffi::gpu_hot_shutdown(self.ptr.as_ptr());
        }
    }
}

/// feR-os runtime - The main interface to PTX-OS
///
/// This provides a safe, RAII wrapper around the GPU runtime.
/// Cloning shares the underlying runtime (reference counted).
#[derive(Clone)]
pub struct RegimeRuntimeCore {
    inner: Arc<Mutex<RuntimeInner>>,
}

/// Backward-compatible alias. Prefer `RegimeRuntimeCore`.
pub type HotRuntime = RegimeRuntimeCore;

/// Re-export the config type for convenience
pub use crate::ffi::GPUHotConfig as RuntimeConfig;

/// Allocation behavior for runtime-managed workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocMode {
    /// Allocate long-lived buffers for a session and reuse them.
    SessionBuffers,
    /// Allow frequent alloc/free calls (CPU-like style) through TLSF.
    CpuLike,
}

/// OS-level runtime policy: pool sizing + allocation behavior.
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    runtime: RuntimeConfig,
    alloc_mode: AllocMode,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            alloc_mode: AllocMode::SessionBuffers,
        }
    }
}

impl RegimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_vram() -> Self {
        Self {
            runtime: RuntimeConfig::max_vram(),
            ..Self::default()
        }
    }

    pub fn pool_fixed_bytes(mut self, size_bytes: usize) -> Self {
        self.runtime = RuntimeConfig::with_fixed_size(size_bytes);
        self
    }

    pub fn pool_fixed_mb(self, size_mb: usize) -> Self {
        self.pool_fixed_bytes(size_mb * 1024 * 1024)
    }

    pub fn pool_fraction(mut self, fraction: f32) -> Self {
        self.runtime = RuntimeConfig::with_fraction(fraction);
        self
    }

    pub fn from_runtime_config(runtime: RuntimeConfig) -> Self {
        Self {
            runtime,
            ..Self::default()
        }
    }

    pub fn quiet(mut self) -> Self {
        self.runtime = self.runtime.quiet();
        self
    }

    pub fn alloc_mode(mut self, alloc_mode: AllocMode) -> Self {
        self.alloc_mode = alloc_mode;
        self
    }

    pub fn runtime_config(&self) -> &RuntimeConfig {
        &self.runtime
    }

    pub fn alloc_mode_value(&self) -> AllocMode {
        self.alloc_mode
    }
}

impl RegimeRuntimeCore {
    fn init_with_optional_config(device_id: i32, config: Option<RuntimeConfig>) -> Result<Self> {
        let ptr = match config {
            Some(cfg) => unsafe {
                ffi::gpu_hot_init_with_config(device_id, std::ptr::null(), &cfg)
            },
            None => unsafe { ffi::gpu_hot_init(device_id, std::ptr::null()) },
        };

        let ptr = NonNull::new(ptr).ok_or(PtxError::InitializationFailed { device_id })?;

        // Enable CUDA allocation hooks - routes ALL cuMemAlloc/cudaMalloc through TLSF
        unsafe {
            ffi::ptx_hook_init(ptr.as_ptr(), false);
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(RuntimeInner { ptr, device_id })),
        })
    }

    /// Initialize a new feR-os runtime on the specified device with default config.
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (0-indexed)
    ///
    /// # Example
    /// ```rust,ignore
    /// let runtime = RegimeRuntimeCore::new(0)?;
    /// ```
    pub fn new(device_id: i32) -> Result<Self> {
        Self::init_with_optional_config(device_id, None)
    }

    /// Initialize a new feR-os runtime with custom configuration.
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (0-indexed)
    /// * `config` - Optional runtime configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// use ptx_os::runtime::{RegimeRuntimeCore, RuntimeConfig};
    ///
    /// // Use 90% of available VRAM
    /// let config = RuntimeConfig::with_fraction(0.90).quiet();
    /// let runtime = RegimeRuntimeCore::with_config(0, Some(config))?;
    ///
    /// // Or use a fixed 4GB pool
    /// let config = RuntimeConfig::with_fixed_size(4 * 1024 * 1024 * 1024);
    /// let runtime = RegimeRuntimeCore::with_config(0, Some(config))?;
    /// ```
    #[deprecated(note = "Use with_regime(device_id, RegimeConfig) as the canonical OS syntax.")]
    pub fn with_config(device_id: i32, config: Option<RuntimeConfig>) -> Result<Self> {
        Self::init_with_optional_config(device_id, config)
    }

    /// Initialize a runtime from OS-level regime policy.
    pub fn with_regime(device_id: i32, regime: RegimeConfig) -> Result<Self> {
        Self::init_with_optional_config(device_id, Some(regime.runtime))
    }

    /// Initialize with maximum available VRAM (90%)
    pub fn max_vram(device_id: i32) -> Result<Self> {
        Self::with_regime(device_id, RegimeConfig::max_vram())
    }

    /// Initialize with a fixed pool size
    pub fn with_pool_size(device_id: i32, size_bytes: usize) -> Result<Self> {
        Self::with_regime(device_id, RegimeConfig::new().pool_fixed_bytes(size_bytes))
    }

    /// Get the device ID this runtime is bound to.
    pub fn device_id(&self) -> i32 {
        self.inner.lock().device_id
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> RuntimeStats {
        let guard = self.inner.lock();
        let mut stats = ffi::GPUHotStats::default();
        unsafe {
            ffi::gpu_hot_get_stats(guard.ptr.as_ptr(), &mut stats);
        }
        stats.into()
    }

    /// Get memory pool statistics.
    pub fn pool_stats(&self) -> PoolStats {
        let guard = self.inner.lock();
        let mut stats = ffi::TLSFPoolStats::default();
        unsafe {
            ffi::gpu_hot_get_tlsf_stats(guard.ptr.as_ptr(), &mut stats);
        }
        stats.into()
    }

    /// Get system state snapshot.
    pub fn system_snapshot(&self) -> SystemSnapshot {
        let guard = self.inner.lock();
        let mut snap = ffi::GPUHotSystemSnapshot::default();
        unsafe {
            ffi::gpu_hot_get_system_snapshot(guard.ptr.as_ptr(), &mut snap);
        }
        snap.into()
    }

    /// Allocate GPU memory from the hot pool.
    ///
    /// Returns a raw pointer to GPU memory. Use `DeviceBox` for safe allocation.
    pub fn alloc_raw(&self, size: usize) -> Result<*mut std::ffi::c_void> {
        let guard = self.inner.lock();
        let ptr = unsafe { ffi::gpu_hot_alloc(guard.ptr.as_ptr(), size) };

        if ptr.is_null() {
            Err(PtxError::AllocationFailed { size })
        } else {
            Ok(ptr)
        }
    }

    /// Free GPU memory.
    ///
    /// # Safety
    /// The pointer must have been allocated by this runtime.
    pub unsafe fn free_raw(&self, ptr: *mut std::ffi::c_void) {
        let guard = self.inner.lock();
        ffi::gpu_hot_free(guard.ptr.as_ptr(), ptr);
    }

    /// Check if an allocation of the given size would succeed.
    pub fn can_allocate(&self, size: usize) -> bool {
        let guard = self.inner.lock();
        unsafe { ffi::gpu_hot_can_allocate(guard.ptr.as_ptr(), size) }
    }

    /// Get the maximum allocatable block size.
    pub fn max_allocatable(&self) -> usize {
        let guard = self.inner.lock();
        unsafe { ffi::gpu_hot_get_max_allocatable(guard.ptr.as_ptr()) }
    }

    /// Defragment the memory pool.
    pub fn defragment(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_defragment_pool(guard.ptr.as_ptr());
        }
    }

    /// Set memory pool warning threshold (0.0 - 100.0).
    pub fn set_warning_threshold(&self, percent: f32) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_set_warning_threshold(guard.ptr.as_ptr(), percent);
        }
    }

    /// Enable or disable automatic defragmentation.
    pub fn set_auto_defrag(&self, enable: bool) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_set_auto_defrag(guard.ptr.as_ptr(), enable);
        }
    }

    /// Print memory pool map to stdout.
    pub fn print_pool_map(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_print_pool_map(guard.ptr.as_ptr());
        }
    }

    /// Synchronize all CUDA streams.
    pub fn sync_all(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_sync_all(guard.ptr.as_ptr());
        }
    }

    /// Get raw CUDA stream pointer for the given stream ID (for PyTorch/tch-rs adapter).
    ///
    /// # Safety
    /// The returned pointer is only valid while this runtime exists.
    pub unsafe fn get_raw_stream(&self, stream_id: i32) -> *mut std::ffi::c_void {
        ffi::gpu_hot_get_stream(self.as_ptr(), stream_id)
    }

    /// Send keepalive signal to GPU.
    pub fn keepalive(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_keepalive(guard.ptr.as_ptr());
        }
    }

    /// Set watchdog timeout in milliseconds.
    pub fn set_watchdog(&self, timeout_ms: i32) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_set_watchdog(guard.ptr.as_ptr(), timeout_ms);
        }
    }

    /// Check if watchdog has tripped.
    pub fn check_watchdog(&self) -> bool {
        let guard = self.inner.lock();
        unsafe { ffi::gpu_hot_check_watchdog(guard.ptr.as_ptr()) }
    }

    /// Reset watchdog state.
    pub fn reset_watchdog(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_reset_watchdog(guard.ptr.as_ptr());
        }
    }

    /// Allocate shared memory segment.
    pub fn shm_alloc(&self, name: &str, size: usize) -> Result<*mut std::ffi::c_void> {
        let guard = self.inner.lock();
        let c_name = CString::new(name).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid segment name".into(),
        })?;

        let ptr = unsafe { ffi::gpu_hot_shm_alloc(guard.ptr.as_ptr(), c_name.as_ptr(), size) };

        if ptr.is_null() {
            Err(PtxError::IpcError {
                message: format!("Failed to allocate shared segment: {}", name),
            })
        } else {
            Ok(ptr)
        }
    }

    /// Open existing shared memory segment.
    pub fn shm_open(&self, name: &str) -> Result<*mut std::ffi::c_void> {
        let guard = self.inner.lock();
        let c_name = CString::new(name).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid segment name".into(),
        })?;

        let ptr = unsafe { ffi::gpu_hot_shm_open(guard.ptr.as_ptr(), c_name.as_ptr()) };

        if ptr.is_null() {
            Err(PtxError::NotFound { name: name.into() })
        } else {
            Ok(ptr)
        }
    }

    /// Unlink shared memory segment.
    pub fn shm_unlink(&self, name: &str) {
        let guard = self.inner.lock();
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                ffi::gpu_hot_shm_unlink(guard.ptr.as_ptr(), c_name.as_ptr());
            }
        }
    }

    /// Boot the persistent GPU OS kernel.
    ///
    /// This launches the GPU-side OS kernel that polls for tasks.
    pub fn boot_os_kernel(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::ptx_os_boot_persistent_kernel(guard.ptr.as_ptr());
        }
    }

    /// Reset system state.
    pub fn reset_system_state(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_reset_system_state(guard.ptr.as_ptr());
        }
    }

    /// Flush task queue.
    pub fn flush_task_queue(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_flush_task_queue(guard.ptr.as_ptr());
        }
    }

    /// Clear signal mask.
    pub fn clear_signal_mask(&self) {
        let guard = self.inner.lock();
        unsafe {
            ffi::gpu_hot_clear_signal_mask(guard.ptr.as_ptr());
        }
    }

    /// Get raw pointer (for FFI with other libraries).
    ///
    /// # Safety
    /// The returned pointer is only valid while this runtime exists.
    pub unsafe fn as_ptr(&self) -> *mut ffi::GPUHotRuntime {
        self.inner.lock().ptr.as_ptr()
    }
}

impl std::fmt::Debug for RegimeRuntimeCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("RegimeRuntimeCore")
            .field("device_id", &self.device_id())
            .field("vram_used", &stats.vram_used)
            .field("vram_free", &stats.vram_free)
            .field("total_ops", &stats.total_ops)
            .finish()
    }
}

/// Typestate marker: runtime has not been initialized yet.
pub struct Uninitialized;
/// Typestate marker: runtime is initialized and fully operational.
pub struct Ready;
/// Typestate marker: runtime is in degraded operation.
pub struct Degraded;
/// Typestate marker: runtime is in locked regime phase (no dynamic alloc/free allowed).
pub struct RegimeLocked;
/// Typestate marker: runtime is shut down and no operations are allowed.
pub struct Shutdown;

#[derive(Debug, Clone, Copy)]
struct RegimeBaseline {
    allocations: u64,
    frees: u64,
}

/// Regime-oriented typestate wrapper for lifecycle-safe runtime usage.
///
/// This makes invalid transitions a compile-time error. For example:
/// - You cannot allocate in `Shutdown`.
/// - You cannot call `boot_os_kernel` before `Ready`.
pub struct RegimeRuntime<State> {
    runtime: Option<RegimeRuntimeCore>,
    regime_baseline: Option<RegimeBaseline>,
    _state: PhantomData<State>,
}

impl RegimeRuntime<Uninitialized> {
    /// Construct an uninitialized runtime handle.
    pub fn new() -> Self {
        Self {
            runtime: None,
            regime_baseline: None,
            _state: PhantomData,
        }
    }

    /// Initialize runtime and transition to `Ready`.
    #[deprecated(note = "Use init_regime(device_id, RegimeConfig) as the canonical OS syntax.")]
    pub fn init(self, device_id: i32, config: Option<RuntimeConfig>) -> Result<RegimeRuntime<Ready>> {
        let rt = RegimeRuntimeCore::init_with_optional_config(device_id, config)?;
        Ok(RegimeRuntime {
            runtime: Some(rt),
            regime_baseline: None,
            _state: PhantomData,
        })
    }

    /// Initialize runtime with OS-level regime policy and transition to `Ready`.
    pub fn init_regime(self, device_id: i32, regime: RegimeConfig) -> Result<RegimeRuntime<Ready>> {
        let rt = RegimeRuntimeCore::with_regime(device_id, regime)?;
        Ok(RegimeRuntime {
            runtime: Some(rt),
            regime_baseline: None,
            _state: PhantomData,
        })
    }
}

impl Default for RegimeRuntime<Uninitialized> {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeRuntime<Ready> {
    fn rt(&self) -> &RegimeRuntimeCore {
        self.runtime.as_ref().expect("runtime must exist in Ready state")
    }

    /// Borrow the underlying runtime for compatibility with existing code.
    pub fn runtime(&self) -> &RegimeRuntimeCore {
        self.rt()
    }

    /// Allocate GPU memory in `Ready` state.
    pub fn alloc_raw(&self, size: usize) -> Result<*mut std::ffi::c_void> {
        self.rt().alloc_raw(size)
    }

    /// Free GPU memory in `Ready` state.
    ///
    /// # Safety
    /// The pointer must have been allocated by this runtime.
    pub unsafe fn free_raw(&self, ptr: *mut std::ffi::c_void) {
        self.rt().free_raw(ptr);
    }

    /// Transition to degraded mode (policy decision by caller/supervisor).
    pub fn degrade(self) -> RegimeRuntime<Degraded> {
        RegimeRuntime {
            runtime: self.runtime,
            regime_baseline: self.regime_baseline,
            _state: PhantomData,
        }
    }

    /// Enter locked regime by freezing allocator counters.
    ///
    /// In `RegimeLocked` state, allocation/free drift is treated as a hard violation.
    pub fn enter_regime_lock(self) -> RegimeRuntime<RegimeLocked> {
        let baseline = self.rt().pool_stats();
        RegimeRuntime {
            runtime: self.runtime,
            regime_baseline: Some(RegimeBaseline {
                allocations: baseline.total_allocations,
                frees: baseline.total_frees,
            }),
            _state: PhantomData,
        }
    }

    /// Shutdown runtime and transition to `Shutdown`.
    pub fn shutdown(mut self) -> RegimeRuntime<Shutdown> {
        self.runtime.take();
        RegimeRuntime {
            runtime: None,
            regime_baseline: None,
            _state: PhantomData,
        }
    }

    pub fn stats(&self) -> RuntimeStats {
        self.rt().stats()
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.rt().pool_stats()
    }

    pub fn system_snapshot(&self) -> SystemSnapshot {
        self.rt().system_snapshot()
    }

    pub fn sync_all(&self) {
        self.rt().sync_all();
    }

    pub fn keepalive(&self) {
        self.rt().keepalive();
    }

    pub fn set_watchdog(&self, timeout_ms: i32) {
        self.rt().set_watchdog(timeout_ms);
    }

    pub fn boot_os_kernel(&self) {
        self.rt().boot_os_kernel();
    }
}

impl RegimeRuntime<Degraded> {
    fn rt(&self) -> &RegimeRuntimeCore {
        self.runtime.as_ref().expect("runtime must exist in Degraded state")
    }

    /// In degraded mode, introspection is still allowed.
    pub fn stats(&self) -> RuntimeStats {
        self.rt().stats()
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.rt().pool_stats()
    }

    pub fn system_snapshot(&self) -> SystemSnapshot {
        self.rt().system_snapshot()
    }

    /// Recover from degraded mode back to `Ready`.
    pub fn recover(self) -> RegimeRuntime<Ready> {
        RegimeRuntime {
            runtime: self.runtime,
            regime_baseline: None,
            _state: PhantomData,
        }
    }

    /// Shutdown runtime from degraded mode.
    pub fn shutdown(mut self) -> RegimeRuntime<Shutdown> {
        self.runtime.take();
        RegimeRuntime {
            runtime: None,
            regime_baseline: None,
            _state: PhantomData,
        }
    }
}

impl RegimeRuntime<RegimeLocked> {
    fn rt(&self) -> &RegimeRuntimeCore {
        self.runtime.as_ref().expect("runtime must exist in RegimeLocked state")
    }

    fn baseline(&self) -> RegimeBaseline {
        self.regime_baseline
            .expect("regime baseline must exist in RegimeLocked state")
    }

    /// Enforce no-allocation/no-free locked-regime contract.
    pub fn check_regime_gate(&self) -> Result<()> {
        let base = self.baseline();
        let now = self.rt().pool_stats();
        if now.total_allocations != base.allocations || now.total_frees != base.frees {
            return Err(PtxError::RegimeAllocationViolation {
                alloc_before: base.allocations,
                alloc_after: now.total_allocations,
                free_before: base.frees,
                free_after: now.total_frees,
            });
        }
        Ok(())
    }

    /// Borrow underlying runtime after successful regime gate check.
    pub fn runtime(&self) -> Result<&RegimeRuntimeCore> {
        self.check_regime_gate()?;
        Ok(self.rt())
    }

    /// Regime-locked synchronization (checks gate before and after).
    pub fn sync_all(&self) -> Result<()> {
        self.check_regime_gate()?;
        self.rt().sync_all();
        self.check_regime_gate()
    }

    pub fn keepalive(&self) -> Result<()> {
        self.check_regime_gate()?;
        self.rt().keepalive();
        self.check_regime_gate()
    }

    pub fn set_watchdog(&self, timeout_ms: i32) -> Result<()> {
        self.check_regime_gate()?;
        self.rt().set_watchdog(timeout_ms);
        self.check_regime_gate()
    }

    pub fn stats(&self) -> Result<RuntimeStats> {
        self.check_regime_gate()?;
        Ok(self.rt().stats())
    }

    pub fn pool_stats(&self) -> Result<PoolStats> {
        self.check_regime_gate()?;
        Ok(self.rt().pool_stats())
    }

    pub fn system_snapshot(&self) -> Result<SystemSnapshot> {
        self.check_regime_gate()?;
        Ok(self.rt().system_snapshot())
    }

    /// Leave locked regime and transition to degraded state after a gate failure
    /// or explicit policy decision.
    pub fn degrade(self) -> RegimeRuntime<Degraded> {
        RegimeRuntime {
            runtime: self.runtime,
            regime_baseline: None,
            _state: PhantomData,
        }
    }

    pub fn shutdown(mut self) -> RegimeRuntime<Shutdown> {
        self.runtime.take();
        RegimeRuntime {
            runtime: None,
            regime_baseline: None,
            _state: PhantomData,
        }
    }
}

impl RegimeRuntime<Shutdown> {
    /// Explicit state probe for orchestration code.
    pub fn is_shutdown(&self) -> bool {
        true
    }
}
