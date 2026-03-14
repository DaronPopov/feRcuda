//! Raw FFI bindings to PTX-OS C/CUDA library

use libc::{c_char, c_int, c_void, size_t, ssize_t};

// Opaque types
#[repr(C)]
pub struct GPUHotRuntime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct VMMState {
    _private: [u8; 0],
}

#[repr(C)]
pub struct VFSState {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PTXTLSFAllocator {
    _private: [u8; 0],
}

// CUDA types (simplified)
pub type CudaStream = *mut c_void;
pub type CudaGraph = *mut c_void;
pub type CudaGraphExec = *mut c_void;

/// GPU Hot Statistics
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct GPUHotStats {
    pub vram_allocated: size_t,
    pub vram_used: size_t,
    pub vram_free: size_t,
    pub gpu_utilization: f32,
    pub active_streams: c_int,
    pub registered_kernels: c_int,
    pub shm_count: c_int,
    pub total_ops: u64,
    pub avg_latency_us: f32,
    pub watchdog_tripped: bool,
}

/// TLSF Pool Statistics
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct TLSFPoolStats {
    pub total_pool_size: size_t,
    pub allocated_bytes: size_t,
    pub free_bytes: size_t,
    pub peak_allocated: size_t,
    pub largest_free_block: size_t,
    pub smallest_free_block: size_t,

    pub total_blocks: u32,
    pub free_blocks: u32,
    pub allocated_blocks: u32,

    pub fallback_count: u32,
    pub utilization_percent: f32,
    pub fragmentation_ratio: f32,
    pub external_fragmentation: f32,

    pub free_list_counts: [u32; 32],

    pub hash_collisions: u32,
    pub max_chain_length: u32,
    pub avg_chain_length: f32,

    pub total_allocations: u64,
    pub total_frees: u64,
    pub total_splits: u32,
    pub total_merges: u32,

    pub is_healthy: bool,
    pub needs_defrag: bool,
}

/// TLSF Health Report
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TLSFHealthReport {
    pub is_valid: bool,
    pub has_memory_leaks: bool,
    pub has_corrupted_blocks: bool,
    pub has_broken_chains: bool,
    pub has_hash_errors: bool,
    pub error_count: c_int,
    pub error_messages: [[c_char; 256]; 16],
}

impl Default for TLSFHealthReport {
    fn default() -> Self {
        Self {
            is_valid: true,
            has_memory_leaks: false,
            has_corrupted_blocks: false,
            has_broken_chains: false,
            has_hash_errors: false,
            error_count: 0,
            error_messages: [[0; 256]; 16],
        }
    }
}

/// System Snapshot
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct GPUHotSystemSnapshot {
    pub total_ops: u64,
    pub active_processes: c_int,
    pub active_tasks: c_int,
    pub max_priority_active: c_int,
    pub total_vram_used: size_t,
    pub watchdog_alert: bool,
    pub kernel_running: bool,
    pub shutdown_requested: bool,
    pub active_priority_level: c_int,
    pub signal_mask: u64,
    pub interrupt_cnt: u32,
    pub queue_head: u32,
    pub queue_tail: u32,
    pub queue_lock: u32,
}

/// VFS Inode (simplified for FFI)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VFSInodeStat {
    pub inode_id: u32,
    pub name: [c_char; 64],
    pub path: [c_char; 256],
    pub node_type: c_int,
    pub size: size_t,
    pub dims: c_int,
    pub shape: [c_int; 8],
    pub dtype: c_int,
    pub created_at: u64,
    pub modified_at: u64,
    pub accessed_at: u64,
    pub mode: u32,
    pub active: bool,
}

impl Default for VFSInodeStat {
    fn default() -> Self {
        Self {
            inode_id: 0,
            name: [0; 64],
            path: [0; 256],
            node_type: 0,
            size: 0,
            dims: 0,
            shape: [0; 8],
            dtype: 0,
            created_at: 0,
            modified_at: 0,
            accessed_at: 0,
            mode: 0,
            active: false,
        }
    }
}

/// feR-os runtime Configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GPUHotConfig {
    /// Fraction of available VRAM to use (0.0-1.0). Set to 0 to use fixed_pool_size.
    pub pool_fraction: f32,
    /// Fixed pool size in bytes. Used if pool_fraction == 0.
    pub fixed_pool_size: size_t,
    /// Minimum pool size in bytes (default: 256MB)
    pub min_pool_size: size_t,
    /// Maximum pool size in bytes (0 = no limit)
    pub max_pool_size: size_t,
    /// VRAM to reserve for CUDA runtime (default: 256MB)
    pub reserve_vram: size_t,
    /// Enable memory leak detection
    pub enable_leak_detection: bool,
    /// Enable pool health monitoring
    pub enable_pool_health: bool,
    /// Utilization threshold to trigger warnings (0.0-1.0)
    pub warning_threshold: f32,
    /// Force daemon mode
    pub force_daemon_mode: bool,
    /// Suppress initialization messages
    pub quiet_init: bool,
}

impl Default for GPUHotConfig {
    fn default() -> Self {
        Self {
            pool_fraction: 0.85,
            fixed_pool_size: 0,
            min_pool_size: 256 * 1024 * 1024,
            max_pool_size: 0,
            reserve_vram: 256 * 1024 * 1024,
            enable_leak_detection: true,
            enable_pool_health: true,
            warning_threshold: 0.9,
            force_daemon_mode: false,
            quiet_init: false,
        }
    }
}

impl GPUHotConfig {
    /// Create a config that uses a fixed pool size
    pub fn with_fixed_size(size_bytes: usize) -> Self {
        Self {
            pool_fraction: 0.0,
            fixed_pool_size: size_bytes,
            ..Default::default()
        }
    }

    /// Create a config that uses a fraction of available VRAM
    pub fn with_fraction(fraction: f32) -> Self {
        Self {
            pool_fraction: fraction.clamp(0.0, 1.0),
            fixed_pool_size: 0,
            ..Default::default()
        }
    }

    /// Use maximum available VRAM (90%)
    pub fn max_vram() -> Self {
        Self::with_fraction(0.90)
    }

    /// Enable quiet mode (minimal output)
    pub fn quiet(mut self) -> Self {
        self.quiet_init = true;
        self
    }

    /// Set the reserve VRAM for CUDA runtime
    pub fn reserve(mut self, bytes: usize) -> Self {
        self.reserve_vram = bytes;
        self
    }

    /// Set the minimum pool size
    pub fn min_pool(mut self, bytes: usize) -> Self {
        self.min_pool_size = bytes;
        self
    }
}

// FFI function declarations
#[link(name = "ptx_core")]
unsafe extern "C" {
    // ========================================================================
    // Core Runtime API
    // ========================================================================

    pub fn gpu_hot_default_config() -> GPUHotConfig;
    pub fn gpu_hot_init(device_id: c_int, token: *const c_char) -> *mut GPUHotRuntime;
    pub fn gpu_hot_init_with_config(
        device_id: c_int,
        token: *const c_char,
        config: *const GPUHotConfig,
    ) -> *mut GPUHotRuntime;
    pub fn gpu_hot_shutdown(runtime: *mut GPUHotRuntime);
    pub fn gpu_hot_keepalive(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // Memory Allocation API
    // ========================================================================

    pub fn gpu_hot_alloc(runtime: *mut GPUHotRuntime, size: size_t) -> *mut c_void;
    pub fn gpu_hot_free(runtime: *mut GPUHotRuntime, ptr: *mut c_void);
    pub fn gpu_hot_can_allocate(runtime: *mut GPUHotRuntime, size: size_t) -> bool;
    pub fn gpu_hot_get_max_allocatable(runtime: *mut GPUHotRuntime) -> size_t;
    pub fn gpu_hot_owns_ptr(runtime: *mut GPUHotRuntime, ptr: *mut c_void) -> bool;

    // ========================================================================
    // CUDA Allocation Hook API
    // ========================================================================

    pub fn ptx_hook_init(runtime: *mut GPUHotRuntime, verbose: bool);
    pub fn ptx_hook_disable();
    pub fn ptx_hook_get_launch_count() -> u64;
    pub fn ptx_hook_reset_launch_count();
    pub fn ptx_hook_is_enabled() -> bool;

    // ========================================================================
    // CUDA Graph API
    // ========================================================================

    pub fn gpu_hot_begin_capture(
        runtime: *mut GPUHotRuntime,
        stream_id: c_int,
        graph_name: *const c_char,
    ) -> c_int;
    pub fn gpu_hot_end_capture(runtime: *mut GPUHotRuntime, stream_id: c_int) -> c_int;
    pub fn gpu_hot_launch_graph(
        runtime: *mut GPUHotRuntime,
        graph_id: c_int,
        stream: CudaStream,
    );
    pub fn gpu_hot_destroy_graph(runtime: *mut GPUHotRuntime, graph_id: c_int);

    // ========================================================================
    // Stream API
    // ========================================================================

    pub fn gpu_hot_get_stream(runtime: *mut GPUHotRuntime, stream_id: c_int) -> CudaStream;
    pub fn gpu_hot_get_priority_stream(runtime: *mut GPUHotRuntime, priority: c_int)
        -> CudaStream;
    pub fn gpu_hot_sync_all(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // Statistics API
    // ========================================================================

    pub fn gpu_hot_get_stats(runtime: *mut GPUHotRuntime, stats: *mut GPUHotStats);
    pub fn gpu_hot_get_tlsf_stats(runtime: *mut GPUHotRuntime, stats: *mut TLSFPoolStats);
    pub fn gpu_hot_validate_tlsf_pool(
        runtime: *mut GPUHotRuntime,
        report: *mut TLSFHealthReport,
    );
    pub fn gpu_hot_print_pool_map(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // Pool Management
    // ========================================================================

    pub fn gpu_hot_defragment_pool(runtime: *mut GPUHotRuntime);
    pub fn gpu_hot_set_warning_threshold(runtime: *mut GPUHotRuntime, threshold_percent: f32);
    pub fn gpu_hot_set_auto_defrag(runtime: *mut GPUHotRuntime, enable: bool);

    // ========================================================================
    // Shared Memory / IPC API
    // ========================================================================

    pub fn gpu_hot_shm_alloc(
        runtime: *mut GPUHotRuntime,
        name: *const c_char,
        size: size_t,
    ) -> *mut c_void;
    pub fn gpu_hot_shm_open(runtime: *mut GPUHotRuntime, name: *const c_char) -> *mut c_void;
    pub fn gpu_hot_shm_close(runtime: *mut GPUHotRuntime, ptr: *mut c_void);
    pub fn gpu_hot_shm_unlink(runtime: *mut GPUHotRuntime, name: *const c_char);

    // ========================================================================
    // Watchdog API
    // ========================================================================

    pub fn gpu_hot_set_watchdog(runtime: *mut GPUHotRuntime, timeout_ms: c_int);
    pub fn gpu_hot_check_watchdog(runtime: *mut GPUHotRuntime) -> bool;
    pub fn gpu_hot_reset_watchdog(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // System State API
    // ========================================================================

    pub fn gpu_hot_get_system_snapshot(
        runtime: *mut GPUHotRuntime,
        snapshot: *mut GPUHotSystemSnapshot,
    );
    pub fn gpu_hot_reset_system_state(runtime: *mut GPUHotRuntime);
    pub fn gpu_hot_flush_task_queue(runtime: *mut GPUHotRuntime);
    pub fn gpu_hot_clear_signal_mask(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // PTX-OS Kernel Boot
    // ========================================================================

    pub fn ptx_os_boot_persistent_kernel(runtime: *mut GPUHotRuntime);

    // ========================================================================
    // VMM API
    // ========================================================================

    pub fn vmm_init(runtime: *mut GPUHotRuntime, swap_size: size_t) -> *mut VMMState;
    pub fn vmm_shutdown(vmm: *mut VMMState);
    pub fn vmm_alloc_page(vmm: *mut VMMState, flags: u32) -> *mut c_void;
    pub fn vmm_free_page(vmm: *mut VMMState, addr: *mut c_void);
    pub fn vmm_swap_out(vmm: *mut VMMState, addr: *mut c_void) -> c_int;
    pub fn vmm_swap_in(vmm: *mut VMMState, addr: *mut c_void) -> c_int;
    pub fn vmm_pin_page(vmm: *mut VMMState, addr: *mut c_void);
    pub fn vmm_unpin_page(vmm: *mut VMMState, addr: *mut c_void);
    pub fn vmm_get_stats(
        vmm: *mut VMMState,
        resident: *mut u64,
        swapped: *mut u64,
        faults: *mut u64,
        evictions: *mut u64,
    );

    // ========================================================================
    // VFS API
    // ========================================================================

    pub fn vfs_init(runtime: *mut GPUHotRuntime) -> *mut VFSState;
    pub fn vfs_shutdown(vfs: *mut VFSState);
    pub fn vfs_mkdir(vfs: *mut VFSState, path: *const c_char, mode: u32) -> c_int;
    pub fn vfs_rmdir(vfs: *mut VFSState, path: *const c_char) -> c_int;
    pub fn vfs_open(vfs: *mut VFSState, path: *const c_char, flags: u32) -> c_int;
    pub fn vfs_close(vfs: *mut VFSState, fd: c_int) -> c_int;
    pub fn vfs_read(vfs: *mut VFSState, fd: c_int, buf: *mut c_void, count: size_t) -> ssize_t;
    pub fn vfs_write(
        vfs: *mut VFSState,
        fd: c_int,
        buf: *const c_void,
        count: size_t,
    ) -> ssize_t;
    pub fn vfs_seek(vfs: *mut VFSState, fd: c_int, offset: size_t, whence: c_int) -> c_int;
    pub fn vfs_create_tensor(
        vfs: *mut VFSState,
        path: *const c_char,
        shape: *const c_int,
        dims: c_int,
        dtype: c_int,
    ) -> c_int;
    pub fn vfs_mmap_tensor(vfs: *mut VFSState, path: *const c_char) -> *mut c_void;
    pub fn vfs_sync_tensor(vfs: *mut VFSState, path: *const c_char) -> c_int;
    pub fn vfs_stat(vfs: *mut VFSState, path: *const c_char, stat_out: *mut VFSInodeStat)
        -> c_int;
}

// Priority constants
pub const PTX_PRIORITY_REALTIME: c_int = 0;
pub const PTX_PRIORITY_HIGH: c_int = 1;
pub const PTX_PRIORITY_NORMAL: c_int = 2;
pub const PTX_PRIORITY_LOW: c_int = 3;

// VMM flags
pub const VMM_FLAG_READ: u32 = 0x01;
pub const VMM_FLAG_WRITE: u32 = 0x02;
pub const VMM_FLAG_EXEC: u32 = 0x04;
pub const VMM_FLAG_SHARED: u32 = 0x08;
pub const VMM_FLAG_PINNED: u32 = 0x10;

// VFS open flags
pub const VFS_O_RDONLY: u32 = 0x01;
pub const VFS_O_WRONLY: u32 = 0x02;
pub const VFS_O_RDWR: u32 = 0x03;
pub const VFS_O_CREAT: u32 = 0x10;
pub const VFS_O_TRUNC: u32 = 0x20;
pub const VFS_O_APPEND: u32 = 0x40;
