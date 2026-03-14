/*
 * PTX-OS GPU Hot Runtime - Header
 * Persistent GPU context with pre-allocated VRAM pools
 */

#ifndef GPU_HOT_RUNTIME_H
#define GPU_HOT_RUNTIME_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Constants
// ============================================================================

#define GPU_HOT_MAX_STREAMS         100000  // Configurable via GPUHotConfig (software queues) - UNLIMITED!
#define GPU_HOT_MAX_KERNELS         256
#define GPU_HOT_MAX_GRAPHS          256     // Increased for massive parallelism
#define GPU_HOT_MAX_NAMED_SEGMENTS  256
#define GPU_HOT_MAX_NAME_LEN        64
#define GPU_HOT_DEFAULT_VRAM_PERCENT 70.0f
#define GPU_HOT_IPC_KEY_PREFIX      "/ptx_os_"
#define GPU_HOT_IPC_KEY_SUFFIX      "_v1"
#define GPU_HOT_IPC_KEY_MAX_LEN     64

// TLSF Allocator Configuration
// FL_INDEX_MAX = 34 supports up to 2^34 = 16GB single allocations
// This is needed for large LLM KV caches (7B model = ~2-4GB KV cache)
#define TLSF_FL_INDEX_MAX           34
#define TLSF_SL_INDEX_COUNT         16
#define TLSF_MIN_BLOCK_SIZE         64
#define TLSF_ALIGNMENT              256

// PTX-OS Task Queue
#define PTX_MAX_QUEUE_SIZE          1024
#define PTX_MAX_TASK_ARGS           8

// Priority Levels
#define PTX_PRIORITY_REALTIME       0
#define PTX_PRIORITY_HIGH           1
#define PTX_PRIORITY_NORMAL         2
#define PTX_PRIORITY_LOW            3

// VMM Configuration
#define VMM_PAGE_SIZE               (64 * 1024)  // 64KB pages
#define VMM_MAX_PAGES               16384
#define VMM_MAX_SWAP_REGIONS        256

// VFS Configuration
#define VFS_MAX_NODES               1024
#define VFS_MAX_PATH_LEN            256
#define VFS_MAX_OPEN_FILES          256

// ============================================================================
// Forward Declarations
// ============================================================================

struct GPUHotRuntime;
typedef struct GPUHotRuntime GPUHotRuntime;

struct PTXTLSFAllocator;
typedef struct PTXTLSFAllocator PTXTLSFAllocator;

// ============================================================================
// TLSF Block Structure (Host-side metadata)
// ============================================================================

typedef struct TLSFBlock {
    void* device_ptr;           // Pointer to GPU memory
    size_t size;                // Block size in bytes
    uint32_t magic;             // Validation magic number

    struct TLSFBlock* prev_phys; // Previous physical block
    struct TLSFBlock* next_phys; // Next physical block
    struct TLSFBlock* prev_free; // Previous in free list
    struct TLSFBlock* next_free; // Next in free list
    struct TLSFBlock* hash_next; // Next in hash chain

    bool is_free;
    bool is_last;
    bool is_pinned;             // Prevent coalescing/eviction

    // Debug info
    uint32_t alloc_id;
    const char* alloc_file;
    int alloc_line;
    uint64_t alloc_timestamp;
    uint32_t owner_id;
} TLSFBlock;

// ============================================================================
// CUDA Graph Handle
// ============================================================================

typedef struct GPUGraphHandle {
    char name[GPU_HOT_MAX_NAME_LEN];
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool is_valid;
} GPUGraphHandle;

// ============================================================================
// Kernel Handle
// ============================================================================

typedef struct GPUKernelHandle {
    void* kernel_func;
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
} GPUKernelHandle;

// ============================================================================
// Global Registry Entry (IPC Shared Memory)
// ============================================================================

typedef struct GPURegistryEntry {
    char name[GPU_HOT_MAX_NAME_LEN];
    cudaIpcMemHandle_t ipc_handle;
    size_t size;
    bool active;
    uint64_t created_at;
    uint32_t ref_count;
    uint32_t owner_pid;
} GPURegistryEntry;

// ============================================================================
// PTX-OS Task Structure
// ============================================================================

typedef struct PTXOSTask {
    uint32_t task_id;
    uint32_t opcode;
    uint32_t priority;
    bool active;
    bool completed;
    void* args[PTX_MAX_TASK_ARGS];
    uint64_t submitted_at;
    uint64_t completed_at;
} PTXOSTask;

// ============================================================================
// PTX-OS Task Queue
// ============================================================================

typedef struct PTXTaskQueue {
    PTXOSTask tasks[PTX_MAX_QUEUE_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t lock;
} PTXTaskQueue;

// ============================================================================
// PTX Kernel Launch Descriptor (for CDP)
// ============================================================================

typedef struct PTXKernelLaunch {
    void* kernel_func;
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
    void* arg_values[PTX_MAX_TASK_ARGS];
} PTXKernelLaunch;

// ============================================================================
// VMM Page Table Entry
// ============================================================================

typedef struct VMMPageEntry {
    void* gpu_addr;             // Current GPU address (may change after swap-in)
    void* original_addr;        // Original address returned to user (for lookup)
    void* host_addr;            // Host backing store (if swapped)
    size_t size;                // Page size
    uint32_t flags;             // Page flags
    uint64_t last_access;       // For LRU eviction
    uint32_t access_count;      // Access frequency
    bool resident;              // In GPU memory
    bool dirty;                 // Modified since last swap
    bool pinned;                // Cannot be evicted
} VMMPageEntry;

// VMM Flags
#define VMM_FLAG_READ       0x01
#define VMM_FLAG_WRITE      0x02
#define VMM_FLAG_EXEC       0x04
#define VMM_FLAG_SHARED     0x08
#define VMM_FLAG_PINNED     0x10

// ============================================================================
// VMM State
// ============================================================================

typedef struct VMMState {
    GPUHotRuntime* runtime;   // Owning runtime for allocations
    VMMPageEntry pages[VMM_MAX_PAGES];
    uint32_t num_pages;
    uint32_t resident_pages;
    uint32_t swapped_pages;

    // Swap regions (host memory backing store)
    void* swap_regions[VMM_MAX_SWAP_REGIONS];
    size_t swap_region_sizes[VMM_MAX_SWAP_REGIONS];
    uint32_t num_swap_regions;
    size_t total_swap_size;
    size_t used_swap_size;

    // Statistics
    uint64_t page_faults;
    uint64_t swap_ins;
    uint64_t swap_outs;
    uint64_t evictions;
} VMMState;

// ============================================================================
// VFS Inode Structure
// ============================================================================

typedef enum VFSNodeType {
    VFS_NODE_FILE,
    VFS_NODE_DIRECTORY,
    VFS_NODE_TENSOR,
    VFS_NODE_STREAM,
    VFS_NODE_DEVICE
} VFSNodeType;

typedef struct VFSInode {
    uint32_t inode_id;
    char name[GPU_HOT_MAX_NAME_LEN];
    char path[VFS_MAX_PATH_LEN];
    VFSNodeType type;

    // Data pointer (GPU memory for tensors)
    void* data;
    size_t size;

    // Tensor metadata (if type == VFS_NODE_TENSOR)
    int dims;
    int shape[8];
    int dtype;  // 0=f32, 1=f16, 2=i32, 3=i8

    // Tree structure
    struct VFSInode* parent;
    struct VFSInode* first_child;
    struct VFSInode* next_sibling;

    // Timestamps
    uint64_t created_at;
    uint64_t modified_at;
    uint64_t accessed_at;

    // Permissions
    uint32_t mode;
    uint32_t uid;
    uint32_t gid;

    bool active;
} VFSInode;

// ============================================================================
// VFS Open File Handle
// ============================================================================

typedef struct VFSFileHandle {
    uint32_t fd;
    VFSInode* inode;
    size_t offset;
    uint32_t flags;
    bool is_open;
} VFSFileHandle;

// VFS Open Flags
#define VFS_O_RDONLY    0x01
#define VFS_O_WRONLY    0x02
#define VFS_O_RDWR      0x03
#define VFS_O_CREAT     0x10
#define VFS_O_TRUNC     0x20
#define VFS_O_APPEND    0x40

// ============================================================================
// VFS State
// ============================================================================

typedef struct VFSState {
    GPUHotRuntime* runtime;   // Owning runtime for allocations
    VFSInode inodes[VFS_MAX_NODES];
    uint32_t num_inodes;
    VFSInode* root;

    VFSFileHandle open_files[VFS_MAX_OPEN_FILES];
    uint32_t num_open_files;
    uint32_t next_fd;

    // Statistics
    uint64_t reads;
    uint64_t writes;
    uint64_t opens;
    uint64_t closes;
} VFSState;

// ============================================================================
// PTX-OS System State (Shared between Host and GPU)
// ============================================================================

typedef struct PTXSystemState {
    // Authentication
    uint32_t auth_token;

    // Process tracking
    volatile int active_processes;
    volatile int active_tasks;
    volatile int max_priority_active;

    // Memory tracking
    volatile size_t total_vram_used;

    // Watchdog
    volatile bool watchdog_alert;
    volatile bool kernel_running;
    volatile bool shutdown_requested;

    // Priority scheduling
    volatile int active_priority_level;

    // Signal handling
    volatile uint64_t signal_mask;
    volatile uint32_t interrupt_cnt;

    // Task queue
    PTXTaskQueue queue;

    // Operation counter
    volatile uint64_t total_ops;

    // VFS node count
    volatile uint32_t fs_node_count;

    // VMM state pointer (GPU accessible)
    VMMState* vmm;

    // VFS state pointer (GPU accessible)
    VFSState* vfs;
} PTXSystemState;

// ============================================================================
// Runtime Statistics
// ============================================================================

typedef struct GPUHotStats {
    size_t vram_allocated;
    size_t vram_used;
    size_t vram_free;
    float gpu_utilization;      // SM utilization (%)
    float mem_utilization;      // Memory controller utilization (%)
    uint32_t sm_clock_mhz;      // Current SM clock
    uint32_t mem_clock_mhz;     // Current memory clock
    float power_w;              // Instant power draw
    int32_t temperature_c;      // GPU temperature
    bool nvml_valid;            // True when NVML hardware poll succeeded
    float hw_ops_per_sec;       // CUPTI counter rate (operations/sec)
    float gflops_total;         // True counter-based GFLOPS when CUPTI is active
    bool cupti_valid;           // True when CUPTI counters are active
    int active_streams;         // Count of streams with pending GPU work
    int stream_poll_count;      // How many streams were polled (min(num_streams, 1024))
    uint8_t stream_busy[128];   // Bitmask — bit i set = stream i has pending work (up to 1024)
    int registered_kernels;
    int shm_count;
    uint64_t total_ops;
    float avg_latency_us;
    bool watchdog_tripped;
} GPUHotStats;

// ============================================================================
// TLSF Pool Statistics
// ============================================================================

typedef struct TLSFPoolStats {
    size_t total_pool_size;
    size_t allocated_bytes;
    size_t free_bytes;
    size_t peak_allocated;
    size_t largest_free_block;
    size_t smallest_free_block;

    uint32_t total_blocks;
    uint32_t free_blocks;
    uint32_t allocated_blocks;

    uint32_t fallback_count;
    float utilization_percent;
    float fragmentation_ratio;
    float external_fragmentation;

    uint32_t free_list_counts[34];  // Must match TLSF_FL_INDEX_MAX

    // Hash table stats
    uint32_t hash_collisions;
    uint32_t max_chain_length;
    float avg_chain_length;

    // Operation counts
    uint64_t total_allocations;
    uint64_t total_frees;
    uint32_t total_splits;
    uint32_t total_merges;

    bool is_healthy;
    bool needs_defrag;
} TLSFPoolStats;

// ============================================================================
// Per-Owner Memory Tracking
// ============================================================================

#define TLSF_MAX_OWNERS 64

typedef struct TLSFOwnerUsage {
    uint32_t owner_id;
    size_t allocated_bytes;
    uint32_t block_count;
} TLSFOwnerUsage;

typedef struct TLSFOwnerStats {
    TLSFOwnerUsage owners[TLSF_MAX_OWNERS];
    uint32_t num_owners;
} TLSFOwnerStats;

// ============================================================================
// Allocation Event Ring Buffer
// ============================================================================

#define TLSF_EVENT_RING_SIZE 256

typedef struct TLSFAllocEvent {
    uint64_t timestamp;
    size_t size;
    uint32_t owner_id;
    uint32_t alloc_id;
    uint8_t event_type;   // 0=alloc, 1=free, 2=realloc
    uint8_t _pad[3];
} TLSFAllocEvent;

typedef struct TLSFEventRing {
    TLSFAllocEvent events[TLSF_EVENT_RING_SIZE];
    uint32_t head;    // next write position (mod RING_SIZE)
    uint32_t count;   // total events ever written
} TLSFEventRing;

// ============================================================================
// TLSF Health Report
// ============================================================================

typedef struct TLSFHealthReport {
    bool is_valid;
    bool has_memory_leaks;
    bool has_corrupted_blocks;
    bool has_broken_chains;
    bool has_hash_errors;
    int error_count;
    char error_messages[16][256];
} TLSFHealthReport;

// ============================================================================
// System Snapshot (for debugging)
// ============================================================================

typedef struct GPUHotSystemSnapshot {
    uint64_t total_ops;
    int active_processes;
    int active_tasks;
    int max_priority_active;
    size_t total_vram_used;
    bool watchdog_alert;
    bool kernel_running;
    bool shutdown_requested;
    int active_priority_level;
    uint64_t signal_mask;
    uint32_t interrupt_cnt;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_lock;
} GPUHotSystemSnapshot;

// ============================================================================
// Runtime Configuration
// ============================================================================

typedef struct GPUHotConfig {
    // Pool sizing options (choose one method)
    float pool_fraction;        // Fraction of available VRAM to use (0.0-1.0)
                                // Set to 0 to use fixed_pool_size instead

    size_t fixed_pool_size;     // Fixed pool size in bytes (used if pool_fraction == 0)
                                // Set to 0 for auto-sizing with pool_fraction

    size_t min_pool_size;       // Minimum pool size in bytes (default: 256MB)
    size_t max_pool_size;       // Maximum pool size in bytes (0 = no limit)

    // Safety margins
    size_t reserve_vram;        // VRAM to reserve for CUDA runtime (default: 256MB)

    // Allocator options
    bool enable_leak_detection; // Enable memory leak detection (default: true)
    bool enable_pool_health;    // Enable pool health monitoring (default: true)
    float warning_threshold;    // Utilization % to trigger warnings (default: 0.9)

    // Behavior flags
    bool force_daemon_mode;     // Force daemon mode even if not first process
    bool quiet_init;            // Suppress initialization messages

    // Stream configuration
    unsigned int max_streams;   // Maximum number of CUDA streams (default: 16, max: GPU_HOT_MAX_STREAMS)

    // Single-pool strict mode: deny competing pool init from child processes
    bool single_pool_strict;    // If true, refuse pool init when PTX_DAEMON_CLIENT=1
} GPUHotConfig;

// Get default configuration
GPUHotConfig gpu_hot_default_config(void);

// ============================================================================
// Core Runtime API
// ============================================================================

GPUHotRuntime* gpu_hot_init(int device_id, const char* token);
GPUHotRuntime* gpu_hot_init_with_config(int device_id, const char* token, const GPUHotConfig* config);
void gpu_hot_shutdown(GPUHotRuntime* runtime);
void gpu_hot_keepalive(GPUHotRuntime* runtime);

// ============================================================================
// Memory Allocation API
// ============================================================================

void* gpu_hot_alloc(GPUHotRuntime* runtime, size_t size);
void gpu_hot_free(GPUHotRuntime* runtime, void* ptr);
void* gpu_hot_alloc_async(GPUHotRuntime* runtime, size_t size, cudaStream_t stream);
void gpu_hot_free_async(GPUHotRuntime* runtime, void* ptr, cudaStream_t stream);
void gpu_hot_poll_deferred(GPUHotRuntime* runtime, int max_drain);
bool gpu_hot_can_allocate(GPUHotRuntime* runtime, size_t size);
size_t gpu_hot_get_max_allocatable(GPUHotRuntime* runtime);
bool gpu_hot_owns_ptr(GPUHotRuntime* runtime, void* ptr);

// Per-owner allocation API
void* gpu_hot_alloc_owned(GPUHotRuntime* runtime, size_t size, uint32_t owner_id);
void gpu_hot_free_owner(GPUHotRuntime* runtime, uint32_t owner_id);
void gpu_hot_get_owner_stats(GPUHotRuntime* runtime, TLSFOwnerStats* stats);

// Allocation event log
void gpu_hot_get_alloc_events(GPUHotRuntime* runtime, TLSFEventRing* ring_out);

// ============================================================================
// CUDA Allocation Hook API (intercepts cuMemAlloc/cudaMalloc)
// ============================================================================

// Initialize hooks - call after gpu_hot_init to enable TLSF for ALL CUDA allocations
void ptx_hook_init(GPUHotRuntime* runtime, bool verbose);

// Disable hooks (for cleanup)
void ptx_hook_disable(void);

// Check if a pointer was allocated by the TLSF pool
bool ptx_hook_owns_ptr(void* ptr);

// ============================================================================
// Kernel Registration & Launch API
// ============================================================================

int gpu_hot_register_kernel(GPUHotRuntime* runtime, void* kernel_func,
                            dim3 grid, dim3 block, size_t shared_mem);
void gpu_hot_launch_kernel(GPUHotRuntime* runtime, int kernel_id, void** args);

// ============================================================================
// CUDA Graph API
// ============================================================================

int gpu_hot_begin_capture(GPUHotRuntime* runtime, int stream_id, const char* graph_name);
int gpu_hot_end_capture(GPUHotRuntime* runtime, int stream_id);
void gpu_hot_launch_graph(GPUHotRuntime* runtime, int graph_id, cudaStream_t stream);
void gpu_hot_destroy_graph(GPUHotRuntime* runtime, int graph_id);

// ============================================================================
// Stream API
// ============================================================================

cudaStream_t gpu_hot_get_stream(GPUHotRuntime* runtime, int stream_id);
cudaStream_t gpu_hot_get_priority_stream(GPUHotRuntime* runtime, int priority);
void gpu_hot_sync_all(GPUHotRuntime* runtime);

// ============================================================================
// Statistics API
// ============================================================================

void gpu_hot_get_stats(GPUHotRuntime* runtime, GPUHotStats* stats);
void gpu_hot_get_tlsf_stats(GPUHotRuntime* runtime, TLSFPoolStats* stats);
void gpu_hot_validate_tlsf_pool(GPUHotRuntime* runtime, TLSFHealthReport* report);
void gpu_hot_print_pool_map(GPUHotRuntime* runtime);

// ============================================================================
// TLSF Pool Management
// ============================================================================

void gpu_hot_defragment_pool(GPUHotRuntime* runtime);
void gpu_hot_set_warning_threshold(GPUHotRuntime* runtime, float threshold_percent);
void gpu_hot_set_auto_defrag(GPUHotRuntime* runtime, bool enable);

// ============================================================================
// Shared Memory / IPC API
// ============================================================================

void* gpu_hot_shm_alloc(GPUHotRuntime* runtime, const char* name, size_t size);
void* gpu_hot_shm_open(GPUHotRuntime* runtime, const char* name);
void gpu_hot_shm_close(GPUHotRuntime* runtime, void* ptr);
void gpu_hot_shm_unlink(GPUHotRuntime* runtime, const char* name);
bool gpu_hot_get_registry_entry(GPUHotRuntime* runtime, int index,
                                char* name_out, size_t* size_out,
                                bool* active_out, unsigned long long* created_out);

// ============================================================================
// Watchdog API
// ============================================================================

void gpu_hot_set_watchdog(GPUHotRuntime* runtime, int timeout_ms);
bool gpu_hot_check_watchdog(GPUHotRuntime* runtime);
void gpu_hot_reset_watchdog(GPUHotRuntime* runtime);

// ============================================================================
// System State API
// ============================================================================

PTXSystemState* gpu_hot_get_system_state(GPUHotRuntime* runtime);
PTXSystemState* gpu_hot_get_host_system_state(GPUHotRuntime* runtime);
void gpu_hot_reset_system_state(GPUHotRuntime* runtime);
void gpu_hot_flush_task_queue(GPUHotRuntime* runtime);
void gpu_hot_clear_signal_mask(GPUHotRuntime* runtime);
void gpu_hot_get_system_snapshot(GPUHotRuntime* runtime, GPUHotSystemSnapshot* snapshot);

// ============================================================================
// Context Export API
// ============================================================================

// Returns the captured primary CUcontext as void* (avoids requiring cuda.h driver header)
void* gpu_hot_get_context(GPUHotRuntime* runtime);

// Exports the CUcontext pointer as PTX_CONTEXT_PTR environment variable
void  gpu_hot_export_context(GPUHotRuntime* runtime);

// ============================================================================
// PTX-OS Kernel Boot
// ============================================================================

void ptx_os_boot_persistent_kernel(GPUHotRuntime* runtime);
int ptx_os_submit_task(GPUHotRuntime* runtime, uint32_t opcode, uint32_t priority, void* args[PTX_MAX_TASK_ARGS]);

// ============================================================================
// CUDA Dynamic Parallelism — Device-Side Self-Scheduling
// ============================================================================

// Run a recursive CDP test: the persistent kernel dispatches a child kernel
// which re-enqueues itself back into the task queue, creating a fully
// autonomous GPU-side execution loop.  Returns the number of iterations
// completed, or a negative error code.
int ptx_cdp_test_recursive(GPUHotRuntime* runtime, int iterations);

// ============================================================================
// VMM API
// ============================================================================

VMMState* vmm_init(GPUHotRuntime* runtime, size_t swap_size);
void vmm_shutdown(VMMState* vmm);
void* vmm_alloc_page(VMMState* vmm, uint32_t flags);
void vmm_free_page(VMMState* vmm, void* addr);
int vmm_swap_out(VMMState* vmm, void* addr);
int vmm_swap_in(VMMState* vmm, void* addr);
void vmm_pin_page(VMMState* vmm, void* addr);
void vmm_unpin_page(VMMState* vmm, void* addr);
void vmm_get_stats(VMMState* vmm, uint64_t* resident, uint64_t* swapped,
                   uint64_t* faults, uint64_t* evictions);
void gpu_hot_set_vmm(GPUHotRuntime* runtime, VMMState* vmm);
int vmm_evict_for_alloc(VMMState* vmm, size_t needed_size);

// ============================================================================
// VFS API
// ============================================================================

VFSState* vfs_init(GPUHotRuntime* runtime);
void vfs_shutdown(VFSState* vfs);

// Path operations
VFSInode* vfs_lookup(VFSState* vfs, const char* path);
int vfs_mkdir(VFSState* vfs, const char* path, uint32_t mode);
int vfs_rmdir(VFSState* vfs, const char* path);
int vfs_unlink(VFSState* vfs, const char* path);

// File operations
int vfs_open(VFSState* vfs, const char* path, uint32_t flags);
int vfs_close(VFSState* vfs, int fd);
ssize_t vfs_read(VFSState* vfs, int fd, void* buf, size_t count);
ssize_t vfs_write(VFSState* vfs, int fd, const void* buf, size_t count);
int vfs_seek(VFSState* vfs, int fd, size_t offset, int whence);

// Tensor operations
int vfs_create_tensor(VFSState* vfs, const char* path, int* shape, int dims, int dtype);
void* vfs_mmap_tensor(VFSState* vfs, const char* path);
int vfs_sync_tensor(VFSState* vfs, const char* path);

// Directory listing
int vfs_readdir(VFSState* vfs, const char* path, char** names, int max_entries);
int vfs_stat(VFSState* vfs, const char* path, VFSInode* stat_out);

#ifdef __cplusplus
}
#endif

#endif // GPU_HOT_RUNTIME_H
