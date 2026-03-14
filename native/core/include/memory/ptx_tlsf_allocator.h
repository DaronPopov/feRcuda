/*
 * PTX-OS Enhanced TLSF Memory Allocator - Header
 *
 * Two-Level Segregated Fit allocator with:
 * - O(1) allocation/deallocation
 * - O(1) block lookup via hash table
 * - Memory leak detection
 * - Comprehensive diagnostics
 */

#ifndef PTX_TLSF_ALLOCATOR_H
#define PTX_TLSF_ALLOCATOR_H

// Include the main runtime header for shared type definitions
#include "gpu/gpu_hot_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Additional Configuration
// ============================================================================

#ifndef TLSF_HASH_TABLE_SIZE
#define TLSF_HASH_TABLE_SIZE    4096
#endif

#ifndef TLSF_BLOCK_MAGIC
#define TLSF_BLOCK_MAGIC        0x544C5346  // "TLSF"
#endif

// ============================================================================
// Hash Bucket (not in main header)
// ============================================================================

typedef struct TLSFHashBucket {
    TLSFBlock* head;
    uint32_t count;
} TLSFHashBucket;

// ============================================================================
// Allocation Tracking Info (for leak detection)
// ============================================================================

typedef struct TLSFAllocationInfo {
    void* device_ptr;
    size_t size;
    const char* file;
    int line;
    uint64_t timestamp;
    uint32_t alloc_id;
} TLSFAllocationInfo;

// ============================================================================
// TLSF Allocator Structure
// ============================================================================

typedef struct PTXTLSFAllocator {
    // GPU memory pool
    void* vram_pool;
    size_t vram_pool_size;
    bool owns_pool;             // True if we allocated the pool, false if external

    // Segregated free lists
    TLSFBlock* free_lists[TLSF_FL_INDEX_MAX][TLSF_SL_INDEX_COUNT];

    // Physical block chain
    TLSFBlock* first_block;
    TLSFBlock* last_block;

    // Hash table for O(1) lookup
    TLSFHashBucket hash_table[TLSF_HASH_TABLE_SIZE];

    // Statistics
    TLSFPoolStats stats;

    // Configuration
    float warning_threshold;
    float warning_step;             // Minimum utilization delta to re-warn (fraction)
    uint64_t warning_interval_us;   // Minimum time between warnings
    uint64_t warning_last_ts;       // Last warning timestamp (us)
    float warning_last_utilization; // Last warned utilization
    bool warnings_enabled;
    bool auto_defrag_enabled;
    bool debug_mode;

    // Allocation ID counter
    uint32_t next_alloc_id;

    // Per-owner memory tracking
    TLSFOwnerStats owner_stats;

    // Allocation event ring buffer
    TLSFEventRing event_ring;
} PTXTLSFAllocator;

// ============================================================================
// Core API
// ============================================================================

PTXTLSFAllocator* ptx_tlsf_create(size_t pool_size, bool debug_mode);
PTXTLSFAllocator* ptx_tlsf_create_from_pool(void* pool, size_t pool_size, bool debug_mode);
void ptx_tlsf_destroy(PTXTLSFAllocator* allocator);

// ============================================================================
// Allocation API
// ============================================================================

void* ptx_tlsf_alloc(PTXTLSFAllocator* allocator, size_t size);
void* ptx_tlsf_alloc_debug(PTXTLSFAllocator* allocator, size_t size,
                            const char* file, int line);
void ptx_tlsf_free(PTXTLSFAllocator* allocator, void* ptr);
void* ptx_tlsf_realloc(PTXTLSFAllocator* allocator, void* ptr, size_t new_size);

// Per-owner allocation API
void* ptx_tlsf_alloc_owned(PTXTLSFAllocator* allocator, size_t size, uint32_t owner_id);
void ptx_tlsf_free_owner(PTXTLSFAllocator* allocator, uint32_t owner_id);
void ptx_tlsf_get_owner_stats(PTXTLSFAllocator* allocator, TLSFOwnerStats* stats);

// Allocation event API
void ptx_tlsf_get_events(PTXTLSFAllocator* allocator, TLSFEventRing* ring_out);

// Debug allocation macro
#ifdef TLSF_DEBUG
#define TLSF_ALLOC(alloc, size) ptx_tlsf_alloc_debug(alloc, size, __FILE__, __LINE__)
#else
#define TLSF_ALLOC(alloc, size) ptx_tlsf_alloc(alloc, size)
#endif

// ============================================================================
// Diagnostics API
// ============================================================================

void ptx_tlsf_get_stats(PTXTLSFAllocator* allocator, TLSFPoolStats* stats);
void ptx_tlsf_validate(PTXTLSFAllocator* allocator, TLSFHealthReport* report);
bool ptx_tlsf_can_allocate(PTXTLSFAllocator* allocator, size_t size);
size_t ptx_tlsf_get_max_allocatable(PTXTLSFAllocator* allocator);
bool ptx_tlsf_owns_ptr(PTXTLSFAllocator* allocator, void* ptr);

// ============================================================================
// Debug API
// ============================================================================

void ptx_tlsf_print_pool_map(PTXTLSFAllocator* allocator);
void ptx_tlsf_print_allocations(PTXTLSFAllocator* allocator);

// ============================================================================
// Maintenance API
// ============================================================================

void ptx_tlsf_defragment(PTXTLSFAllocator* allocator);
void ptx_tlsf_compact(PTXTLSFAllocator* allocator);
void ptx_tlsf_set_warning_threshold(PTXTLSFAllocator* allocator, float threshold);
void ptx_tlsf_set_warning_step(PTXTLSFAllocator* allocator, float step);
void ptx_tlsf_set_warning_interval_ms(PTXTLSFAllocator* allocator, uint64_t interval_ms);
void ptx_tlsf_set_warnings_enabled(PTXTLSFAllocator* allocator, bool enabled);
void ptx_tlsf_set_auto_defrag(PTXTLSFAllocator* allocator, bool enable);
void ptx_tlsf_reset_stats(PTXTLSFAllocator* allocator);

// ============================================================================
// Advanced API
// ============================================================================

void ptx_tlsf_pin_block(PTXTLSFAllocator* allocator, void* ptr);
void ptx_tlsf_unpin_block(PTXTLSFAllocator* allocator, void* ptr);
bool ptx_tlsf_get_block_info(PTXTLSFAllocator* allocator, void* ptr,
                              size_t* size_out, bool* is_free_out);
bool ptx_tlsf_expand_pool(PTXTLSFAllocator* allocator, size_t additional_size);
bool ptx_tlsf_shrink_pool(PTXTLSFAllocator* allocator);

#ifdef __cplusplus
}
#endif

#endif // PTX_TLSF_ALLOCATOR_H
