/*
 * Elastic GPU Memory Pool
 *
 * Two-tier memory system:
 * - Tier 1: TLSF pool for mutable allocations (KV cache, activations)
 * - Tier 2: Immutable weight region that grows dynamically during model load
 *
 * The immutable region polls available VRAM and expands as needed.
 */

#ifndef ELASTIC_POOL_CUH
#define ELASTIC_POOL_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex>
#include <vector>
#include <atomic>

// Forward declare TLSF
struct PTXTLSFAllocator;

// Memory region types
enum class RegionType : uint8_t {
    MUTABLE = 0,    // TLSF-managed, for KV cache/activations
    IMMUTABLE = 1,  // Append-only, for model weights
};

// Immutable allocation record (no free-list needed)
struct ImmutableBlock {
    void* ptr;
    size_t size;
    uint32_t layer_id;      // Which layer this belongs to
    uint32_t tensor_id;     // Tensor within layer
    bool is_quantized;      // True if storing raw quantized data
};

// Statistics for the elastic pool
struct ElasticPoolStats {
    // TLSF (mutable) region
    size_t tlsf_total;
    size_t tlsf_used;
    size_t tlsf_free;

    // Immutable region
    size_t immutable_total;
    size_t immutable_used;
    size_t immutable_committed;  // Actually allocated from CUDA
    uint32_t immutable_blocks;

    // System
    size_t vram_total;
    size_t vram_available;
    size_t vram_reserved;        // Reserved for CUDA runtime

    // Performance
    float compression_ratio;     // Avg compression of immutable weights
    uint64_t total_loads;
    float avg_load_time_us;
};

// Configuration for elastic pool
struct ElasticPoolConfig {
    // TLSF pool sizing
    size_t tlsf_initial_size;    // Initial TLSF pool (default: 512MB)
    size_t tlsf_max_size;        // Max TLSF can grow to (default: 2GB)

    // Immutable region
    size_t immutable_chunk_size; // Grow immutable region in chunks (default: 256MB)
    size_t immutable_reserve;    // Keep this much free for safety (default: 256MB)

    // System
    size_t cuda_reserve;         // Reserve for CUDA runtime (default: 256MB)
    bool verbose;

    static ElasticPoolConfig defaults() {
        return {
            .tlsf_initial_size = 512 * 1024 * 1024,      // 512MB
            .tlsf_max_size = 2ULL * 1024 * 1024 * 1024,  // 2GB
            .immutable_chunk_size = 256 * 1024 * 1024,   // 256MB
            .immutable_reserve = 256 * 1024 * 1024,      // 256MB
            .cuda_reserve = 256 * 1024 * 1024,           // 256MB
            .verbose = false,
        };
    }
};

/*
 * ElasticPool - Dynamic two-tier GPU memory manager
 */
class ElasticPool {
public:
    ElasticPool(int device_id, ElasticPoolConfig config = ElasticPoolConfig::defaults());
    ~ElasticPool();

    // ========================================================================
    // Mutable allocations (via TLSF)
    // ========================================================================

    void* alloc_mutable(size_t size);
    void free_mutable(void* ptr);
    bool owns_mutable(void* ptr) const;

    // ========================================================================
    // Immutable allocations (append-only weight storage)
    // ========================================================================

    // Allocate space for a weight tensor (grows pool if needed)
    void* alloc_immutable(size_t size, uint32_t layer_id, uint32_t tensor_id, bool is_quantized = true);

    // Load quantized data directly into immutable region
    void* load_quantized_weight(const void* host_data, size_t size,
                                 uint32_t layer_id, uint32_t tensor_id);

    // Get pointer to an immutable block
    void* get_immutable(uint32_t layer_id, uint32_t tensor_id) const;

    // Check if we can fit more immutable data
    bool can_grow_immutable(size_t additional_size) const;

    // ========================================================================
    // Memory management
    // ========================================================================

    // Poll available VRAM and expand if possible
    size_t poll_and_expand();

    // Defragment TLSF region (immutable region doesn't fragment)
    void defragment_mutable();

    // Get current stats
    ElasticPoolStats get_stats() const;

    // Print memory map
    void print_memory_map() const;

    // ========================================================================
    // Accessors
    // ========================================================================

    int device_id() const { return device_id_; }
    PTXTLSFAllocator* tlsf() { return tlsf_; }

private:
    int device_id_;
    ElasticPoolConfig config_;

    // TLSF for mutable allocations
    PTXTLSFAllocator* tlsf_;
    void* tlsf_base_;
    size_t tlsf_size_;

    // Immutable region - contiguous chunks
    struct ImmutableChunk {
        void* base;
        size_t size;
        size_t used;
    };
    std::vector<ImmutableChunk> immutable_chunks_;
    std::vector<ImmutableBlock> immutable_blocks_;
    size_t immutable_total_used_;

    // Thread safety
    mutable std::mutex mutex_;

    // Stats
    std::atomic<uint64_t> total_loads_{0};
    std::atomic<uint64_t> total_load_time_us_{0};

    // Internal helpers
    bool expand_immutable_region(size_t min_size);
    size_t query_available_vram() const;
    void* find_immutable_space(size_t size);
};

// ============================================================================
// C API for FFI
// ============================================================================

extern "C" {
    // Create/destroy
    ElasticPool* elastic_pool_create(int device_id, const ElasticPoolConfig* config);
    void elastic_pool_destroy(ElasticPool* pool);

    // Mutable allocations
    void* elastic_pool_alloc_mutable(ElasticPool* pool, size_t size);
    void elastic_pool_free_mutable(ElasticPool* pool, void* ptr);

    // Immutable allocations
    void* elastic_pool_alloc_immutable(ElasticPool* pool, size_t size,
                                        uint32_t layer_id, uint32_t tensor_id, bool is_quantized);
    void* elastic_pool_load_weight(ElasticPool* pool, const void* host_data, size_t size,
                                    uint32_t layer_id, uint32_t tensor_id);
    void* elastic_pool_get_immutable(ElasticPool* pool, uint32_t layer_id, uint32_t tensor_id);

    // Management
    size_t elastic_pool_poll_expand(ElasticPool* pool);
    void elastic_pool_defragment(ElasticPool* pool);
    void elastic_pool_get_stats(ElasticPool* pool, ElasticPoolStats* stats);
    void elastic_pool_print_map(ElasticPool* pool);

    // Queries
    bool elastic_pool_can_grow(ElasticPool* pool, size_t additional);
    bool elastic_pool_owns_mutable(ElasticPool* pool, void* ptr);
}

#endif // ELASTIC_POOL_CUH
